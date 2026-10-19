"""Colour/shape extraction with separate course ink and forbidden masks.

The output is a candidate interpretation. Grey buildings, blue water, dark
green and detected purple hatching are hard obstacles; ambiguous markings
remain visible for review. Source pixels are never overwritten.
"""

from pathlib import Path
import struct

import cv2
import numpy as np

from .reference import polygon_properties
from .symbols import classify_course_symbols, course_ink_mask
from .overprint import annotation_mask, restore
from .protocol import TERRAIN_SCHEMA


def read_image(path):
    # Check dimensions BEFORE decompression. The existing metadata reader
    # reads only PNG/JPEG headers and also works for uploaded .img files.
    from tools.inspect_inputs import image_header
    try:
        header = image_header(Path(path))
    except (ValueError, OSError, KeyError, struct.error) as exc:
        raise ValueError('无法读取地图，支持 JPEG/PNG') from exc
    if header['width_px'] * header['height_px'] > 30_000_000:
        raise ValueError('地图不能超过三千万像素')
    raw = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > 30_000_000:
        raise ValueError("无法读取地图，支持不超过三千万像素的 JPEG/PNG")
    return image


def colour_masks(image):
    b, g, r = (channel.astype(np.int16) for channel in cv2.split(image))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    purple = ((r-g > 22) & (b-g > 18) & (r > 85) & (b > 75)).astype(np.uint8)*255
    water = (((b-r > 24) & (g-r > 15) & (g > 95) & (h > 78) & (h < 111)) & (purple == 0)).astype(np.uint8)*255
    # The supplied roof fill straddles V=179 after JPEG compression (and
    # after removing purple overprint). A cutoff at that value breaks a
    # solid roof into traversable speckles. Keep the medium grey palette;
    # light grey canopies/passages (about V=215) remain separate.
    building = ((s < 48) & (v > 58) & (v < 196)).astype(np.uint8)*255
    green = ((g-r > 9) & (g-b > 8) & (h >= 32) & (h <= 90)).astype(np.uint8)*255
    dark_green = ((green > 0) & (v < 159) & (s > 80)).astype(np.uint8)*255
    black = ((v < 108) & (s < 115)).astype(np.uint8)*255
    # Olive green is the ISSprOM settlement/out-of-bounds fill, unlike yellow
    # open land or light green vegetation. Bright red custom course ink is
    # unknown, never treated as a new road.
    olive = ((h >= 22) & (h <= 39) & (s > 62) & (v < 216) & (v > 90)).astype(np.uint8)*255
    return dict(purple=purple, water=water, building=building, green=green,
                dark_green=dark_green, black=black, olive=olive,
                course_text=cv2.bitwise_and(purple, (s >= 100).astype(np.uint8)*255))


def water_obstacles(mask):
    """Keep narrow water except very long straight blue north-guide strokes.

    Opening is used only to recover solid water under a guide, not to delete
    all narrow waterways from the obstacle map.
    """
    height, width = mask.shape
    native = max(5, int(round(np.sqrt(height*width)/420)) | 1)
    bodies = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((native, native), np.uint8))
    guides = np.zeros_like(mask)
    lines = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=max(60, int(height*.3)),
                            minLineLength=max(150, int(height*.62)), maxLineGap=int(height*.12))
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2-x1) < abs(y2-y1)*.12:
                cv2.line(guides, (int(x1), int(y1)), (int(x2), int(y2)), 255, native)
    # Wide blue bodies survive even where a magnetic-north line crosses them.
    return cv2.bitwise_or(bodies, cv2.bitwise_and(mask, cv2.bitwise_not(guides)))


def region_features(mask, kind, image_area, close=3):
    cleaned = mask
    if kind == 'water':
        # Remove long north guides explicitly. Opening before closing would
        # also erase the thin blue strips BETWEEN purple forbidden hatching
        # and lose the actual lake as a geographic landmark.
        cleaned = water_obstacles(mask)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not max(90, image_area*0.00004) < area < image_area*0.10:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.006*perimeter, True).reshape(-1, 2)
        properties = polygon_properties(polygon)
        if properties and properties['aspect'] < 9:
            features.append({'kind': kind, **properties})
    return sorted(features, key=lambda f: -f['area'])


def detect_controls(purple):
    height, width = purple.shape
    scale = min(1., 1800/max(height, width))
    mask = cv2.resize(purple, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = (mask > 75).astype(np.uint8)*255
    smoothed = cv2.GaussianBlur(mask, (3, 3), 0.8)
    distance = cv2.distanceTransform(255-mask, cv2.DIST_L2, 3)
    min_radius = max(6, round(max(mask.shape)*0.011))
    max_radius = max(min_radius+2, round(max(mask.shape)*0.028))
    circles = cv2.HoughCircles(smoothed, cv2.HOUGH_GRADIENT, dp=1,
                              minDist=min_radius*1.6, param1=80, param2=13,
                              minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    candidates = []
    theta = np.linspace(0, 2*np.pi, 120, endpoint=False)
    for x, y, radius in circles[0]:
        if x-radius < 0 or y-radius < 0 or x+radius >= mask.shape[1] or y+radius >= mask.shape[0]:
            continue
        # Refine the Hough radius and centre on the original ink. A one-pixel
        # error can suppress a genuine thin circle at this working scale.
        best_ring = (-1., float(x), float(y), float(radius))
        for delta_x in (-4., -2., 0., 2., 4.):
            for delta_y in (-4., -2., 0., 2., 4.):
                for delta_r in (-3., -2., -1., 0., 1., 2., 3.):
                    ring_x = np.rint(x+delta_x+(radius+delta_r)*np.cos(theta)).astype(int)
                    ring_y = np.rint(y+delta_y+(radius+delta_r)*np.sin(theta)).astype(int)
                    if ring_x.min() < 0 or ring_y.min() < 0 or ring_x.max() >= mask.shape[1] or ring_y.max() >= mask.shape[0]:
                        continue
                    value = float(np.mean(mask[ring_y, ring_x] > 0))
                    if value > best_ring[0]:
                        best_ring = (value, float(x+delta_x), float(y+delta_y), float(radius+delta_r))
        _, x, y, radius = best_ring
        offsets = np.linspace(-2, 2, 5)
        samples = []
        for delta in offsets:
            xx = np.clip(np.rint(x+(radius+delta)*np.cos(theta)).astype(int), 0, mask.shape[1]-1)
            yy = np.clip(np.rint(y+(radius+delta)*np.sin(theta)).astype(int), 0, mask.shape[0]-1)
            samples.append(distance[yy, xx])
        covered = np.min(samples, axis=0) < 1.2
        coverage = float(covered.mean())
        quadrants = [chunk.mean() for chunk in np.array_split(covered, 4)]
        inner_r = max(2, round(radius*.55))
        inner = mask[max(0, round(y)-inner_r):round(y)+inner_r,
                     max(0, round(x)-inner_r):round(x)+inner_r]
        density = np.mean(inner > 0) if inner.size else 1
        ring_samples = []
        for delta in (-1, 0, 1):
            xx = np.rint(x+(radius+delta)*np.cos(theta)).astype(int)
            yy = np.rint(y+(radius+delta)*np.sin(theta)).astype(int)
            ring_samples.append(mask[yy, xx] > 0)
        ring_density = float(np.mean(ring_samples))
        if coverage < .94 or min(quadrants) < .80 or density > .32 or ring_density-density < .25:
            continue
        candidates.append({'pixel': [float(x/scale), float(y/scale)],
                           'radiusPx': float(radius/scale), 'confidence': round(coverage, 3),
                           'ringDensity': round(ring_density, 3),
                           'state': 'candidate', 'labelSource': 'unread', 'label': ''})
    # Nested rings at the finish are ONE checkpoint. Keep the stronger fit.
    accepted = []
    for item in sorted(candidates, key=lambda c: -c['confidence']):
        if any(np.linalg.norm(np.array(item['pixel'])-other['pixel']) < max(item['radiusPx'], other['radiusPx'])*.8
               for other in accepted):
            continue
        accepted.append(item)
    accepted.sort(key=lambda c: (c['pixel'][1], c['pixel'][0]))
    for index, control in enumerate(accepted):
        control['id'] = f'c{index+1}'
        control['label'] = f'待核{index+1}'
        control['required'] = True
        control['serviceBaseSec'] = 5
        control['servicePerPersonSec'] = 1
    return accepted


def purple_restrictions(purple, controls, annotations=None):
    """Hatching has sustained local density; a circle or course line does not.

    Course-circle interiors are deliberately not erased: forbidden hatching
    inside a control circle must still prohibit passage.
    """
    ink = cv2.bitwise_and(purple, cv2.bitwise_not(annotations if annotations is not None else course_ink_mask(purple, controls)))
    h, w = ink.shape
    radius = max(7, round(max(h, w)*.006))
    density = cv2.boxFilter((ink > 0).astype(np.float32), -1, (2*radius+1, 2*radius+1))
    regions = ((density > .32) & (ink > 0)).astype(np.uint8)*255
    closed = cv2.morphologyEx(regions, cv2.MORPH_CLOSE, np.ones((radius, radius), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(ink)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > radius*radius*3:
            cv2.drawContours(result, [contour], -1, 255, cv2.FILLED)
    return result


def analyze(image, labeler=None):
    masks = colour_masks(image)
    height, width = image.shape[:2]
    features = region_features(masks['building'], 'building', height*width, close=5)
    features += region_features(masks['water'], 'water', height*width, close=7)
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    pitch_ink = ((h >= 12) & (h <= 34) & (s >= 65) & (s <= 180) & (v > 200)).astype(np.uint8)*255
    for feature in region_features(pitch_ink, 'pitch', height*width, close=5):
        rect = cv2.minAreaRect(np.asarray(feature['points'], np.float32))
        rectangularity = feature['area']/max(1, rect[1][0]*rect[1][1])
        if rectangularity > .72 and feature['area'] > height*width*.0004 and feature['aspect'] < 3.5:
            feature['rectangularity'] = float(rectangularity)
            features.append(feature)
    controls = detect_controls(masks['purple'])
    classify_course_symbols(masks['purple'], controls)
    ocr = labeler(masks['purple'], controls, masks['course_text']) if labeler else None
    annotations = annotation_mask(masks['purple'], controls, masks['course_text'])
    restrictions = purple_restrictions(masks['purple'], controls, annotations)
    recovered, hidden, continued_barriers = restore(image, masks['purple'], annotations, restrictions, masks['black'], controls)
    terrain_masks = colour_masks(recovered)
    # Preserve observed obstacle pixels, while restoring the hidden portions
    # of water/roofs/walls rather than turning purple ink into free ground.
    for kind in ('water', 'building', 'dark_green', 'olive', 'black'):
        terrain_masks[kind] |= cv2.bitwise_and(masks[kind], cv2.bitwise_not(hidden))
    terrain_masks['black'] |= continued_barriers
    # Do not fill an entire external contour: it may contain a courtyard,
    # portico, garden, or an opening through a U-shaped building. Roof fill
    # and its black outline remain hard; real white openings remain visible.
    buildings = terrain_masks['building'].copy()
    # Anti-aliasing beside thin grey roofs leaves pale grey road pixels that
    # the loose grey threshold also accepts. A real roof is a filled region,
    # so drop building pixels whose 3x3 neighbourhood is not all grey; white
    # roads and courtyards then stay walkable instead of becoming false walls.
    support = cv2.erode(buildings, np.ones((3, 3), np.uint8))
    buildings = cv2.dilate(support, np.ones((3, 3), np.uint8))
    # A black roof outline blended with yellow paving is dark brown in the
    # scan, not neutral black. Retain that one-pixel boundary where a solid
    # roof supports it, without filling bright gates or light-grey porticos.
    _, terrain_s, terrain_v = cv2.split(cv2.cvtColor(recovered, cv2.COLOR_BGR2HSV))
    roof_edge = ((terrain_v < 165) & (terrain_s < 150)
                 & (cv2.dilate(buildings, np.ones((3, 3), np.uint8)) > 0))
    buildings[roof_edge] = 255
    water_regions = water_obstacles(terrain_masks['water'])
    hard = cv2.bitwise_or(buildings, water_regions)
    for kind in ('dark_green', 'olive', 'black'):
        hard = cv2.bitwise_or(hard, terrain_masks[kind])
    hard = cv2.bitwise_or(hard, restrictions)
    unknown_ink = cv2.bitwise_and(masks['purple'], cv2.bitwise_not(cv2.bitwise_or(annotations, hidden)))
    hard = cv2.bitwise_or(hard, unknown_ink)
    # Tiny isolated compression flecks are not walls. A 2x2 black pillar on
    # the supplied low-resolution map is already a real obstacle, though.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    keep = np.array([False]+[int(stats[i, cv2.CC_STAT_AREA]) >= 4 for i in range(1, count)])
    hard = keep[labels].astype(np.uint8)*255
    costs = np.ones((height, width), np.float32)
    costs[terrain_masks['green'] > 0] = 1.5
    costs[hard > 0] = 0
    return {'width': width, 'height': height, 'controls': controls, 'features': features,
            'masks': masks, 'blocked': hard, 'costs': costs,
            'semantics': {'building': buildings > 0, 'water': water_regions > 0,
                          'point_ink': terrain_masks['black'] > 0, 'vegetation': terrain_masks['dark_green'] > 0,
                          'restricted': (restrictions > 0) | (terrain_masks['olive'] > 0),
                          'course_ink': hidden > 0, 'uncertain_ink': unknown_ink > 0},
            'statistics': {'buildingCandidates': sum(f['kind']=='building' for f in features),
                           'waterCandidates': sum(f['kind']=='water' for f in features),
                           'controlCandidates': sum(c.get('role') == 'control' for c in controls),
                           'courseSymbolCandidates': len(controls),
                           'forbiddenPixelFraction': round(float(np.mean(hard > 0)), 4),
                           'restoredCourseInkPixels': int(np.count_nonzero(hidden)),
                           'continuedBarrierPixels': int(np.count_nonzero(continued_barriers)),
                           **({'ocr': ocr} if ocr else {})}, 'terrainSchema': TERRAIN_SCHEMA,
            'warnings': ['自动提取结果待核验：细墙、门洞、桥隧层级及紫色禁区可能存在歧义。',
                         '控制点编号需核对；圈心和到访位置分别保存。']}


def save_analysis(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory/'blocked.png'), result['blocked'])
    np.savez_compressed(directory/'terrain.npz', costs=result['costs'], **result.get('semantics', {}))
    overlay = np.zeros((result['height'], result['width'], 4), np.uint8)
    overlay[result['blocked'] > 0] = [67, 66, 221, 125]
    cv2.imwrite(str(directory/'terrain.png'), overlay)


def preview(path, output, max_side=1200):
    image = read_image(path)
    scale = min(1, max_side/max(image.shape[:2]))
    cv2.imwrite(str(output), cv2.resize(image, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 82])
