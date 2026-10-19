"""Automatic semantic matching, robust alignment, local correction, and warp."""

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import Frame, Registration, affine_points, fit_affine, mercator, unmercator


def principal_angle(points):
    contour = np.asarray(points, np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    if abs(moments['m00']) < 1:
        return 0.
    covariance = np.array([[moments['mu20'], moments['mu11']],
                           [moments['mu11'], moments['mu02']]])/moments['m00']
    _, vectors = np.linalg.eigh(covariance)
    return math.atan2(vectors[1, -1], vectors[0, -1])


class Matcher:
    def __init__(self, features, reference):
        self.source = {kind: [f for f in features if f['kind'] == kind] for kind in ('water', 'building', 'pitch')}
        self.target = {kind: [f for f in reference if f['kind'] == kind] for kind in self.source}
        self.trees = {kind: cKDTree([f['center'] for f in values]) for kind, values in self.target.items() if values}
        self.arrays = {}
        for kind, values in self.target.items():
            self.arrays[kind] = (np.array([f['area'] for f in values]), np.array([f['aspect'] for f in values]))

    def matches(self, matrix, max_distance=45):
        pairs = []
        score = 0.
        scale2 = abs(np.linalg.det(matrix[:, :2]))
        for kind in self.source:
            sources, targets = self.source[kind], self.target[kind]
            if not sources or not targets:
                continue
            points = affine_points(matrix, [f['center'] for f in sources])
            distance, indexes = self.trees[kind].query(points, k=min(5, len(targets)))
            if distance.ndim == 1:
                distance, indexes = distance[:, None], indexes[:, None]
            area, aspect = self.arrays[kind]
            area_error = np.abs(np.log(np.array([f['area'] for f in sources])[:, None]*scale2 / area[indexes]))
            aspect_error = np.abs(np.log(np.array([f['aspect'] for f in sources])[:, None] / aspect[indexes]))
            combined = distance + 15*area_error + 10*aspect_error
            best = np.argmin(combined, axis=1)
            used = set()
            order = np.argsort(combined[np.arange(len(sources)), best])
            for i in order:
                j = int(best[i])
                target_index = int(indexes[i, j])
                if target_index in used or distance[i, j] > max_distance or area_error[i, j] > 1.1 or aspect_error[i, j] > .8:
                    continue
                used.add(target_index)
                weight = 4. if kind == 'water' else 1.
                score += weight*math.exp(-.5*(combined[i, j]/25)**2)
                pairs.append({'pixel': sources[i]['center'], 'local': targets[target_index]['center'],
                              'sourceIndex': int(i), 'referenceId': targets[target_index]['id'], 'kind': kind,
                              'errorM': float(distance[i, j]), 'shapeError': float(area_error[i, j]+aspect_error[i, j])})
        return score, pairs

    def initial(self, width, height):
        sources = sorted(self.source['water'], key=lambda f: -f['area'])[:7]
        targets = [f for f in self.target['water'] if f['area'] > 400]
        if not sources or not targets:
            raise ValueError("缺少可匹配水体。请补充建筑/路口同名点，或导入校园参考。")
        best = (-1., None, [])
        reflect = np.diag([1., -1.])
        candidates = 0
        for source in sources:
            a = principal_angle(np.asarray(source['points']) @ reflect)
            for target in targets:
                if abs(math.log(source['aspect']/target['aspect'])) > .75:
                    continue
                scale = math.sqrt(target['area']/source['area'])
                if not 250 < max(width, height)*scale < 4200:
                    continue
                b = principal_angle(target['points'])
                for flip in (0, math.pi):
                    for delta in (-.12, 0, .12):
                        angle = b-a+flip+delta
                        rotate = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                        for factor in (.9, 1, 1.1):
                            linear = scale*factor*rotate @ reflect
                            shift = np.array(target['center'])-linear @ source['center']
                            matrix = np.column_stack((linear, shift))
                            score, pairs = self.matches(matrix)
                            candidates += 1
                            if score > best[0]:
                                best = (score, matrix, pairs)
        if best[1] is None or len(best[2]) < 6:
            raise ValueError("自动地物匹配证据不足，请在原图与底图补充同名点")
        return best[1], candidates


def _corners(points, tolerance):
    points = np.asarray(points, dtype=np.float32)
    if len(points) > 1 and np.linalg.norm(points[0]-points[-1]) < 1e-5:
        points = points[:-1]
    if len(points) < 3:
        return np.empty((0, 2))
    points = cv2.approxPolyDP(points.reshape(-1, 1, 2), tolerance, True).reshape(-1, 2)
    previous, following = np.roll(points, 1, axis=0)-points, np.roll(points, -1, axis=0)-points
    norms = np.linalg.norm(previous, axis=1)*np.linalg.norm(following, axis=1)
    cosine = np.sum(previous*following, axis=1)/np.maximum(norms, 1e-8)
    # Remove nearly straight coastline/contour samples; retain real corners.
    keep = (cosine > -.91) & (norms > tolerance*tolerance*4)
    return points[keep]


def _outline_iou(first, second):
    first, second = np.asarray(first), np.asarray(second)
    lower = np.floor(np.minimum(first.min(axis=0), second.min(axis=0)))-2
    upper = np.ceil(np.maximum(first.max(axis=0), second.max(axis=0)))+2
    size = np.maximum(1, np.ceil(upper-lower).astype(int))
    if size.prod() > 1_000_000:
        return 0.
    a, b = np.zeros((size[1], size[0]), np.uint8), np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(a, [np.rint(first-lower).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.rint(second-lower).astype(np.int32)], 1)
    return float(np.sum(a & b)/max(1, np.sum(a | b)))


def feature_corner_matches(matcher, matrix, matches):
    """Match corners within already matched, shape-checked real buildings.

    These are additional observed correspondences, not invented outer hull
    anchors or affine extrapolations. Whole buildings stay in one fit/holdout
    partition so corners of one object cannot leak across the validation set.
    """
    references = {f['id']: f for f in matcher.target['building']}
    scale = math.sqrt(abs(np.linalg.det(matrix[:, :2])))
    result = []
    for match in matches:
        if match['kind'] != 'building':
            continue
        source = matcher.source['building'][match['sourceIndex']]
        target = references[match['referenceId']]
        projected = affine_points(matrix, source['points'])
        iou = _outline_iou(projected, target['points'])
        if iou < .52:
            continue
        a = _corners(source['points'], 1.3/max(.1, scale))
        b = _corners(target['points'], 1.3)
        if len(a) < 3 or len(b) < 3:
            continue
        world = affine_points(matrix, a)
        distances, forward = cKDTree(b).query(world)
        _, backward = cKDTree(world).query(b)
        accepted = [(i, int(j)) for i, j in enumerate(forward) if distances[i] < 13 and backward[j] == i]
        if len(accepted) < 3 or np.median([distances[i] for i, _ in accepted]) > 8:
            continue
        for i, j in accepted:
            result.append({'pixel': a[i].astype(float).tolist(), 'local': b[j].astype(float).tolist(),
                           'referenceId': match['referenceId'], 'kind': 'building_corner',
                           'outlineIoU': round(iou, 3), 'errorM': round(float(distances[i]), 3)})
    return result


def partial_feature_corner_matches(matcher, matrix, width, height):
    """Use visible measured edges when a building or pitch is cropped.

    The artificial cut edge is never an anchor. At least two real corners
    must agree on one reference object, edge direction and physical length.
    This avoids the centroid bias of partially visible stadium pitches.
    """
    scale = math.sqrt(abs(np.linalg.det(matrix[:, :2])))
    result = []
    for kind in ('building', 'pitch'):
        for feature in matcher.source[kind]:
            raw = np.array(feature['points'])
            clipped = np.any(raw[:, 0] <= 2) or np.any(raw[:, 1] <= 2) or np.any(raw[:, 0] >= width-3) or np.any(raw[:, 1] >= height-3)
            corners = _corners(raw, 1.2/max(.1, scale))
            if kind == 'building':
                corners = corners[(corners[:, 0] > 3) & (corners[:, 0] < width-4) & (corners[:, 1] > 3) & (corners[:, 1] < height-4)]
            if len(corners) < 2:
                continue
            world = affine_points(matrix, corners)
            projected_outline = affine_points(matrix, raw)
            options = []
            for target in matcher.target[kind]:
                area_ratio = feature['area']*abs(np.linalg.det(matrix[:, :2]))/target['area']
                if kind == 'building' and not clipped and area_ratio > .70:
                    continue  # a partial outline may end at an inset/photo, not the sheet border
                if np.linalg.norm(np.mean(world, axis=0)-target['center']) > max(60, math.sqrt(target['area'])*1.5):
                    continue
                reference_corners = _corners(target['points'], 1.2)
                if len(reference_corners) < 2:
                    continue
                distances, indexes = cKDTree(reference_corners).query(world)
                _, reverse = cKDTree(world).query(reference_corners)
                pairs = [(i, int(j)) for i, j in enumerate(indexes) if distances[i] <= 12 and reverse[j] == i]
                if len(pairs) < 2:
                    continue
                coherent = False
                for a, b in __import__('itertools').combinations(pairs, 2):
                    left, right = world[b[0]]-world[a[0]], reference_corners[b[1]]-reference_corners[a[1]]
                    length_a, length_b = np.linalg.norm(left), np.linalg.norm(right)
                    if min(length_a, length_b) >= 10 and .85 <= length_a/length_b <= 1.15 and np.dot(left, right)/(length_a*length_b) >= .98:
                        # Extra near-collinear vertices may lie on a real
                        # edge. Test the observed outline itself, rather than
                        # requiring consecutive indices or accepting a diagonal.
                        edge_is_observed = True
                        for begin, vector, outline, tolerance in ((world[a[0]], left, projected_outline, min(4., max(2., 6*scale, length_a*.04))),
                                                                  (reference_corners[a[1]], right, target['points'], min(4., max(2., length_b*.04)))):
                            contour = np.asarray(outline, np.float32).reshape(-1, 1, 2)
                            for ratio in np.linspace(.1, .9, 9):
                                sample = tuple((begin+vector*ratio).astype(float))
                                if abs(cv2.pointPolygonTest(contour, sample, True)) > tolerance:
                                    edge_is_observed = False
                                    break
                        if edge_is_observed:
                            coherent = True
                if not coherent:
                    continue
                iou = _outline_iou(projected_outline, target['points'])
                if not .20 <= area_ratio <= 1.6 or iou < .28:
                    continue
                score = float(np.mean([distances[i] for i, _ in pairs]))+(1-iou)*3
                options.append((score, pairs, target, reference_corners, iou))
            options.sort(key=lambda option: option[0])
            if not options or (len(options) > 1 and options[1][0]-options[0][0] < 3):
                continue
            score, pairs, target, reference_corners, iou = options[0]
            for i, j in pairs:
                result.append({'pixel': corners[i].astype(float).tolist(), 'local': reference_corners[j].astype(float).tolist(),
                               'kind': f'visible_{kind}_corner', 'referenceId': target['id'],
                               'outlineIoU': round(iou, 3), 'errorM': round(float(np.linalg.norm(world[i]-reference_corners[j])), 3)})
    return result


def fit_observed_landmarks(pairs, extra):
    """Fit measured outlines, keeping every point of an object in one split.

    Boundary observations belong to the fit so that an arbitrary holdout
    does not shrink the supported map. No synthetic/extrapolated anchor is
    added. The reported holdout is explicitly an interior object holdout.
    """
    outlined = {p['referenceId'] for p in extra}
    observations = [dict(p) for p in pairs if p['referenceId'] not in outlined]
    observations += [dict(p) for p in extra]
    groups = defaultdict(list)
    for observation in observations:
        groups[observation['referenceId']].append(observation)
    ordered = sorted(groups, key=lambda key: tuple(np.mean([p['pixel'] for p in groups[key]], axis=0)[::-1]))
    sources = np.array([p['pixel'] for p in observations], np.float32)
    hull = cv2.convexHull(sources, returnPoints=False).ravel()
    boundary_ids = {observations[i]['referenceId'] for i in hull}
    interior = [key for key in ordered if key not in boundary_ids]
    validation_ids = set(interior[1::3])
    fit_ids = [key for key in ordered if key not in validation_ids]
    # An object with many corners gets the same global-fit vote as one
    # compact landmark. Its measured corners still refine the local mesh.
    centers = lambda key, field: np.mean([p[field] for p in groups[key]], axis=0)
    matrix, keep = fit_affine([centers(key, 'pixel') for key in fit_ids],
                              [centers(key, 'local') for key in fit_ids], threshold=15)
    rejected = {key for key, accepted in zip(fit_ids, keep) if not accepted}
    fitting = [p for p in observations if p['referenceId'] not in validation_ids | rejected]
    validation = [p for p in observations if p['referenceId'] in validation_ids]
    if len(fitting) < 8 or not validation:
        raise ValueError('独立地物或覆盖范围不足，不能建立可核查的配准')
    # Remove coincident corner observations deterministically. They must not
    # give a singular mesh or be counted as separate accuracy evidence.
    unique = {}
    for p in fitting:
        key = tuple(np.round(p['pixel'], 3))
        if key not in unique:
            unique[key] = p
    fitting = list(unique.values())
    fallback = False
    try:
        reg = Registration(matrix, [p['pixel'] for p in fitting], [p['local'] for p in fitting])
    except ValueError:
        reg = Registration(matrix, [p['pixel'] for p in fitting], affine_points(matrix, [p['pixel'] for p in fitting]))
        fallback = True
    predicted, inside = reg.forward([p['pixel'] for p in validation], strict=False)
    errors = np.linalg.norm(predicted-np.array([p['local'] for p in validation]), axis=1)
    for p in fitting:
        p['purpose'] = 'fit'
    for p, error, valid in zip(validation, errors, inside):
        p.update(purpose='validation', errorM=float(error), insideHull=bool(valid))
    return reg, {'fitCount': len(fitting), 'fitObjectCount': len({p['referenceId'] for p in fitting}),
                 'validationCount': len(validation), 'validationObjectCount': len(validation_ids),
                 'validationInsideHull': int(inside.sum()), 'rejectedObjectIds': sorted(rejected),
                 'medianErrorM': round(float(np.median(errors)), 2),
                 'p95ErrorM': round(float(np.percentile(errors, 95)), 2),
                 'maxErrorM': round(float(errors.max()), 2), 'affineFallback': fallback,
                 'foldedTriangles': reg.folded, 'landmarks': fitting+validation}


def auto_register(features, reference, width, height):
    matcher = Matcher(features, reference)
    matrix, candidates = matcher.initial(width, height)
    for tolerance in (45, 35, 28, 23):
        _, pairs = matcher.matches(matrix, max_distance=tolerance)
        if len(pairs) < 6:
            break
        candidate, keep = fit_affine([p['pixel'] for p in pairs], [p['local'] for p in pairs], threshold=18)
        if keep.sum() >= 6:
            matrix = candidate
    _, pairs = matcher.matches(matrix, max_distance=22)
    if len(pairs) < 8:
        raise ValueError(f"自动配准仅找到 {len(pairs)} 组稳定地物，需补充同名点")
    corners = feature_corner_matches(matcher, matrix, pairs)
    corners += partial_feature_corner_matches(matcher, matrix, width, height)
    reg, quality = fit_observed_landmarks(pairs, corners)
    quality.update(state='needs_review', method='water_seed_observed_outline_ransac_mesh',
                   candidateTransforms=candidates, observedCornerCount=len(corners),
                   validationScope='按地物整体留出内部核查点，实测边界角点用于拟合；自动对应关系未人工核对，不等价于独立现场定位精度。')
    return reg, quality


def manual_register(anchors, frame=None):
    frame = frame or Frame()
    fitting = [p for p in anchors if p.get('purpose', 'fit') == 'fit']
    validation = [p for p in anchors if p.get('purpose') == 'validation']
    source = np.array([p['pixel'] for p in fitting])
    target = frame.to_local([p['lnglat'] for p in fitting])
    matrix, keep = fit_affine(source, target, threshold=20)
    if not np.all(keep):
        raise ValueError("手动同名点中存在冲突，请检查坐标和对应关系")
    reg = Registration(matrix, source, target, regularization=.2)
    errors = np.array([])
    if validation:
        predicted, inside = reg.forward([p['pixel'] for p in validation])
        if not np.all(inside):
            raise ValueError("核查点超出配准点包围范围")
        errors = np.linalg.norm(predicted-frame.to_local([p['lnglat'] for p in validation]), axis=1)
    return reg, {'state': 'needs_review', 'method': 'reviewed_landmarks_mesh', 'fitCount': len(source),
                 'validationCount': len(errors), 'medianErrorM': float(np.median(errors)) if len(errors) else None,
                 'p95ErrorM': float(np.percentile(errors, 95)) if len(errors) else None,
                 'maxErrorM': float(errors.max()) if len(errors) else None,
                 'foldedTriangles': reg.folded, 'landmarks': anchors,
                 'validationScope': '教师录入的独立核查点；发布前仍需检查通行语义。'}


def register_to_map(image, base_image, base_registration, base_id, base_quality):
    """Transfer a campus reference through automatically matched map detail.

    Useful for a small/low-resolution course sheet with too few OSM features.
    A transfer inherits the base map's geographic uncertainty.
    """
    from .vision import colour_masks
    scales = [min(1., 1600/max(im.shape[:2])) for im in (image, base_image)]
    resized = [cv2.resize(im, None, fx=s, fy=s) for im, s in zip((image, base_image), scales)]
    detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=.025)
    keypoints, descriptors = [], []
    for im in resized:
        purple = colour_masks(im)['purple']
        valid = 255-cv2.dilate(purple, np.ones((5, 5), np.uint8))
        kp, desc = detector.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), valid)
        keypoints.append(kp); descriptors.append(desc)
    if any(desc is None for desc in descriptors):
        raise ValueError("两张地图之间没有足够的共同地物")
    matches = cv2.BFMatcher().knnMatch(descriptors[0], descriptors[1], k=2)
    good = [a for pair in matches if len(pair) == 2 for a, b in [pair] if a.distance < .68*b.distance]
    if len(good) < 16:
        raise ValueError("两图共同地物匹配不足")
    source = np.array([keypoints[0][m.queryIdx].pt for m in good])/scales[0]
    target_pixels = np.array([keypoints[1][m.trainIdx].pt for m in good])/scales[1]
    _, inliers = fit_affine(source, target_pixels, threshold=5/scales[1])
    source, target_pixels = source[inliers], target_pixels[inliers]
    target, inside = base_registration.forward(target_pixels)
    source, target = source[inside], target[inside]
    # SIFT can report several orientations at the same pixel. Keep only one
    # before preserving the observed hull, otherwise duplicate mesh vertices
    # make a valid low-resolution transfer fail.
    _, distinct = np.unique(np.round(source, 4), axis=0, return_index=True)
    source, target = source[distinct], target[distinct]
    # Avoid hundreds of virtually coincident SIFT landmarks dominating a fit.
    bins = np.floor(source/50).astype(int)
    _, unique = np.unique(bins, axis=0, return_index=True)
    boundary = cv2.convexHull(source.astype(np.float32), returnPoints=False).ravel()
    unique = np.unique(np.concatenate((unique, boundary)))
    source, target = source[unique], target[unique]
    if len(source) < 12:
        raise ValueError("两图有效覆盖区的匹配点不足")
    indices = np.argsort(source[:, 1])
    source, target = source[indices], target[indices]
    boundary = set(cv2.convexHull(source.astype(np.float32), returnPoints=False).ravel().tolist())
    validation = np.zeros(len(source), dtype=bool)
    validation[[i for i in range(len(source)) if i not in boundary][2::5]] = True
    matrix, keep = fit_affine(source[~validation], target[~validation], threshold=8)
    fit_source, fit_target = source[~validation][keep], target[~validation][keep]
    try:
        reg = Registration(matrix, fit_source, fit_target, regularization=1.5)
    except ValueError:
        reg = Registration(matrix, fit_source, affine_points(matrix, fit_source))
    prediction, inside = reg.forward(source[validation], strict=False)
    errors = np.linalg.norm(prediction-target[validation], axis=1)
    quality = {'state': 'needs_review', 'method': 'sift_cross_map_ransac_mesh',
               'baseAssetId': base_id, 'fitCount': len(fit_source), 'validationCount': len(errors),
               'medianErrorM': round(float(np.median(errors)), 2), 'p95ErrorM': round(float(np.percentile(errors, 95)), 2),
               'maxErrorM': round(float(errors.max()), 2), 'foldedTriangles': reg.folded,
               'inheritedGeographicP95M': base_quality.get('inheritedGeographicP95M', base_quality.get('p95ErrorM')),
               'validationScope': '与已配准原图的内部留出接缝误差，边界由实测共同地物确定；绝对位置误差继承基准图。',
               'landmarks': [{'pixel': p.tolist(), 'local': q.tolist(), 'purpose': 'fit'} for p, q in zip(fit_source, fit_target)]
                    + [{'pixel': p.tolist(), 'local': q.tolist(), 'purpose': 'validation', 'errorM': float(e), 'insideHull': bool(v)}
                       for p, q, e, v in zip(source[validation], target[validation], errors, inside)]}
    return reg, quality


def extend_transfer_boundary(reg, quality, geographic_quality):
    """Supplement a copied map with observed geographic corners it lacks.

    A low-resolution SIFT match may end before the start symbol. Only fitted,
    shape-checked geographic corners may extend that hull; neither holdout
    points nor synthetic image corners can be used as boundary evidence.
    """
    fitting = [p for p in quality['landmarks'] if p['purpose'] == 'fit']
    validation = [p for p in quality['landmarks'] if p['purpose'] == 'validation']
    extra = []
    known = {tuple(np.round(p['pixel'], 3)) for p in fitting}
    for p in (geographic_quality or {}).get('landmarks', []):
        if p.get('purpose') != 'fit' or 'corner' not in p.get('kind', ''):
            continue
        point = np.asarray(p['pixel'])
        if tuple(np.round(point, 3)) in known or reg.forward([point])[1][0]:
            continue
        if np.linalg.norm(affine_points(reg.affine, [point])[0]-p['local']) > 12:
            continue
        extra.append({**p, 'source': 'geographic_outline'})
        known.add(tuple(np.round(point, 3)))
    if not extra:
        return reg, quality
    combined = fitting+extra
    try:
        updated = Registration(reg.affine, [p['pixel'] for p in combined], [p['local'] for p in combined])
    except ValueError:
        return reg, quality
    prediction, valid = updated.forward([p['pixel'] for p in validation], strict=False)
    errors = np.linalg.norm(prediction-np.array([p['local'] for p in validation]), axis=1)
    for p, error, inside in zip(validation, errors, valid):
        p.update(errorM=float(error), insideHull=bool(inside))
    return updated, {**quality, 'method': 'sift_cross_map_observed_boundary_mesh',
                     'fitCount': len(combined), 'geographicBoundaryAnchors': len(extra),
                     'medianErrorM': round(float(np.median(errors)), 2),
                     'p95ErrorM': round(float(np.percentile(errors, 95)), 2),
                     'maxErrorM': round(float(errors.max()), 2), 'landmarks': combined+validation}


def warp_overlay(image, reg, output, frame=None, max_side=1700, nearest=False):
    """Inverse sampling to EPSG:3857, cropped to the valid registration hull."""
    frame = frame or Frame()
    geo = mercator(frame.to_lnglat(reg.target))
    lower, upper = geo.min(axis=0), geo.max(axis=0)
    span = upper-lower
    factor = max_side/max(span)
    width, height = np.maximum(1, np.ceil(span*factor).astype(int))
    output_image = np.zeros((height, width, 4), np.uint8)
    source = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) if image.shape[2] == 3 else image
    # Small stripes avoid giant XY/triangle arrays in memory.
    for y in range(0, height, 100):
        rows = min(100, height-y)
        xx, yy = np.meshgrid(lower[0]+(np.arange(width)+.5)/width*span[0],
                             upper[1]-(np.arange(y, y+rows)+.5)/height*span[1])
        world = np.column_stack((xx.ravel(), yy.ravel()))
        pixels = reg.inverse(frame.to_local(unmercator(world)))
        valid = np.isfinite(pixels).all(axis=1)
        pixels[~valid] = -1000
        mx = pixels[:, 0].reshape(rows, width).astype(np.float32)
        my = pixels[:, 1].reshape(rows, width).astype(np.float32)
        output_image[y:y+rows] = cv2.remap(source, mx, my,
                                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    cv2.imwrite(str(Path(output)), output_image, [cv2.IMWRITE_PNG_COMPRESSION, 7])
    return [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]
