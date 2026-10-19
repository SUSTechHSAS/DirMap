"""Separate course annotations from ground symbols and repair their occlusion.

The recovered colours are an interpretation, never a modification of the
source scan. A course connection supplies no edge to the navigation graph.
"""
import itertools

import cv2
import numpy as np


def annotation_mask(purple, controls, text_ink=None):
    """Confirmed circles/triangles, OCR text, and supported circle connections."""
    strokes = np.zeros_like(purple)
    for control in controls:
        radius = float(control['radiusPx'])
        center = tuple(np.rint(control['pixel']).astype(int))
        geometry = control.get('symbolGeometry', {})
        for ring in [radius, *geometry.get('ringRadii', [])]:
            cv2.circle(strokes, center, round(ring), 255, max(5, round(radius*.4)))
        if geometry.get('triangle'):
            cv2.polylines(strokes, [np.rint(geometry['triangle']).astype(np.int32)], True, 255, max(5, round(radius*.35)))
        box = control.get('labelPixelBox')
        if box:
            x0, y0 = np.maximum(0, np.floor(np.array(box[:2])-2).astype(int))
            x1, y1 = np.minimum([purple.shape[1], purple.shape[0]], np.ceil(np.array(box[2:])+2).astype(int))
            if x1 > x0 and y1 > y0:
                # Bold text and its anti-aliasing may cover pale restriction
                # hatching. Do not erase all purple pixels inside its box.
                ink = (text_ink if text_ink is not None else purple)[y0:y1, x0:x1]
                strokes[y0:y1, x0:x1] |= cv2.dilate(ink, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(255-purple, cv2.DIST_L2, 3)
    for left, right in itertools.combinations(controls, 2):
        a, b = np.array(left['pixel']), np.array(right['pixel'])
        length = float(np.linalg.norm(b-a))
        radius = min(left['radiusPx'], right['radiusPx'])
        if length < radius*5:
            continue
        start, stop = left['radiusPx']*1.2/length, 1-right['radiusPx']*1.2/length
        if start >= stop:
            continue
        samples = np.rint(a+np.linspace(start, stop, max(20, round(length)))[:, None]*(b-a)).astype(int)
        if (samples < 0).any() or (samples[:, 0] >= purple.shape[1]).any() or (samples[:, 1] >= purple.shape[0]).any():
            continue
        hit = distance[samples[:, 1], samples[:, 0]] <= max(1.5, radius*.1)
        if np.mean(hit) >= .9 and min(np.mean(part) for part in np.array_split(hit, 4)) >= .8:
            cv2.line(strokes, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), 255, max(5, round(radius*.45)))
    return cv2.bitwise_and(strokes, purple)


def _line_kernel(size, angle):
    result = np.zeros((size, size), np.uint8)
    center = np.array([size//2, size//2])
    vector = np.array([np.cos(angle), np.sin(angle)])*(size//2)
    cv2.line(result, tuple(np.rint(center-vector).astype(int)), tuple(np.rint(center+vector).astype(int)), 1, 1)
    return result


def continue_barriers(black, unknown, maximum_gap):
    """Continue supported linear fragments only inside hidden pixels.

    Directional support prevents joining two parallel road borders into a
    cross-road wall. Known white gates outside the overprint remain intact.
    """
    visible = cv2.bitwise_and(black, cv2.bitwise_not(unknown))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(visible, 8)
    linear = np.array([False]+[max(stats[i, 2:4]) >= 20 or max(stats[i, 2:4]) >= 3*max(1, min(stats[i, 2:4]))
                               for i in range(1, count)])
    strokes = linear[labels].astype(np.uint8)*255
    grown = cv2.dilate(strokes, np.ones((3, 3), np.uint8))
    repaired = np.zeros_like(black)
    size = int(maximum_gap) | 1
    for angle in np.arange(0, np.pi, np.pi/12):
        direction = _line_kernel(7, angle)
        supported = cv2.morphologyEx(grown, cv2.MORPH_OPEN, direction)
        joined = cv2.morphologyEx(supported, cv2.MORPH_CLOSE, _line_kernel(size, angle))
        repaired |= cv2.bitwise_and(joined, unknown)
    return repaired


def restore(image, purple, annotations, restrictions, black, controls):
    # Include purple-coloured antialiasing beside confirmed ink. Its hue can
    # lie between a water pixel and purple, outside the initial purple test.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    nearby = cv2.dilate(annotations, np.ones((5, 5), np.uint8)) > 0
    fringe = nearby & (hsv[:, :, 0] >= 111) & (hsv[:, :, 0] <= 170) & (hsv[:, :, 1] >= 30)
    unknown = ((annotations > 0) | fringe) & (restrictions == 0)
    unknown = unknown.astype(np.uint8)*255
    if not unknown.any():
        return image.copy(), unknown, np.zeros_like(unknown)
    # Continue colour edges at native resolution. A wide averaging stencil
    # mixes the roofs on both sides of a covered passage into its hidden
    # centre. Navier-Stokes inpainting follows the visible isophotes instead;
    # observed pixels and independently continued black barriers still win.
    recovered = cv2.inpaint(image, unknown, 1, cv2.INPAINT_NS)
    radii = [c['radiusPx'] for c in controls]
    gap = max(9, min(49, round((np.median(radii) if radii else 12)*1.2)))
    barriers = continue_barriers(black, unknown, gap)
    return recovered, unknown, barriers
