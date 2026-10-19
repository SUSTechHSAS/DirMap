"""Recognize course symbols without erasing cartographic forbidden areas."""

import cv2
import numpy as np


def course_symbol(purple, control):
    center = np.array(control['pixel'], dtype=float)
    radius = float(control['radiusPx'])
    height, width = purple.shape
    pad = int(np.ceil(radius*2.1))
    origin = np.maximum(0, np.floor(center-pad).astype(int))
    end = np.minimum([width, height], np.ceil(center+pad).astype(int))
    crop = purple[origin[1]:end[1], origin[0]:end[0]]
    point = center-origin
    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    radial = np.linspace(.5, 1.65, 93)*radius
    ring_points = point+radial[:, None, None]*np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    xx = np.clip(np.rint(ring_points[..., 0]).astype(int), 0, crop.shape[1]-1)
    yy = np.clip(np.rint(ring_points[..., 1]).astype(int), 0, crop.shape[0]-1)
    profile = (crop[yy, xx] > 0).mean(axis=1)
    # Separate connected peaks, rather than counting the two edges of one
    # thick circle as a double finish ring.
    groups = np.split(np.flatnonzero(profile >= .65), np.flatnonzero(np.diff(np.flatnonzero(profile >= .65)) > 1)+1)
    radii = []
    for group in groups:
        if len(group):
            radii.append(float(radial[group[np.argmax(profile[group])]]))
    double = any(abs(a-b) > radius*.16 for a in radii for b in radii)
    distance = cv2.distanceTransform(255-crop, cv2.DIST_L2, 3)
    grid_y, grid_x = np.indices(crop.shape)
    inner = (grid_x-point[0])**2+(grid_y-point[1])**2 < (radius*.84)**2
    inner_density = np.mean(crop[inner] > 0) if inner.any() else 0
    best = None
    if .035 < inner_density < .8:
        t = np.linspace(.14, .86, 27)
        for angle in np.arange(-90, 30, 6)*np.pi/180:
            unit = np.stack((np.cos(angle+np.arange(3)*2*np.pi/3), np.sin(angle+np.arange(3)*2*np.pi/3)), axis=-1)
            for relative_radius in np.arange(.85, 1.71, .05):
                for dx, dy in ((0, 0), (-.07, 0), (.07, 0), (0, -.07), (0, .07)):
                    vertices = point+[dx*radius, dy*radius]+unit*radius*relative_radius
                    next_vertices = np.roll(vertices, -1, axis=0)
                    sides = vertices[:, None, :]+(next_vertices-vertices)[:, None, :]*t[None, :, None]
                    vectors = next_vertices-vertices
                    normal = np.stack((-vectors[:, 1], vectors[:, 0]), axis=-1)
                    normal /= np.linalg.norm(normal, axis=1)[:, None]
                    offset = normal[:, None, :]*max(4., radius*.22)

                    def hit(points, tolerance):
                        x = np.rint(points[..., 0]).astype(int)
                        y = np.rint(points[..., 1]).astype(int)
                        valid = (x >= 0) & (y >= 0) & (x < crop.shape[1]) & (y < crop.shape[0])
                        result = np.zeros(x.shape, dtype=bool)
                        result[valid] = distance[y[valid], x[valid]] <= tolerance
                        return result.mean(axis=1)

                    coverage = hit(sides, max(1., radius*.045))
                    if coverage.min() < .80:
                        continue
                    background = (hit(sides+offset, .5)+hit(sides-offset, .5))/2
                    contrast = coverage-background
                    if contrast.min() < .24:
                        continue
                    score = float(np.mean(coverage)-.4*np.mean(background))
                    if best is None or score > best[0]:
                        best = (score, vertices+origin, coverage.tolist())
    if best is not None and not double:
        # In a low-resolution compound symbol, triangle strokes can bridge
        # the valley between the two radial peaks. The independently fitted
        # Hough ring is still evidence of the outer circle.
        outer_support = float(profile[np.argmin(np.abs(radial-radius))])
        if outer_support >= .65 and any(abs(r-radius) > radius*.2 for r in radii):
            radii.append(radius)
            double = True
    role = 'start_finish' if best is not None and double else 'start' if best is not None else 'finish' if double else 'control'
    return {'role': role, 'ringRadii': radii or [radius],
            'triangle': best[1].tolist() if best else None,
            'triangleCoverage': best[2] if best else None,
            'confidence': round(best[0], 3) if best else round(float(profile.max()), 3)}


def classify_course_symbols(purple, controls):
    for control in controls:
        symbol = course_symbol(purple, control)
        control['role'] = symbol['role']
        control['symbolGeometry'] = symbol
        if symbol['role'] != 'control':
            control['label'] = {'start': '起点', 'finish': '终点', 'start_finish': '起终点'}[symbol['role']]
            control['labelSource'] = 'course_symbol'
            control['required'] = False


def course_ink_mask(purple, controls):
    """Mask detected symbol strokes, not the disk enclosed by a control."""
    strokes = np.zeros_like(purple)
    for control in controls:
        center = tuple(np.rint(control['pixel']).astype(int))
        radius = control['radiusPx']
        geometry = control.get('symbolGeometry', {})
        for ring in [radius, *geometry.get('ringRadii', [])]:
            cv2.circle(strokes, center, round(ring), 255, max(3, round(radius*.15)))
        if geometry.get('triangle'):
            triangle = np.rint(geometry['triangle']).astype(np.int32)
            cv2.polylines(strokes, [triangle], True, 255, max(3, round(radius*.16)))
    return cv2.bitwise_and(strokes, purple)
