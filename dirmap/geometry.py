"""Explicit WGS84/local metre coordinates and an invertible registration mesh."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class Frame:
    lng: float = 113.9304
    lat: float = 22.535

    @property
    def scales(self):
        phi = math.radians(self.lat)
        e2 = 0.00669437999014
        n = 6378137 / math.sqrt(1 - e2 * math.sin(phi) ** 2)
        m = 6378137 * (1-e2) / (1-e2 * math.sin(phi) ** 2) ** 1.5
        return np.array([n * math.cos(phi), m]) * math.pi / 180

    def to_local(self, lnglat):
        points = np.asarray(lnglat, dtype=float)
        if not np.isfinite(points).all() or np.any(np.abs(points[..., 0]) > 180) or np.any(np.abs(points[..., 1]) > 90):
            raise ValueError("经纬度无效")
        return (points - [self.lng, self.lat]) * self.scales

    def to_lnglat(self, xy):
        return np.asarray(xy, dtype=float) / self.scales + [self.lng, self.lat]

    def as_dict(self):
        return {"crs": "WGS84", "origin": [self.lng, self.lat], "unit": "metre", "axes": "east_north"}


def mercator(lnglat):
    points = np.asarray(lnglat, dtype=float)
    lat = np.clip(points[..., 1], -85.05112878, 85.05112878)
    return np.stack([np.radians(points[..., 0]) * 6378137,
                     np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * 6378137], axis=-1)


def unmercator(xy):
    points = np.asarray(xy, dtype=float)
    return np.stack([np.degrees(points[..., 0] / 6378137),
                     np.degrees(2 * np.arctan(np.exp(points[..., 1] / 6378137)) - np.pi / 2)], axis=-1)


def affine_points(matrix, points):
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


def fit_affine(source, target, threshold=15):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if len(source) < 3 or source.shape != target.shape or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("至少需要三组有效的同名点")
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("配准点不能共线")
    cv2.setRNGSeed(7)
    matrix, inliers = cv2.estimateAffine2D(source, target, method=cv2.RANSAC,
                                        ransacReprojThreshold=threshold, maxIters=4000,
                                        confidence=0.999, refineIters=20)
    if matrix is None or abs(np.linalg.det(matrix[:, :2])) < 1e-10:
        raise ValueError("无法得到稳定配准")
    return matrix, inliers.ravel().astype(bool)


class Registration:
    """Global affine plus a smoothed, nonfolding piecewise affine correction.

    Pixel y is down; local y is north. Signed areas are compared to the
    global affine determinant, so a deliberate axis reflection is allowed.
    Points outside the measured hull remain explicitly uncalibrated.
    """

    def __init__(self, affine, source=None, target=None, regularization=1.5):
        self.affine = np.asarray(affine, dtype=float)
        if self.affine.shape != (2, 3) or not np.isfinite(self.affine).all() or abs(np.linalg.det(self.affine[:, :2])) < 1e-10:
            raise ValueError("配准矩阵无效")
        self.source = np.asarray(source if source is not None else [], dtype=float).reshape(-1, 2)
        self.target = np.asarray(target if target is not None else [], dtype=float).reshape(-1, 2)
        self.tri = None
        self._inverse_grid = None
        self.folded = 0
        if len(self.source) >= 3:
            if len(self.source) != len(self.target) or not np.isfinite(self.source).all() or not np.isfinite(self.target).all():
                raise ValueError("配准网格点无效")
            if len(np.unique(self.source, axis=0)) != len(self.source):
                raise ValueError("配准点不能重复")
            self.tri = Delaunay(self.source)
            lap = np.zeros((len(self.source), len(self.source)))
            edges = set()
            for triangle in self.tri.simplices:
                for i, j in zip(triangle, np.roll(triangle, -1)):
                    edges.add(tuple(sorted((int(i), int(j)))))
            for i, j in edges:
                lap[i, i] += 1
                lap[j, j] += 1
                lap[i, j] -= 1
                lap[j, i] -= 1
            base = affine_points(self.affine, self.source)
            delta = self.target - base
            self.target = base + np.linalg.solve(np.eye(len(self.source)) + regularization * lap, delta)
            reference_sign = np.sign(np.linalg.det(self.affine[:, :2]))
            for ids in self.tri.simplices:
                a, b = self.source[ids], self.target[ids]
                area_a = np.linalg.det(np.stack((a[1]-a[0], a[2]-a[0])))
                area_b = np.linalg.det(np.stack((b[1]-b[0], b[2]-b[0])))
                if abs(area_b) < 1e-5 or np.sign(area_a * area_b) != reference_sign:
                    self.folded += 1
            # A locally orientation-preserving mesh with a simple, convex
            # boundary has no nonlocal overlap. Reject a nonconvex boundary
            # instead of silently accepting a possibly self-intersecting warp.
            hull = cv2.convexHull(self.source.astype(np.float32), returnPoints=False).ravel()
            boundary = self.target[hull].astype(np.float32)
            if self.folded or not cv2.isContourConvex(boundary):
                raise ValueError("局部配准出现折叠或边界重叠，请修正同名点")

    def forward(self, points, strict=True):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        result = affine_points(self.affine, points)
        valid = np.zeros(len(points), dtype=bool)
        if self.tri is not None:
            simplex = self.tri.find_simplex(points)
            valid = simplex >= 0
            transforms = self.tri.transform[simplex[valid]]
            bary = np.einsum('nij,nj->ni', transforms[:, :2], points[valid] - transforms[:, 2])
            weights = np.column_stack((bary, 1-bary.sum(axis=1)))
            result[valid] = np.einsum('ni,nij->nj', weights, self.target[self.tri.simplices[simplex[valid]]])
        if strict:
            result[~valid] = np.nan
        return result, valid

    def inverse(self, points):
        """Invert the SAME triangles; retriangulating target vertices is wrong."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        result = np.full_like(points, np.nan)
        if self.tri is None:
            return result
        if self._inverse_grid is None:
            lower, upper = self.target.min(axis=0)-1e-6, self.target.max(axis=0)+1e-6
            span = upper-lower
            cell_count = max(1, len(self.tri.simplices)//3)
            shape = np.clip(np.rint(np.sqrt(cell_count)*np.sqrt(span/span[::-1])).astype(int), 1, 64)
            buckets = {}
            transforms, anchors = [], []
            for triangle, ids in enumerate(self.tri.simplices):
                vertices = self.target[ids]
                a, b, c = vertices
                transforms.append(np.linalg.inv(np.column_stack((a-c, b-c))))
                anchors.append(c)
                lo, hi = np.floor((np.array([vertices.min(axis=0), vertices.max(axis=0)])-lower)/span*shape).astype(int)
                lo, hi = np.maximum(0, lo), np.minimum(shape-1, hi)
                for y in range(lo[1], hi[1]+1):
                    for x in range(lo[0], hi[0]+1):
                        buckets.setdefault(int(y*shape[0]+x), []).append(triangle)
            self._inverse_grid = lower, span, shape, buckets, np.array(transforms), np.array(anchors)
        lower, span, shape, buckets, transforms, anchors = self._inverse_grid
        finite = np.isfinite(points).all(axis=1)
        candidates = np.flatnonzero(finite & np.all((points >= lower) & (points <= lower+span), axis=1))
        cells = np.minimum(shape-1, np.floor((points[candidates]-lower)/span*shape).astype(int))
        keys = cells[:, 1]*shape[0]+cells[:, 0]
        order = np.argsort(keys)
        keys, candidates = keys[order], candidates[order]
        unique, offsets = np.unique(keys, return_index=True)
        for key, begin, end in zip(unique, offsets, np.r_[offsets[1:], len(keys)]):
            indexes = candidates[begin:end]
            for triangle in buckets.get(int(key), []):
                weights01 = (points[indexes]-anchors[triangle]) @ transforms[triangle].T
                weights = np.column_stack((weights01, 1-weights01.sum(axis=1)))
                keep = np.all(weights >= -1e-7, axis=1)
                result[indexes[keep]] = weights[keep] @ self.source[self.tri.simplices[triangle]]
        return result

    def as_dict(self):
        return {"affine": self.affine.tolist(), "source": self.source.tolist(),
                "target": self.target.tolist(), "triangles": self.tri.simplices.tolist() if self.tri is not None else [],
                "model": "regularized_piecewise_affine", "foldedTriangles": self.folded}

    @classmethod
    def from_dict(cls, value):
        return cls(value['affine'], value['source'], value['target'], regularization=0)
