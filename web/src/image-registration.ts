/** Automatic semantic map registration, manual anchor calibration and overlay warp.
 *
 * TypeScript port of `dirmap/registration.py`, plus the mesh construction half of
 * `dirmap/geometry.py::Registration` (evaluation is delegated to `geometry.ts`'s
 * `Mesh`). The module is DOM-free, uses no Node APIs and runs both in the browser
 * and under `node --experimental-strip-types`.
 *
 * Deliberate deviations from the Python reference (documented, not accidental):
 * - No SIFT. `registerToMap` matches analysed map-detail features against the base
 *   map's features projected through the base registration (same `Matcher`
 *   machinery: kind / area / aspect / principal angle), then reuses the
 *   observed-boundary transfer bookkeeping. opencv.js has no SIFT, so the SIFT
 *   descriptor path cannot be ported; quality `method` values say
 *   `feature_cross_map_*` instead of `sift_cross_map_*`.
 * - RANSAC is a deterministic custom implementation (seeded LCG, seed 7 like
 *   `cv2.setRNGSeed(7)`, 3-point samples, LS refine). opencv.js' RNG cannot be
 *   seeded, so `cv.estimateAffine2D` is not used.
 * - approxPolyDP / moments / fillPoly / convexHull / pointPolygonTest are native
 *   TypeScript ports of the OpenCV routines (deterministic, no WASM load);
 *   cross-checked against Python cv2 while porting.
 * - Delaunay uses our own incremental triangulation + Lawson flips (scipy uses
 *   Qhull); the triangulation is valid and orientation-preserving but can differ
 *   from scipy's on (near-)cocircular point sets.
 */
import { Mesh, frameLngLat, localScales } from './geometry.ts';
import type { RegistrationData } from './geometry.ts';
import type { Anchor, Control, MapState, Pixel } from './types.ts';
import type { RegionFeature, RgbaImage } from './vision.ts';

export type { RegionFeature, RgbaImage };

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Region feature produced by `vision.analyze()`; coordinates are native pixels. */
export type SourceFeature = RegionFeature;

/** Campus reference feature from `web/public/data/reference-features.json` (local metres). */
export interface ReferenceFeature extends RegionFeature {
  id: string;
}

/** Observed correspondence kept in `quality.landmarks` (shape mirrors Python dicts). */
export interface RegistrationLandmark {
  pixel: Pixel;
  local?: Pixel;
  purpose?: 'fit' | 'validation';
  referenceId?: string;
  kind?: string;
  sourceIndex?: number;
  errorM?: number;
  shapeError?: number;
  outlineIoU?: number;
  insideHull?: boolean;
  source?: string;
  [key: string]: unknown;
}

/**
 * Quality block of a registration. Always carries the `MapState['quality']`
 * fields; the additional fields are the ones the Python pipeline records.
 */
export type RegistrationQuality = Omit<NonNullable<MapState['quality']>, 'inheritedGeographicP95M'> & {
  inheritedGeographicP95M?: number | null;
  candidateTransforms?: number;
  observedCornerCount?: number;
  fitObjectCount?: number;
  validationObjectCount?: number;
  validationInsideHull?: number;
  rejectedObjectIds?: string[];
  affineFallback?: boolean;
  foldedTriangles?: number;
  maxErrorM?: number | null;
  landmarks?: RegistrationLandmark[];
  geographicBoundaryAnchors?: number;
  baseVersion?: number;
  baseSourceHash?: string;
  baseRegistrationHash?: string;
};

export interface AutoRegisterResult {
  registration: RegistrationData;
  quality: RegistrationQuality;
}

/** Already-registered base map used by the cross-map transfer fallback. */
export interface RegisterBase {
  features: SourceFeature[];
  registration: RegistrationData;
  quality: RegistrationQuality | null;
  id: string;
  sourceHash?: string;
}

export interface WarpOverlayOptions {
  /** Long side of the produced overlay. Default 1700 (bilinear) / 1200 (nearest). */
  maxSide?: number;
  /** Nearest-neighbour sampling (terrain overlay path). */
  nearest?: boolean;
  /** Frame origin override; defaults to the project frame [113.9304, 22.535]. */
  frameOrigin?: Pixel;
}

export interface WarpOverlayResult {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  extent: [number, number, number, number];
}

/** Result of the custom deterministic RANSAC affine fit. */
export interface AffineFit {
  matrix: number[][];
  inliers: boolean[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default WGS84 frame origin of the DirMap project (`geometry.Frame()`). */
const DEFAULT_ORIGIN: Pixel = [113.9304, 22.535];
const WGS84_RADIUS = 6378137;
const MAX_LATITUDE = 85.05112878;
const NUM_EPS = 2.220446049250313e-16;

const CHINESE = {
  insufficientPoints: '至少需要三组有效的同名点',
  collinear: '配准点不能共线',
  noStableAffine: '无法得到稳定配准',
  noWaterSeed: '缺少可匹配水体。请补充建筑/路口同名点，或导入校园参考。',
  weakEvidence: '自动地物匹配证据不足，请在原图与底图补充同名点',
  invalidAffine: '配准矩阵无效',
  invalidMeshPoints: '配准网格点无效',
  duplicatePoints: '配准点不能重复',
  foldedMesh: '局部配准出现折叠或边界重叠，请修正同名点',
  weakHoldout: '独立地物或覆盖范围不足，不能建立可核查的配准',
  conflictingAnchors: '手动同名点中存在冲突，请检查坐标和对应关系',
  validationOutside: '核查点超出配准点包围范围',
  invalidLngLat: '经纬度无效',
  noCommonFeatures: '两张地图之间没有足够的共同地物',
  insufficientTransfer: '两图共同地物匹配不足',
  insufficientCoverage: '两图有效覆盖区的匹配点不足'
} as const;

function weakEvidenceMessage(count: number): string {
  return `自动配准仅找到 ${count} 组稳定地物，需补充同名点`;
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

/** 2x3 affine matrix, row-major: [a, b, c, d, e, f] maps (x, y) -> (a x + b y + c, d x + e y + f). */
type Affine6 = [number, number, number, number, number, number];

function affineOf(matrix: number[][]): Affine6 {
  if (!Array.isArray(matrix) || matrix.length !== 2
    || !Array.isArray(matrix[0]) || matrix[0].length !== 3
    || !Array.isArray(matrix[1]) || matrix[1].length !== 3) {
    throw new Error(CHINESE.invalidAffine);
  }
  const values: Affine6 = [
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2],
  ];
  if (!values.every(value => Number.isFinite(value))) throw new Error(CHINESE.invalidAffine);
  return values;
}

function matrixOf(affine: Affine6): number[][] {
  return [[affine[0], affine[1], affine[2]], [affine[3], affine[4], affine[5]]];
}

function determinant2(affine: Affine6): number {
  return affine[0] * affine[4] - affine[1] * affine[3];
}

function applyAffine(affine: Affine6, x: number, y: number): Pixel {
  return [
    affine[0] * x + affine[1] * y + affine[2],
    affine[3] * x + affine[4] * y + affine[5],
  ];
}

function affinePoints(affine: Affine6, points: readonly Pixel[]): Pixel[] {
  const result: Pixel[] = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    const x = points[i][0];
    const y = points[i][1];
    result[i] = [
      affine[0] * x + affine[1] * y + affine[2],
      affine[3] * x + affine[4] * y + affine[5],
    ];
  }
  return result;
}

function isFinitePoint(point: readonly number[] | undefined | null): boolean {
  return !!point && Number.isFinite(point[0]) && Number.isFinite(point[1]);
}

function allFinitePoints(points: readonly Pixel[]): boolean {
  for (const point of points) {
    if (!isFinitePoint(point)) return false;
  }
  return true;
}

function clamp(value: number, lower: number, upper: number): number {
  return value < lower ? lower : value > upper ? upper : value;
}

/** numpy `np.round` (banker's rounding) on an integer-valued coordinate. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Python `round(value, digits)` for the quality report (2 decimal places). */
function round2(value: number): number {
  if (!Number.isFinite(value)) return value;
  const scaled = value * 100;
  const rounded = Math.round(scaled);
  // Normalise -0 like Python's round.
  return (rounded === 0 ? 0 : rounded) / 100;
}

function approxEqual(a: number, b: number, tolerance = 1e-9): boolean {
  return Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b));
}

/** numpy median (average of the two middle values for even counts). */
function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** numpy `percentile(values, p)` with the default linear interpolation. */
function percentile(values: number[], p: number): number {
  if (values.length === 0) return NaN;
  const sorted = values.slice().sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * (p / 100);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/** numpy `argsort` (stable, ascending). */
function argsortStable(values: number[]): number[] {
  const indices = new Array<number>(values.length);
  for (let i = 0; i < values.length; i++) indices[i] = i;
  indices.sort((a, b) => {
    const delta = values[a] - values[b];
    return delta < 0 ? -1 : delta > 0 ? 1 : a - b;
  });
  return indices;
}

// ---------------------------------------------------------------------------
// WGS84 / local-metre frame helpers (dirmap/geometry.py)
// ---------------------------------------------------------------------------

function frameScales(origin: Pixel): Pixel {
  return localScales(origin);
}

function toLocal(point: Pixel, origin: Pixel): Pixel {
  if (!isFinitePoint(point) || Math.abs(point[0]) > 180 || Math.abs(point[1]) > 90) {
    throw new Error(CHINESE.invalidLngLat);
  }
  const scales = frameScales(origin);
  return [(point[0] - origin[0]) * scales[0], (point[1] - origin[1]) * scales[1]];
}

function toLngLat(local: Pixel, origin: Pixel): Pixel {
  const scales = frameScales(origin);
  return [local[0] / scales[0] + origin[0], local[1] / scales[1] + origin[1]];
}

/** EPSG:3857 spherical Mercator (x, y) metres. */
function mercator(lnglat: Pixel): Pixel {
  const latitude = clamp(lnglat[1], -MAX_LATITUDE, MAX_LATITUDE);
  return [
    (lnglat[0] * Math.PI / 180) * WGS84_RADIUS,
    Math.log(Math.tan(Math.PI / 4 + (latitude * Math.PI / 180) / 2)) * WGS84_RADIUS,
  ];
}

function unmercator(xy: Pixel): Pixel {
  return [
    (xy[0] / WGS84_RADIUS) * 180 / Math.PI,
    (2 * Math.atan(Math.exp(xy[1] / WGS84_RADIUS)) - Math.PI / 2) * 180 / Math.PI,
  ];
}

// ---------------------------------------------------------------------------
// Principal angle (cv2.moments on a polygon contour)
// ---------------------------------------------------------------------------

interface PolygonMoments {
  m00: number;
  m10: number;
  m01: number;
  mu20: number;
  mu11: number;
  mu02: number;
}

/** Polygon moments, exactly the formulas cv2.moments uses for a point contour. */
function polygonMoments(points: readonly Pixel[]): PolygonMoments | null {
  const count = points.length;
  if (count < 3) return null;
  const xs = new Float64Array(count);
  const ys = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    xs[i] = points[i][0];
    ys[i] = points[i][1];
  }
  let m00 = 0;
  let m10 = 0;
  let m01 = 0;
  let m20 = 0;
  let m02 = 0;
  let m11 = 0;
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    const cross = xs[i] * ys[j] - xs[j] * ys[i];
    m00 += cross;
    m10 += (xs[i] + xs[j]) * cross;
    m01 += (ys[i] + ys[j]) * cross;
    m20 += (xs[i] * xs[i] + xs[i] * xs[j] + xs[j] * xs[j]) * cross;
    m02 += (ys[i] * ys[i] + ys[i] * ys[j] + ys[j] * ys[j]) * cross;
    m11 += (2 * xs[i] * ys[i] + xs[i] * ys[j] + xs[j] * ys[i] + 2 * xs[j] * ys[j]) * cross;
  }
  m00 /= 2;
  if (!Number.isFinite(m00)) return null;
  const m00Abs = Math.abs(m00);
  if (m00Abs < 1e-12) return { m00, m10: 0, m01: 0, mu20: 0, mu11: 0, mu02: 0 };
  m10 /= 6;
  m01 /= 6;
  m20 /= 12;
  m02 /= 12;
  m11 /= 24;
  const cx = m10 / m00;
  const cy = m01 / m00;
  return {
    m00,
    m10,
    m01,
    mu20: m20 - m00 * cx * cx,
    mu11: m11 - m00 * cx * cy,
    mu02: m02 - m00 * cy * cy,
  };
}

/**
 * Principal orientation (radians) of a polygon outline, from its second moments.
 * Same value as Python's `principal_angle` modulo pi (the initial Matcher grid
 * covers both directions via its 0/pi flip loop, so equivalent).
 */
export function principalAngle(points: readonly Pixel[]): number {
  const moments = polygonMoments(points);
  if (!moments || Math.abs(moments.m00) < 1) return 0;
  const covarianceXX = moments.mu20 / moments.m00;
  const covarianceXY = moments.mu11 / moments.m00;
  const covarianceYY = moments.mu02 / moments.m00;
  // Closed-form eigendecomposition of the symmetric 2x2 covariance matrix;
  // theta is the direction of the eigenvector of the largest eigenvalue.
  const theta = 0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY);
  return Number.isFinite(theta) ? theta : 0;
}

// ---------------------------------------------------------------------------
// Ramer-Douglas-Peucker (cv2.approxPolyDP closed=True) and corner filtering
// ---------------------------------------------------------------------------

/**
 * Port of OpenCV's `approxPolyDP` for closed contours (CV_32F semantics:
 * coordinates are float32, epsilon is a double). Returns the approximated
 * polygon vertices in input order.
 */
export function approxPolyDPClosed(points: readonly Pixel[], epsilon: number): Pixel[] {
  const count = points.length;
  if (count === 0) return [];
  if (!(epsilon >= 0)) throw new Error('epsilon');
  let eps = epsilon;
  eps *= eps;

  const src: Float32Array = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    src[i * 2] = points[i][0];
    src[i * 2 + 1] = points[i][1];
  }
  const px = (index: number) => src[(index % count) * 2];
  const py = (index: number) => src[(index % count) * 2 + 1];

  const dst: Pixel[] = [];
  const write = (point: Pixel) => { dst.push([point[0], point[1]]); };

  // 1. Find approximately two farthest points of the contour (3 init iterations).
  let rightStart = 0;
  let leftStart = 0;
  let pos = 0;
  let leEps = false;
  let startPoint: Pixel = [px(0), py(0)];
  for (let iteration = 0; iteration < 3; iteration++) {
    let maxDist = 0;
    pos = (pos + rightStart) % count;
    startPoint = [px(pos), py(pos)];
    pos = (pos + 1) % count;
    for (let j = 1; j < count; j++) {
      const x = px(pos);
      const y = py(pos);
      pos = (pos + 1) % count;
      const dx = x - startPoint[0];
      const dy = y - startPoint[1];
      const dist = dx * dx + dy * dy;
      if (dist > maxDist) {
        maxDist = dist;
        rightStart = j;
      }
    }
    leEps = maxDist <= eps;
  }

  interface Slice { start: number; end: number }
  const stack: Slice[] = [];
  if (!leEps) {
    const rightSlice: Slice = { start: 0, end: 0 };
    const slice: Slice = { start: 0, end: 0 };
    rightSlice.end = pos % count;
    slice.start = pos % count;
    slice.end = (rightStart + slice.start) % count;
    rightSlice.start = slice.end;
    // LIFO: push right first, slice second so slice is processed first.
    stack.push(rightSlice);
    stack.push(slice);
  } else {
    write(startPoint);
  }

  // 2. Recursive subdivision.
  while (stack.length > 0) {
    const slice = stack.pop() as Slice;
    const endPoint: Pixel = [px(slice.end), py(slice.end)];
    let cursor = slice.start;
    startPoint = [px(cursor), py(cursor)];
    cursor = (cursor + 1) % count;
    if (cursor !== slice.end) {
      const dx = endPoint[0] - startPoint[0];
      const dy = endPoint[1] - startPoint[1];
      let maxDist = 0;
      let splitStart = 0;
      while (cursor !== slice.end) {
        const x = px(cursor);
        const y = py(cursor);
        cursor = (cursor + 1) % count;
        const dist = Math.abs((y - startPoint[1]) * dx - (x - startPoint[0]) * dy);
        if (dist > maxDist) {
          maxDist = dist;
          splitStart = (cursor + count - 1) % count;
        }
      }
      leEps = maxDist * maxDist <= eps * (dx * dx + dy * dy);
      if (!leEps) {
        const rightSlice: Slice = { start: splitStart, end: slice.end };
        const leftSlice: Slice = { start: slice.start, end: splitStart };
        stack.push(rightSlice);
        stack.push(leftSlice);
        continue;
      }
    } else {
      startPoint = [px(slice.start), py(slice.start)];
    }
    write(startPoint);
  }

  // 3. Final cleanup: remove extra points on almost straight lines.
  const resultCount = dst.length;
  if (resultCount <= 2) return dst;
  const buffer: Pixel[] = dst.slice();
  let newCount = resultCount;
  let cleanupPos = resultCount - 1;
  let start: Pixel = buffer[cleanupPos];
  cleanupPos = (cleanupPos + 1) % resultCount;
  let writePos = cleanupPos;
  let current: Pixel = buffer[cleanupPos];
  cleanupPos = (cleanupPos + 1) % resultCount;
  for (let i = 0; i < resultCount && newCount > 2; i++) {
    const end: Pixel = buffer[cleanupPos];
    cleanupPos = (cleanupPos + 1) % resultCount;
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const dist = Math.abs((current[0] - start[0]) * dy - (current[1] - start[1]) * dx);
    const successiveInnerProduct = (current[0] - start[0]) * (end[0] - current[0])
      + (current[1] - start[1]) * (end[1] - current[1]);
    if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx !== 0 && dy !== 0
      && successiveInnerProduct >= 0) {
      newCount--;
      buffer[writePos] = end;
      start = end;
      writePos = (writePos + 1) % resultCount;
      current = buffer[cleanupPos];
      cleanupPos = (cleanupPos + 1) % resultCount;
      i++;
      continue;
    }
    buffer[writePos] = current;
    start = current;
    writePos = (writePos + 1) % resultCount;
    current = end;
  }
  return buffer.slice(0, newCount);
}

/**
 * Port of `dirmap.registration._corners`: approximate the outline and keep only
 * real corners (drop near-straight samples of coastlines and walls).
 */
export function corners(points: readonly Pixel[], tolerance: number): Pixel[] {
  let outline: Pixel[] = points.map(point => [Math.fround(point[0]), Math.fround(point[1])]);
  const count = outline.length;
  if (count > 1 && Math.hypot(outline[0][0] - outline[count - 1][0], outline[0][1] - outline[count - 1][1]) < 1e-5) {
    outline = outline.slice(0, -1);
  }
  if (outline.length < 3) return [];
  const approximated = approxPolyDPClosed(outline, tolerance);
  if (approximated.length === 0) return [];
  const previous = new Float64Array(approximated.length * 2);
  const following = new Float64Array(approximated.length * 2);
  for (let i = 0; i < approximated.length; i++) {
    const before = approximated[(i + approximated.length - 1) % approximated.length];
    const after = approximated[(i + 1) % approximated.length];
    previous[i * 2] = before[0] - approximated[i][0];
    previous[i * 2 + 1] = before[1] - approximated[i][1];
    following[i * 2] = after[0] - approximated[i][0];
    following[i * 2 + 1] = after[1] - approximated[i][1];
  }
  const kept: Pixel[] = [];
  const limit = tolerance * tolerance * 4;
  for (let i = 0; i < approximated.length; i++) {
    const previousLength = Math.hypot(previous[i * 2], previous[i * 2 + 1]);
    const followingLength = Math.hypot(following[i * 2], following[i * 2 + 1]);
    const norms = previousLength * followingLength;
    const cosine = (previous[i * 2] * following[i * 2] + previous[i * 2 + 1] * following[i * 2 + 1])
      / Math.max(norms, 1e-8);
    if (cosine > -0.91 && norms > limit) kept.push(approximated[i]);
  }
  return kept;
}

// ---------------------------------------------------------------------------
// Polygon rasterisation (cv2.fillPoly: outline + scanline spans) and IoU
// ---------------------------------------------------------------------------

/** Bresenham line, 8-connected, both endpoints inclusive (cv::Line semantics). */
function drawLine(mask: Uint8Array, width: number, height: number, x0: number, y0: number, x1: number, y1: number): void {
  let x = x0;
  let y = y0;
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let error = dx - dy;
  for (;;) {
    if (x >= 0 && x < width && y >= 0 && y < height) mask[y * width + x] = 1;
    if (x === x1 && y === y1) break;
    const doubled = 2 * error;
    if (doubled > -dy) {
      error -= dy;
      x += sx;
    }
    if (doubled < dx) {
      error += dx;
      y += sy;
    }
  }
}

interface FillEdge {
  y0: number;
  y1: number;
  x: number;
  dx: number;
}

/**
 * Pixel coverage of `cv2.fillPoly(mask, [polygon], 1)` for integer coordinates:
 * Bresenham outline plus scanline span filling at integer rows. Matches OpenCV's
 * edge-collection rasterisation closely (fixed-point x is reproduced exactly).
 */
function rasterizePolygon(points: readonly Pixel[], width: number, height: number, mask: Uint8Array): void {
  const count = points.length;
  if (count === 0 || width <= 0 || height <= 0) return;
  const xs = new Int32Array(count);
  const ys = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    xs[i] = roundHalfEven(points[i][0]);
    ys[i] = roundHalfEven(points[i][1]);
  }
  // Outline first (cv2.fillPoly draws the contour before filling spans).
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    drawLine(mask, width, height, xs[i], ys[i], xs[j], ys[j]);
  }
  const edges: FillEdge[] = [];
  const fixed = 65536;
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    if (ys[i] === ys[j]) continue;
    const fromX = xs[i] * fixed;
    const toX = xs[j] * fixed;
    const dx = Math.trunc((toX - fromX) / (ys[j] - ys[i]));
    if (ys[i] < ys[j]) {
      edges.push({ y0: ys[i], y1: ys[j], x: fromX, dx });
    } else {
      edges.push({ y0: ys[j], y1: ys[i], x: toX, dx });
    }
  }
  if (edges.length < 2) return;
  edges.sort((a, b) => (a.y0 - b.y0) || (a.x - b.x) || (a.dx - b.dx));
  const firstY = edges[0].y0;
  let lastY = edges[0].y1;
  for (const edge of edges) if (edge.y1 > lastY) lastY = edge.y1;
  const endY = Math.min(lastY, height);
  const active: FillEdge[] = [];
  let next = 0;
  for (let y = firstY; y < endY; y++) {
    if (y < 0) continue;
    // Remove edges that end at this row.
    for (let i = active.length - 1; i >= 0; i--) {
      if (active[i].y1 === y) active.splice(i, 1);
    }
    // Insert edges that start at this row, keeping the active list x-sorted.
    const beginning: FillEdge[] = [];
    while (next < edges.length && edges[next].y0 === y) {
      beginning.push(edges[next]);
      next++;
    }
    if (beginning.length > 0) {
      for (const edge of beginning) {
        let position = active.length;
        for (let i = 0; i < active.length; i++) {
          if (active[i].x >= edge.x) { position = i; break; }
        }
        active.splice(position, 0, edge);
      }
    }
    for (let i = 0; i + 1 < active.length; i += 2) {
      const first = active[i];
      const second = active[i + 1];
      const xMin = first.x < second.x ? first.x : second.x;
      const xMax = first.x < second.x ? second.x : first.x;
      let x1 = Math.floor((xMin + (fixed - 1)) / fixed);
      let x2 = Math.floor(xMax / fixed);
      if (x1 < 0) x1 = 0;
      if (x2 >= width) x2 = width - 1;
      const row = y * width;
      for (let x = x1; x <= x2; x++) mask[row + x] = 1;
    }
    for (const edge of active) edge.x += edge.dx;
  }
}

/**
 * Port of `dirmap.registration._outline_iou`: rasterise both outlines on a shared
 * grid and return intersection/union pixel ratio.
 */
export function outlineIou(first: readonly Pixel[], second: readonly Pixel[]): number {
  if (first.length === 0 || second.length === 0) return 0;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of first) {
    if (point[0] < minX) minX = point[0];
    if (point[1] < minY) minY = point[1];
    if (point[0] > maxX) maxX = point[0];
    if (point[1] > maxY) maxY = point[1];
  }
  for (const point of second) {
    if (point[0] < minX) minX = point[0];
    if (point[1] < minY) minY = point[1];
    if (point[0] > maxX) maxX = point[0];
    if (point[1] > maxY) maxY = point[1];
  }
  const lowerX = Math.floor(minX) - 2;
  const lowerY = Math.floor(minY) - 2;
  const upperX = Math.ceil(maxX) + 2;
  const upperY = Math.ceil(maxY) + 2;
  const width = Math.max(1, Math.ceil(upperX - lowerX));
  const height = Math.max(1, Math.ceil(upperY - lowerY));
  if (width * height > 1_000_000) return 0;
  const shiftedFirst: Pixel[] = first.map(point => [point[0] - lowerX, point[1] - lowerY]);
  const shiftedSecond: Pixel[] = second.map(point => [point[0] - lowerX, point[1] - lowerY]);
  const a = new Uint8Array(width * height);
  const b = new Uint8Array(width * height);
  rasterizePolygon(shiftedFirst, width, height, a);
  rasterizePolygon(shiftedSecond, width, height, b);
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const valueA = a[i];
    const valueB = b[i];
    if (valueA && valueB) intersection++;
    if (valueA || valueB) union++;
  }
  return intersection / Math.max(1, union);
}

// ---------------------------------------------------------------------------
// Convex hull / convexity / point-polygon distance (cv2 equivalents)
// ---------------------------------------------------------------------------

/** Monotone-chain convex hull; returns the indices of hull vertices (no collinear points). */
function convexHullIndices(points: readonly Pixel[]): number[] {
  const count = points.length;
  if (count < 3) {
    const trivial = new Array<number>(count);
    for (let i = 0; i < count; i++) trivial[i] = i;
    return trivial;
  }
  const order = new Array<number>(count);
  for (let i = 0; i < count; i++) order[i] = i;
  order.sort((a, b) => (points[a][0] - points[b][0]) || (points[a][1] - points[b][1]) || (a - b));
  const cross = (o: number, a: number, b: number) =>
    (points[a][0] - points[o][0]) * (points[b][1] - points[o][1])
    - (points[a][1] - points[o][1]) * (points[b][0] - points[o][0]);
  const lower: number[] = [];
  for (const index of order) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], index) <= 0) lower.pop();
    lower.push(index);
  }
  const upper: number[] = [];
  for (let i = order.length - 1; i >= 0; i--) {
    const index = order[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], index) <= 0) upper.pop();
    upper.push(index);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** cv2.isContourConvex: all turns must have the same orientation. */
function isContourConvex(points: readonly Pixel[]): boolean {
  const count = points.length;
  if (count < 3) return false;
  let orientation = 0;
  for (let i = 0; i < count; i++) {
    const previous = points[(i + count - 1) % count];
    const current = points[i];
    const next = points[(i + 1) % count];
    const dx1 = current[0] - previous[0];
    const dy1 = current[1] - previous[1];
    const dx2 = next[0] - previous[0];
    const dy2 = next[1] - previous[1];
    const turn = dx1 * dy2 - dy1 * dx2;
    if (turn === 0) continue;
    const sign = turn > 0 ? 1 : -1;
    if (orientation === 0) orientation = sign;
    else if (orientation !== sign) return false;
  }
  return true;
}

/** Distance from a point to the closed polygon boundary (mirrors |cv2.pointPolygonTest(.., True)|). */
function pointPolygonDistance(points: readonly Pixel[], x: number, y: number): number {
  let best = Infinity;
  const count = points.length;
  for (let i = 0; i < count; i++) {
    const a = points[i];
    const b = points[(i + 1) % count];
    const abx = b[0] - a[0];
    const aby = b[1] - a[1];
    const lengthSquared = abx * abx + aby * aby;
    let t = 0;
    if (lengthSquared > 0) t = clamp(((x - a[0]) * abx + (y - a[1]) * aby) / lengthSquared, 0, 1);
    const dx = x - (a[0] + t * abx);
    const dy = y - (a[1] + t * aby);
    const distance = Math.hypot(dx, dy);
    if (distance < best) best = distance;
  }
  return best;
}

// ---------------------------------------------------------------------------
// Nearest-neighbour search over feature centres (scipy cKDTree equivalent)
// ---------------------------------------------------------------------------

interface KdLeaf {
  leaf: true;
  indices: number[];
}

interface KdBranch {
  leaf: false;
  axis: 0 | 1;
  split: number;
  left: KdNode;
  right: KdNode;
}

type KdNode = KdLeaf | KdBranch;

const KD_LEAF_SIZE = 8;

class FeatureKdTree {
  private readonly xs: Float64Array;
  private readonly ys: Float64Array;
  private readonly root: KdNode;

  constructor(points: readonly Pixel[]) {
    const count = points.length;
    this.xs = new Float64Array(count);
    this.ys = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      this.xs[i] = points[i][0];
      this.ys[i] = points[i][1];
    }
    const indices = new Array<number>(count);
    for (let i = 0; i < count; i++) indices[i] = i;
    this.root = this.build(indices, 0);
  }

  private build(indices: number[], depth: number): KdNode {
    if (indices.length <= KD_LEAF_SIZE) return { leaf: true, indices };
    const axis: 0 | 1 = (depth % 2) as 0 | 1;
    const coordinates = axis === 0 ? this.xs : this.ys;
    indices.sort((a, b) => (coordinates[a] - coordinates[b]) || (a - b));
    const middle = indices.length >> 1;
    return {
      leaf: false,
      axis,
      split: coordinates[indices[middle]],
      left: this.build(indices.slice(0, middle), depth + 1),
      right: this.build(indices.slice(middle), depth + 1),
    };
  }

  /** k nearest neighbours; returns distances and indices ordered by distance (then index). */
  query(x: number, y: number, k: number): { distances: Float64Array; indexes: Int32Array } {
    const count = Math.max(1, Math.min(k, this.xs.length));
    const bestDistances = new Float64Array(count).fill(Infinity);
    const bestIndexes = new Int32Array(count).fill(-1);
    this.search(this.root, x, y, bestDistances, bestIndexes, count);
    return { distances: bestDistances, indexes: bestIndexes };
  }

  private search(node: KdNode, x: number, y: number, bestDistances: Float64Array,
    bestIndexes: Int32Array, count: number): void {
    if (node.leaf) {
      for (const index of node.indices) {
        const dx = x - this.xs[index];
        const dy = y - this.ys[index];
        const distance = dx * dx + dy * dy;
        if (distance >= bestDistances[count - 1]) continue;
        let position = count - 1;
        while (position > 0 && (bestDistances[position - 1] > distance
          || (bestDistances[position - 1] === distance && bestIndexes[position - 1] > index))) {
          bestDistances[position] = bestDistances[position - 1];
          bestIndexes[position] = bestIndexes[position - 1];
          position--;
        }
        bestDistances[position] = distance;
        bestIndexes[position] = index;
      }
      return;
    }
    const coordinate = node.axis === 0 ? x : y;
    const delta = coordinate - node.split;
    const near = delta <= 0 ? node.left : node.right;
    const far = delta <= 0 ? node.right : node.left;
    this.search(near, x, y, bestDistances, bestIndexes, count);
    if (delta * delta < bestDistances[count - 1]) {
      this.search(far, x, y, bestDistances, bestIndexes, count);
    }
  }
}

/** Brute-force nearest neighbour (small point sets; same ordering semantics as cKDTree.query). */
function nearestIndex(points: readonly Pixel[], x: number, y: number): { distance: number; index: number } {
  let bestDistance = Infinity;
  let bestIndex = -1;
  for (let i = 0; i < points.length; i++) {
    const dx = x - points[i][0];
    const dy = y - points[i][1];
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance || (distance === bestDistance && (bestIndex < 0 || i < bestIndex))) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return { distance: Math.sqrt(bestDistance), index: bestIndex };
}

// ---------------------------------------------------------------------------
// Deterministic RANSAC affine fit (dirmap/geometry.py::fit_affine)
// ---------------------------------------------------------------------------

function determinant3(matrix: readonly number[]): number {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
    - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
    + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

function solveLinear3(matrix: readonly number[], rhs: readonly number[]): [number, number, number] | null {
  const determinant = determinant3(matrix);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-14) return null;
  const replace = (column: number): number[] => {
    const copy = matrix.slice();
    copy[column] = rhs[0];
    copy[3 + column] = rhs[1];
    copy[6 + column] = rhs[2];
    return copy;
  };
  return [
    determinant3(replace(0)) / determinant,
    determinant3(replace(1)) / determinant,
    determinant3(replace(2)) / determinant,
  ];
}

/** Affine from three point pairs (6 equations, Cramer's rule). */
function affineFromThree(correspondences: readonly [number, number, number, number][]): Affine6 | null {
  const rowX: number[] = [];
  const rowY: number[] = [];
  for (const [x, y, targetX, targetY] of correspondences) {
    rowX.push(x, y, 1, targetX);
    rowY.push(x, y, 1, targetY);
  }
  // Same 3x3 design matrix for both output coordinates.
  const design = [rowX[0], rowX[1], rowX[2], rowX[4], rowX[5], rowX[6], rowX[8], rowX[9], rowX[10]];
  const first = solveLinear3(design, [rowX[3], rowX[7], rowX[11]]);
  if (!first) return null;
  const second = solveLinear3(design, [rowY[3], rowY[7], rowY[11]]);
  if (!second) return null;
  const affine: Affine6 = [first[0], first[1], first[2], second[0], second[1], second[2]];
  if (!affine.every(Number.isFinite) || Math.abs(determinant2(affine)) < 1e-12) return null;
  return affine;
}

/** Least-squares affine on a subset (normal equations, closes over all output rows). */
function affineLeastSquares(source: readonly Pixel[], target: readonly Pixel[], use: Uint8Array | null): Affine6 | null {
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  let sumYY = 0;
  let sumTargetX = 0;
  let sumTargetY = 0;
  let sumXTargetX = 0;
  let sumYTargetX = 0;
  let sumXTargetY = 0;
  let sumYTargetY = 0;
  for (let i = 0; i < source.length; i++) {
    if (use !== null && !use[i]) continue;
    const x = source[i][0];
    const y = source[i][1];
    const targetX = target[i][0];
    const targetY = target[i][1];
    count++;
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumXY += x * y;
    sumYY += y * y;
    sumTargetX += targetX;
    sumTargetY += targetY;
    sumXTargetX += x * targetX;
    sumYTargetX += y * targetX;
    sumXTargetY += x * targetY;
    sumYTargetY += y * targetY;
  }
  if (count < 3) return null;
  const design = [sumXX, sumXY, sumX, sumXY, sumYY, sumY, sumX, sumY, count];
  const first = solveLinear3(design, [sumXTargetX, sumYTargetX, sumTargetX]);
  if (!first) return null;
  const second = solveLinear3(design, [sumXTargetY, sumYTargetY, sumTargetY]);
  if (!second) return null;
  const affine: Affine6 = [first[0], first[1], first[2], second[0], second[1], second[2]];
  if (!affine.every(Number.isFinite) || Math.abs(determinant2(affine)) < 1e-12) return null;
  return affine;
}

function countInliers(source: readonly Pixel[], target: readonly Pixel[], affine: Affine6, threshold: number): { count: number; mask: Uint8Array } {
  const mask = new Uint8Array(source.length);
  const thresholdSquared = threshold * threshold;
  let count = 0;
  for (let i = 0; i < source.length; i++) {
    const x = source[i][0];
    const y = source[i][1];
    const errorX = affine[0] * x + affine[1] * y + affine[2] - target[i][0];
    const errorY = affine[3] * x + affine[4] * y + affine[5] - target[i][1];
    if (errorX * errorX + errorY * errorY <= thresholdSquared) {
      mask[i] = 1;
      count++;
    }
  }
  return { count, mask };
}

/** Deterministic LCG matching the spirit of `cv2.setRNGSeed(7)` (seed fixed, repeatable). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * Robust affine fit: deterministic custom RANSAC (3-point samples, seeded LCG,
 * reprojection threshold, least-squares refinement), returning the model and
 * the inlier mask. Semantics match `dirmap.geometry.fit_affine` but the random
 * sampling sequence intentionally differs (opencv.js RNG is not seedable).
 */
export function fitAffine(source: readonly Pixel[], target: readonly Pixel[], threshold = 15): AffineFit {
  const count = source.length;
  if (count < 3 || target.length !== count || !allFinitePoints(source) || !allFinitePoints(target)) {
    throw new Error(CHINESE.insufficientPoints);
  }
  // Rank of the centred source point set must be 2 (points not collinear).
  let meanX = 0;
  let meanY = 0;
  for (const point of source) {
    meanX += point[0];
    meanY += point[1];
  }
  meanX /= count;
  meanY /= count;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const point of source) {
    const dx = point[0] - meanX;
    const dy = point[1] - meanY;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const trace = sxx + syy;
  const determinant = sxx * syy - sxy * sxy;
  const discriminant = Math.max(0, trace * trace / 4 - determinant);
  const largest = trace / 2 + Math.sqrt(discriminant);
  const smallest = Math.max(0, trace / 2 - Math.sqrt(discriminant));
  const rankTolerance = count * NUM_EPS * Math.sqrt(Math.max(0, largest));
  if (Math.sqrt(smallest) <= rankTolerance) throw new Error(CHINESE.collinear);

  const maximumIterations = 4000;
  const confidence = 0.999;
  const random = createRandom(7);
  let bestAffine: Affine6 | null = null;
  let bestMask: Uint8Array | null = null;
  let bestCount = -1;
  let iterations = maximumIterations;

  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    if (iteration >= iterations) break;
    // Sample three distinct correspondences.
    const sampleIndexes: number[] = [];
    let guard = 0;
    while (sampleIndexes.length < 3 && guard++ < 100) {
      const index = Math.floor(random() * count);
      if (index >= count || sampleIndexes.includes(index)) continue;
      sampleIndexes.push(index);
    }
    if (sampleIndexes.length < 3) continue;
    const sample: [number, number, number, number][] = sampleIndexes.map(index =>
      [source[index][0], source[index][1], target[index][0], target[index][1]]);
    const candidate = affineFromThree(sample);
    if (!candidate) continue;
    const inliers = countInliers(source, target, candidate, threshold);
    if (inliers.count > bestCount) {
      bestCount = inliers.count;
      bestAffine = candidate;
      bestMask = inliers.mask;
      const ratio = inliers.count / count;
      if (ratio > 0 && ratio < 1) {
        const sampleProbability = ratio * ratio * ratio;
        const denominator = Math.log(1 - sampleProbability);
        if (denominator < 0) {
          const required = Math.log(1 - confidence) / denominator;
          if (Number.isFinite(required) && required < iterations) iterations = Math.ceil(required);
        }
      } else if (ratio === 1) {
        iterations = iteration + 1;
      }
    }
    if (iteration + 1 >= iterations) break;
  }

  if (!bestAffine || !bestMask) throw new Error(CHINESE.noStableAffine);
  // `refineIters=20` in Python: iteratively re-fit on the inliers.
  let currentAffine = bestAffine;
  let currentMask = bestMask;
  let currentCount = bestCount;
  for (let refine = 0; refine < 20; refine++) {
    const refined = affineLeastSquares(source, target, currentMask);
    if (!refined) break;
    const inliers = countInliers(source, target, refined, threshold);
    if (inliers.count < currentCount) break;
    const stable = inliers.count === currentCount;
    currentAffine = refined;
    currentMask = inliers.mask;
    currentCount = inliers.count;
    if (stable) break;
  }
  if (!currentAffine.every(Number.isFinite) || Math.abs(determinant2(currentAffine)) < 1e-10) {
    throw new Error(CHINESE.noStableAffine);
  }
  const inliers: boolean[] = new Array(source.length);
  for (let i = 0; i < source.length; i++) inliers[i] = currentMask[i] === 1;
  return { matrix: matrixOf(currentAffine), inliers };
}

// ---------------------------------------------------------------------------
// Delaunay triangulation (own incremental triangulation + Lawson flips)
// ---------------------------------------------------------------------------

function cross2(ax: number, ay: number, bx: number, by: number): number {
  return ax * by - ay * bx;
}

function signedArea(a: Pixel, b: Pixel, c: Pixel): number {
  return cross2(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]);
}

function inCircle(ax: number, ay: number, bx: number, by: number,
  cx: number, cy: number, dx: number, dy: number): number {
  const adx = ax - dx;
  const ady = ay - dy;
  const bdx = bx - dx;
  const bdy = by - dy;
  const cdx = cx - dx;
  const cdy = cy - dy;
  const bdxcdy = bdx * cdy;
  const cdxbdy = cdx * bdy;
  const alift = adx * adx + ady * ady;
  const blift = bdx * bdx + bdy * bdy;
  const clift = cdx * cdx + cdy * cdy;
  return (bdxcdy - cdxbdy) * alift + (cdx * ady - adx * cdy) * blift + (adx * bdy - bdx * ady) * clift;
}

/**
 * Delaunay triangulation of `points`. Returns index triples with positive
 * orientation (counter-clockwise in the coordinate system). Deterministic:
 * hull fan first, then incremental insertion in input order, then Lawson flips.
 */
export function delaunayTriangles(points: readonly Pixel[]): number[][] {
  const count = points.length;
  if (count < 3) return [];
  const hull = convexHullIndices(points);
  if (hull.length < 3) throw new Error(CHINESE.collinear);
  const triangles: number[][] = [];
  const pushTriangle = (a: number, b: number, c: number): void => {
    if (a === b || b === c || a === c) return;
    const area = signedArea(points[a], points[b], points[c]);
    if (Math.abs(area) < 1e-12) return;
    triangles.push(area > 0 ? [a, b, c] : [a, c, b]);
  };
  for (let i = 1; i + 1 < hull.length; i++) pushTriangle(hull[0], hull[i], hull[i + 1]);
  const onHull = new Set(hull);

  // Insert the non-hull points, splitting the containing triangle (or the two
  // triangles sharing an edge when the point falls exactly on an edge).
  for (let index = 0; index < count; index++) {
    if (onHull.has(index)) continue;
    const point = points[index];
    let located = false;
    for (let triangleIndex = 0; triangleIndex < triangles.length && !located; triangleIndex++) {
      const [ia, ib, ic] = triangles[triangleIndex];
      const a = points[ia];
      const b = points[ib];
      const c = points[ic];
      // Signed perpendicular distances to the three edges (positive inside).
      const dab = signedArea(a, b, point) / Math.hypot(b[0] - a[0], b[1] - a[1]);
      const dbc = signedArea(b, c, point) / Math.hypot(c[0] - b[0], c[1] - b[1]);
      const dca = signedArea(c, a, point) / Math.hypot(a[0] - c[0], a[1] - c[1]);
      if (dab < -1e-9 || dbc < -1e-9 || dca < -1e-9) continue;
      const scale = Math.max(1, Math.max(Math.abs(point[0]), Math.abs(point[1])));
      const edgeTolerance = 1e-12 * scale;
      if (Math.abs(dab) <= edgeTolerance) {
        // On edge a-b: split both adjacent triangles.
        triangles.splice(triangleIndex, 1);
        pushTriangle(ia, index, ic);
        pushTriangle(index, ib, ic);
        for (let other = 0; other < triangles.length; other++) {
          const candidate = triangles[other];
          if (candidate.includes(ia) && candidate.includes(ib) && !candidate.includes(ic)) {
            const opposite = candidate.find(vertex => vertex !== ia && vertex !== ib) as number;
            triangles.splice(other, 1);
            pushTriangle(ia, index, opposite);
            pushTriangle(index, ib, opposite);
            break;
          }
        }
        located = true;
      } else if (Math.abs(dbc) <= edgeTolerance) {
        triangles.splice(triangleIndex, 1);
        pushTriangle(ib, index, ia);
        pushTriangle(index, ic, ia);
        for (let other = 0; other < triangles.length; other++) {
          const candidate = triangles[other];
          if (candidate.includes(ib) && candidate.includes(ic) && !candidate.includes(ia)) {
            const opposite = candidate.find(vertex => vertex !== ib && vertex !== ic) as number;
            triangles.splice(other, 1);
            pushTriangle(ib, index, opposite);
            pushTriangle(index, ic, opposite);
            break;
          }
        }
        located = true;
      } else if (Math.abs(dca) <= edgeTolerance) {
        triangles.splice(triangleIndex, 1);
        pushTriangle(ic, index, ib);
        pushTriangle(index, ia, ib);
        for (let other = 0; other < triangles.length; other++) {
          const candidate = triangles[other];
          if (candidate.includes(ic) && candidate.includes(ia) && !candidate.includes(ib)) {
            const opposite = candidate.find(vertex => vertex !== ic && vertex !== ia) as number;
            triangles.splice(other, 1);
            pushTriangle(ic, index, opposite);
            pushTriangle(index, ia, opposite);
            break;
          }
        }
        located = true;
      } else {
        triangles.splice(triangleIndex, 1);
        pushTriangle(ia, ib, index);
        pushTriangle(ib, ic, index);
        pushTriangle(ic, ia, index);
        located = true;
      }
    }
    if (!located) {
      // Numerically outside its hull (should not happen); insert it on the hull
      // boundary by connecting it to every visible hull edge.
      for (let i = 0; i < count; i++) {
        if (i !== index && onHull.has(i)) continue;
      }
      throw new Error(CHINESE.collinear);
    }
  }

  // Lawson flips until the triangulation is Delaunay (or a sweep cap is hit).
  const maxSweeps = 200;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    if (!legalizeSweep(points, triangles)) break;
  }
  return triangles;
}

/** One pass of Delaunay edge legalisation; returns true when something flipped. */
function legalizeSweep(points: readonly Pixel[], triangles: number[][]): boolean {
  // Edge key -> up to two triangle indices.
  const edgeMap = new Map<number, number[]>();
  const n = points.length;
  for (let t = 0; t < triangles.length; t++) {
    const triangle = triangles[t];
    for (let k = 0; k < 3; k++) {
      const a = triangle[k];
      const b = triangle[(k + 1) % 3];
      const key = Math.min(a, b) * n + Math.max(a, b);
      const bucket = edgeMap.get(key);
      if (bucket) bucket.push(t);
      else edgeMap.set(key, [t]);
    }
  }
  let changed = false;
  for (const [key, bucket] of edgeMap) {
    if (bucket.length !== 2) continue;
    const [t1, t2] = bucket;
    const triangle1 = triangles[t1];
    const triangle2 = triangles[t2];
    // Find the shared edge and the opposite vertices.
    let u = -1;
    let v = -1;
    let apex1 = -1;
    for (let k = 0; k < 3; k++) {
      const a = triangle1[k];
      const b = triangle1[(k + 1) % 3];
      if (triangle2.includes(a) && triangle2.includes(b)) {
        u = a;
        v = b;
        apex1 = triangle1[(k + 2) % 3];
        break;
      }
    }
    if (u < 0 || apex1 < 0) continue;
    const apex2 = triangle2.find(vertex => vertex !== u && vertex !== v) as number;
    if (apex2 === undefined) continue;
    void key;
    const pa = points[u];
    const pb = points[v];
    const pc = points[apex1];
    const pd = points[apex2];
    // Convex quadrilateral: u and v must lie on opposite sides of apex1-apex2.
    const sideU = signedArea(pc, pd, pa);
    const sideV = signedArea(pc, pd, pb);
    if (sideU === 0 || sideV === 0 || Math.sign(sideU) === Math.sign(sideV)) continue;
    const determinant = inCircle(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], pd[0], pd[1]);
    // Relative tolerance so exactly cocircular points do not flip.
    const permanent = Math.abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1]))
      * ((pd[0] - pa[0]) ** 2 + (pd[1] - pa[1]) ** 2);
    if (determinant <= 1e-14 * Math.max(1, permanent)) continue;
    // Flip edge u-v to apex1-apex2.
    triangles[t1] = [apex1, u, apex2];
    triangles[t2] = [apex1, apex2, v];
    // Fix orientation if the flip produced a reversed triangle.
    for (const index of [t1, t2]) {
      const [a, b, c] = triangles[index];
      if (signedArea(points[a], points[b], points[c]) < 0) triangles[index] = [a, c, b];
    }
    changed = true;
  }
  return changed;
}

// ---------------------------------------------------------------------------
// Registration mesh construction (dirmap/geometry.py::Registration)
// ---------------------------------------------------------------------------

interface RegistrationBuild {
  registration: RegistrationData;
  folded: number;
}

/** Dense `(I + regularization * L) x = delta` solve via Cholesky factorisation. */
function solveRegularizedLaplacian(pointCount: number, triangles: readonly number[][],
  deltaX: Float64Array, deltaY: Float64Array, regularization: number): { x: Float64Array; y: Float64Array } {
  const size = pointCount;
  const matrix = new Float64Array(size * size);
  for (let i = 0; i < size; i++) matrix[i * size + i] = 1;
  const edgeKeys = new Set<number>();
  for (const triangle of triangles) {
    for (let k = 0; k < 3; k++) {
      const i = triangle[k];
      const j = triangle[(k + 1) % 3];
      const low = Math.min(i, j);
      const high = Math.max(i, j);
      const key = low * size + high;
      if (edgeKeys.has(key)) continue;
      edgeKeys.add(key);
      // Graph Laplacian: L[i][i] += 1, L[j][j] += 1, L[i][j] -= 1, L[j][i] -= 1.
      matrix[low * size + low] += regularization;
      matrix[high * size + high] += regularization;
      matrix[low * size + high] -= regularization;
      matrix[high * size + low] -= regularization;
    }
  }
  // Cholesky factorisation (in place, lower triangular).
  for (let j = 0; j < size; j++) {
    let diagonal = matrix[j * size + j];
    for (let k = 0; k < j; k++) diagonal -= matrix[j * size + k] * matrix[j * size + k];
    const pivot = Math.sqrt(Math.max(diagonal, 1e-15));
    matrix[j * size + j] = pivot;
    for (let i = j + 1; i < size; i++) {
      let value = matrix[i * size + j];
      for (let k = 0; k < j; k++) value -= matrix[i * size + k] * matrix[j * size + k];
      matrix[i * size + j] = value / pivot;
    }
  }
  const forwardBack = (rhs: Float64Array): Float64Array => {
    const intermediate = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      let value = rhs[i];
      for (let k = 0; k < i; k++) value -= matrix[i * size + k] * intermediate[k];
      intermediate[i] = value / matrix[i * size + i];
    }
    const result = new Float64Array(size);
    for (let i = size - 1; i >= 0; i--) {
      let value = intermediate[i];
      for (let k = i + 1; k < size; k++) value -= matrix[k * size + i] * result[k];
      result[i] = value / matrix[i * size + i];
    }
    return result;
  };
  return { x: forwardBack(deltaX), y: forwardBack(deltaY) };
}

/**
 * Build the registration mesh: global affine plus a regularized piecewise-affine
 * correction, rejecting folded triangles or a non-convex measured boundary.
 * Equivalent to `Registration.__init__` in `dirmap/geometry.py`.
 */
export function buildRegistration(affineMatrix: number[][], sourcePoints: readonly Pixel[],
  targetPoints: readonly Pixel[], regularization = 1.5): RegistrationBuild {
  const affine = affineOf(affineMatrix);
  if (Math.abs(determinant2(affine)) < 1e-10) throw new Error(CHINESE.invalidAffine);
  const source: Pixel[] = sourcePoints.map(point => [point[0], point[1]]);
  const target: Pixel[] = targetPoints.map(point => [point[0], point[1]]);
  if (source.length < 3) {
    return { registration: { affine: matrixOf(affine), source, target, triangles: [] }, folded: 0 };
  }
  if (source.length !== target.length || !allFinitePoints(source) || !allFinitePoints(target)) {
    throw new Error(CHINESE.invalidMeshPoints);
  }
  // Exact-duplicate check like np.unique(source, axis=0).
  const seen = new Set<string>();
  for (const point of source) {
    const key = `${point[0]},${point[1]}`;
    if (seen.has(key)) throw new Error(CHINESE.duplicatePoints);
    seen.add(key);
  }
  const triangles = delaunayTriangles(source);
  if (triangles.length === 0) throw new Error(CHINESE.collinear);
  const base = affinePoints(affine, source);
  const deltaX = new Float64Array(source.length);
  const deltaY = new Float64Array(source.length);
  for (let i = 0; i < source.length; i++) {
    deltaX[i] = target[i][0] - base[i][0];
    deltaY[i] = target[i][1] - base[i][1];
  }
  const correction = solveRegularizedLaplacian(source.length, triangles, deltaX, deltaY, regularization);
  const smoothed: Pixel[] = new Array(source.length);
  for (let i = 0; i < source.length; i++) {
    smoothed[i] = [base[i][0] + correction.x[i], base[i][1] + correction.y[i]];
  }
  const referenceSign = Math.sign(determinant2(affine));
  let folded = 0;
  for (const triangle of triangles) {
    const [a, b, c] = triangle;
    const areaSource = signedArea(source[a], source[b], source[c]);
    const areaTarget = signedArea(smoothed[a], smoothed[b], smoothed[c]);
    if (Math.abs(areaTarget) < 1e-5 || Math.sign(areaSource * areaTarget) !== referenceSign) folded++;
  }
  const hull = convexHullIndices(source);
  const boundary: Pixel[] = hull.map(index => smoothed[index]);
  if (folded > 0 || !isContourConvex(boundary)) throw new Error(CHINESE.foldedMesh);
  return {
    registration: { affine: matrixOf(affine), source, target: smoothed, triangles } as RegistrationData,
    folded,
  };
}

/** Forward-map pixels through a registration mesh; `strict=false` uses the affine outside the hull. */
function forwardPoints(registration: RegistrationData, points: readonly Pixel[]): { local: Pixel[]; inside: boolean[] } {
  const mesh = new Mesh(registration);
  const local: Pixel[] = new Array(points.length);
  const inside: boolean[] = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    const mapped = mesh.forward(points[i]);
    local[i] = mapped.local;
    inside[i] = mapped.inside;
  }
  return { local, inside };
}

// ---------------------------------------------------------------------------
// Matcher (dirmap/registration.py::Matcher)
// ---------------------------------------------------------------------------

type FeatureKind = 'water' | 'building' | 'pitch';
const FEATURE_KINDS: FeatureKind[] = ['water', 'building', 'pitch'];

export interface FeatureMatch {
  pixel: Pixel;
  local: Pixel;
  sourceIndex: number;
  referenceId: string;
  kind: FeatureKind;
  errorM: number;
  shapeError: number;
}

export interface MatcherInitial {
  matrix: number[][];
  candidates: number;
}

/**
 * Semantic feature matcher: source features in native pixels, target features in
 * local metres. `initial` enumerates water-seeded candidate transforms over
 * reflection / rotation / scale, then `matches` scores a transform by
 * distance + area/aspect agreement.
 */
export class Matcher {
  readonly source: Record<FeatureKind, SourceFeature[]>;
  readonly target: Record<FeatureKind, ReferenceFeature[]>;
  private readonly trees: Partial<Record<FeatureKind, FeatureKdTree>>;
  private readonly targetArea: Partial<Record<FeatureKind, Float64Array>>;
  private readonly targetAspect: Partial<Record<FeatureKind, Float64Array>>;
  private readonly sourceArea: Record<FeatureKind, Float64Array>;
  private readonly sourceAspect: Record<FeatureKind, Float64Array>;

  constructor(features: readonly SourceFeature[], reference: readonly ReferenceFeature[]) {
    this.source = { water: [], building: [], pitch: [] };
    this.target = { water: [], building: [], pitch: [] };
    this.trees = {};
    this.targetArea = {};
    this.targetAspect = {};
    this.sourceArea = { water: new Float64Array(0), building: new Float64Array(0), pitch: new Float64Array(0) };
    this.sourceAspect = { water: new Float64Array(0), building: new Float64Array(0), pitch: new Float64Array(0) };
    for (const kind of FEATURE_KINDS) {
      this.source[kind] = features.filter(feature => feature.kind === kind);
      this.target[kind] = reference.filter(feature => feature.kind === kind);
      const areas = new Float64Array(this.target[kind].length);
      const aspects = new Float64Array(this.target[kind].length);
      for (let i = 0; i < this.target[kind].length; i++) {
        areas[i] = this.target[kind][i].area;
        aspects[i] = this.target[kind][i].aspect;
      }
      this.targetArea[kind] = areas;
      this.targetAspect[kind] = aspects;
      if (this.target[kind].length > 0) this.trees[kind] = new FeatureKdTree(this.target[kind].map(feature => feature.center));
      const sourceAreas = new Float64Array(this.source[kind].length);
      const sourceAspects = new Float64Array(this.source[kind].length);
      for (let i = 0; i < this.source[kind].length; i++) {
        sourceAreas[i] = this.source[kind][i].area;
        sourceAspects[i] = this.source[kind][i].aspect;
      }
      this.sourceArea[kind] = sourceAreas;
      this.sourceAspect[kind] = sourceAspects;
    }
  }

  /**
   * Score `matrix` against the feature sets and return accepted correspondences.
   * `score` weights water matches x4 (water outlines are the strongest seed).
   */
  matches(matrix: number[][], maxDistance = 45): { score: number; pairs: FeatureMatch[] } {
    const affine = affineOf(matrix);
    const scaleSquared = Math.abs(determinant2(affine));
    const pairs: FeatureMatch[] = [];
    let score = 0;
    for (const kind of FEATURE_KINDS) {
      const sources = this.source[kind];
      const targets = this.target[kind];
      const tree = this.trees[kind];
      if (sources.length === 0 || targets.length === 0 || !tree) continue;
      const width = Math.min(5, targets.length);
      const targetArea = this.targetArea[kind] as Float64Array;
      const targetAspect = this.targetAspect[kind] as Float64Array;
      const sourceArea = this.sourceArea[kind];
      const sourceAspect = this.sourceAspect[kind];
      const sourceCount = sources.length;
      const projectedX = new Float64Array(sourceCount);
      const projectedY = new Float64Array(sourceCount);
      for (let i = 0; i < sourceCount; i++) {
        const center = sources[i].center;
        projectedX[i] = affine[0] * center[0] + affine[1] * center[1] + affine[2];
        projectedY[i] = affine[3] * center[0] + affine[4] * center[1] + affine[5];
      }
      const combined = new Float64Array(sourceCount * width);
      const distances = new Float64Array(sourceCount * width);
      const areaErrors = new Float64Array(sourceCount * width);
      const aspectErrors = new Float64Array(sourceCount * width);
      const indexes = new Int32Array(sourceCount * width);
      for (let i = 0; i < sourceCount; i++) {
        const result = tree.query(projectedX[i], projectedY[i], width);
        for (let j = 0; j < width; j++) {
          const targetIndex = result.indexes[j];
          const distance = Math.sqrt(result.distances[j]);
          const offset = i * width + j;
          indexes[offset] = targetIndex;
          distances[offset] = distance;
          if (targetIndex < 0) {
            areaErrors[offset] = Infinity;
            aspectErrors[offset] = Infinity;
            combined[offset] = Infinity;
            continue;
          }
          const areaError = Math.abs(Math.log(sourceArea[i] * scaleSquared / targetArea[targetIndex]));
          const aspectError = Math.abs(Math.log(sourceAspect[i] / targetAspect[targetIndex]));
          areaErrors[offset] = areaError;
          aspectErrors[offset] = aspectError;
          combined[offset] = distance + 15 * areaError + 10 * aspectError;
        }
      }
      const best = new Int32Array(sourceCount);
      const bestValues = new Float64Array(sourceCount);
      for (let i = 0; i < sourceCount; i++) {
        let chosen = 0;
        let chosenValue = Infinity;
        for (let j = 0; j < width; j++) {
          const value = combined[i * width + j];
          if (value < chosenValue) {
            chosenValue = value;
            chosen = j;
          }
        }
        best[i] = chosen;
        bestValues[i] = chosenValue;
      }
      const order = argsortStable(Array.from(bestValues));
      const used = new Set<number>();
      for (const i of order) {
        const j = best[i];
        const targetIndex = indexes[i * width + j];
        if (targetIndex < 0 || used.has(targetIndex)) continue;
        const distance = distances[i * width + j];
        if (distance > maxDistance) continue;
        if (areaErrors[i * width + j] > 1.1) continue;
        if (aspectErrors[i * width + j] > 0.8) continue;
        used.add(targetIndex);
        const weight = kind === 'water' ? 4 : 1;
        const value = combined[i * width + j];
        score += weight * Math.exp(-0.5 * (value / 25) ** 2);
        pairs.push({
          pixel: [sources[i].center[0], sources[i].center[1]],
          local: [targets[targetIndex].center[0], targets[targetIndex].center[1]],
          sourceIndex: i,
          referenceId: targets[targetIndex].id,
          kind,
          errorM: distance,
          shapeError: areaErrors[i * width + j] + aspectErrors[i * width + j],
        });
      }
    }
    return { score, pairs };
  }

  /**
   * Enumerate water-seeded candidate transforms (reflection, two rotations,
   * jittered angle, three scales) and keep the best scoring one.
   */
  initial(width: number, height: number): MatcherInitial {
    const sources = this.source.water
      .slice()
      .sort((a, b) => b.area - a.area)
      .slice(0, 7);
    const targets = this.target.water.filter(feature => feature.area > 400);
    if (sources.length === 0 || targets.length === 0) throw new Error(CHINESE.noWaterSeed);
    let bestScore = -1;
    let bestMatrix: number[][] | null = null;
    let bestPairs: FeatureMatch[] = [];
    let candidates = 0;
    for (const source of sources) {
      // np.asarray(source['points']) @ diag(1, -1): flip the pixel y axis.
      const reflected = source.points.map(point => [point[0], -point[1]] as Pixel);
      const angleA = principalAngle(reflected);
      for (const target of targets) {
        if (Math.abs(Math.log(source.aspect / target.aspect)) > 0.75) continue;
        const scale = Math.sqrt(target.area / source.area);
        const extent = Math.max(width, height) * scale;
        if (!(extent > 250 && extent < 4200)) continue;
        const angleB = principalAngle(target.points);
        for (const flip of [0, Math.PI]) {
          for (const delta of [-0.12, 0, 0.12]) {
            const angle = angleB - angleA + flip + delta;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            // rotate @ reflect = [[cos, sin], [sin, -cos]]
            const r0 = cos;
            const r1 = sin;
            const r2 = sin;
            const r3 = -cos;
            for (const factor of [0.9, 1, 1.1]) {
              const scaleTimesFactor = scale * factor;
              const linear: Affine6 = [
                scaleTimesFactor * r0, scaleTimesFactor * r1, 0,
                scaleTimesFactor * r2, scaleTimesFactor * r3, 0,
              ];
              const shiftX = target.center[0] - (linear[0] * source.center[0] + linear[1] * source.center[1]);
              const shiftY = target.center[1] - (linear[3] * source.center[0] + linear[4] * source.center[1]);
              linear[2] = shiftX;
              linear[5] = shiftY;
              const matrix = matrixOf(linear);
              const result = this.matches(matrix);
              candidates++;
              if (result.score > bestScore) {
                bestScore = result.score;
                bestMatrix = matrix;
                bestPairs = result.pairs;
              }
            }
          }
        }
      }
    }
    if (!bestMatrix || bestPairs.length < 6) throw new Error(CHINESE.weakEvidence);
    return { matrix: bestMatrix, candidates };
  }
}

// ---------------------------------------------------------------------------
// Observed-landmark corner correspondences
// ---------------------------------------------------------------------------

/** Port of `feature_corner_matches`: corners inside already matched, shape-checked buildings. */
export function featureCornerMatches(matcher: Matcher, matrix: number[][], matches: readonly FeatureMatch[]): RegistrationLandmark[] {
  const affine = affineOf(matrix);
  const references = new Map<string, ReferenceFeature>();
  for (const feature of matcher.target.building) references.set(feature.id, feature);
  const scale = Math.sqrt(Math.abs(determinant2(affine)));
  const result: RegistrationLandmark[] = [];
  for (const match of matches) {
    if (match.kind !== 'building') continue;
    const source = matcher.source.building[match.sourceIndex];
    const target = references.get(match.referenceId);
    if (!source || !target) continue;
    const projected = affinePoints(affine, source.points);
    const iou = outlineIou(projected, target.points);
    if (iou < 0.52) continue;
    const a = corners(source.points, 1.3 / Math.max(0.1, scale));
    const b = corners(target.points, 1.3);
    if (a.length < 3 || b.length < 3) continue;
    const world = affinePoints(affine, a);
    const forward: number[] = new Array(world.length);
    const forwardDistances: number[] = new Array(world.length);
    for (let i = 0; i < world.length; i++) {
      const nearest = nearestIndex(b, world[i][0], world[i][1]);
      forward[i] = nearest.index;
      forwardDistances[i] = nearest.distance;
    }
    const backward: number[] = new Array(b.length);
    for (let j = 0; j < b.length; j++) backward[j] = nearestIndex(world, b[j][0], b[j][1]).index;
    const accepted: [number, number][] = [];
    for (let i = 0; i < world.length; i++) {
      const j = forward[i];
      if (forwardDistances[i] < 13 && backward[j] === i) accepted.push([i, j]);
    }
    if (accepted.length < 3) continue;
    const acceptedDistances = accepted.map(([i]) => forwardDistances[i]).sort((x, y) => x - y);
    if (median(acceptedDistances) > 8) continue;
    for (const [i, j] of accepted) {
      result.push({
        pixel: [a[i][0], a[i][1]],
        local: [b[j][0], b[j][1]],
        referenceId: match.referenceId,
        kind: 'building_corner',
        outlineIoU: round2(iou),
        errorM: round2(forwardDistances[i]),
      });
    }
  }
  return result;
}

/** Port of `partial_feature_corner_matches`: use visible measured edges of cropped objects. */
export function partialFeatureCornerMatches(matcher: Matcher, matrix: number[][], width: number, height: number): RegistrationLandmark[] {
  const affine = affineOf(matrix);
  const scale = Math.sqrt(Math.abs(determinant2(affine)));
  const result: RegistrationLandmark[] = [];
  for (const kind of ['building', 'pitch'] as const) {
    for (const feature of matcher.source[kind]) {
      const raw = feature.points;
      let clipped = false;
      for (const point of raw) {
        if (point[0] <= 2 || point[1] <= 2 || point[0] >= width - 3 || point[1] >= height - 3) {
          clipped = true;
          break;
        }
      }
      let chosenCorners = corners(raw, 1.2 / Math.max(0.1, scale));
      if (kind === 'building') {
        chosenCorners = chosenCorners.filter(point => point[0] > 3 && point[0] < width - 4
          && point[1] > 3 && point[1] < height - 4);
      }
      if (chosenCorners.length < 2) continue;
      const world = affinePoints(affine, chosenCorners);
      const projectedOutline = affinePoints(affine, raw);
      let worldMeanX = 0;
      let worldMeanY = 0;
      for (const point of world) {
        worldMeanX += point[0];
        worldMeanY += point[1];
      }
      worldMeanX /= world.length;
      worldMeanY /= world.length;
      interface Option {
        score: number;
        pairs: [number, number][];
        target: ReferenceFeature;
        referenceCorners: Pixel[];
        iou: number;
      }
      const options: Option[] = [];
      for (const target of matcher.target[kind]) {
        const areaRatio = feature.area * Math.abs(determinant2(affine)) / target.area;
        if (kind === 'building' && !clipped && areaRatio > 0.70) continue;
        const centerDistance = Math.hypot(worldMeanX - target.center[0], worldMeanY - target.center[1]);
        if (centerDistance > Math.max(60, Math.sqrt(target.area) * 1.5)) continue;
        const referenceCorners = corners(target.points, 1.2);
        if (referenceCorners.length < 2) continue;
        const forward: number[] = new Array(world.length);
        const forwardDistances: number[] = new Array(world.length);
        for (let i = 0; i < world.length; i++) {
          const nearest = nearestIndex(referenceCorners, world[i][0], world[i][1]);
          forward[i] = nearest.index;
          forwardDistances[i] = nearest.distance;
        }
        const reverse: number[] = new Array(referenceCorners.length);
        for (let j = 0; j < referenceCorners.length; j++) {
          reverse[j] = nearestIndex(world, referenceCorners[j][0], referenceCorners[j][1]).index;
        }
        const pairs: [number, number][] = [];
        for (let i = 0; i < world.length; i++) {
          const j = forward[i];
          if (forwardDistances[i] <= 12 && reverse[j] === i) pairs.push([i, j]);
        }
        if (pairs.length < 2) continue;
        let coherent = false;
        for (let first = 0; first < pairs.length && !coherent; first++) {
          for (let second = first + 1; second < pairs.length && !coherent; second++) {
            const a = pairs[first];
            const b = pairs[second];
            const leftX = world[b[0]][0] - world[a[0]][0];
            const leftY = world[b[0]][1] - world[a[0]][1];
            const rightX = referenceCorners[b[1]][0] - referenceCorners[a[1]][0];
            const rightY = referenceCorners[b[1]][1] - referenceCorners[a[1]][1];
            const lengthA = Math.hypot(leftX, leftY);
            const lengthB = Math.hypot(rightX, rightY);
            if (!(Math.min(lengthA, lengthB) >= 10)) continue;
            if (!(lengthA / lengthB >= 0.85 && lengthA / lengthB <= 1.15)) continue;
            if ((leftX * rightX + leftY * rightY) / (lengthA * lengthB) < 0.98) continue;
            let edgeIsObserved = true;
            const tests: { begin: Pixel; vector: Pixel; outline: Pixel[]; tolerance: number }[] = [
              {
                begin: world[a[0]], vector: [leftX, leftY], outline: projectedOutline,
                tolerance: Math.min(4, Math.max(2, 6 * scale, lengthA * 0.04)),
              },
              {
                begin: referenceCorners[a[1]], vector: [rightX, rightY], outline: target.points,
                tolerance: Math.min(4, Math.max(2, lengthB * 0.04)),
              },
            ];
            for (const test of tests) {
              for (let step = 0; step < 9 && edgeIsObserved; step++) {
                const ratio = 0.1 + step * 0.1;
                const sampleX = test.begin[0] + test.vector[0] * ratio;
                const sampleY = test.begin[1] + test.vector[1] * ratio;
                if (Math.abs(pointPolygonDistance(test.outline, sampleX, sampleY)) > test.tolerance) {
                  edgeIsObserved = false;
                }
              }
              if (!edgeIsObserved) break;
            }
            if (edgeIsObserved) coherent = true;
          }
        }
        if (!coherent) continue;
        const iou = outlineIou(projectedOutline, target.points);
        if (!(areaRatio >= 0.20 && areaRatio <= 1.6) || iou < 0.28) continue;
        let distanceSum = 0;
        for (const [i] of pairs) distanceSum += forwardDistances[i];
        const score = distanceSum / pairs.length + (1 - iou) * 3;
        options.push({ score, pairs, target, referenceCorners, iou });
      }
      options.sort((a, b) => a.score - b.score);
      if (options.length === 0) continue;
      if (options.length > 1 && options[1].score - options[0].score < 3) continue;
      const best = options[0];
      for (const [i, j] of best.pairs) {
        result.push({
          pixel: [chosenCorners[i][0], chosenCorners[i][1]],
          local: [best.referenceCorners[j][0], best.referenceCorners[j][1]],
          kind: `visible_${kind}_corner`,
          referenceId: best.target.id,
          outlineIoU: round2(best.iou),
          errorM: round2(Math.hypot(world[i][0] - best.referenceCorners[j][0], world[i][1] - best.referenceCorners[j][1])),
        });
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// fit_observed_landmarks
// ---------------------------------------------------------------------------

interface Observation {
  pixel: Pixel;
  local: Pixel;
  referenceId: string;
  kind?: string;
  sourceIndex?: number;
  errorM?: number;
  shapeError?: number;
  outlineIoU?: number;
  purpose?: 'fit' | 'validation';
  insideHull?: boolean;
  source?: string;
  [key: string]: unknown;
}

function cloneObservation(landmark: RegistrationLandmark & { referenceId?: string }): Observation | null {
  if (!landmark.referenceId) return null;
  const observation: Observation = {
    pixel: [landmark.pixel[0], landmark.pixel[1]],
    local: landmark.local ? [landmark.local[0], landmark.local[1]] : [NaN, NaN],
    referenceId: landmark.referenceId,
    purpose: landmark.purpose,
  };
  for (const [key, value] of Object.entries(landmark)) {
    if (key === 'pixel' || key === 'local' || key === 'referenceId' || key === 'purpose') continue;
    observation[key] = value;
  }
  if (Number.isFinite(observation.local[0]) === false || Number.isFinite(observation.local[1]) === false) return null;
  return observation;
}

/**
 * Fit measured outlines, keeping every point of an object in one fit/holdout
 * split. Port of `dirmap.registration.fit_observed_landmarks`.
 */
export function fitObservedLandmarks(pairs: readonly FeatureMatch[],
  extra: readonly RegistrationLandmark[]): AutoRegisterResult {
  const outlined = new Set<string>();
  for (const landmark of extra) if (landmark.referenceId) outlined.add(landmark.referenceId);
  const observations: Observation[] = [];
  for (const pair of pairs) {
    if (outlined.has(pair.referenceId)) continue;
    const observation = cloneObservation({
      pixel: pair.pixel, local: pair.local, referenceId: pair.referenceId, kind: pair.kind,
      sourceIndex: pair.sourceIndex, errorM: pair.errorM, shapeError: pair.shapeError,
    });
    if (observation) observations.push(observation);
  }
  for (const landmark of extra) {
    const observation = cloneObservation(landmark);
    if (observation) observations.push(observation);
  }
  const groups = new Map<string, Observation[]>();
  for (const observation of observations) {
    const group = groups.get(observation.referenceId);
    if (group) group.push(observation);
    else groups.set(observation.referenceId, [observation]);
  }
  const means = new Map<string, Pixel>();
  for (const [key, group] of groups) {
    let sumX = 0;
    let sumY = 0;
    for (const observation of group) {
      sumX += observation.pixel[0];
      sumY += observation.pixel[1];
    }
    means.set(key, [sumX / group.length, sumY / group.length]);
  }
  // sorted(groups, key=lambda key: (mean_pixel Y, mean_pixel X))
  const ordered = Array.from(groups.keys()).sort((a, b) => {
    const first = means.get(a) as Pixel;
    const second = means.get(b) as Pixel;
    return (first[1] - second[1]) || (first[0] - second[0]);
  });
  const sourcePixels: Pixel[] = observations.map(observation => observation.pixel);
  const hull = convexHullIndices(sourcePixels);
  const boundaryIds = new Set<string>();
  for (const index of hull) boundaryIds.add(observations[index].referenceId);
  const interior = ordered.filter(key => !boundaryIds.has(key));
  const validationIds = new Set<string>();
  for (let i = 1; i < interior.length; i += 3) validationIds.add(interior[i]);
  const fitIds = ordered.filter(key => !validationIds.has(key));
  const groupCenter = (key: string, field: 'pixel' | 'local'): Pixel => {
    const group = groups.get(key) as Observation[];
    let sumX = 0;
    let sumY = 0;
    for (const observation of group) {
      sumX += observation[field][0];
      sumY += observation[field][1];
    }
    return [sumX / group.length, sumY / group.length];
  };
  const fit = fitAffine(fitIds.map(key => groupCenter(key, 'pixel')),
    fitIds.map(key => groupCenter(key, 'local')), 15);
  const rejected = new Set<string>();
  for (let i = 0; i < fitIds.length; i++) if (!fit.inliers[i]) rejected.add(fitIds[i]);
  const fitting = observations.filter(observation =>
    !validationIds.has(observation.referenceId) && !rejected.has(observation.referenceId));
  const validation = observations.filter(observation => validationIds.has(observation.referenceId));
  if (fitting.length < 8 || validation.length === 0) throw new Error(CHINESE.weakHoldout);

  // Remove coincident corner observations deterministically.
  const unique = new Map<string, Observation>();
  for (const observation of fitting) {
    const key = `${round2(observation.pixel[0] * 1000)},${round2(observation.pixel[1] * 1000)}`;
    if (!unique.has(key)) unique.set(key, observation);
  }
  const uniqueFitting = Array.from(unique.values());
  const fittingPixels = uniqueFitting.map(observation => observation.pixel);
  const fittingLocals = uniqueFitting.map(observation => observation.local);
  let registration: RegistrationData;
  let fallback = false;
  try {
    registration = buildRegistration(fit.matrix, fittingPixels, fittingLocals, 1.5).registration;
  } catch {
    registration = buildRegistration(fit.matrix, fittingPixels, affinePoints(affineOf(fit.matrix), fittingPixels), 1.5).registration;
    fallback = true;
  }
  const validationPixels = validation.map(observation => observation.pixel);
  const predicted = forwardPoints(registration, validationPixels);
  const localTargets = validation.map(observation => observation.local);
  const errors: number[] = predicted.local.map((point, index) =>
    Math.hypot(point[0] - localTargets[index][0], point[1] - localTargets[index][1]));
  let insideCount = 0;
  for (const inside of predicted.inside) if (inside) insideCount++;
  for (const observation of uniqueFitting) observation.purpose = 'fit';
  for (let i = 0; i < validation.length; i++) {
    validation[i].purpose = 'validation';
    validation[i].errorM = errors[i];
    validation[i].insideHull = predicted.inside[i];
  }
  const fittingObjects = new Set<string>();
  for (const observation of uniqueFitting) fittingObjects.add(observation.referenceId);
  const landmarks: RegistrationLandmark[] = [...uniqueFitting, ...validation] as RegistrationLandmark[];
  const quality = {
    state: 'needs_review',
    method: 'water_seed_observed_outline_ransac_mesh',
    fitCount: uniqueFitting.length,
    fitObjectCount: fittingObjects.size,
    validationCount: validation.length,
    validationObjectCount: validationIds.size,
    validationInsideHull: insideCount,
    rejectedObjectIds: Array.from(rejected).sort(),
    medianErrorM: round2(median(errors)),
    p95ErrorM: round2(percentile(errors, 95)),
    maxErrorM: round2(Math.max(...errors)),
    affineFallback: fallback,
    foldedTriangles: 0,
    validationScope: '',
    landmarks,
  };
  return { registration, quality: quality as RegistrationQuality };
}

// ---------------------------------------------------------------------------
// autoRegister
// ---------------------------------------------------------------------------

/**
 * Full automatic registration: water-seeded initial transform, iterative RANSAC
 * refinement, measured building/pitch corner correspondences, object-grouped
 * holdout and a regularized non-folding mesh. Port of `auto_register`.
 */
export async function autoRegister(features: readonly SourceFeature[], reference: readonly ReferenceFeature[],
  width: number, height: number,
  options: { progress?: (percent: number, message: string) => void } = {}): Promise<AutoRegisterResult> {
  const progress = options.progress ?? (() => undefined);
  progress(5, '匹配水体种子与校园参照');
  const matcher = new Matcher(features, reference);
  const initial = matcher.initial(width, height);
  let matrix = initial.matrix;
  progress(35, `评估 ${initial.candidates} 个候选变换`);
  for (const tolerance of [45, 35, 28, 23]) {
    const { pairs } = matcher.matches(matrix, tolerance);
    if (pairs.length < 6) break;
    const candidate = fitAffine(pairs.map(pair => pair.pixel), pairs.map(pair => pair.local), 18);
    let inlierCount = 0;
    for (const inlier of candidate.inliers) if (inlier) inlierCount++;
    if (inlierCount >= 6) matrix = candidate.matrix;
  }
  const { pairs } = matcher.matches(matrix, 22);
  if (pairs.length < 8) throw new Error(weakEvidenceMessage(pairs.length));
  progress(60, '提取实测地物角点');
  const corners = featureCornerMatches(matcher, matrix, pairs);
  corners.push(...partialFeatureCornerMatches(matcher, matrix, width, height));
  progress(75, '留出独立地物并拟合局部网格');
  const fitted = fitObservedLandmarks(pairs, corners);
  const quality = fitted.quality;
  quality.candidateTransforms = initial.candidates;
  quality.observedCornerCount = corners.length;
  quality.validationScope = '按地物整体留出内部核查点，实测边界角点用于拟合；自动对应关系未人工核对，不等价于独立现场定位精度。';
  progress(95, '配准完成');
  return { registration: fitted.registration, quality };
}

// ---------------------------------------------------------------------------
// registerToMap (cross-map transfer, feature matcher instead of SIFT)
// ---------------------------------------------------------------------------

function projectBaseFeatures(base: RegisterBase): ReferenceFeature[] {
  const mesh = new Mesh(base.registration);
  const projected: ReferenceFeature[] = [];
  for (let index = 0; index < base.features.length; index++) {
    const feature = base.features[index];
    const center = mesh.forward(feature.center);
    if (!center.inside) continue;
    const localPoints: Pixel[] = [];
    let allInside = true;
    for (const point of feature.points) {
      const mapped = mesh.forward(point);
      if (!mapped.inside) {
        allInside = false;
        break;
      }
      localPoints.push(mapped.local);
    }
    if (!allInside || localPoints.length < 3) continue;
    const moments = polygonMoments(localPoints);
    if (!moments) continue;
    const area = Math.abs(moments.m00);
    if (!(area > 0)) continue;
    projected.push({
      id: `base-${index}`,
      kind: feature.kind,
      center: [moments.m10 / moments.m00, moments.m01 / moments.m00],
      area,
      // Affine-like transfer preserves the pixel-space aspect closely; the
      // reference aspect is the same quantity in local metres for the base map.
      aspect: feature.aspect,
      points: localPoints,
    });
  }
  return projected;
}

function registrationHash(registration: RegistrationData | null): string {
  if (!registration) return 'none';
  return JSON.stringify({ affine: registration.affine, source: registration.source, target: registration.target });
}

/**
 * Cross-map fallback without SIFT: match analysed map-detail features against
 * the already registered base map's features projected into local metres, then
 * reuse the SIFT transfer bookkeeping (fit/holdout split by convex-hull index,
 * documented transfer quality). Best effort for low-resolution sheets.
 */
export async function registerToMap(source: readonly SourceFeature[], base: RegisterBase,
  options: { progress?: (percent: number, message: string) => void } = {}): Promise<AutoRegisterResult> {
  const progress = options.progress ?? (() => undefined);
  progress(5, '配准基准图：投影共同地物');
  if (!base.registration || base.registration.triangles.length === 0) throw new Error(CHINESE.noCommonFeatures);
  const reference = projectBaseFeatures(base);
  const buildingCount = reference.filter(feature => feature.kind === 'building').length;
  const waterCount = reference.filter(feature => feature.kind === 'water').length;
  if (buildingCount + waterCount < 20) throw new Error(CHINESE.noCommonFeatures);
  const width = Math.max(...source.map(feature => Math.max(...feature.points.map(point => point[0]))) , 1);
  const height = Math.max(...source.map(feature => Math.max(...feature.points.map(point => point[1]))) , 1);
  const matcher = new Matcher(source, reference);
  let matrix: number[][];
  let candidates = 0;
  try {
    progress(15, '枚举跨图初始变换');
    const initial = matcher.initial(width, height);
    matrix = initial.matrix;
    candidates = initial.candidates;
  } catch {
    throw new Error(CHINESE.insufficientTransfer);
  }
  for (const tolerance of [45, 35, 28, 23]) {
    const { pairs } = matcher.matches(matrix, tolerance);
    if (pairs.length < 6) break;
    const candidate = fitAffine(pairs.map(pair => pair.pixel), pairs.map(pair => pair.local), 18);
    let inlierCount = 0;
    for (const inlier of candidate.inliers) if (inlier) inlierCount++;
    if (inlierCount >= 6) matrix = candidate.matrix;
  }
  progress(45, '筛选稳定共同地物');
  let { pairs } = matcher.matches(matrix, 22);
  if (pairs.length < 16) throw new Error(CHINESE.insufficientTransfer);
  // Remove duplicate observations, then avoid dense coincident landmarks.
  let sourcePoints: Pixel[] = pairs.map(pair => pair.pixel);
  let targetPoints: Pixel[] = pairs.map(pair => pair.local);
  const uniqueByCoordinate = new Map<string, number>();
  for (let index = 0; index < sourcePoints.length; index++) {
    const key = `${Math.round(sourcePoints[index][0] * 10000)},${Math.round(sourcePoints[index][1] * 10000)}`;
    if (!uniqueByCoordinate.has(key)) uniqueByCoordinate.set(key, index);
  }
  let keepIndexes = Array.from(uniqueByCoordinate.values());
  sourcePoints = keepIndexes.map(index => sourcePoints[index]);
  targetPoints = keepIndexes.map(index => targetPoints[index]);
  const bins = new Map<string, number>();
  const binned: number[] = [];
  for (let index = 0; index < sourcePoints.length; index++) {
    const key = `${Math.floor(sourcePoints[index][0] / 50)},${Math.floor(sourcePoints[index][1] / 50)}`;
    if (!bins.has(key)) {
      bins.set(key, index);
      binned.push(index);
    }
  }
  const boundaryIndexes = convexHullIndices(sourcePoints);
  const keepSet = new Set<number>([...binned, ...boundaryIndexes]);
  keepIndexes = Array.from(keepSet).sort((a, b) => a - b);
  sourcePoints = keepIndexes.map(index => sourcePoints[index]);
  targetPoints = keepIndexes.map(index => targetPoints[index]);
  if (sourcePoints.length < 12) throw new Error(CHINESE.insufficientCoverage);
  // Sort by source y (stable), then hold out interior points for validation.
  const yOrder = sourcePoints.map((point, index) => ({ y: point[1], index }))
    .sort((a, b) => (a.y - b.y) || (a.index - b.index))
    .map(entry => entry.index);
  sourcePoints = yOrder.map(index => sourcePoints[index]);
  targetPoints = yOrder.map(index => targetPoints[index]);
  const boundary = new Set(convexHullIndices(sourcePoints));
  const validationMask = new Uint8Array(sourcePoints.length);
  let interiorCount = 0;
  for (let index = 0; index < sourcePoints.length; index++) {
    if (boundary.has(index)) continue;
    if (interiorCount % 5 === 2) validationMask[index] = 1;
    interiorCount++;
  }
  const fitPoints: Pixel[] = [];
  const fitTargets: Pixel[] = [];
  for (let index = 0; index < sourcePoints.length; index++) {
    if (validationMask[index]) continue;
    fitPoints.push(sourcePoints[index]);
    fitTargets.push(targetPoints[index]);
  }
  if (fitPoints.length < 12) throw new Error(CHINESE.insufficientCoverage);
  progress(65, 'RANSAC 拟合跨图变换');
  const fit = fitAffine(fitPoints, fitTargets, 8);
  const keptSource: Pixel[] = [];
  const keptTarget: Pixel[] = [];
  for (let index = 0; index < fitPoints.length; index++) {
    if (!fit.inliers[index]) continue;
    keptSource.push(fitPoints[index]);
    keptTarget.push(fitTargets[index]);
  }
  let registration: RegistrationData;
  let folded = 0;
  try {
    const built = buildRegistration(fit.matrix, keptSource, keptTarget, 1.5);
    registration = built.registration;
    folded = built.folded;
  } catch {
    registration = buildRegistration(fit.matrix, keptSource, affinePoints(affineOf(fit.matrix), keptSource), 1.5).registration;
  }
  const validationSources: Pixel[] = [];
  const validationTargets: Pixel[] = [];
  for (let index = 0; index < sourcePoints.length; index++) {
    if (!validationMask[index]) continue;
    validationSources.push(sourcePoints[index]);
    validationTargets.push(targetPoints[index]);
  }
  const predicted = forwardPoints(registration, validationSources);
  const errors = predicted.local.map((point, index) =>
    Math.hypot(point[0] - validationTargets[index][0], point[1] - validationTargets[index][1]));
  const landmarks: RegistrationLandmark[] = keptSource.map((point, index) => ({
    pixel: point, local: keptTarget[index], purpose: 'fit',
  }));
  for (let index = 0; index < validationSources.length; index++) {
    landmarks.push({
      pixel: validationSources[index], local: validationTargets[index], purpose: 'validation',
      errorM: errors[index], insideHull: predicted.inside[index],
    });
  }
  const baseQuality = base.quality;
  const inherited = baseQuality
    ? (baseQuality.inheritedGeographicP95M ?? baseQuality.p95ErrorM ?? null)
    : null;
  // Best effort for low-resolution sheets: the Matcher may only find a handful
  // of whole-feature centres, which does not cover the sheet. Observed,
  // shape-checked outline corners of the matched features extend the measured
  // hull (same evidence class as Python's observed-boundary transfer); the
  // holdout stays object-grouped and the inherited absolute error is unchanged.
  if (pairs.length >= 8) {
    const observedCorners = featureCornerMatches(matcher, matrix, pairs);
    observedCorners.push(...partialFeatureCornerMatches(matcher, matrix, width, height));
    if (observedCorners.length >= 6) {
      try {
        const observed = fitObservedLandmarks(pairs, observedCorners);
        const observedQuality = observed.quality;
        observedQuality.method = 'feature_cross_map_observed_boundary_mesh';
        observedQuality.baseAssetId = base.id;
        observedQuality.baseSourceHash = base.sourceHash;
        observedQuality.baseRegistrationHash = registrationHash(base.registration);
        observedQuality.inheritedGeographicP95M = inherited === undefined ? null : inherited;
        observedQuality.candidateTransforms = candidates;
        observedQuality.observedCornerCount = observedCorners.length;
        observedQuality.validationScope = '与已配准原图的内部留出接缝误差，实测共同地物轮廓角点用于拟合；绝对位置误差继承基准图。';
        progress(95, '跨图配准完成');
        return { registration: observed.registration, quality: observedQuality };
      } catch {
        // Too few independent objects for a verifiable holdout: keep the
        // centre-transfer result below (still honest, but lower coverage).
      }
    }
  }
  const quality = {
    state: 'needs_review',
    // No SIFT in opencv.js: this is the feature-matcher transfer variant.
    method: 'feature_cross_map_ransac_mesh',
    baseAssetId: base.id,
    baseSourceHash: base.sourceHash,
    baseRegistrationHash: registrationHash(base.registration),
    fitCount: keptSource.length,
    validationCount: errors.length,
    medianErrorM: round2(median(errors)),
    p95ErrorM: round2(percentile(errors, 95)),
    maxErrorM: round2(Math.max(...errors)),
    foldedTriangles: folded,
    inheritedGeographicP95M: inherited === undefined ? null : inherited,
    validationScope: '与已配准原图的内部留出接缝误差，边界由实测共同地物确定；绝对位置误差继承基准图。',
    landmarks,
    candidateTransforms: candidates,
  };
  progress(95, '跨图配准完成');
  return { registration, quality: quality as RegistrationQuality };
}

/** Port of `extend_transfer_boundary`: supplement a transfer with observed geographic corners. */
export async function extendTransferBoundary(registration: RegistrationData, quality: RegistrationQuality,
  geographicQuality: RegistrationQuality | null): Promise<AutoRegisterResult> {
  const landmarks = quality.landmarks ?? [];
  const fitting = landmarks.filter(landmark => landmark.purpose === 'fit');
  const validation = landmarks.filter(landmark => landmark.purpose === 'validation');
  const extra: RegistrationLandmark[] = [];
  const known = new Set<string>();
  for (const landmark of fitting) known.add(`${round2(landmark.pixel[0] * 1000)},${round2(landmark.pixel[1] * 1000)}`);
  const mesh = new Mesh(registration);
  const affine = affineOf(registration.affine);
  for (const landmark of geographicQuality?.landmarks ?? []) {
    if (landmark.purpose !== 'fit' || !(landmark.kind ?? '').includes('corner')) continue;
    if (!landmark.local) continue;
    const key = `${round2(landmark.pixel[0] * 1000)},${round2(landmark.pixel[1] * 1000)}`;
    if (known.has(key) || mesh.forward(landmark.pixel).inside) continue;
    const projected = applyAffine(affine, landmark.pixel[0], landmark.pixel[1]);
    if (Math.hypot(projected[0] - landmark.local[0], projected[1] - landmark.local[1]) > 12) continue;
    extra.push({ ...landmark, source: 'geographic_outline' });
    known.add(key);
  }
  if (extra.length === 0) return { registration, quality };
  const combined = [...fitting, ...extra];
  if (combined.some(landmark => !landmark.local)) return { registration, quality };
  let updated: RegistrationData;
  try {
    updated = buildRegistration(registration.affine, combined.map(landmark => landmark.pixel),
      combined.map(landmark => landmark.local as Pixel), 1.5).registration;
  } catch {
    return { registration, quality };
  }
  const predicted = forwardPoints(updated, validation.map(landmark => landmark.pixel));
  const errors = predicted.local.map((point, index) => {
    const local = validation[index].local as Pixel;
    return Math.hypot(point[0] - local[0], point[1] - local[1]);
  });
  for (let index = 0; index < validation.length; index++) {
    validation[index].errorM = errors[index];
    validation[index].insideHull = predicted.inside[index];
  }
  const updatedQuality: RegistrationQuality = {
    ...quality,
    method: 'feature_cross_map_observed_boundary_mesh',
    fitCount: combined.length,
    geographicBoundaryAnchors: extra.length,
    medianErrorM: round2(median(errors)),
    p95ErrorM: round2(percentile(errors, 95)),
    maxErrorM: round2(Math.max(...errors)),
    landmarks: [...combined, ...validation],
  };
  return { registration: updated, quality: updatedQuality };
}

// ---------------------------------------------------------------------------
// manualRegister
// ---------------------------------------------------------------------------

/**
 * Teacher anchor calibration: RANSAC affine + regularized mesh, rejecting
 * conflicting anchors and validation points outside the measured hull.
 */
export async function manualRegister(anchors: readonly Anchor[], frameOrigin: Pixel = DEFAULT_ORIGIN): Promise<AutoRegisterResult> {
  const fitting = anchors.filter(anchor => (anchor.purpose ?? 'fit') === 'fit');
  const validation = anchors.filter(anchor => anchor.purpose === 'validation');
  const source: Pixel[] = fitting.map(anchor => [anchor.pixel[0], anchor.pixel[1]]);
  const target: Pixel[] = fitting.map(anchor => toLocal(anchor.lnglat, frameOrigin));
  const fit = fitAffine(source, target, 20);
  if (!fit.inliers.every(Boolean)) throw new Error(CHINESE.conflictingAnchors);
  const built = buildRegistration(fit.matrix, source, target, 0.2);
  const errors: number[] = [];
  if (validation.length > 0) {
    const predicted = forwardPoints(built.registration, validation.map(anchor => anchor.pixel));
    for (const inside of predicted.inside) if (!inside) throw new Error(CHINESE.validationOutside);
    for (let index = 0; index < validation.length; index++) {
      const expected = toLocal(validation[index].lnglat, frameOrigin);
      errors.push(Math.hypot(predicted.local[index][0] - expected[0], predicted.local[index][1] - expected[1]));
    }
  }
  const quality: RegistrationQuality = {
    state: 'needs_review',
    method: 'reviewed_landmarks_mesh',
    fitCount: source.length,
    validationCount: errors.length,
    medianErrorM: errors.length > 0 ? median(errors) : null,
    p95ErrorM: errors.length > 0 ? percentile(errors, 95) : null,
    maxErrorM: errors.length > 0 ? Math.max(...errors) : null,
    foldedTriangles: built.folded,
    landmarks: anchors as unknown as RegistrationLandmark[],
    validationScope: '教师录入的独立核查点；发布前仍需检查通行语义。',
  };
  return { registration: built.registration, quality };
}

// ---------------------------------------------------------------------------
// projectControls
// ---------------------------------------------------------------------------

/**
 * Project control points through the registration mesh: sets `lnglat` and
 * `calibrated` (points outside the measured hull stay uncalibrated).
 */
export function projectControls(controls: Control[], registration: RegistrationData | null,
  frameOrigin: Pixel = DEFAULT_ORIGIN): void {
  if (!registration) return;
  const mesh = new Mesh(registration);
  for (const control of controls) {
    const mapped = mesh.forward(control.pixel);
    if (mapped.inside) {
      control.lnglat = frameLngLat(mapped.local, frameOrigin);
      control.calibrated = true;
    } else {
      control.lnglat = null;
      control.calibrated = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Overlay warp (inverse sampling into EPSG:3857)
// ---------------------------------------------------------------------------

interface InverseIndex {
  lower: Pixel;
  span: Pixel;
  shape: [number, number];
  buckets: Map<number, number[]>;
  transforms: Float64Array;
  anchors: Float64Array;
  source: Float64Array;
  triangles: Int32Array;
}

function buildInverseIndex(registration: RegistrationData): InverseIndex {
  const target = registration.target;
  const triangles = registration.triangles;
  if (target.length === 0 || triangles.length === 0) throw new Error(CHINESE.invalidAffine);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of target) {
    if (point[0] < minX) minX = point[0];
    if (point[1] < minY) minY = point[1];
    if (point[0] > maxX) maxX = point[0];
    if (point[1] > maxY) maxY = point[1];
  }
  const lower: Pixel = [minX - 1e-6, minY - 1e-6];
  const upper: Pixel = [maxX + 1e-6, maxY + 1e-6];
  const span: Pixel = [Math.max(upper[0] - lower[0], 1e-9), Math.max(upper[1] - lower[1], 1e-9)];
  const triangleCount = triangles.length;
  const cellCount = Math.max(1, Math.floor(triangleCount / 3));
  const baseShape = Math.sqrt(cellCount);
  const shape: [number, number] = [
    clamp(roundHalfEven(baseShape * Math.sqrt(span[0] / span[1])), 1, 64),
    clamp(roundHalfEven(baseShape * Math.sqrt(span[1] / span[0])), 1, 64),
  ];
  const buckets = new Map<number, number[]>();
  const transforms = new Float64Array(triangleCount * 4);
  const anchors = new Float64Array(triangleCount * 2);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const [ia, ib, ic] = triangles[triangle];
    const a = target[ia];
    const b = target[ib];
    const c = target[ic];
    const m00 = a[0] - c[0];
    const m01 = b[0] - c[0];
    const m10 = a[1] - c[1];
    const m11 = b[1] - c[1];
    const determinant = m00 * m11 - m01 * m10;
    if (Math.abs(determinant) < 1e-12) continue;
    const inverseDeterminant = 1 / determinant;
    transforms[triangle * 4] = m11 * inverseDeterminant;
    transforms[triangle * 4 + 1] = -m01 * inverseDeterminant;
    transforms[triangle * 4 + 2] = -m10 * inverseDeterminant;
    transforms[triangle * 4 + 3] = m00 * inverseDeterminant;
    anchors[triangle * 2] = c[0];
    anchors[triangle * 2 + 1] = c[1];
    let triangleMinX = Infinity;
    let triangleMinY = Infinity;
    let triangleMaxX = -Infinity;
    let triangleMaxY = -Infinity;
    for (const point of [a, b, c]) {
      if (point[0] < triangleMinX) triangleMinX = point[0];
      if (point[1] < triangleMinY) triangleMinY = point[1];
      if (point[0] > triangleMaxX) triangleMaxX = point[0];
      if (point[1] > triangleMaxY) triangleMaxY = point[1];
    }
    const lowX = clamp(Math.floor((triangleMinX - lower[0]) / span[0] * shape[0]), 0, shape[0] - 1);
    const lowY = clamp(Math.floor((triangleMinY - lower[1]) / span[1] * shape[1]), 0, shape[1] - 1);
    const highX = clamp(Math.floor((triangleMaxX - lower[0]) / span[0] * shape[0]), 0, shape[0] - 1);
    const highY = clamp(Math.floor((triangleMaxY - lower[1]) / span[1] * shape[1]), 0, shape[1] - 1);
    for (let y = lowY; y <= highY; y++) {
      for (let x = lowX; x <= highX; x++) {
        const key = y * shape[0] + x;
        const bucket = buckets.get(key);
        if (bucket) bucket.push(triangle);
        else buckets.set(key, [triangle]);
      }
    }
  }
  const source = new Float64Array(registration.source.length * 2);
  for (let i = 0; i < registration.source.length; i++) {
    source[i * 2] = registration.source[i][0];
    source[i * 2 + 1] = registration.source[i][1];
  }
  const flatTriangles = new Int32Array(triangleCount * 3);
  for (let i = 0; i < triangleCount; i++) {
    flatTriangles[i * 3] = triangles[i][0];
    flatTriangles[i * 3 + 1] = triangles[i][1];
    flatTriangles[i * 3 + 2] = triangles[i][2];
  }
  return { lower, span, shape, buckets, transforms, anchors, source, triangles: flatTriangles };
}

/** Invert local-metre points through the same triangles (NaN when uncalibrated). */
function inversePoints(index: InverseIndex, points: Float64Array): Float64Array {
  const count = points.length >> 1;
  const result = new Float64Array(count * 2);
  result.fill(NaN);
  const { lower, span, shape, buckets, transforms, anchors, source, triangles } = index;
  const candidateIndexes: number[] = [];
  const keys: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = points[i * 2];
    const y = points[i * 2 + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < lower[0] || x > lower[0] + span[0] || y < lower[1] || y > lower[1] + span[1]) continue;
    const cellX = Math.min(shape[0] - 1, Math.floor((x - lower[0]) / span[0] * shape[0]));
    const cellY = Math.min(shape[1] - 1, Math.floor((y - lower[1]) / span[1] * shape[1]));
    candidateIndexes.push(i);
    keys.push(cellY * shape[0] + cellX);
  }
  const order = argsortStable(keys);
  let begin = 0;
  while (begin < order.length) {
    const key = keys[order[begin]];
    let end = begin + 1;
    while (end < order.length && keys[order[end]] === key) end++;
    const bucket = buckets.get(key);
    if (bucket) {
      for (const triangle of bucket) {
        const offset = triangle * 4;
        const m00 = transforms[offset];
        const m01 = transforms[offset + 1];
        const m10 = transforms[offset + 2];
        const m11 = transforms[offset + 3];
        const anchorX = anchors[triangle * 2];
        const anchorY = anchors[triangle * 2 + 1];
        const ia = triangles[triangle * 3];
        const ib = triangles[triangle * 3 + 1];
        const ic = triangles[triangle * 3 + 2];
        for (let k = begin; k < end; k++) {
          const i = order[k];
          const dx = points[i * 2] - anchorX;
          const dy = points[i * 2 + 1] - anchorY;
          const u = m00 * dx + m01 * dy;
          const v = m10 * dx + m11 * dy;
          const w = 1 - u - v;
          if (u < -1e-7 || v < -1e-7 || w < -1e-7) continue;
          result[i * 2] = u * source[ia * 2] + v * source[ib * 2] + w * source[ic * 2];
          result[i * 2 + 1] = u * source[ia * 2 + 1] + v * source[ib * 2 + 1] + w * source[ic * 2 + 1];
        }
      }
    }
    begin = end;
  }
  return result;
}

function sampleBilinear(image: RgbaImage, x: number, y: number, out: Uint8ClampedArray, offset: number): void {
  const width = image.width;
  const height = image.height;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const data = image.data;
  for (let channel = 0; channel < 4; channel++) {
    let value = 0;
    for (let dy = 0; dy <= 1; dy++) {
      for (let dx = 0; dx <= 1; dx++) {
        const sampleX = x0 + dx;
        const sampleY = y0 + dy;
        if (sampleX < 0 || sampleX >= width || sampleY < 0 || sampleY >= height) continue;
        const weight = (dx === 0 ? 1 - fx : fx) * (dy === 0 ? 1 - fy : fy);
        value += data[(sampleY * width + sampleX) * 4 + channel] * weight;
      }
    }
    out[offset + channel] = value;
  }
}

function sampleNearest(image: RgbaImage, x: number, y: number, out: Uint8ClampedArray, offset: number): void {
  const sampleX = Math.round(x);
  const sampleY = Math.round(y);
  if (sampleX < 0 || sampleX >= image.width || sampleY < 0 || sampleY >= image.height) return;
  const sourceOffset = (sampleY * image.width + sampleX) * 4;
  out[offset] = image.data[sourceOffset];
  out[offset + 1] = image.data[sourceOffset + 1];
  out[offset + 2] = image.data[sourceOffset + 2];
  out[offset + 3] = image.data[sourceOffset + 3];
}

/**
 * Port of `warp_overlay`: inverse-sample `image` through the registration into
 * the EPSG:3857 extent of the registration target, cropped to the measured hull.
 * Long side defaults to 1700 px (bilinear overlay) or 1200 px (nearest terrain).
 */
export function warpOverlay(image: RgbaImage, registration: RegistrationData,
  options: WarpOverlayOptions = {}): WarpOverlayResult {
  const maxSide = options.maxSide ?? (options.nearest ? 1200 : 1700);
  const frameOrigin = options.frameOrigin ?? DEFAULT_ORIGIN;
  const target = registration.target;
  if (!target || target.length === 0) throw new Error(CHINESE.invalidAffine);
  let lowerX = Infinity;
  let lowerY = Infinity;
  let upperX = -Infinity;
  let upperY = -Infinity;
  for (const point of target) {
    const geo = mercator(toLngLat(point, frameOrigin));
    if (geo[0] < lowerX) lowerX = geo[0];
    if (geo[1] < lowerY) lowerY = geo[1];
    if (geo[0] > upperX) upperX = geo[0];
    if (geo[1] > upperY) upperY = geo[1];
  }
  const spanX = upperX - lowerX;
  const spanY = upperY - lowerY;
  const factor = maxSide / Math.max(spanX, spanY);
  const width = Math.max(1, Math.ceil(spanX * factor));
  const height = Math.max(1, Math.ceil(spanY * factor));
  const rgba = new Uint8ClampedArray(width * height * 4);
  const index = buildInverseIndex(registration);
  const scales = frameScales(frameOrigin);
  const stripeHeight = 100;
  for (let stripeY = 0; stripeY < height; stripeY += stripeHeight) {
    const rows = Math.min(stripeHeight, height - stripeY);
    const world = new Float64Array(rows * width * 2);
    for (let row = 0; row < rows; row++) {
      const mercatorY = upperY - ((stripeY + row + 0.5) / height) * spanY;
      const latitude = (2 * Math.atan(Math.exp(mercatorY / WGS84_RADIUS)) - Math.PI / 2) * 180 / Math.PI;
      for (let x = 0; x < width; x++) {
        const mercatorX = lowerX + ((x + 0.5) / width) * spanX;
        const longitude = (mercatorX / WGS84_RADIUS) * 180 / Math.PI;
        const offset = (row * width + x) * 2;
        // local = (lnglat - origin) * scales
        world[offset] = (longitude - frameOrigin[0]) * scales[0];
        world[offset + 1] = (latitude - frameOrigin[1]) * scales[1];
      }
    }
    const pixels = inversePoints(index, world);
    const rowsData = rows * width;
    if (options.nearest) {
      for (let i = 0; i < rowsData; i++) {
        const pixelX = pixels[i * 2];
        const pixelY = pixels[i * 2 + 1];
        if (!Number.isFinite(pixelX) || !Number.isFinite(pixelY)) continue;
        const outputOffset = ((stripeY + (i / width | 0)) * width + (i % width)) * 4;
        sampleNearest(image, pixelX, pixelY, rgba, outputOffset);
      }
    } else {
      for (let i = 0; i < rowsData; i++) {
        const pixelX = pixels[i * 2];
        const pixelY = pixels[i * 2 + 1];
        if (!Number.isFinite(pixelX) || !Number.isFinite(pixelY)) continue;
        const outputOffset = ((stripeY + (i / width | 0)) * width + (i % width)) * 4;
        sampleBilinear(image, pixelX, pixelY, rgba, outputOffset);
      }
    }
  }
  return { rgba, width, height, extent: [lowerX, lowerY, upperX, upperY] };
}
