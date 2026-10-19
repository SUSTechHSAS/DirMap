/** WGS84/local-metre frame conversions and the invertible registration mesh.
 *
 * Ports the parts of `dirmap/geometry.py` the static browser build needs. The
 * mesh keeps the exact triangle topology produced by the baker, so forward and
 * inverse mapping use the same triangles (retriangulating the target vertices
 * would silently change the warp).
 */
import { localScales as frameScales } from './types.ts';
import type { Pixel } from './types.ts';

export interface FrameInfo {
  crs: string;
  origin: Pixel;
}

export interface RegistrationData {
  affine: number[][];
  source: Pixel[];
  target: Pixel[];
  triangles: number[][];
  /** Optional raster dimensions for client-processed maps (browser builds). */
  width?: number;
  height?: number;
}

export { frameScales as localScales };

/** Tolerance used by the browser-side point-in-triangle test. */
const BARYCENTRIC_TOLERANCE = 1e-7;

/** Longitude/latitude degrees -> local east/north metres. */
export function frameLocal(lnglat: Pixel, origin: Pixel): Pixel {
  const scales = frameScales(origin);
  return [(lnglat[0] - origin[0]) * scales[0], (lnglat[1] - origin[1]) * scales[1]];
}

/** Local east/north metres -> longitude/latitude degrees. */
export function frameLngLat(local: Pixel, origin: Pixel): Pixel {
  const scales = frameScales(origin);
  return [local[0] / scales[0] + origin[0], local[1] / scales[1] + origin[1]];
}

/** Row-major uniform grid over triangle bounding boxes. */
interface TriangleIndex {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  cellX: number;
  cellY: number;
  cols: number;
  rows: number;
  offsets: Uint32Array;
  entries: Uint32Array;
}

function buildTriangleIndex(vertices: Float64Array, triangles: Int32Array, triangleCount: number): TriangleIndex {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let t = 0; t < triangleCount; t++) {
    for (let k = 0; k < 3; k++) {
      const vertex = triangles[t * 3 + k];
      const x = vertices[vertex * 2];
      const y = vertices[vertex * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX) || maxX <= minX || maxY <= minY) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0, cellX: 1, cellY: 1, cols: 1, rows: 1,
      offsets: new Uint32Array(2), entries: new Uint32Array(0) };
  }
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const wanted = Math.max(1, Math.min(128, Math.ceil(Math.sqrt(triangleCount))));
  const aspect = spanX / spanY;
  const cols = Math.max(1, Math.min(256, Math.ceil(wanted * Math.sqrt(aspect))));
  const rows = Math.max(1, Math.min(256, Math.ceil(wanted / Math.sqrt(aspect))));
  const cellX = spanX / cols;
  const cellY = spanY / rows;
  const cellCount = (cols + 1) * (rows + 1);
  const counts = new Uint32Array(cellCount);
  const cellOf = (x: number, y: number): [number, number] => {
    let cx = Math.floor((x - minX) / cellX);
    let cy = Math.floor((y - minY) / cellY);
    if (cx < 0) cx = 0;
    if (cx > cols) cx = cols;
    if (cy < 0) cy = 0;
    if (cy > rows) cy = rows;
    return [cx, cy];
  };
  for (let t = 0; t < triangleCount; t++) {
    let tMinX = Infinity;
    let tMinY = Infinity;
    let tMaxX = -Infinity;
    let tMaxY = -Infinity;
    for (let k = 0; k < 3; k++) {
      const vertex = triangles[t * 3 + k];
      const x = vertices[vertex * 2];
      const y = vertices[vertex * 2 + 1];
      if (x < tMinX) tMinX = x;
      if (x > tMaxX) tMaxX = x;
      if (y < tMinY) tMinY = y;
      if (y > tMaxY) tMaxY = y;
    }
    const [x0, y0] = cellOf(tMinX, tMinY);
    const [x1, y1] = cellOf(tMaxX, tMaxY);
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) counts[cy * (cols + 1) + cx]++;
    }
  }
  const offsets = new Uint32Array(cellCount + 1);
  for (let i = 0; i < cellCount; i++) offsets[i + 1] = offsets[i] + counts[i];
  const entries = new Uint32Array(offsets[cellCount]);
  const cursor = offsets.slice(0, cellCount);
  for (let t = 0; t < triangleCount; t++) {
    let tMinX = Infinity;
    let tMinY = Infinity;
    let tMaxX = -Infinity;
    let tMaxY = -Infinity;
    for (let k = 0; k < 3; k++) {
      const vertex = triangles[t * 3 + k];
      const x = vertices[vertex * 2];
      const y = vertices[vertex * 2 + 1];
      if (x < tMinX) tMinX = x;
      if (x > tMaxX) tMaxX = x;
      if (y < tMinY) tMinY = y;
      if (y > tMaxY) tMaxY = y;
    }
    const [x0, y0] = cellOf(tMinX, tMinY);
    const [x1, y1] = cellOf(tMaxX, tMaxY);
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) entries[cursor[cy * (cols + 1) + cx]++] = t;
    }
  }
  return { minX, minY, maxX, maxY, cellX, cellY, cols, rows, offsets, entries };
}

/** Barycentric coordinates u, v of (px, py) inside the triangle, or null. */
function barycentric(
  px: number, py: number,
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
): [number, number] | null {
  const determinant = (ax - cx) * (by - cy) - (bx - cx) * (ay - cy);
  if (Math.abs(determinant) < 1e-10) return null;
  const u = ((px - cx) * (by - cy) - (bx - cx) * (py - cy)) / determinant;
  const v = ((ax - cx) * (py - cy) - (px - cx) * (ay - cy)) / determinant;
  if (u < -BARYCENTRIC_TOLERANCE || v < -BARYCENTRIC_TOLERANCE || u + v > 1 + BARYCENTRIC_TOLERANCE) return null;
  return [u, v];
}

/** Index of the triangle containing the point, or -1. */
function locateTriangle(
  index: TriangleIndex | null, vertices: Float64Array, triangles: Int32Array,
  px: number, py: number,
): number {
  if (index === null || !Number.isFinite(px) || !Number.isFinite(py)) return -1;
  if (px < index.minX - 1e-6 || px > index.maxX + 1e-6 || py < index.minY - 1e-6 || py > index.maxY + 1e-6) return -1;
  let cx = Math.floor((px - index.minX) / index.cellX);
  let cy = Math.floor((py - index.minY) / index.cellY);
  if (cx < 0) cx = 0;
  if (cx > index.cols) cx = index.cols;
  if (cy < 0) cy = 0;
  if (cy > index.rows) cy = index.rows;
  const cell = cy * (index.cols + 1) + cx;
  const start = index.offsets[cell];
  const end = index.offsets[cell + 1];
  for (let e = start; e < end; e++) {
    const t = index.entries[e];
    const a = triangles[t * 3];
    const b = triangles[t * 3 + 1];
    const c = triangles[t * 3 + 2];
    if (barycentric(px, py, vertices[a * 2], vertices[a * 2 + 1], vertices[b * 2], vertices[b * 2 + 1],
      vertices[c * 2], vertices[c * 2 + 1]) !== null) return t;
  }
  return -1;
}

function interpolate(vertices: Float64Array, a: number, b: number, c: number, u: number, v: number): [number, number] {
  const w = 1 - u - v;
  return [
    u * vertices[a * 2] + v * vertices[b * 2] + w * vertices[c * 2],
    u * vertices[a * 2 + 1] + v * vertices[b * 2 + 1] + w * vertices[c * 2 + 1],
  ];
}

/**
 * The registration mesh: global affine for points outside the measured hull,
 * piecewise-affine interpolation of the baked triangles inside it.
 */
export class Mesh {
  private readonly affine: number[][];
  private readonly source: Float64Array;
  private readonly target: Float64Array;
  private readonly triangles: Int32Array;
  private readonly triangleCount: number;
  private readonly sourceIndex: TriangleIndex | null;
  private targetIndex: TriangleIndex | null;

  constructor(registration: RegistrationData | null) {
    if (!registration) {
      this.affine = [[1, 0, 0], [0, 1, 0]];
      this.source = new Float64Array(2);
      this.target = new Float64Array(2);
      this.triangles = new Int32Array(0);
      this.triangleCount = 0;
      this.sourceIndex = null;
      this.targetIndex = null;
      return;
    }
    const affine = registration.affine;
    const validAffine = Array.isArray(affine) && affine.length === 2
      && affine.every(row => Array.isArray(row) && row.length === 3 && row.every(value => Number.isFinite(value)));
    if (!validAffine) throw new Error('配准矩阵无效');
    const determinant = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0];
    if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-10) throw new Error('配准矩阵无效');
    this.affine = [affine[0].slice(), affine[1].slice()];
    const source = Array.isArray(registration.source) ? registration.source : [];
    const target = Array.isArray(registration.target) ? registration.target : [];
    const pointCount = Math.min(source.length, target.length);
    this.source = new Float64Array(pointCount * 2);
    this.target = new Float64Array(pointCount * 2);
    for (let i = 0; i < pointCount; i++) {
      this.source[i * 2] = source[i][0];
      this.source[i * 2 + 1] = source[i][1];
      this.target[i * 2] = target[i][0];
      this.target[i * 2 + 1] = target[i][1];
    }
    const ids: number[] = [];
    for (const triangle of Array.isArray(registration.triangles) ? registration.triangles : []) {
      if (!Array.isArray(triangle) || triangle.length < 3) continue;
      const a = triangle[0];
      const b = triangle[1];
      const c = triangle[2];
      if (a >= 0 && b >= 0 && c >= 0 && a < pointCount && b < pointCount && c < pointCount) ids.push(a, b, c);
    }
    this.triangles = new Int32Array(ids);
    this.triangleCount = this.triangles.length / 3;
    this.sourceIndex = this.triangleCount > 0 ? buildTriangleIndex(this.source, this.triangles, this.triangleCount) : null;
    this.targetIndex = null;
  }

  /** Whether the registration actually provides a triangle mesh. */
  get hasMesh(): boolean {
    return this.triangleCount > 0;
  }

  /** Forward-map many native-pixel points to local metres. */
  forwardMany(pixels: Float64Array): { local: Float64Array; inside: Uint8Array } {
    const pointCount = pixels.length >> 1;
    const local = new Float64Array(pointCount * 2);
    const inside = new Uint8Array(pointCount);
    const [a0, a1, a2] = [this.affine[0][0], this.affine[0][1], this.affine[0][2]];
    const [b0, b1, b2] = [this.affine[1][0], this.affine[1][1], this.affine[1][2]];
    const index = this.sourceIndex;
    const triangles = this.triangles;
    for (let i = 0; i < pointCount; i++) {
      const x = pixels[i * 2];
      const y = pixels[i * 2 + 1];
      local[i * 2] = a0 * x + a1 * y + a2;
      local[i * 2 + 1] = b0 * x + b1 * y + b2;
      if (index === null) continue;
      const triangle = locateTriangle(index, this.source, triangles, x, y);
      if (triangle < 0) continue;
      const triangleA = triangles[triangle * 3];
      const triangleB = triangles[triangle * 3 + 1];
      const triangleC = triangles[triangle * 3 + 2];
      const weights = barycentric(x, y, this.source[triangleA * 2], this.source[triangleA * 2 + 1],
        this.source[triangleB * 2], this.source[triangleB * 2 + 1], this.source[triangleC * 2], this.source[triangleC * 2 + 1]);
      if (weights === null) continue;
      const point = interpolate(this.target, triangleA, triangleB, triangleC, weights[0], weights[1]);
      local[i * 2] = point[0];
      local[i * 2 + 1] = point[1];
      inside[i] = 1;
    }
    return { local, inside };
  }

  /** Forward-map one native-pixel point. */
  forward(pixel: Pixel): { local: Pixel; inside: boolean } {
    const x = pixel[0];
    const y = pixel[1];
    const affineLocal: Pixel = [
      this.affine[0][0] * x + this.affine[0][1] * y + this.affine[0][2],
      this.affine[1][0] * x + this.affine[1][1] * y + this.affine[1][2],
    ];
    if (this.sourceIndex === null) return { local: affineLocal, inside: false };
    const triangle = locateTriangle(this.sourceIndex, this.source, this.triangles, x, y);
    if (triangle < 0) return { local: affineLocal, inside: false };
    const triangleA = this.triangles[triangle * 3];
    const triangleB = this.triangles[triangle * 3 + 1];
    const triangleC = this.triangles[triangle * 3 + 2];
    const weights = barycentric(x, y, this.source[triangleA * 2], this.source[triangleA * 2 + 1],
      this.source[triangleB * 2], this.source[triangleB * 2 + 1], this.source[triangleC * 2], this.source[triangleC * 2 + 1]);
    if (weights === null) return { local: affineLocal, inside: false };
    return { local: interpolate(this.target, triangleA, triangleB, triangleC, weights[0], weights[1]), inside: true };
  }

  /** Invert the same triangles (local metres -> native pixels). */
  inverse(local: Pixel): Pixel | null {
    const x = local[0];
    const y = local[1];
    if (!Number.isFinite(x) || !Number.isFinite(y) || this.triangleCount === 0) return null;
    if (this.targetIndex === null) this.targetIndex = buildTriangleIndex(this.target, this.triangles, this.triangleCount);
    const triangle = locateTriangle(this.targetIndex, this.target, this.triangles, x, y);
    if (triangle < 0) return null;
    const triangleA = this.triangles[triangle * 3];
    const triangleB = this.triangles[triangle * 3 + 1];
    const triangleC = this.triangles[triangle * 3 + 2];
    const weights = barycentric(x, y, this.target[triangleA * 2], this.target[triangleA * 2 + 1],
      this.target[triangleB * 2], this.target[triangleB * 2 + 1], this.target[triangleC * 2], this.target[triangleC * 2 + 1]);
    if (weights === null) return null;
    return interpolate(this.source, triangleA, triangleB, triangleC, weights[0], weights[1]);
  }

  /** Native pixel -> WGS84 degrees through the mesh (null when uncalibrated). */
  pixelToLngLat(pixel: Pixel, origin: Pixel): Pixel | null {
    const mapped = this.forward(pixel);
    return mapped.inside ? frameLngLat(mapped.local, origin) : null;
  }

  /** WGS84 degrees -> native pixel through the inverse mesh (null when uncalibrated). */
  lngLatToPixel(lnglat: Pixel, origin: Pixel): Pixel | null {
    return this.inverse(frameLocal(lnglat, origin));
  }
}
