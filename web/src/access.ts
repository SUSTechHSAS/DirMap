/** Legal approach candidates for cartographic point symbols and building edges.
 *
 * Browser port of `dirmap/access.py`. The control's circle centre remains the
 * target's exact map position; approaches end on walkable ground and never add
 * an edge through an obstacle. A long wall, a forbidden area, conflicting
 * sides, and a large enclosing feature need review.
 *
 * The module is browser-compatible: all image work goes through `cv.ts`
 * (opencv.js WASM wrappers) on plain typed arrays, and no Node API is used.
 * The pure 2-D helpers (`rintEven`, covariance, union-of-masks bookkeeping)
 * live here so they are testable without a browser.
 */
import {
  arcLength,
  connectedComponents,
  connectedComponentsWithStats,
  contourArea,
  copyMakeBorder,
  dilate,
  fillContours,
  findContours,
  findContoursWithHierarchy,
  getCv,
  loadOpenCV,
  maskOf,
  minEnclosingCircle,
  morphology,
  pointPolygonTest,
} from './cv.ts';
import type { Contour, MaskImage, ComponentsResult } from './cv.ts';
import { frameLngLat } from './geometry.ts';
import type { RegistrationData } from './geometry.ts';
import type { Control, Pixel, TerrainEdit } from './types.ts';

/** Fixed WGS84 frame origin used by `dirmap.geometry.Frame` (degrees). */
export const FRAME_ORIGIN: Pixel = [113.9304, 22.535];

/** Semantic layers consumed by the resolver (`terrain.npz` / `AnalysisSemantics`). */
export type AccessSemanticsKind = 'building' | 'water' | 'point_ink' | 'vegetation' | 'restricted';

/** 0/1 (or 0/255) per-pixel masks, length width*height. Missing layer -> null. */
export interface AccessSemantics {
  building?: Uint8Array | null;
  water?: Uint8Array | null;
  point_ink?: Uint8Array | null;
  vegetation?: Uint8Array | null;
  restricted?: Uint8Array | null;
}

/** Semantic masks decoded from `maps/<id>/semantics.bin.gz` (DMSEM1). */
export interface SemanticsBundle {
  width: number;
  height: number;
  masks: AccessSemantics;
}

/** The subset of `RasterGraph` the resolver needs (browser graph or edited graph). */
export interface AccessGraph {
  readonly width: number;
  readonly height: number;
  readonly costs: Float32Array;
  readonly pixels: Float64Array;
  readonly local: Float64Array;
  readonly components: Uint32Array;
  /** Full-map teacher closures (`raster-graph.ts` field; null = none). */
  readonly closedEdits?: Uint8Array | null;
  snap(pixel: Pixel, maxDistancePx?: number): number;
  segmentPixels(a: Pixel, b: Pixel): Pixel[];
  queryNodesNear(point: Pixel, maxDistancePx: number): number[];
}

/** Forward mapping used for local metres / lnglat (structural `Mesh`). */
export interface MeshLike {
  forward(pixel: Pixel): { local: Pixel; inside: boolean };
}

/** Everything the resolver reads from the processed/baked map. */
export interface AccessContext {
  graph: AccessGraph;
  mesh: MeshLike;
  semantics: AccessSemantics;
  edits?: TerrainEdit[];
  /** Optional override; otherwise `graph.closedEdits` is used. */
  closedEdits?: Uint8Array | null;
  /** Registration affine gives the exact pixel->metre scale (`sqrt(|det|)`). */
  registration?: RegistrationData;
}

export interface AccessOptions {
  /** Component of the canonical start/finish node (Addendum A re-resolve rule). */
  startComponent?: number;
  /** WGS84 frame origin for `lnglat`; defaults to the department frame. */
  frameOrigin?: Pixel;
}

/** Successful baked-compatible approach (same shape as `approaches.json` entries). */
export interface ResolvedApproach {
  ok: true;
  node: number;
  pixel: Pixel;
  local: Pixel;
  method: string;
  automatic: boolean;
  offsetM: number;
  lnglat: Pixel;
  targetLnglat: Pixel | null;
  featureKind?: string;
  reviewRequired?: boolean;
  targetPixel: Pixel;
  component: number;
}

/** Failed approach with the exact Chinese explanation the Python server produced. */
export interface UnresolvedApproach {
  ok: false;
  message: string;
}

export type ApproachResolution = ResolvedApproach | UnresolvedApproach;

/** Internal `resolve()` result (a superset of `BakedApproach`, without lnglat). */
interface InternalResolution {
  node: number;
  pixel: Pixel;
  local: Pixel;
  method: string;
  automatic: boolean;
  offsetM: number;
  featureKind?: string;
  reviewRequired?: boolean;
  targetPixel?: Pixel;
}

interface Feature {
  origin: Pixel;
  mask: MaskImage;
  limitM: number;
  method: string;
  kind: AccessSemanticsKind;
  symbol: MaskImage;
  symbolOrigin: Pixel;
  stroke?: boolean;
  strokeLabel?: number;
}

interface ComponentsParts {
  labels: Int32Array;
  stats: ComponentsResult['stats'];
  width: number;
  height: number;
}

/** Offsets closer than this are the same distance for candidate selection.
 *
 * The registration mesh is piecewise affine, so two pixel centres symmetric
 * about a half-integer control pixel map to points *exactly* equidistant from
 * the control. Double-precision mesh evaluation leaves ~1e-14 m of rounding
 * noise there; a stable tie-break must not depend on it.
 */
const OFFSET_TIE_EPSILON = 1e-12;

/** Resolution failure whose message is shown to the user verbatim. */
export class ApproachError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApproachError';
  }
}

/** NumPy `np.rint` / Python `round`: round half to even. */
export function rintEven(value: number): number {
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** 0/1 mask -> 0/255 [`MaskImage`] copy (cv2 only cares about nonzero). */
function binaryMask(data: Uint8Array, width: number, height: number): MaskImage {
  const image = maskOf(width, height);
  for (let i = 0; i < data.length; i++) if (data[i]) image.data[i] = 255;
  return image;
}

/** Eigenvalues (ascending) of the 2x2 sample covariance, NumPy `np.cov` ddof=1. */
function covarianceEigenvalues(xs: number[], ys: number[]): [number, number] {
  const count = xs.length;
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < count; i++) {
    meanX += xs[i];
    meanY += ys[i];
  }
  meanX /= count;
  meanY /= count;
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (let i = 0; i < count; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    xx += dx * dx;
    yy += dy * dy;
    xy += dx * dy;
  }
  const divisor = Math.max(1, count - 1);
  xx /= divisor;
  yy /= divisor;
  xy /= divisor;
  const half = (xx + yy) / 2;
  const difference = (xx - yy) / 2;
  const spread = Math.sqrt(difference * difference + xy * xy);
  return [half - spread, half + spread];
}

/** Decode `maps/<id>/semantics.bin.gz` (after gzip inflate) into 0/1 masks. */
export function decodeSemantics(buffer: ArrayBuffer | Uint8Array): SemanticsBundle {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length < 17) throw new Error('语义数据格式无效');
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
  if (magic !== 'DMSEM1') throw new Error('语义数据格式无效');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint16(6, true);
  if (version !== 1) throw new Error(`语义数据版本不受支持（${version}）`);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const count = bytes[16];
  let offset = 17;
  const names: string[] = [];
  for (let i = 0; i < count; i++) {
    if (offset >= bytes.length) throw new Error('语义数据格式无效');
    const length = bytes[offset];
    offset += 1;
    if (offset + length > bytes.length) throw new Error('语义数据格式无效');
    names.push(new TextDecoder().decode(bytes.subarray(offset, offset + length)));
    offset += length;
  }
  const pixels = width * height;
  const maskBytes = Math.ceil(pixels / 8);
  if (offset + count * maskBytes > bytes.length) throw new Error('语义数据长度不足');
  const masks: AccessSemantics = {};
  for (let i = 0; i < count; i++) {
    const name = names[i];
    const row = new Uint8Array(pixels);
    const start = offset + i * maskBytes;
    for (let p = 0; p < pixels; p++) {
      if ((bytes[start + (p >> 3)] >> (p & 7)) & 1) row[p] = 1;
    }
    if (name === 'building' || name === 'water' || name === 'point_ink' || name === 'vegetation' || name === 'restricted') {
      masks[name] = row;
    }
  }
  return { width, height, masks };
}

/** Await opencv.js; call once before the synchronous resolve functions. */
export async function prepareAccess(): Promise<void> {
  await loadOpenCV();
}

/** Whether the CV runtime has been loaded (`resolveApproaches` needs it). */
export function accessCvReady(): boolean {
  try {
    getCv();
    return true;
  } catch {
    return false;
  }
}

/**
 * Estimate metres-per-pixel from graph edges when the registration affine is
 * not supplied. Each CSR edge pairs a pixel distance with a measured local
 * distance; the median ratio is the affine determinant scale to first order.
 */
export function estimateScaleFromGraph(graph: AccessGraph): number {
  const nodeCount = Math.floor(graph.local.length / 2);
  const stride = Math.max(1, Math.floor(nodeCount / 2000));
  const ratios: number[] = [];
  for (let node = 0; node + 1 < nodeCount; node += stride) {
    const other = node + 1;
    const pixelDistance = Math.hypot(graph.pixels[other * 2] - graph.pixels[node * 2],
      graph.pixels[other * 2 + 1] - graph.pixels[node * 2 + 1]);
    if (pixelDistance < 1e-9 || pixelDistance > 24) continue;
    const localDistance = Math.hypot(graph.local[other * 2] - graph.local[node * 2],
      graph.local[other * 2 + 1] - graph.local[node * 2 + 1]);
    if (localDistance < 1e-9) continue;
    ratios.push(localDistance / pixelDistance);
  }
  if (!ratios.length) return 1;
  ratios.sort((left, right) => left - right);
  return ratios[Math.floor(ratios.length / 2)];
}

/**
 * Port of `dirmap.access.AccessResolver`.
 *
 * Instantiate per `resolveApproaches` call: semantic connected components are
 * cached per instance (full-map masks are expensive), which matches Python's
 * per-map resolver lifetime.
 */
export class AccessResolver {
  readonly graph: AccessGraph;
  readonly mesh: MeshLike;
  readonly semantics: AccessSemantics;
  readonly scale: number;
  private readonly closedEdits: Uint8Array | null;
  private readonly componentsCache = new Map<AccessSemanticsKind, ComponentsParts | null>();

  constructor(context: AccessContext) {
    this.graph = context.graph;
    this.mesh = context.mesh;
    this.semantics = context.semantics ?? {};
    this.closedEdits = context.closedEdits ?? context.graph.closedEdits ?? null;
    const affine = context.registration?.affine;
    const validAffine = Array.isArray(affine) && affine.length === 2
      && affine.every(row => Array.isArray(row) && row.length === 3 && row.every(value => Number.isFinite(value)));
    if (validAffine) {
      const determinant = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0];
      if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-10) throw new Error('配准矩阵无效');
      this.scale = Math.sqrt(Math.abs(determinant));
    } else {
      this.scale = estimateScaleFromGraph(context.graph);
    }
  }

  /** `_components`: semantic mask -> (water close 5x5) -> labels/stats, cached. */
  private componentParts(kind: AccessSemanticsKind): ComponentsParts | null {
    const cached = this.componentsCache.get(kind);
    if (cached !== undefined) return cached;
    const raw = this.semantics[kind];
    if (raw === null || raw === undefined) {
      this.componentsCache.set(kind, null);
      return null;
    }
    const width = this.graph.width;
    const height = this.graph.height;
    if (raw.length !== width * height) throw new Error('语义掩膜尺寸与通行图不一致');
    let mask = binaryMask(raw, width, height);
    if (kind === 'water') {
      // This semantic footprint only joins the quadrants of a tiny fountain's
      // cross symbol. It never modifies terrain.
      mask = morphology(mask, 'close', 5);
    }
    const result = connectedComponentsWithStats(mask, 8);
    const parts: ComponentsParts = { labels: result.labels, stats: result.stats, width, height };
    this.componentsCache.set(kind, parts);
    return parts;
  }

  /** Build the 0/255 mask of one connected-component label inside a bbox. */
  private labelMask(parts: ComponentsParts, label: number, bx: number, by: number, width: number, height: number): MaskImage {
    const mask = maskOf(width, height);
    for (let row = 0; row < height; row++) {
      const source = (by + row) * parts.width + bx;
      const target = row * width;
      for (let column = 0; column < width; column++) {
        if (parts.labels[source + column] === label) mask.data[target + column] = 255;
      }
    }
    return mask;
  }

  /** `_feature`: per-kind candidate rules; returns the first qualifying component. */
  private feature(kind: AccessSemanticsKind, point: Pixel, circleRadius: number): Feature | null {
    const parts = this.componentParts(kind);
    if (parts === null) return null;
    const labels = parts.labels;
    const stats = parts.stats;
    const width = parts.width;
    const height = parts.height;
    const x = rintEven(point[0]);
    const y = rintEven(point[1]);
    if (x < 0 || y < 0 || x >= width || y >= height) return null;
    const ids: number[] = [];
    const centreLabel = labels[y * width + x];
    if (centreLabel) ids.push(centreLabel);
    const nearLabels = new Set<number>();
    for (let row = Math.max(0, y - 2); row <= Math.min(height - 1, y + 2); row++) {
      for (let column = Math.max(0, x - 2); column <= Math.min(width - 1, x + 2); column++) {
        const label = labels[row * width + column];
        if (label) nearLabels.add(label);
      }
    }
    for (const label of [...nearLabels].sort((left, right) => left - right)) {
      if (!ids.includes(label)) ids.push(label);
    }
    // White interiors of a small ring/cross can be isolated graph nodes; look
    // for an enclosing point-symbol contour, not an enclosing wall.
    const radius = Math.max(8, rintEven(Math.min(circleRadius, 12 / Math.max(this.scale, 0.1))));
    const x0 = Math.max(0, x - radius);
    const y0 = Math.max(0, y - radius);
    const x1 = Math.min(width, x + radius + 1);
    const y1 = Math.min(height, y + radius + 1);
    const cropWidth = x1 - x0;
    const cropHeight = y1 - y0;
    const crop = maskOf(cropWidth, cropHeight);
    for (let row = 0; row < cropHeight; row++) {
      const source = (y0 + row) * width + x0;
      const target = row * cropWidth;
      for (let column = 0; column < cropWidth; column++) {
        if (labels[source + column] > 0) crop.data[target + column] = 255;
      }
    }
    for (const contour of findContours(crop, 'external')) {
      if (pointPolygonTest(contour, [point[0] - x0, point[1] - y0], false) < 0) continue;
      const first = contour.points[0];
      if (!first) continue;
      const sampleX = rintEven(first[0]) + x0;
      const sampleY = rintEven(first[1]) + y0;
      const label = labels[sampleY * width + sampleX];
      if (label && !ids.includes(label)) ids.push(label);
    }
    for (const label of ids) {
      const stat = stats[label];
      if (!stat) continue;
      const bx = stat.x;
      const by = stat.y;
      const boxWidth = stat.width;
      const boxHeight = stat.height;
      const area = stat.area;
      const largest = Math.max(boxWidth, boxHeight) * this.scale;
      const smallest = Math.min(boxWidth, boxHeight) * this.scale;
      let limit: number;
      let method: string;
      let local: MaskImage;
      if (kind === 'point_ink' || kind === 'vegetation') {
        if (largest > (kind === 'point_ink' ? 18 : 8) || Math.max(boxWidth, boxHeight) > Math.max(12, circleRadius)) continue;
        local = this.labelMask(parts, label, bx, by, boxWidth, boxHeight);
        const xs: number[] = [];
        const ys: number[] = [];
        for (let row = 0; row < boxHeight; row++) {
          for (let column = 0; column < boxWidth; column++) {
            if (local.data[row * boxWidth + column]) {
              xs.push(column);
              ys.push(row);
            }
          }
        }
        if (xs.length < 3) continue;
        const eigenvalues = covarianceEigenvalues(xs, ys);
        if (eigenvalues[0] < 0.08 * Math.max(1e-6, eigenvalues[1])) continue; // narrow line / fence
        if (kind === 'point_ink' && area / (boxWidth * boxHeight) < 0.22) continue; // sparse angular wall
        limit = Math.min(8, Math.max(3, largest / 2 + 1));
        method = 'point_symbol_edge';
      } else if (kind === 'building') {
        // Reject a narrow elongated grey wall as well as a linear black barrier.
        if (smallest < 3.5 && largest > 3 * Math.max(0.1, smallest)) continue;
        if (area * this.scale * this.scale < 8) continue;
        local = this.labelMask(parts, label, bx, by, boxWidth, boxHeight);
        limit = 3;
        method = 'building_edge';
        if (largest <= 26 && Math.max(boxWidth, boxHeight) <= circleRadius * 1.5 && area / (boxWidth * boxHeight) < 0.55) {
          const detailed = findContoursWithHierarchy(local, 'ccomp');
          const rings: Contour[] = [];
          for (let index = 0; index < detailed.contours.length; index++) {
            const relation = detailed.hierarchy[index];
            if (!relation || relation.parent < 0) continue;
            const contour = detailed.contours[index];
            const circle = minEnclosingCircle(contour);
            const ringArea = contourArea(contour);
            if (ringArea >= 6 && ringArea / Math.max(1, Math.PI * circle.radius * circle.radius) >= 0.75) rings.push(contour);
          }
          if (rings.length >= 2
            && rings.some(contour => pointPolygonTest(contour, [point[0] - bx, point[1] - by], false) >= 0)) {
            limit = 8;
            method = 'compound_point_symbol_edge';
          }
        }
      } else {
        local = this.labelMask(parts, label, bx, by, boxWidth, boxHeight);
        const outline = findContours(local, 'external');
        if (!outline.length) continue;
        let contour = outline[0];
        let contourSize = contourArea(contour);
        for (const candidate of outline) {
          const candidateArea = contourArea(candidate);
          if (candidateArea > contourSize) {
            contour = candidate;
            contourSize = candidateArea;
          }
        }
        const perimeter = arcLength(contour, true);
        const circularity = 4 * Math.PI * contourSize / Math.max(1, perimeter * perimeter);
        if (largest <= 18 && smallest >= largest * 0.6 && area * this.scale * this.scale <= 240 && circularity >= 0.65) {
          limit = Math.min(9, largest / 2 + 1.5);
          method = 'small_water_feature_edge';
        } else {
          // A bank can be approached only immediately at its edge; an interior
          // lake point cannot be moved to a distant road.
          limit = 3;
          method = 'water_edge';
        }
      }
      // A symbol footprint may be filled to locate its outside edge; this fill
      // is NEVER applied to the graph's navigable cost mask.
      const contourList = findContours(local, 'external');
      const footprint = maskOf(boxWidth, boxHeight);
      fillContours(footprint, contourList, 255);
      const padded = dilate(copyMakeBorder(footprint, 2, 0), 3);
      const origin: Pixel = [bx - 2, by - 2];
      const atX = rintEven(point[0] - origin[0]);
      const atY = rintEven(point[1] - origin[1]);
      if (!(atX >= 0 && atY >= 0 && atX < padded.width && atY < padded.height
        && padded.data[atY * padded.width + atX])) continue;
      // The filled silhouette is the symbol itself; the dilated footprint only
      // bounds the search. Crossing the symbol's own body is expected.
      const symbol = maskOf(boxWidth, boxHeight);
      fillContours(symbol, contourList, 255);
      return { origin, mask: padded, limitM: limit, method, kind, symbol, symbolOrigin: [bx, by] };
    }
    return null;
  }

  /** `_in_feature`: each point inside the (dilated) footprint search mask. */
  private inFeature(feature: Feature, points: Pixel[]): boolean[] {
    return points.map(point => {
      const x = rintEven(point[0] - feature.origin[0]);
      const y = rintEven(point[1] - feature.origin[1]);
      if (x < 0 || y < 0 || x >= feature.mask.width || y >= feature.mask.height) return false;
      return feature.mask.data[y * feature.mask.width + x] > 0;
    });
  }

  /** `_over_symbol`: whether each pixel lies on the feature itself, not its halo. */
  private overSymbol(feature: Feature, points: Pixel[]): boolean[] {
    const symbol = feature.symbol;
    return points.map(point => {
      const x = rintEven(point[0] - feature.symbolOrigin[0]);
      const y = rintEven(point[1] - feature.symbolOrigin[1]);
      if (x < 0 || y < 0 || x >= symbol.width || y >= symbol.height) return false;
      return symbol.data[y * symbol.width + x] > 0;
    });
  }

  /** `_complete_outline`: recover a small symbol's dark rim, never a connected wall. */
  private completeOutline(feature: Feature, circleRadius: number): Feature {
    if (feature.stroke || feature.method === 'building_edge' || feature.method === 'water_edge') return feature;
    const symbol = feature.symbol;
    const [symbolX, symbolY] = feature.symbolOrigin;
    const width = this.graph.width;
    const height = this.graph.height;
    const padding = Math.max(2, Math.min(4, rintEven(circleRadius * 0.10)));
    const x0 = Math.max(0, symbolX - padding - 1);
    const y0 = Math.max(0, symbolY - padding - 1);
    const x1 = Math.min(width, symbolX + symbol.width + padding + 1);
    const y1 = Math.min(height, symbolY + symbol.height + padding + 1);
    const patchWidth = x1 - x0;
    const patchHeight = y1 - y0;
    const hard = maskOf(patchWidth, patchHeight);
    for (let row = 0; row < patchHeight; row++) {
      const source = (y0 + row) * width + x0;
      const target = row * patchWidth;
      for (let column = 0; column < patchWidth; column++) {
        if (!(this.graph.costs[source + column] > 0)) hard.data[target + column] = 255;
      }
    }
    const labels = connectedComponentsWithStats(hard, 8).labels;
    const owned = maskOf(patchWidth, patchHeight);
    const offsetX = symbolX - x0;
    const offsetY = symbolY - y0;
    for (let row = 0; row < symbol.height; row++) {
      for (let column = 0; column < symbol.width; column++) {
        if (symbol.data[row * symbol.width + column]) owned.data[(offsetY + row) * patchWidth + offsetX + column] = 255;
      }
    }
    const ownedLabels = new Set<number>();
    for (let i = 0; i < owned.data.length; i++) if (owned.data[i]) ownedLabels.add(labels[i]);
    const restricted = this.semantics.restricted ?? null;
    const closed = this.closedEdits ?? null;
    const sortedLabels = [...ownedLabels].filter(label => label > 0).sort((left, right) => left - right);
    for (const label of sortedLabels) {
      // Complete component must have a free border on all four sides.
      let minX = patchWidth;
      let minY = patchHeight;
      let maxX = -1;
      let maxY = -1;
      const pixels: number[] = [];
      for (let row = 0; row < patchHeight; row++) {
        for (let column = 0; column < patchWidth; column++) {
          const index = row * patchWidth + column;
          if (labels[index] !== label) continue;
          pixels.push(index);
          if (column < minX) minX = column;
          if (column > maxX) maxX = column;
          if (row < minY) minY = row;
          if (row > maxY) maxY = row;
        }
      }
      if (minX === 0 || minY === 0 || maxX >= patchWidth - 1 || maxY >= patchHeight - 1) continue;
      let blocked = false;
      for (const index of pixels) {
        const row = Math.floor(index / patchWidth);
        const column = index - row * patchWidth;
        const absolute = (y0 + row) * width + x0 + column;
        if ((restricted !== null && restricted[absolute]) || (closed !== null && closed[absolute])) {
          blocked = true;
          break;
        }
      }
      if (blocked) continue;
      for (const index of pixels) owned.data[index] = 255;
    }
    const contours = findContours(owned, 'external');
    fillContours(owned, contours, 255);
    return {
      ...feature,
      symbol: owned,
      symbolOrigin: [x0, y0],
      mask: dilate(owned, 3),
      origin: [x0, y0],
    };
  }

  /** `_ink_feature`: footprint of thin black strokes under a circle centre. */
  private inkFeature(point: Pixel, circleRadius: number): Feature | null {
    const ink = this.semantics.point_ink;
    if (ink === null || ink === undefined) return null;
    const width = this.graph.width;
    const height = this.graph.height;
    if (ink.length !== width * height) throw new Error('语义掩膜尺寸与通行图不一致');
    const radius = Math.max(6, rintEven(Math.min(circleRadius, 12)));
    const x = rintEven(point[0]);
    const y = rintEven(point[1]);
    const x0 = Math.max(0, x - radius);
    const y0 = Math.max(0, y - radius);
    const x1 = Math.min(width, x + radius + 1);
    const y1 = Math.min(height, y + radius + 1);
    const patchWidth = x1 - x0;
    const patchHeight = y1 - y0;
    const local = maskOf(patchWidth, patchHeight);
    let hasInk = false;
    for (let row = 0; row < patchHeight; row++) {
      const source = (y0 + row) * width + x0;
      const target = row * patchWidth;
      for (let column = 0; column < patchWidth; column++) {
        if (ink[source + column]) {
          local.data[target + column] = 255;
          hasInk = true;
        }
      }
    }
    if (!hasInk) return null;
    // Only the stroke under the circle counts. A wall a few pixels away must
    // never be bridged just because the checkpoint happens to be near ink.
    const result = connectedComponentsWithStats(local, 8);
    const centreIndex = (rintEven(point[1]) - y0) * patchWidth + (rintEven(point[0]) - x0);
    const label = result.labels[centreIndex];
    if (!label) return null;
    const stat = result.stats[label];
    if (Math.max(stat.width, stat.height) > Math.max(6, radius * 1.5)) return null;
    const stroke = this.labelMask({ labels: result.labels, stats: result.stats, width: patchWidth, height: patchHeight },
      label, 0, 0, patchWidth, patchHeight);
    // The stroke's own anti-aliased edge belongs to the symbol; the wider
    // footprint only bounds the search.
    const symbol = dilate(stroke, 3);
    const grown = dilate(stroke, 5);
    if (!grown.data[centreIndex]) return null;
    return {
      origin: [x0, y0],
      mask: grown,
      limitM: 4,
      method: 'ink_stroke_edge',
      kind: 'point_ink',
      stroke: true,
      strokeLabel: label,
      symbol,
      symbolOrigin: [x0, y0],
    };
  }

  /** `_stroke_component`: true when every blocked pixel lies in the own stroke. */
  strokeComponent(feature: Feature, blocked: Pixel[]): boolean | null {
    const symbol = feature.symbol;
    if (!symbol) return null;
    for (const point of blocked) {
      const x = rintEven(point[0] - feature.symbolOrigin[0]);
      const y = rintEven(point[1] - feature.symbolOrigin[1]);
      if (x < 0 || y < 0 || x >= symbol.width || y >= symbol.height) return null;
      if (!(symbol.data[y * symbol.width + x] > 0)) return null;
    }
    return true;
  }

  /** Port of `AccessResolver.resolve`; throws [`ApproachError`] with the Python text. */
  resolve(control: Control, requiredComponent: number | null = null): InternalResolution {
    const point: Pixel = control.approachPixel
      ? [control.approachPixel[0], control.approachPixel[1]]
      : [control.pixel[0], control.pixel[1]];
    const world = this.mesh.forward(point);
    if (!world.inside) throw new ApproachError('目标超出有效配准范围，需要补充参照');
    const width = this.graph.width;
    const height = this.graph.height;
    const x = rintEven(point[0]);
    const y = rintEven(point[1]);
    if (x < 0 || y < 0 || x >= width || y >= height) throw new ApproachError('到访位置超出原图');
    const closed = this.closedEdits ?? null;
    if (closed !== null && closed[y * width + x]) throw new ApproachError('教师已封闭该位置，不能自动改用地物外沿');
    try {
      const node = this.graph.snap(point, 4);
      if (requiredComponent === null || this.graph.components[node] === requiredComponent) {
        return {
          node,
          pixel: [point[0], point[1]],
          local: [world.local[0], world.local[1]],
          method: control.approachPixel ? 'manual' : 'exact',
          offsetM: 0,
          automatic: false,
        };
      }
    } catch {
      // Fall through to the feature search, exactly like Python.
    }
    if (control.approachPixel) throw new ApproachError('指定到访位置不可达，请核对门洞和到访侧');
    if (this.semantics.restricted !== null && this.semantics.restricted !== undefined && this.semantics.restricted[y * width + x]) {
      throw new ApproachError('圈心位于禁入区域，不能自动移到外部道路');
    }
    const radius = control.radiusPx ?? 15;
    let feature: Feature | null = null;
    for (const kind of ['water', 'vegetation', 'point_ink', 'building'] as AccessSemanticsKind[]) {
      feature = this.feature(kind, point, radius);
      if (feature) break;
    }
    if (feature === null) feature = this.inkFeature(point, radius);
    if (feature === null) throw new ApproachError('圈心处的地物或通行侧不明确，请核对实际到访位置');
    feature = this.completeOutline(feature, radius);
    const limitScale = Math.max(this.scale, 0.1);
    const nearby = this.graph.queryNodesNear(point, feature.limitM / limitScale + 2);
    let nodes = nearby;
    if (!feature.stroke) {
      nodes = nearby.filter(node => !this.inFeature(feature as Feature, [[this.graph.pixels[node * 2], this.graph.pixels[node * 2 + 1]]])[0]);
    }
    const candidates: { offset: number; node: number }[] = [];
    const restricted = this.semantics.restricted ?? null;
    const costs = this.graph.costs;
    for (const node of nodes) {
      const offset = Math.hypot(this.graph.local[node * 2] - world.local[0],
        this.graph.local[node * 2 + 1] - world.local[1]);
      if (offset > feature.limitM) continue;
      const end: Pixel = [this.graph.pixels[node * 2], this.graph.pixels[node * 2 + 1]];
      const pixels = this.graph.segmentPixels(point, end);
      // This ray associates a target with its approach; it is NEVER a route
      // edge. Only the target's bounded silhouette may be crossed.
      const overSymbol = this.overSymbol(feature, pixels);
      let rejected = false;
      for (let index = 0; index < pixels.length; index++) {
        const [pixelX, pixelY] = pixels[index];
        if (pixelX < 0 || pixelY < 0 || pixelX >= width || pixelY >= height) {
          rejected = true;
          break;
        }
        const absolute = pixelY * width + pixelX;
        if (costs[absolute] === 0 && !overSymbol[index]) {
          rejected = true;
          break;
        }
        if ((restricted !== null && restricted[absolute]) || (closed !== null && closed[absolute])) {
          rejected = true;
          break;
        }
      }
      if (rejected) continue;
      candidates.push({ offset, node });
    }
    if (!candidates.length) {
      if (restricted !== null) {
        const reach = Math.max(2, rintEven(feature.limitM / limitScale));
        const x0 = Math.max(0, x - reach);
        const y0 = Math.max(0, y - reach);
        const x1 = Math.min(width, x + reach + 1);
        const y1 = Math.min(height, y + reach + 1);
        let total = 0;
        let forbidden = 0;
        for (let row = y0; row < y1; row++) {
          for (let column = x0; column < x1; column++) {
            total++;
            if (restricted[row * width + column]) forbidden++;
          }
        }
        if (total > 0 && forbidden / total > 0.45) {
          throw new ApproachError('目标地物被禁入区域包围，请核对点位；不能跨禁区自动接入道路');
        }
      }
      throw new ApproachError('地物外沿没有无障碍的近距离到访位置');
    }
    // Local topology, not just global connectivity: the two sides of a wall can
    // connect far away and still be ambiguous at this control.
    const localRadius = Math.max(8,
      Math.ceil(feature.limitM * 2 / limitScale),
      Math.max(feature.symbol.width, feature.symbol.height) + 3);
    const x0 = Math.max(0, x - localRadius);
    const y0 = Math.max(0, y - localRadius);
    const x1 = Math.min(width, x + localRadius + 1);
    const y1 = Math.min(height, y + localRadius + 1);
    const patchWidth = x1 - x0;
    const patchHeight = y1 - y0;
    const localFree = maskOf(patchWidth, patchHeight);
    for (let row = 0; row < patchHeight; row++) {
      const source = (y0 + row) * width + x0;
      const target = row * patchWidth;
      for (let column = 0; column < patchWidth; column++) {
        if (costs[source + column] > 0) localFree.data[target + column] = 255;
      }
    }
    const localLabels = connectedComponents(localFree, 4).labels;
    const sideOf = new Map<number, number>();
    for (const candidate of candidates) {
      const nodeX = rintEven(this.graph.pixels[candidate.node * 2]) - x0;
      const nodeY = rintEven(this.graph.pixels[candidate.node * 2 + 1]) - y0;
      const side = nodeX >= 0 && nodeY >= 0 && nodeX < patchWidth && nodeY < patchHeight
        ? localLabels[nodeY * patchWidth + nodeX]
        : -1;
      sideOf.set(candidate.node, side);
    }
    let active = candidates;
    if (feature.method === 'point_symbol_edge' || feature.method === 'small_water_feature_edge'
      || feature.method === 'compound_point_symbol_edge') {
      // The white holes inside a printed sculpture/fountain are not ground on
      // another approach side; keep the local exterior of the complete symbol.
      const exterior = new Set<number>();
      for (let column = 0; column < patchWidth; column++) {
        exterior.add(localLabels[column]);
        exterior.add(localLabels[(patchHeight - 1) * patchWidth + column]);
      }
      for (let row = 0; row < patchHeight; row++) {
        exterior.add(localLabels[row * patchWidth]);
        exterior.add(localLabels[row * patchWidth + patchWidth - 1]);
      }
      active = active.filter(candidate => exterior.has(sideOf.get(candidate.node) ?? -1));
      if (!active.length) throw new ApproachError('地物符号外部没有可确认的到访位置');
    }
    const groups = new Set<number>();
    for (const candidate of active) groups.add(sideOf.get(candidate.node) ?? -1);
    if (groups.size !== 1) throw new ApproachError('地物附近存在不同通行侧，需要选择实际到访侧');
    if (requiredComponent !== null) {
      active = active.filter(candidate => this.graph.components[candidate.node] === requiredComponent);
    }
    if (!active.length) throw new ApproachError('合法到访侧与起点不连通，请核对通道');
    let best = active[0];
    for (const candidate of active) {
      const difference = candidate.offset - best.offset;
      const better = Math.abs(difference) <= OFFSET_TIE_EPSILON
        ? candidate.node < best.node
        : difference < 0;
      if (better) best = candidate;
    }
    return {
      node: best.node,
      pixel: [this.graph.pixels[best.node * 2], this.graph.pixels[best.node * 2 + 1]],
      local: [this.graph.local[best.node * 2], this.graph.local[best.node * 2 + 1]],
      method: feature.method,
      featureKind: feature.kind,
      offsetM: best.offset,
      automatic: true,
      reviewRequired: true,
      targetPixel: [control.pixel[0], control.pixel[1]],
    };
  }

  /** Resolve one control into a baked-compatible entry (never throws on failure). */
  entry(control: Control, requiredComponent: number | null, frameOrigin: Pixel): ApproachResolution {
    try {
      const resolved = this.resolve(control, requiredComponent);
      const node = resolved.node;
      const local: Pixel = [resolved.local[0], resolved.local[1]];
      const target = this.mesh.forward(control.pixel);
      const entry: ResolvedApproach = {
        ok: true,
        node,
        pixel: [resolved.pixel[0], resolved.pixel[1]],
        local,
        method: resolved.method,
        automatic: resolved.automatic,
        offsetM: resolved.offsetM,
        lnglat: frameLngLat(local, frameOrigin),
        targetLnglat: target.inside ? frameLngLat(target.local, frameOrigin) : null,
        targetPixel: [control.pixel[0], control.pixel[1]],
        component: this.graph.components[node],
      };
      if (resolved.featureKind !== undefined) entry.featureKind = resolved.featureKind;
      if (resolved.reviewRequired !== undefined) entry.reviewRequired = resolved.reviewRequired;
      return entry;
    } catch (error) {
      if (error instanceof ApproachError) return { ok: false, message: error.message };
      throw error;
    }
  }

  /** Resolve with the Addendum A canonical-start retry rule. */
  approach(control: Control, startComponent: number | null, frameOrigin: Pixel): ApproachResolution {
    let resolution = this.entry(control, null, frameOrigin);
    if (resolution.ok && startComponent !== null && resolution.component !== startComponent) {
      resolution = this.entry(control, startComponent, frameOrigin);
    }
    return resolution;
  }
}

/**
 * Resolve legal approach positions for all checkpoint controls.
 *
 * Mirrors `tools/bake_static.py::bake_approaches`: start/finish symbols and
 * rejected controls are skipped; an `ok` approach whose component differs from
 * `options.startComponent` is re-resolved with that component required.
 * Returns the exact `BakedApproach` entry shape (Chinese `message` on failure).
 */
export function resolveApproaches(
  context: AccessContext,
  controls: Control[],
  options: AccessOptions = {},
): Record<string, ApproachResolution> {
  const resolver = new AccessResolver(context);
  const frameOrigin = options.frameOrigin ?? FRAME_ORIGIN;
  const startComponent = options.startComponent ?? null;
  const entries: Record<string, ApproachResolution> = {};
  for (const control of controls) {
    if ((control.role ?? 'control') !== 'control') continue;
    if (control.state === 'rejected') continue;
    entries[control.id] = resolver.approach(control, startComponent, frameOrigin);
  }
  return entries;
}

/** Resolve a single checkpoint control (convenience for editor/UI flows). */
export function resolveApproach(
  context: AccessContext,
  control: Control,
  options: AccessOptions = {},
): ApproachResolution {
  const resolver = new AccessResolver(context);
  return resolver.approach(control, options.startComponent ?? null, options.frameOrigin ?? FRAME_ORIGIN);
}
