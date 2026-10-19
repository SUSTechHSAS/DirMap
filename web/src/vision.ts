/** In-browser map recognition: faithful TypeScript port of the Python
 * `dirmap.vision` / `dirmap.symbols` / `dirmap.overprint` pipeline.
 *
 * Reference implementation: `dirmap/vision.py` (`colour_masks`,
 * `water_obstacles`, `region_features`, `detect_controls`,
 * `purple_restrictions`, `analyze`), `dirmap/symbols.py` (`course_symbol`,
 * `classify_course_symbols`, `course_ink_mask`), `dirmap/overprint.py`
 * (`annotation_mask`, `_line_kernel`, `continue_barriers`, `restore`) and the
 * OCR hook of `dirmap/ocr.py` (implemented in `web/src/ocr.ts`).
 *
 * The module is DOM-free and worker-compatible: it only uses typed arrays and
 * the plain surface of `web/src/cv.ts` (opencv.js). Function names in comments
 * such as `vision.py:42` point at the Python statement being reproduced.
 *
 * Conventions: every mask is a native-pixel `Uint8Array` (0/255 unless the
 * field is documented as 0/1); coordinates are `[x, y]` with x right / y down,
 * matching `web/src/types.ts`. Deviations from Python are documented inline.
 */
import {
  type BgrImage,
  type Contour,
  type MaskImage,
  type RgbaImage,
  approxPolyDP,
  arcLength,
  bgrFromRgba,
  bitwise,
  boxFilterMean,
  circle,
  connectedComponentsWithStats,
  contourArea,
  contourMoments,
  distanceTransformL2,
  dilate,
  erode,
  fillContours,
  findContours,
  gaussianBlur,
  getCv,
  houghCircles,
  houghLinesP,
  hsvFromBgr,
  hsvFromRgba,
  inpaintBgr,
  line,
  loadOpenCV,
  maskNot,
  maskOf,
  minAreaRect,
  morphology,
  morphologyWithMaskKernel,
  polylines,
  resizeByScale,
  threshold,
} from './cv.ts';
import type { Control } from './types.ts';

export type { BgrImage, MaskImage, RgbaImage } from './cv.ts';

// ---------------------------------------------------------------------------
// Public result types (docs/static-architecture.md Addendum C.1)
// ---------------------------------------------------------------------------

/** A building / water body / sports pitch found in the scan (native pixels). */
export interface RegionFeature {
  kind: 'building' | 'water' | 'pitch';
  center: [number, number];
  area: number;
  aspect: number;
  points: [number, number][];
  rectangularity?: number;
}

/** Geometry of the recognized course symbol around a control circle. */
export interface CourseSymbolGeometry {
  role: 'control' | 'start' | 'finish' | 'start_finish';
  ringRadii: number[];
  triangle: number[][] | null;
  triangleCoverage: number[] | null;
  confidence: number;
}

/** Control candidate plus recognition bookkeeping (a superset of `Control`). */
export interface AnalyzedControl extends Control {
  labelSource: 'unread' | 'tesseract' | 'course_symbol';
  labelConfidence?: number;
  labelPixelBox?: [number, number, number, number];
  labelAmbiguous?: boolean;
  ringDensity?: number;
  symbolGeometry?: CourseSymbolGeometry;
}

export interface OcrResult {
  available: boolean;
  recognized?: number;
  preserved?: number;
  method?: string;
}

/** Per-pixel analysis semantics (0/1 masks, native size). */
export interface AnalysisSemantics {
  building: Uint8Array;
  water: Uint8Array;
  point_ink: Uint8Array;
  vegetation: Uint8Array;
  restricted: Uint8Array;
  course_ink: Uint8Array;
  uncertain_ink: Uint8Array;
}

export interface AnalysisStatistics {
  buildingCandidates: number;
  waterCandidates: number;
  controlCandidates: number;
  courseSymbolCandidates: number;
  forbiddenPixelFraction: number;
  restoredCourseInkPixels: number;
  continuedBarrierPixels: number;
  ocr?: OcrResult;
}

/** Initial colour masks (0/255), exactly like Python `result['masks']`. */
export interface ColourMaskSet {
  purple: Uint8Array;
  water: Uint8Array;
  building: Uint8Array;
  green: Uint8Array;
  dark_green: Uint8Array;
  black: Uint8Array;
  olive: Uint8Array;
  course_text: Uint8Array;
}

export interface AnalysisResult {
  width: number;
  height: number;
  controls: AnalyzedControl[];
  features: RegionFeature[];
  /** Palette index per pixel: 0 -> 0 (blocked), 1 -> 1 (walkable), 2 -> 1.5 (vegetation). */
  costsIndex: Uint8Array;
  costsPalette: [number, number, number];
  blocked: Uint8Array;
  semantics: AnalysisSemantics;
  statistics: AnalysisStatistics;
  warnings: string[];
  /** W*H*4 RGBA visualization: blocked -> [221, 66, 67, 125] (see Python save_analysis). */
  terrainRgba: Uint8ClampedArray;
  masks: ColourMaskSet;
}

/** Optional OCR labeler; the default is `labelControls` from `web/src/ocr.ts`. */
export type Labeler = (
  purple: MaskImage,
  controls: AnalyzedControl[],
  textInk: MaskImage,
) => OcrResult | Promise<OcrResult>;

export interface AnalyzeOptions {
  labeler?: Labeler | null;
  progress?: (percent: number, message: string) => void;
}

const WARNINGS = [
  '自动提取结果待核验：细墙、门洞、桥隧层级及紫色禁区可能存在歧义。',
  '控制点编号需核对；圈心和到访位置分别保存。',
];

// ---------------------------------------------------------------------------
// Small numeric helpers matching numpy/Python semantics
// ---------------------------------------------------------------------------

/** `np.rint`: round half to even (JS Math.round rounds half up). */
function npRound(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference > 0.5) return floor + 1;
  if (difference < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** `np.rint(...).astype(int)` for values already (nearly) integral. */
function rintInt(value: number): number {
  return Math.trunc(npRound(value));
}

/** Python `round(value, digits)` (round half to even at the requested precision). */
function roundDigits(value: number, digits: number): number {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  const scaled = value * scale;
  const floor = Math.floor(scaled);
  if (Math.abs(scaled - floor - 0.5) > 1e-9) return Math.round(scaled) / scale;
  return (floor % 2 === 0 ? floor : floor + 1) / scale;
}

/** `np.linspace(start, stop, count)` with endpoint (inclusive). */
function linspace(start: number, stop: number, count: number): Float64Array {
  const values = new Float64Array(count);
  const step = (stop - start) / Math.max(1, count - 1);
  for (let index = 0; index < count; index++) values[index] = start + index * step;
  return values;
}

/** `np.arange(start, stop, step)` (end exclusive). */
function arange(start: number, stop: number, step: number): number[] {
  const values: number[] = [];
  for (let index = 0; ; index++) {
    const value = start + index * step;
    if (step >= 0 ? value >= stop - 1e-12 : value <= stop + 1e-12) break;
    values.push(value);
  }
  return values;
}

/** Python `int(value)` (truncation toward zero). */
const trunc = (value: number): number => Math.trunc(value);

/** numpy median (average of the two middle values for even counts). */
function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** `mask & ~other` on two 0/255 (or 0/1) masks. */
function andNotMask(mask: Uint8Array, other: Uint8Array): Uint8Array {
  const result = new Uint8Array(mask.length);
  for (let index = 0; index < mask.length; index++) result[index] = other[index] ? 0 : mask[index];
  return result;
}

/** `a | b` on two 0/255 masks. */
function orMask(a: Uint8Array, b: Uint8Array): Uint8Array {
  const result = new Uint8Array(a.length);
  for (let index = 0; index < a.length; index++) result[index] = a[index] | b[index];
  return result;
}

/** `a & b` on two 0/255 masks. */
function andMask(a: Uint8Array, b: Uint8Array): Uint8Array {
  const result = new Uint8Array(a.length);
  for (let index = 0; index < a.length; index++) result[index] = a[index] & b[index];
  return result;
}

/** Copy `source` into `target` where `condition` is nonzero (cv2 bitwise_or semantics). */
function orInto(target: Uint8Array, source: Uint8Array, condition?: Uint8Array): void {
  if (condition) {
    for (let index = 0; index < target.length; index++) if (condition[index]) target[index] |= source[index];
    return;
  }
  for (let index = 0; index < target.length; index++) target[index] |= source[index];
}

/** 0/1 view of a 0/255 mask (fresh array, used for `AnalysisSemantics`). */
function binaryMask(mask: Uint8Array): Uint8Array {
  const result = new Uint8Array(mask.length);
  for (let index = 0; index < mask.length; index++) result[index] = mask[index] ? 1 : 0;
  return result;
}

/** Crop with numpy slicing clamping (`mask[y0:y1, x0:x1]`). */
function cropMask(mask: MaskImage, x0: number, y0: number, x1: number, y1: number): MaskImage {
  const left = Math.max(0, Math.min(mask.width, x0));
  const top = Math.max(0, Math.min(mask.height, y0));
  const right = Math.max(left, Math.min(mask.width, x1));
  const bottom = Math.max(top, Math.min(mask.height, y1));
  const width = right - left;
  const height = bottom - top;
  const data = new Uint8Array(width * height);
  for (let row = 0; row < height; row++) {
    const source = (top + row) * mask.width + left;
    data.set(mask.data.subarray(source, source + width), row * width);
  }
  return { data, width, height };
}

const yieldToEventLoop = (): Promise<void> =>
  new Promise(resolve => {
    const timer = (globalThis as { setTimeout?: (fn: () => void, ms: number) => unknown }).setTimeout;
    if (typeof timer === 'function') timer(resolve, 0);
    else resolve();
  });

/** Report progress and give the browser a chance to paint between stages. */
async function stage(report: ((percent: number, message: string) => void) | undefined, percent: number, message: string): Promise<void> {
  report?.(percent, message);
  await yieldToEventLoop();
}

// ---------------------------------------------------------------------------
// Colour masks (vision.py colour_masks)
// ---------------------------------------------------------------------------

interface ColourChannels {
  masks: ColourMaskSet;
  h: Uint8Array;
  s: Uint8Array;
  v: Uint8Array;
}

type PixelSource = 'rgba' | 'bgr';

/**
 * Exact port of `vision.py::colour_masks`.
 *
 * The formulas use signed channel differences (`b - r`, `r - g`, ...), hence the
 * per-pixel integer arithmetic on `int16` values like the numpy reference.
 * HSV comes from OpenCV (8-bit: h 0..179, s/v 0..255), identical in cv2 and
 * opencv.js around the thresholds used here.
 */
function colourMasks(image: RgbaImage | BgrImage, source: PixelSource): ColourChannels {
  const { h, s, v } = source === 'rgba' ? hsvFromRgba(image as RgbaImage) : hsvFromBgr(image as BgrImage);
  const count = image.width * image.height;
  const purple = new Uint8Array(count);
  const water = new Uint8Array(count);
  const building = new Uint8Array(count);
  const green = new Uint8Array(count);
  const darkGreen = new Uint8Array(count);
  const black = new Uint8Array(count);
  const olive = new Uint8Array(count);
  const courseText = new Uint8Array(count);
  const data = image.data;
  const step = source === 'rgba' ? 4 : 3;
  const redOffset = source === 'rgba' ? 0 : 2;
  const blueOffset = source === 'rgba' ? 2 : 0;
  for (let index = 0, offset = 0; index < count; index++, offset += step) {
    const r = data[offset + redOffset];
    const g = data[offset + 1];
    const b = data[offset + blueOffset];
    const hue = h[index];
    const saturation = s[index];
    const value = v[index];
    const isPurple = r - g > 22 && b - g > 18 && r > 85 && b > 75;
    purple[index] = isPurple ? 255 : 0;
    water[index] = !isPurple && b - r > 24 && g - r > 15 && g > 95 && hue > 78 && hue < 111 ? 255 : 0;
    building[index] = saturation < 48 && value > 58 && value < 196 ? 255 : 0;
    const isGreen = g - r > 9 && g - b > 8 && hue >= 32 && hue <= 90;
    green[index] = isGreen ? 255 : 0;
    darkGreen[index] = isGreen && value < 159 && saturation > 80 ? 255 : 0;
    black[index] = value < 108 && saturation < 115 ? 255 : 0;
    olive[index] = hue >= 22 && hue <= 39 && saturation > 62 && value < 216 && value > 90 ? 255 : 0;
    courseText[index] = isPurple && saturation >= 100 ? 255 : 0;
  }
  return {
    masks: { purple, water, building, green, dark_green: darkGreen, black, olive, course_text: courseText },
    h,
    s,
    v,
  };
}

// ---------------------------------------------------------------------------
// Water obstacles (vision.py water_obstacles)
// ---------------------------------------------------------------------------

/** Keep narrow water except very long straight blue north-guide strokes. */
function waterObstacles(mask: MaskImage): MaskImage {
  const { width, height } = mask;
  const native = Math.max(5, trunc(npRound(Math.sqrt(height * width) / 420)) | 1);
  const bodies = morphology(mask, 'open', native);
  const guides = maskOf(width, height);
  const lines = houghLinesP(mask, {
    rho: 1,
    theta: Math.PI / 720,
    threshold: Math.max(60, trunc(height * 0.3)),
    minLineLength: Math.max(150, trunc(height * 0.62)),
    maxLineGap: trunc(height * 0.12),
  });
  for (const segment of lines) {
    if (Math.abs(segment.x2 - segment.x1) < Math.abs(segment.y2 - segment.y1) * 0.12) {
      line(guides, [segment.x1, segment.y1], [segment.x2, segment.y2], native, 255);
    }
  }
  return { data: orMask(bodies.data, andNotMask(mask.data, guides.data)), width, height };
}

// ---------------------------------------------------------------------------
// Region features (vision.py region_features + reference.polygon_properties)
// ---------------------------------------------------------------------------

interface PolygonProperties {
  center: [number, number];
  area: number;
  aspect: number;
  points: [number, number][];
}

/** `reference.py::polygon_properties` on a pixel-space polygon. */
function polygonProperties(points: number[][]): PolygonProperties | null {
  const moments = contourMoments(points);
  const area = Math.abs(moments.m00);
  if (area < 1) return null;
  const rectangle = minAreaRect(points);
  const longSide = Math.max(rectangle.size[0], rectangle.size[1]);
  const shortSide = Math.min(rectangle.size[0], rectangle.size[1]);
  const aspect = longSide / Math.max(1e-6, shortSide);
  return {
    center: [moments.cx, moments.cy],
    area,
    aspect,
    points: points.map(point => [point[0], point[1]] as [number, number]),
  };
}

function regionFeatures(mask: MaskImage, kind: RegionFeature['kind'], imageArea: number, close = 3): RegionFeature[] {
  let cleaned = mask;
  if (kind === 'water') cleaned = waterObstacles(mask);
  cleaned = morphology(cleaned, 'close', close);
  cleaned = morphology(cleaned, 'open', 3);
  const contours = findContours(cleaned, 'external');
  const features: RegionFeature[] = [];
  for (const contour of contours) {
    const area = contourArea(contour);
    if (!(Math.max(90, imageArea * 0.00004) < area && area < imageArea * 0.1)) continue;
    const perimeter = arcLength(contour, true);
    const polygon = approxPolyDP(contour, 0.006 * perimeter, true).points;
    const properties = polygonProperties(polygon);
    if (properties && properties.aspect < 9) features.push({ kind, ...properties });
  }
  return features.sort((a, b) => b.area - a.area);
}

// ---------------------------------------------------------------------------
// Control circles (vision.py detect_controls + symbols.py course_symbol)
// ---------------------------------------------------------------------------

function detectControls(purple: MaskImage): AnalyzedControl[] {
  const { width, height } = purple;
  const scale = Math.min(1, 1800 / Math.max(height, width));
  // Python always resizes (fx=fy=scale); a scale of 1 is an exact identity in
  // cv2.resize, so skip the copy there (`vision.py:106`).
  let mask = scale < 1 ? resizeByScale(purple, scale, scale, getCv().INTER_AREA) : purple;
  mask = threshold(mask, 75);
  const smoothed = gaussianBlur(mask, 3, 0.8);
  const distance = distanceTransformL2(maskNot(mask), 3);
  const rows = mask.height;
  const cols = mask.width;
  const minRadius = Math.max(6, npRound(Math.max(rows, cols) * 0.011));
  const maxRadius = Math.max(minRadius + 2, npRound(Math.max(rows, cols) * 0.028));
  const circles = houghCircles(smoothed, {
    dp: 1,
    minDist: minRadius * 1.6,
    param1: 80,
    param2: 13,
    minRadius,
    maxRadius,
  });
  if (!circles.length) return [];

  const thetaCount = 120;
  const theta = new Float64Array(thetaCount);
  for (let index = 0; index < thetaCount; index++) theta[index] = (index * 2 * Math.PI) / thetaCount;
  const cosTheta = new Float64Array(thetaCount);
  const sinTheta = new Float64Array(thetaCount);
  for (let index = 0; index < thetaCount; index++) {
    cosTheta[index] = Math.cos(theta[index]);
    sinTheta[index] = Math.sin(theta[index]);
  }
  const maskData = mask.data;
  const distanceData = distance.data;
  const ringX = new Int32Array(thetaCount);
  const ringY = new Int32Array(thetaCount);

  interface Candidate {
    pixel: [number, number];
    radiusPx: number;
    confidence: number;
    ringDensity: number;
  }
  const candidates: Candidate[] = [];

  for (const circle of circles) {
    const centerX = circle.x;
    const centerY = circle.y;
    let radius = circle.radius;
    if (centerX - radius < 0 || centerY - radius < 0 || centerX + radius >= cols || centerY + radius >= rows) continue;
    // vision.py:124-135 — refine Hough radius/centre on the original ink.
    let bestCoverage = -1;
    let bestX = centerX;
    let bestY = centerY;
    let bestRadius = radius;
    for (const deltaX of [-4, -2, 0, 2, 4]) {
      for (const deltaY of [-4, -2, 0, 2, 4]) {
        for (const deltaRadius of [-3, -2, -1, 0, 1, 2, 3]) {
          const ringRadius = radius + deltaRadius;
          const originX = centerX + deltaX;
          const originY = centerY + deltaY;
          let valid = true;
          let hits = 0;
          for (let index = 0; index < thetaCount; index++) {
            const x = rintInt(originX + ringRadius * cosTheta[index]);
            const y = rintInt(originY + ringRadius * sinTheta[index]);
            if (x < 0 || y < 0 || x >= cols || y >= rows) {
              valid = false;
              break;
            }
            ringX[index] = x;
            ringY[index] = y;
            if (maskData[y * cols + x] > 0) hits++;
          }
          if (!valid) continue;
          const value = hits / thetaCount;
          if (value > bestCoverage) {
            bestCoverage = value;
            bestX = originX;
            bestY = originY;
            bestRadius = ringRadius;
          }
        }
      }
    }
    if (bestCoverage < 0) continue;
    // vision.py:136-155 — coverage over five radial offsets, quadrant balance,
    // inner density and ring density checks.
    const offsets = [-2, -1, 0, 1, 2];
    const covered = new Uint8Array(thetaCount);
    const sampleMin = new Float32Array(thetaCount);
    sampleMin.fill(Number.POSITIVE_INFINITY);
    for (const delta of offsets) {
      for (let index = 0; index < thetaCount; index++) {
        let x = rintInt(bestX + (bestRadius + delta) * cosTheta[index]);
        let y = rintInt(bestY + (bestRadius + delta) * sinTheta[index]);
        if (x < 0) x = 0;
        if (x > cols - 1) x = cols - 1;
        if (y < 0) y = 0;
        if (y > rows - 1) y = rows - 1;
        const value = distanceData[y * cols + x];
        if (value < sampleMin[index]) sampleMin[index] = value;
      }
    }
    for (let index = 0; index < thetaCount; index++) covered[index] = sampleMin[index] < 1.2 ? 1 : 0;
    let coveredCount = 0;
    for (let index = 0; index < thetaCount; index++) coveredCount += covered[index];
    const coverage = coveredCount / thetaCount;
    const quadrantSize = Math.floor(thetaCount / 4);
    const quadrantMeans: number[] = [];
    for (let quadrant = 0; quadrant < 4; quadrant++) {
      const start = Math.round((quadrant * thetaCount) / 4);
      const end = Math.round(((quadrant + 1) * thetaCount) / 4);
      let sum = 0;
      for (let index = start; index < end; index++) sum += covered[index];
      quadrantMeans.push(end > start ? sum / (end - start) : 0);
    }
    const radiusRound = rintInt(bestRadius);
    const innerRadius = Math.max(2, npRound(bestRadius * 0.55));
    const centerXRound = rintInt(bestX);
    const centerYRound = rintInt(bestY);
    const y0 = Math.max(0, centerYRound - innerRadius);
    const y1 = Math.min(rows, centerYRound + innerRadius);
    const x0 = Math.max(0, centerXRound - innerRadius);
    const x1 = Math.min(cols, centerXRound + innerRadius);
    let innerCount = 0;
    let innerPositive = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        innerCount++;
        if (maskData[y * cols + x] > 0) innerPositive++;
      }
    }
    const density = innerCount ? innerPositive / innerCount : 1;
    if (coverage < 0.94 || Math.min(...quadrantMeans) < 0.8 || density > 0.32) continue;
    let ringHits = 0;
    for (const delta of [-1, 0, 1]) {
      for (let index = 0; index < thetaCount; index++) {
        // Python indexes the raw array (numpy wraps negative indices); use the
        // same wrap so a circle one pixel inside the border matches Python.
        let x = rintInt(bestX + (bestRadius + delta) * cosTheta[index]);
        let y = rintInt(bestY + (bestRadius + delta) * sinTheta[index]);
        if (x < 0) x += cols;
        if (y < 0) y += rows;
        if (x >= cols || y >= rows) continue;
        if (maskData[y * cols + x] > 0) ringHits++;
      }
    }
    const ringDensity = ringHits / (3 * thetaCount);
    if (ringDensity - density < 0.25) continue;
    void radiusRound;
    candidates.push({
      pixel: [bestX / scale, bestY / scale],
      radiusPx: bestRadius / scale,
      confidence: roundDigits(coverage, 3),
      ringDensity: roundDigits(ringDensity, 3),
    });
  }

  // vision.py:161-175 — nested rings at the finish are one checkpoint; sort by
  // (row, column) afterwards and assign stable ids.
  const accepted: Candidate[] = [];
  for (const candidate of [...candidates].sort((a, b) => b.confidence - a.confidence)) {
    const nested = accepted.some(other => {
      const dx = candidate.pixel[0] - other.pixel[0];
      const dy = candidate.pixel[1] - other.pixel[1];
      const limit = Math.max(candidate.radiusPx, other.radiusPx) * 0.8;
      return Math.sqrt(dx * dx + dy * dy) < limit;
    });
    if (!nested) accepted.push(candidate);
  }
  accepted.sort((a, b) => a.pixel[1] - b.pixel[1] || a.pixel[0] - b.pixel[0]);
  return accepted.map((candidate, index) => ({
    id: `c${index + 1}`,
    label: `待核${index + 1}`,
    pixel: candidate.pixel,
    radiusPx: candidate.radiusPx,
    required: true,
    state: 'candidate' as const,
    labelSource: 'unread' as const,
    confidence: candidate.confidence,
    ringDensity: candidate.ringDensity,
    serviceBaseSec: 5,
    servicePerPersonSec: 1,
  }));
}

/** Render one control's symbol geometry (triangles/rings) into `strokes`. */
function courseInkMask(purple: MaskImage, controls: AnalyzedControl[]): Uint8Array {
  const strokes = maskOf(purple.width, purple.height);
  for (const control of controls) {
    const center: [number, number] = [rintInt(control.pixel[0]), rintInt(control.pixel[1])];
    const radius = control.radiusPx;
    const geometry = control.symbolGeometry;
    const rings = [radius, ...(geometry?.ringRadii ?? [])];
    for (const ring of rings) circle(strokes, center, npRound(ring), Math.max(3, npRound(radius * 0.15)), 255);
    if (geometry?.triangle) {
      const triangle = geometry.triangle.map(point => [rintInt(point[0]), rintInt(point[1])]);
      polylines(strokes, triangle, true, Math.max(3, npRound(radius * 0.16)), 255);
    }
  }
  const result = new Uint8Array(purple.data.length);
  for (let index = 0; index < result.length; index++) result[index] = strokes.data[index] & purple.data[index];
  return result;
}

// ---------------------------------------------------------------------------
// Course symbols (symbols.py course_symbol / classify_course_symbols)
// ---------------------------------------------------------------------------

function courseSymbol(purple: MaskImage, control: AnalyzedControl): CourseSymbolGeometry {
  const centerX = control.pixel[0];
  const centerY = control.pixel[1];
  const radius = control.radiusPx;
  const { width, height } = purple;
  const pad = Math.ceil(radius * 2.1);
  const originX = Math.max(0, Math.floor(centerX - pad));
  const originY = Math.max(0, Math.floor(centerY - pad));
  const endX = Math.min(width, Math.ceil(centerX + pad));
  const endY = Math.min(height, Math.ceil(centerY + pad));
  const crop = cropMask(purple, originX, originY, endX, endY);
  const pointX = centerX - originX;
  const pointY = centerY - originY;
  const cropWidth = crop.width;
  const cropHeight = crop.height;
  const angles = linspace(0, 2 * Math.PI, 180);
  const radial = linspace(0.5, 1.65, 93).map(value => value * radius);
  // symbols.py:16-21 — radial ink profile around the detected circle.
  const profile = new Float64Array(radial.length);
  for (let ringIndex = 0; ringIndex < radial.length; ringIndex++) {
    let hits = 0;
    for (let angleIndex = 0; angleIndex < angles.length; angleIndex++) {
      let x = rintInt(pointX + radial[ringIndex] * Math.cos(angles[angleIndex]));
      let y = rintInt(pointY + radial[ringIndex] * Math.sin(angles[angleIndex]));
      if (x < 0) x = 0;
      if (x > cropWidth - 1) x = cropWidth - 1;
      if (y < 0) y = 0;
      if (y > cropHeight - 1) y = cropHeight - 1;
      if (crop.data[y * cropWidth + x] > 0) hits++;
    }
    profile[ringIndex] = hits / angles.length;
  }
  // symbols.py:23-29 — separate connected peaks (a thick circle has two edges).
  const radii: number[] = [];
  let groupStart = -1;
  for (let index = 0; index <= profile.length; index++) {
    const active = index < profile.length && profile[index] >= 0.65;
    if (active && groupStart < 0) groupStart = index;
    if (!active && groupStart >= 0) {
      let best = groupStart;
      for (let inner = groupStart; inner < index; inner++) if (profile[inner] > profile[best]) best = inner;
      radii.push(radial[best]);
      groupStart = -1;
    }
  }
  let double = false;
  for (const left of radii) for (const right of radii) if (Math.abs(left - right) > radius * 0.16) double = true;

  const distance = distanceTransformL2(maskNot(crop), 3);
  const distanceData = distance.data;
  let innerCount = 0;
  let innerPositive = 0;
  const innerRadius = radius * 0.84;
  const innerLimit = innerRadius * innerRadius;
  for (let y = 0; y < cropHeight; y++) {
    const dy = y - pointY;
    for (let x = 0; x < cropWidth; x++) {
      const dx = x - pointX;
      if (dx * dx + dy * dy >= innerLimit) continue;
      innerCount++;
      if (crop.data[y * cropWidth + x] > 0) innerPositive++;
    }
  }
  const innerDensity = innerCount ? innerPositive / innerCount : 0;

  interface TriangleFit {
    score: number;
    vertices: number[][];
    coverage: number[];
  }
  let best: TriangleFit | null = null;
  if (innerDensity > 0.035 && innerDensity < 0.8) {
    const t = linspace(0.14, 0.86, 27);
    const sampleAngleCount = angles.length;
    const samplesX = new Float64Array(3 * t.length);
    const samplesY = new Float64Array(3 * t.length);
    const offsetSamplesX = new Float64Array(3 * t.length);
    const offsetSamplesY = new Float64Array(3 * t.length);
    const hitSide = (xs: Float64Array, ys: Float64Array, tolerance: number): Float64Array => {
      const coverage = new Float64Array(3);
      for (let side = 0; side < 3; side++) {
        let hits = 0;
        for (let sample = 0; sample < t.length; sample++) {
          const x = rintInt(xs[side * t.length + sample]);
          const y = rintInt(ys[side * t.length + sample]);
          if (x < 0 || y < 0 || x >= cropWidth || y >= cropHeight) continue;
          if (distanceData[y * cropWidth + x] <= tolerance) hits++;
        }
        coverage[side] = hits / t.length;
      }
      return coverage;
    };
    for (const angleDegrees of arange(-90, 30, 6)) {
      const angle = (angleDegrees * Math.PI) / 180;
      const unit: [number, number][] = [0, 1, 2].map(step => {
        const vertexAngle = angle + (step * 2 * Math.PI) / 3;
        return [Math.cos(vertexAngle), Math.sin(vertexAngle)];
      });
      for (const relativeRadius of arange(0.85, 1.71, 0.05)) {
        for (const [offsetX, offsetY] of [
          [0, 0],
          [-0.07, 0],
          [0.07, 0],
          [0, -0.07],
          [0, 0.07],
        ]) {
          const vertices: [number, number][] = unit.map(([ux, uy]) => [
            pointX + offsetX * radius + ux * radius * relativeRadius,
            pointY + offsetY * radius + uy * radius * relativeRadius,
          ]);
          for (let side = 0; side < 3; side++) {
            const next = vertices[(side + 1) % 3];
            const current = vertices[side];
            const vectorX = next[0] - current[0];
            const vectorY = next[1] - current[1];
            const normalLength = Math.hypot(vectorX, vectorY) || 1;
            const normalX = -vectorY / normalLength;
            const normalY = vectorX / normalLength;
            const offset = Math.max(4, radius * 0.22);
            for (let sample = 0; sample < t.length; sample++) {
              const fraction = t[sample];
              const x = current[0] + vectorX * fraction;
              const y = current[1] + vectorY * fraction;
              samplesX[side * t.length + sample] = x;
              samplesY[side * t.length + sample] = y;
              offsetSamplesX[side * t.length + sample] = x + normalX * offset;
              offsetSamplesY[side * t.length + sample] = y + normalY * offset;
            }
          }
          const coverage = hitSide(samplesX, samplesY, Math.max(1, radius * 0.045));
          if (Math.min(coverage[0], coverage[1], coverage[2]) < 0.8) continue;
          const positiveBackground = hitSide(offsetSamplesX, offsetSamplesY, 0.5);
          // symbols.py:60 samples `sides - offset` by reusing the same normal
          // array with a negative multiplier; recompute both offsets exactly.
          const negativeX = new Float64Array(3 * t.length);
          const negativeY = new Float64Array(3 * t.length);
          for (let side = 0; side < 3; side++) {
            const next = vertices[(side + 1) % 3];
            const current = vertices[side];
            const vectorX = next[0] - current[0];
            const vectorY = next[1] - current[1];
            const normalLength = Math.hypot(vectorX, vectorY) || 1;
            const normalX = -vectorY / normalLength;
            const normalY = vectorX / normalLength;
            const offset = Math.max(4, radius * 0.22);
            for (let sample = 0; sample < t.length; sample++) {
              const fraction = t[sample];
              negativeX[side * t.length + sample] = current[0] + vectorX * fraction - normalX * offset;
              negativeY[side * t.length + sample] = current[1] + vectorY * fraction - normalY * offset;
            }
          }
          const negativeBackground = hitSide(negativeX, negativeY, 0.5);
          const background = [
            (positiveBackground[0] + negativeBackground[0]) / 2,
            (positiveBackground[1] + negativeBackground[1]) / 2,
            (positiveBackground[2] + negativeBackground[2]) / 2,
          ];
          let minContrast = Number.POSITIVE_INFINITY;
          for (let side = 0; side < 3; side++) minContrast = Math.min(minContrast, coverage[side] - background[side]);
          if (minContrast < 0.24) continue;
          const score = (coverage[0] + coverage[1] + coverage[2]) / 3 - 0.4 * ((background[0] + background[1] + background[2]) / 3);
          if (!best || score > best.score) {
            best = {
              score,
              vertices: vertices.map(vertex => [vertex[0] + originX, vertex[1] + originY]),
              coverage: [coverage[0], coverage[1], coverage[2]],
            };
          }
          void sampleAngleCount;
        }
      }
    }
  }
  if (best && !double) {
    // symbols.py:67-74 — a compound symbol's triangle strokes can bridge the
    // valley between the two radial peaks; the Hough ring is still evidence.
    let nearest = 0;
    for (let index = 1; index < radial.length; index++) {
      if (Math.abs(radial[index] - radius) < Math.abs(radial[nearest] - radius)) nearest = index;
    }
    const outerSupport = profile[nearest];
    if (outerSupport >= 0.65 && radii.some(value => Math.abs(value - radius) > radius * 0.2)) {
      radii.push(radius);
      double = true;
    }
  }
  const role: CourseSymbolGeometry['role'] = best && double ? 'start_finish' : best ? 'start' : double ? 'finish' : 'control';
  let profileMax = 0;
  for (const value of profile) if (value > profileMax) profileMax = value;
  return {
    role,
    ringRadii: radii.length ? radii : [radius],
    triangle: best ? best.vertices : null,
    triangleCoverage: best ? best.coverage : null,
    confidence: roundDigits(best ? best.score : profileMax, 3),
  };
}

function classifyCourseSymbols(purple: MaskImage, controls: AnalyzedControl[]): void {
  const labels: Record<string, string> = { start: '起点', finish: '终点', start_finish: '起终点' };
  for (const control of controls) {
    const symbol = courseSymbol(purple, control);
    control.role = symbol.role;
    control.symbolGeometry = symbol;
    if (symbol.role !== 'control') {
      control.label = labels[symbol.role];
      control.labelSource = 'course_symbol';
      control.required = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Overprint handling (overprint.py)
// ---------------------------------------------------------------------------

/** Confirmed circles/triangles, OCR text and supported circle connections. */
function annotationMask(purple: MaskImage, controls: AnalyzedControl[], textInk?: Uint8Array): MaskImage {
  const strokes = maskOf(purple.width, purple.height);
  for (const control of controls) {
    const radius = control.radiusPx;
    const center: [number, number] = [rintInt(control.pixel[0]), rintInt(control.pixel[1])];
    const geometry = control.symbolGeometry;
    for (const ring of [radius, ...(geometry?.ringRadii ?? [])]) {
      circle(strokes, center, npRound(ring), Math.max(5, npRound(radius * 0.4)), 255);
    }
    if (geometry?.triangle) {
      polylines(strokes, geometry.triangle.map(point => [rintInt(point[0]), rintInt(point[1])]), true, Math.max(5, npRound(radius * 0.35)), 255);
    }
    const box = control.labelPixelBox;
    if (box) {
      const x0 = Math.max(0, Math.floor(box[0] - 2));
      const y0 = Math.max(0, Math.floor(box[1] - 2));
      const x1 = Math.min(purple.width, Math.ceil(box[2] + 2));
      const y1 = Math.min(purple.height, Math.ceil(box[3] + 2));
      if (x1 > x0 && y1 > y0) {
        // Bold text and its anti-aliasing may cover pale restriction hatching:
        // only OR the dilated text ink, never erase the whole box (overprint.py:23-31).
        const ink = textInk ?? purple.data;
        const crop = cropMask({ data: ink, width: purple.width, height: purple.height }, x0, y0, x1, y1);
        const dilated = dilate(crop, 3);
        for (let row = 0; row < dilated.height; row++) {
          for (let column = 0; column < dilated.width; column++) {
            const target = (y0 + row) * purple.width + x0 + column;
            strokes.data[target] |= dilated.data[row * dilated.width + column];
          }
        }
      }
    }
  }
  const distance = distanceTransformL2(maskNot(purple), 3);
  for (let leftIndex = 0; leftIndex < controls.length; leftIndex++) {
    for (let rightIndex = leftIndex + 1; rightIndex < controls.length; rightIndex++) {
      const left = controls[leftIndex];
      const right = controls[rightIndex];
      const ax = left.pixel[0];
      const ay = left.pixel[1];
      const bx = right.pixel[0];
      const by = right.pixel[1];
      const length = Math.hypot(bx - ax, by - ay);
      const radius = Math.min(left.radiusPx, right.radiusPx);
      if (length < radius * 5) continue;
      const start = (left.radiusPx * 1.2) / length;
      const stop = 1 - (right.radiusPx * 1.2) / length;
      if (start >= stop) continue;
      const sampleCount = Math.max(20, Math.round(length));
      const samples: number[][] = [];
      let outside = false;
      for (let sample = 0; sample < sampleCount; sample++) {
        const fraction = start + (sample * (stop - start)) / (sampleCount - 1);
        const x = rintInt(ax + fraction * (bx - ax));
        const y = rintInt(ay + fraction * (by - ay));
        if (x < 0 || y < 0 || x >= purple.width || y >= purple.height) {
          outside = true;
          break;
        }
        samples.push([x, y]);
      }
      if (outside) continue;
      let hitCount = 0;
      const hit = new Uint8Array(samples.length);
      const tolerance = Math.max(1.5, radius * 0.1);
      for (let index = 0; index < samples.length; index++) {
        const value = distance.data[samples[index][1] * purple.width + samples[index][0]];
        if (value <= tolerance) {
          hit[index] = 1;
          hitCount++;
        }
      }
      if (hitCount / samples.length < 0.9) continue;
      let minimumPart = 1;
      for (let part = 0; part < 4; part++) {
        const from = Math.round((part * samples.length) / 4);
        const to = Math.round(((part + 1) * samples.length) / 4);
        if (to <= from) continue;
        let sum = 0;
        for (let index = from; index < to; index++) sum += hit[index];
        minimumPart = Math.min(minimumPart, sum / (to - from));
      }
      if (minimumPart >= 0.8) {
        line(strokes, [rintInt(ax), rintInt(ay)], [rintInt(bx), rintInt(by)], Math.max(5, npRound(radius * 0.45)), 255);
      }
    }
  }
  const result = new Uint8Array(purple.data.length);
  for (let index = 0; index < result.length; index++) result[index] = strokes.data[index] & purple.data[index];
  return { data: result, width: purple.width, height: purple.height };
}

/**
 * Hatching has sustained local density; a circle or course line does not.
 * Course-circle interiors are deliberately not erased (overprint.py:178-196).
 */
function purpleRestrictions(purple: MaskImage, controls: AnalyzedControl[], annotations?: MaskImage): MaskImage {
  const annotationsData = annotations?.data ?? courseInkMask(purple, controls);
  const ink = andNotMask(purple.data, annotationsData);
  const radius = Math.max(7, npRound(Math.max(purple.height, purple.width) * 0.006));
  const density = boxFilterMean({ data: ink, width: purple.width, height: purple.height }, 2 * radius + 1);
  const regions = maskOf(purple.width, purple.height);
  for (let index = 0; index < regions.data.length; index++) {
    regions.data[index] = density.data[index] > 0.32 && ink[index] > 0 ? 255 : 0;
  }
  const closed = morphology(regions, 'close', radius);
  const contours = findContours(closed, 'external');
  const result = maskOf(purple.width, purple.height);
  for (const contour of contours) {
    if (contourArea(contour) > radius * radius * 3) fillContours(result, [contour], 255);
  }
  return result;
}

/** `_line_kernel`: 1-pixel anti-aliased line through the centre of a square kernel. */
function lineKernel(size: number, angle: number): MaskImage {
  const kernel = maskOf(size, size);
  const half = Math.floor(size / 2);
  const vectorX = Math.cos(angle) * half;
  const vectorY = Math.sin(angle) * half;
  line(kernel, [rintInt(half - vectorX), rintInt(half - vectorY)], [rintInt(half + vectorX), rintInt(half + vectorY)], 1, 1);
  return kernel;
}

/**
 * Continue supported linear fragments only inside hidden pixels (overprint.py:59-78).
 * Directional support prevents joining two parallel road borders into a cross-road wall.
 */
function continueBarriers(black: MaskImage, unknown: MaskImage, maximumGap: number): MaskImage {
  const visibleData = andNotMask(black.data, unknown.data);
  const visible = { data: visibleData, width: black.width, height: black.height };
  const components = connectedComponentsWithStats(visible, 8);
  const linear = new Uint8Array(components.count);
  for (let label = 1; label < components.count; label++) {
    const stat = components.stats[label];
    const longSide = Math.max(stat.width, stat.height);
    const shortSide = Math.min(stat.width, stat.height);
    linear[label] = longSide >= 20 || longSide >= 3 * Math.max(1, shortSide) ? 1 : 0;
  }
  const strokes = maskOf(black.width, black.height);
  for (let index = 0; index < strokes.data.length; index++) strokes.data[index] = linear[components.labels[index]] ? 255 : 0;
  const grown = dilate(strokes, 3);
  const repaired = maskOf(black.width, black.height);
  const size = trunc(maximumGap) | 1;
  for (const angle of arange(0, Math.PI, Math.PI / 12)) {
    const direction = lineKernel(7, angle);
    const supported = morphologyWithMaskKernel(grown, 'open', direction);
    const joined = morphologyWithMaskKernel(supported, 'close', lineKernel(size, angle));
    for (let index = 0; index < repaired.data.length; index++) {
      if (unknown.data[index]) repaired.data[index] |= joined.data[index];
    }
  }
  return repaired;
}

/** Inpaint the hidden overprint and continue black barriers (overprint.py:81-99). */
function restore(
  image: BgrImage,
  annotations: MaskImage,
  restrictions: MaskImage,
  black: MaskImage,
  controls: AnalyzedControl[],
): { recovered: BgrImage; hidden: MaskImage; barriers: MaskImage } {
  const hsv = hsvFromBgr(image);
  const nearby = dilate(annotations, 5);
  const unknown = maskOf(image.width, image.height);
  let anyUnknown = false;
  for (let index = 0; index < unknown.data.length; index++) {
    const isNearby = nearby.data[index] !== 0;
    const isFringe = isNearby && hsv.h[index] >= 111 && hsv.h[index] <= 170 && hsv.s[index] >= 30;
    if ((annotations.data[index] !== 0 || isFringe) && restrictions.data[index] === 0) {
      unknown.data[index] = 255;
      anyUnknown = true;
    }
  }
  if (!anyUnknown) {
    return { recovered: image, hidden: unknown, barriers: maskOf(image.width, image.height) };
  }
  // Navier-Stokes inpainting follows the visible isophotes instead of averaging
  // across a covered passage (overprint.py:95).
  const recovered = inpaintBgr(image, unknown, 1);
  const radii = controls.map(control => control.radiusPx);
  const gap = Math.max(9, Math.min(49, npRound((radii.length ? median(radii) : 12) * 1.2)));
  const barriers = continueBarriers(black, unknown, gap);
  return { recovered, hidden: unknown, barriers };
}

// ---------------------------------------------------------------------------
// analyze() — main pipeline (vision.py analyze)
// ---------------------------------------------------------------------------

/**
 * Run the full recognition pipeline on an RGBA raster.
 *
 * Stages run in the same order as `dirmap.vision.analyze()`; between stages the
 * promise yields so the browser can paint progress. With `labeler: null` no OCR
 * runs (Node goldens); without it the default `labelControls` (tesseract.js) is
 * used and failures degrade to `待核N` labels.
 */
export async function analyze(image: RgbaImage, options: AnalyzeOptions = {}): Promise<AnalysisResult> {
  const report = options.progress;
  const width = image.width;
  const height = image.height;
  const imageArea = width * height;

  // The pipeline is async anyway: loading the WASM module here makes `analyze`
  // self-contained (Node tests / processing.ts do not need a separate await).
  await loadOpenCV();
  await stage(report, 2, '提取颜色掩膜');
  const initial = colourMasks(image, 'rgba');
  const masks = initial.masks;

  await stage(report, 8, '识别建筑物与水系');
  const features: RegionFeature[] = [];
  features.push(...regionFeatures({ data: masks.building, width, height }, 'building', imageArea, 5));
  features.push(...regionFeatures({ data: masks.water, width, height }, 'water', imageArea, 7));

  await stage(report, 16, '识别运动场地');
  const pitchInk = maskOf(width, height);
  for (let index = 0; index < pitchInk.data.length; index++) {
    const hue = initial.h[index];
    const saturation = initial.s[index];
    const value = initial.v[index];
    pitchInk.data[index] = hue >= 12 && hue <= 34 && saturation >= 65 && saturation <= 180 && value > 200 ? 255 : 0;
  }
  for (const feature of regionFeatures(pitchInk, 'pitch', imageArea, 5)) {
    const rectangle = minAreaRect(feature.points);
    const rectangularity = feature.area / Math.max(1, rectangle.size[0] * rectangle.size[1]);
    if (rectangularity > 0.72 && feature.area > imageArea * 0.0004 && feature.aspect < 3.5) {
      feature.rectangularity = rectangularity;
      features.push(feature);
    }
  }
  // The initial HSV planes are only needed for the pitch test; release them
  // before the terrain stage so a 6.6 MP sheet keeps a smaller live set.
  initial.h = new Uint8Array(0);
  initial.s = new Uint8Array(0);
  initial.v = new Uint8Array(0);

  await stage(report, 24, '检测控制点圆圈');
  const purpleMask: MaskImage = { data: masks.purple, width, height };
  const controls = detectControls(purpleMask);

  await stage(report, 32, '识别起终点符号');
  classifyCourseSymbols(purpleMask, controls);

  let ocr: OcrResult | null = null;
  if (options.labeler !== null && controls.length) {
    await stage(report, 38, '识别控制点编号');
    try {
      const labeler = options.labeler ?? (await defaultLabeler());
      if (labeler) {
        const textInk: MaskImage = { data: masks.course_text, width, height };
        ocr = await labeler(purpleMask, controls, textInk);
      }
    } catch {
      // OCR is always optional: keep `待核N` labels and report it unavailable.
      ocr = { available: false, recognized: 0, method: 'tesseract_global_and_digit_crops' };
    }
  }

  await stage(report, 46, '分离课程标注与禁区');
  const annotations = annotationMask(purpleMask, controls, masks.course_text);
  const restrictions = purpleRestrictions(purpleMask, controls, annotations);

  await stage(report, 56, '修复被遮挡地物');
  const blackMask: MaskImage = { data: masks.black, width, height };
  const restored = restore(bgrFromRgba(image), annotations, restrictions, blackMask, controls);
  const recovered = restored.recovered;
  const hidden = restored.hidden;
  const continuedBarriers = restored.barriers;

  await stage(report, 70, '重建地形掩膜');
  const recoveredMasks = colourMasks(recovered, 'bgr');
  // `recovered` is only an intermediate (Python keeps it for the roof rule but
  // the roof rule only needs the S/V planes), so the ~20 MB BGR copy and the
  // unused H plane can be released: 6.6 MP sheets otherwise peak much higher.
  recovered.data = new Uint8Array(0);
  recoveredMasks.h = new Uint8Array(0);
  const terrain = recoveredMasks.masks;
  for (const kind of ['water', 'building', 'dark_green', 'olive', 'black'] as const) {
    const observed = masks[kind];
    orInto(terrain[kind], observed, hidden.data);
  }
  // `terrain_masks['black'] |= continued_barriers` (vision.py:223).
  orInto(terrain.black, continuedBarriers.data);

  // Roof fill rule: drop building pixels whose 3x3 neighbourhood is not all
  // grey (anti-aliasing beside thin roofs), then keep hard one-pixel roof edges.
  // vision.py:227-240.
  const support = erode({ data: terrain.building, width, height }, 3);
  let buildings = dilate(support, 3);
  const terrainS = recoveredMasks.s;
  const terrainV = recoveredMasks.v;
  const roofEdge = maskOf(width, height);
  const dilatedBuildings = dilate(buildings, 3);
  for (let index = 0; index < roofEdge.data.length; index++) {
    roofEdge.data[index] = terrainV[index] < 165 && terrainS[index] < 150 && dilatedBuildings.data[index] > 0 ? 255 : 0;
  }
  for (let index = 0; index < buildings.data.length; index++) {
    if (roofEdge.data[index]) buildings.data[index] = 255;
  }

  const waterRegions = waterObstacles({ data: terrain.water, width, height });

  await stage(report, 84, '生成禁行与通行代价');
  let hard = orMask(buildings.data, waterRegions.data);
  for (const kind of ['dark_green', 'olive', 'black'] as const) hard = orMask(hard, terrain[kind]);
  hard = orMask(hard, restrictions.data);
  const unknownInk = maskOf(width, height);
  for (let index = 0; index < unknownInk.data.length; index++) {
    const annotatedOrHidden = annotations.data[index] !== 0 || hidden.data[index] !== 0;
    unknownInk.data[index] = !annotatedOrHidden ? masks.purple[index] : 0;
  }
  hard = orMask(hard, unknownInk.data);

  // Tiny isolated compression flecks are not walls; >= 4 px components stay.
  // vision.py:250-252.
  const components = connectedComponentsWithStats({ data: hard, width, height }, 8);
  const keep = new Uint8Array(components.count);
  for (let label = 1; label < components.count; label++) keep[label] = components.stats[label].area >= 4 ? 1 : 0;
  const filtered = maskOf(width, height);
  for (let index = 0; index < filtered.data.length; index++) filtered.data[index] = keep[components.labels[index]] ? 255 : 0;
  hard = filtered.data;

  // Costs: palette index 0 -> 0 (blocked), 1 -> 1 (walkable), 2 -> 1.5 (green).
  // vision.py:253-255: ones, then green = 1.5, then hard = 0.
  const costsIndex = new Uint8Array(width * height);
  const greenMask = terrain.green;
  for (let index = 0; index < costsIndex.length; index++) {
    costsIndex[index] = hard[index] > 0 ? 0 : greenMask[index] > 0 ? 2 : 1;
  }

  await stage(report, 94, '汇总语义与统计');
  let forbidden = 0;
  let restoredCourseInkPixels = 0;
  let continuedBarrierPixels = 0;
  for (let index = 0; index < hard.length; index++) {
    if (hard[index] > 0) forbidden++;
    if (hidden.data[index] > 0) restoredCourseInkPixels++;
    if (continuedBarriers.data[index] > 0) continuedBarrierPixels++;
  }
  const statistics: AnalysisStatistics = {
    buildingCandidates: features.filter(feature => feature.kind === 'building').length,
    waterCandidates: features.filter(feature => feature.kind === 'water').length,
    controlCandidates: controls.filter(control => control.role === 'control').length,
    courseSymbolCandidates: controls.length,
    forbiddenPixelFraction: roundDigits(forbidden / hard.length, 4),
    restoredCourseInkPixels,
    continuedBarrierPixels,
  };
  if (ocr) statistics.ocr = ocr;

  const semantics: AnalysisSemantics = {
    building: binaryMask(buildings.data),
    water: binaryMask(waterRegions.data),
    point_ink: binaryMask(terrain.black),
    vegetation: binaryMask(terrain.dark_green),
    restricted: new Uint8Array(width * height),
    course_ink: binaryMask(hidden.data),
    uncertain_ink: binaryMask(unknownInk.data),
  };
  for (let index = 0; index < semantics.restricted.length; index++) {
    semantics.restricted[index] = restrictions.data[index] > 0 || terrain.olive[index] > 0 ? 1 : 0;
  }

  const terrainRgba = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < hard.length; index++) {
    if (!hard[index]) continue;
    const offset = index * 4;
    terrainRgba[offset] = 221;
    terrainRgba[offset + 1] = 66;
    terrainRgba[offset + 2] = 67;
    terrainRgba[offset + 3] = 125;
  }

  report?.(100, '识别完成');
  return {
    width,
    height,
    controls,
    features,
    costsIndex,
    costsPalette: [0, 1, 1.5],
    blocked: hard,
    semantics,
    statistics,
    warnings: [...WARNINGS],
    terrainRgba,
    masks,
  };
}

/** Lazily import the tesseract.js labeler so OCR failures cannot break analysis. */
async function defaultLabeler(): Promise<Labeler | null> {
  try {
    const module = await import('./ocr.ts');
    return module.labelControls;
  } catch {
    return null;
  }
}
