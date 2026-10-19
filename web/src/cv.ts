/** opencv.js (WASM) loader and a plain-JavaScript computer-vision surface.
 *
 * The browser build never bundles OpenCV: `loadOpenCV()` lazily appends the
 * classic script `${BASE}vendor/opencv.js` (copied from
 * `node_modules/@techstark/opencv-js` by `tools/vendor-assets.mjs`) and awaits
 * the module. Node unit tests (`node --experimental-strip-types --test`) import
 * the npm package directly instead, which resolves to the same WASM build.
 *
 * All functions below are **synchronous once the module is loaded** (call
 * `await loadOpenCV()` first) and exchange plain JavaScript typed arrays:
 * masks are `{data: Uint8Array, width, height}` with 0/255 convention, images
 * are RGBA or BGR byte arrays. Internally every wrapper allocates its Mats,
 * copies the result out and deletes every Mat in a `finally` block, so callers
 * never see or leak OpenCV heap objects and never need to call `.delete()`.
 *
 * The handful of `mat*`/`*Mat` helpers exposed for performance-critical
 * in-module use return caller-owned Mats (`withMats` is a convenient scope);
 * the plain-array API above is the supported contract for other modules.
 *
 * Conventions (identical to cv2):
 * - mask foreground is any non-zero byte; binary helpers preserve 0/255
 * - `findContours` returns OpenCV `CHAIN_APPROX_SIMPLE` integer pixel points
 *   in traversal order, without a repeated closing point
 * - `distanceTransformL2` returns the distance of every pixel to the nearest
 *   **zero** pixel of the supplied mask (i.e. plain `cv2.distanceTransform`)
 * - `boxFilterMean` returns the local mean of `mask > 0` as float32
 */
export type CvModule = any;

export interface RgbaImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Interleaved blue/green/red bytes (OpenCV 8UC3 channel order). */
export interface BgrImage {
  data: Uint8Array;
  width: number;
  height: number;
}

/** Single-channel 0/255 binary mask (any non-zero byte is foreground). */
export interface MaskImage {
  data: Uint8Array;
  width: number;
  height: number;
}

/** Single-channel float32 image (distance transforms, local densities, ...). */
export interface FloatImage {
  data: Float32Array;
  width: number;
  height: number;
}

/** One OpenCV contour: integer pixel coordinates in `(x, y)` order. */
export interface Contour {
  points: number[][];
}

/** `cv2.findContours` hierarchy node (`RETR_CCOMP`: parent >= 0 means hole). */
export interface ContourNode {
  next: number;
  prev: number;
  firstChild: number;
  parent: number;
}

export interface ContourHierarchy {
  contours: Contour[];
  hierarchy: ContourNode[];
}

export interface ComponentStats {
  x: number;
  y: number;
  width: number;
  height: number;
  area: number;
}

export interface ComponentsResult {
  width: number;
  height: number;
  /** Number of labels including the background label 0 (cv2 return value). */
  count: number;
  /** Row-major label image; 0 is background. */
  labels: Int32Array;
  /** `stats[label]`, stats[0] is the background component like cv2. */
  stats: ComponentStats[];
  centroids: { x: number; y: number }[];
}

export interface HoughCircle {
  x: number;
  y: number;
  radius: number;
}

export interface HoughLine {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RotatedRectResult {
  center: [number, number];
  size: [number, number];
  angle: number;
}

export interface MomentsResult {
  m00: number;
  m10: number;
  m01: number;
  m20: number;
  m11: number;
  m02: number;
  m30: number;
  m21: number;
  m12: number;
  m03: number;
  cx: number;
  cy: number;
}

type ContourMode = 'external' | 'list' | 'ccomp' | 'tree';
type MorphOp = 'open' | 'close' | 'gradient' | 'tophat' | 'blackhat';
type BitwiseOp = 'and' | 'or' | 'xor' | 'not';

let cvModulePromise: Promise<CvModule> | null = null;
let cvInstance: CvModule | null = null;

const isThenable = (value: unknown): value is PromiseLike<CvModule> =>
  !!value && typeof (value as { then?: unknown }).then === 'function';

/** Browser: load the classic-script build from `public/vendor/opencv.js`. */
async function loadBrowserOpenCV(): Promise<CvModule> {
  const global = globalThis as { cv?: unknown };
  if (global.cv !== undefined) return resolveCvCandidate(global.cv);
  if (typeof document === 'undefined') throw new Error('当前环境不支持加载 opencv.js');
  const env = (import.meta as unknown as { env?: { BASE_URL?: string } }).env;
  const override = (globalThis as { __DIRMAP_OPENCV_URL__?: string }).__DIRMAP_OPENCV_URL__;
  const base = env?.BASE_URL ?? '/';
  const url = override ?? `${base.endsWith('/') ? base : `${base}/`}vendor/opencv.js`;
  await new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    script.dataset.dirmapOpencv = '1';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('无法加载开源视觉库 opencv.js（vendor/opencv.js），请检查静态资源部署'));
    document.head.appendChild(script);
  });
  if (global.cv === undefined) throw new Error('opencv.js 加载完成但未导出模块');
  return resolveCvCandidate(global.cv);
}

/** Node tests: the npm package exports a thenable module in 4.11.x. */
async function loadNodeOpenCV(): Promise<CvModule> {
  const specifier = '@techstark/opencv-js';
  const module = (await import(/* @vite-ignore */ specifier)) as {
    default?: unknown;
    cv?: unknown;
  };
  return resolveCvCandidate(module.default ?? module.cv ?? module);
}

/** Wait for `onRuntimeInitialized` when a build exposes the module before WASM is ready. */
function waitForRuntime(cv: CvModule): Promise<CvModule> {
  if (cv.Mat) return Promise.resolve(cv);
  return new Promise<CvModule>((resolve, reject) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve(cv);
    };
    try {
      cv.onRuntimeInitialized = finish;
    } catch {
      /* some builds expose a read-only hook; polling below still works */
    }
    const started = Date.now();
    const poll = () => {
      if (settled) return;
      if (cv.Mat) {
        finish();
        return;
      }
      if (Date.now() - started > 120_000) {
        settled = true;
        reject(new Error('opencv.js 初始化超时'));
        return;
      }
      const timer = (globalThis as { setTimeout?: (fn: () => void, ms: number) => unknown }).setTimeout;
      if (typeof timer === 'function') timer(poll, 25);
    };
    poll();
  });
}

async function resolveCvCandidate(candidate: unknown): Promise<CvModule> {
  let resolved: unknown = candidate;
  // @techstark/opencv-js 4.11 returns a promise; older builds expose the module
  // object with an `onRuntimeInitialized` hook.
  for (let depth = 0; isThenable(resolved) && depth < 4; depth++) resolved = await resolved;
  const cv = (resolved ?? candidate) as CvModule;
  if (!cv || typeof cv !== 'object') throw new Error('opencv.js 初始化失败');
  return waitForRuntime(cv);
}

/** Load (once) and return the opencv.js module. Safe to call concurrently. */
export function loadOpenCV(): Promise<CvModule> {
  if (!cvModulePromise) {
    cvModulePromise = (async () => {
      const candidate = typeof document !== 'undefined' ? await loadBrowserOpenCV() : await loadNodeOpenCV();
      cvInstance = await resolveCvCandidate(candidate);
      return cvInstance;
    })();
  }
  return cvModulePromise;
}

/** The loaded module; throws a Chinese error when `loadOpenCV()` was not awaited. */
export function getCv(): CvModule {
  if (!cvInstance) throw new Error('opencv.js 尚未加载，请先 await loadOpenCV()');
  return cvInstance;
}

/** Run `fn` with a scope that deletes every tracked Mat afterwards (reverse order). */
export function withMats<T>(fn: (track: <M>(mat: M) => M) => T): T {
  const mats: { delete?: () => void }[] = [];
  const track = <M>(mat: M): M => {
    mats.push(mat as unknown as { delete?: () => void });
    return mat;
  };
  try {
    return fn(track);
  } finally {
    for (let i = mats.length - 1; i >= 0; i--) {
      try {
        mats[i].delete?.();
      } catch {
        /* already deleted by the caller: not an error in a scope helper */
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Plain images and masks
// ---------------------------------------------------------------------------

/** Allocate a 0/255 mask filled with `fill`. */
export function maskOf(width: number, height: number, fill = 0): MaskImage {
  const data = new Uint8Array(width * height);
  if (fill) data.fill(fill);
  return { data, width, height };
}

/** Copy an RGBA image into a caller-owned `CV_8UC4` Mat. */
export function matFromRgba(image: RgbaImage): any {
  const cv = getCv();
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC4);
  mat.data.set(image.data);
  return mat;
}

/** Copy a BGR image into a caller-owned `CV_8UC3` Mat. */
export function matFromBgr(image: BgrImage): any {
  const cv = getCv();
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC3);
  mat.data.set(image.data);
  return mat;
}

/** Build a `CV_8UC3` Mat from RGBA bytes without an intermediate BGR array. */
export function bgrMatFromRgba(image: RgbaImage): any {
  const cv = getCv();
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC3);
  const dst = mat.data as Uint8Array;
  const src = image.data;
  for (let i = 0, j = 0; i < src.length; i += 4, j += 3) {
    dst[j] = src[i + 2];
    dst[j + 1] = src[i + 1];
    dst[j + 2] = src[i];
  }
  return mat;
}

/** Copy a caller-owned `CV_8UC4` Mat into a fresh RGBA image. */
export function rgbaFromMat(mat: any): RgbaImage {
  const data = new Uint8ClampedArray(mat.rows * mat.cols * 4);
  data.set((mat.data as Uint8Array).subarray(0, data.length));
  return { data, width: mat.cols, height: mat.rows };
}

/** Copy a caller-owned `CV_8UC1` Mat into a fresh mask. */
export function maskFromMat(mat: any): MaskImage {
  const data = new Uint8Array(mat.rows * mat.cols);
  data.set((mat.data as Uint8Array).subarray(0, data.length));
  return { data, width: mat.cols, height: mat.rows };
}

/** Copy a mask into a caller-owned `CV_8UC1` Mat. */
export function matFromMask(mask: MaskImage): any {
  const cv = getCv();
  const mat = new cv.Mat(mask.height, mask.width, cv.CV_8UC1);
  mat.data.set(mask.data);
  return mat;
}

/** Alias used by the architecture contract (`grayFromRgba` counterpart). */
export function matFromGray(mask: MaskImage): any {
  return matFromMask(mask);
}

/** RGBA -> BGR channel swap (pure JS, no Mat). */
export function bgrFromRgba(image: RgbaImage): BgrImage {
  const data = new Uint8Array(image.width * image.height * 3);
  const src = image.data;
  for (let i = 0, j = 0; i < src.length; i += 4, j += 3) {
    data[j] = src[i + 2];
    data[j + 1] = src[i + 1];
    data[j + 2] = src[i];
  }
  return { data, width: image.width, height: image.height };
}

/** BGR -> RGBA channel swap (pure JS, no Mat). */
export function rgbaFromBgr(image: BgrImage): RgbaImage {
  const data = new Uint8ClampedArray(image.width * image.height * 4);
  const src = image.data;
  for (let i = 0, j = 0; i < src.length; i += 3, j += 4) {
    data[j] = src[i + 2];
    data[j + 1] = src[i + 1];
    data[j + 2] = src[i];
    data[j + 3] = 255;
  }
  return { data, width: image.width, height: image.height };
}

/** Luma-ish grey image (BT.601, cv2.COLOR_RGB2GRAY integer weights). */
export function grayFromRgba(image: RgbaImage): MaskImage {
  const data = new Uint8Array(image.width * image.height);
  const src = image.data;
  for (let i = 0, j = 0; i < src.length; i += 4, j++) {
    data[j] = ((src[i] * 77 + src[i + 1] * 150 + src[i + 2] * 29 + 128) >> 8) & 0xff;
  }
  return { data, width: image.width, height: image.height };
}

/** Create a caller-owned Mat by applying one `cv.cvtColor` conversion. */
export function cvtColor(src: any, code: number): any {
  const cv = getCv();
  const dst = new cv.Mat();
  cv.cvtColor(src, dst, code);
  return dst;
}

/** Split a caller-owned Mat into fresh one-channel `Uint8Array` planes. */
export function split(mat: any): Uint8Array[] {
  const cv = getCv();
  const planes = new cv.MatVector();
  try {
    cv.split(mat, planes);
    const result: Uint8Array[] = [];
    for (let i = 0; i < planes.size(); i++) {
      const plane = planes.get(i);
      try {
        const data = new Uint8Array(plane.rows * plane.cols);
        data.set((plane.data as Uint8Array).subarray(0, data.length));
        result.push(data);
      } finally {
        plane.delete();
      }
    }
    return result;
  } finally {
    planes.delete();
  }
}

/** HSV planes of an RGBA image (OpenCV 8-bit HSV: h in 0..179, s/v in 0..255). */
export function hsvFromRgba(image: RgbaImage): { h: Uint8Array; s: Uint8Array; v: Uint8Array } {
  const cv = getCv();
  const bgr = bgrMatFromRgba(image);
  const hsv = new cv.Mat();
  try {
    cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);
    const [h, s, v] = split(hsv);
    return { h, s, v };
  } finally {
    hsv.delete();
    bgr.delete();
  }
}

/** HSV planes of a BGR image. */
export function hsvFromBgr(image: BgrImage): { h: Uint8Array; s: Uint8Array; v: Uint8Array } {
  const cv = getCv();
  const bgr = matFromBgr(image);
  const hsv = new cv.Mat();
  try {
    cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);
    const [h, s, v] = split(hsv);
    return { h, s, v };
  } finally {
    hsv.delete();
    bgr.delete();
  }
}

/** Read-only BGR planes from an interleaved BGR image. */
export function bgrPlanes(image: BgrImage): [Uint8Array, Uint8Array, Uint8Array] {
  const count = image.width * image.height;
  const b = new Uint8Array(count);
  const g = new Uint8Array(count);
  const r = new Uint8Array(count);
  const src = image.data;
  for (let i = 0, j = 0; i < count; i++, j += 3) {
    b[i] = src[j];
    g[i] = src[j + 1];
    r[i] = src[j + 2];
  }
  return [b, g, r];
}

// ---------------------------------------------------------------------------
// Mask operations
// ---------------------------------------------------------------------------

const MORPH_OPS: Record<MorphOp, string> = {
  open: 'MORPH_OPEN',
  close: 'MORPH_CLOSE',
  gradient: 'MORPH_GRADIENT',
  tophat: 'MORPH_TOPHAT',
  blackhat: 'MORPH_BLACKHAT',
};

/** Rectangular structuring element as a caller-owned Mat. */
export function getStructuringElement(size: number, shape: 'rect' | 'ellipse' | 'cross' = 'rect'): any {
  const cv = getCv();
  const code = shape === 'ellipse' ? cv.MORPH_ELLIPSE : shape === 'cross' ? cv.MORPH_CROSS : cv.MORPH_RECT;
  return cv.getStructuringElement(code, new cv.Size(size, size));
}

function runMorphology(mask: MaskImage, opName: string, kernel: any, deleteKernel: boolean): MaskImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.morphologyEx(src, dst, cv[opName], kernel);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
    if (deleteKernel) kernel.delete();
  }
}

/** Morphology with a square `kernelSize` element (cv2.morphologyEx + np.ones). */
export function morphology(mask: MaskImage, op: MorphOp, kernelSize: number): MaskImage {
  const kernel = getStructuringElement(kernelSize);
  return runMorphology(mask, MORPH_OPS[op], kernel, true);
}

/** Morphology with a caller-owned kernel Mat (e.g. directional line kernels). */
export function morphologyWithKernel(mask: MaskImage, op: MorphOp, kernel: any): MaskImage {
  return runMorphology(mask, MORPH_OPS[op], kernel, false);
}

/** Morphology with a mask-shaped kernel (converted to a Mat internally). */
export function morphologyWithMaskKernel(mask: MaskImage, op: MorphOp, kernel: MaskImage): MaskImage {
  const mat = matFromMask(kernel);
  try {
    return runMorphology(mask, MORPH_OPS[op], mat, false);
  } finally {
    mat.delete();
  }
}

/** cv2.dilate with a square kernel (default 3x3). */
export function dilate(mask: MaskImage, kernelSize = 3): MaskImage {
  const cv = getCv();
  const kernel = getStructuringElement(kernelSize);
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.dilate(src, dst, kernel);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
    kernel.delete();
  }
}

/** cv2.erode with a square kernel (default 3x3). */
export function erode(mask: MaskImage, kernelSize = 3): MaskImage {
  const cv = getCv();
  const kernel = getStructuringElement(kernelSize);
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.erode(src, dst, kernel);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
    kernel.delete();
  }
}

/** cv2.GaussianBlur on a mask (keeps 8-bit values, e.g. for HoughCircles). */
export function gaussianBlur(mask: MaskImage, kernelSize = 3, sigma = 0): MaskImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.GaussianBlur(src, dst, new cv.Size(kernelSize, kernelSize), sigma);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
  }
}

/** cv2.resize with an explicit destination size. */
export function resize(mask: MaskImage, width: number, height: number, interpolation?: number): MaskImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.resize(src, dst, new cv.Size(width, height), 0, 0, interpolation ?? cv.INTER_LINEAR);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
  }
}

/**
 * `cv2.resize(mask, None, fx=scale, fy=scale, interpolation)` exactly as the
 * Python pipeline calls it: the destination `(0, 0)` keeps OpenCV's own
 * `cvRound` sizing and INTER_AREA scale semantics (no caller rounding drift).
 */
export function resizeByScale(mask: MaskImage, fx: number, fy: number, interpolation?: number): MaskImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.resize(src, dst, new cv.Size(0, 0), fx, fy, interpolation ?? cv.INTER_LINEAR);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
  }
}

/** cv2.threshold (binary by default). */
export function threshold(mask: MaskImage, thresh: number, maxval = 255, type?: number): MaskImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.threshold(src, dst, thresh, maxval, type ?? cv.THRESH_BINARY);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
  }
}

/** Element-wise bitwise op between two masks (`not` ignores `other`). */
export function bitwise(mask: MaskImage, other: MaskImage | null, op: BitwiseOp): MaskImage {
  const cv = getCv();
  const a = matFromMask(mask);
  const b = other ? matFromMask(other) : null;
  const dst = new cv.Mat();
  try {
    if (b && mask.width !== other!.width) throw new Error('掩膜尺寸不一致');
    if (op === 'and') cv.bitwise_and(a, b, dst);
    else if (op === 'or') cv.bitwise_or(a, b, dst);
    else if (op === 'xor') cv.bitwise_xor(a, b, dst);
    else cv.bitwise_not(a, dst);
    return maskFromMat(dst);
  } finally {
    dst.delete();
    b?.delete();
    a.delete();
  }
}

/** cv2.bitwise_not on one mask. */
export function maskNot(mask: MaskImage): MaskImage {
  return bitwise(mask, null, 'not');
}

/** cv2.copyMakeBorder with the same constant border on every side. */
export function copyMakeBorder(mask: MaskImage, pad: number, value = 0): MaskImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.copyMakeBorder(src, dst, pad, pad, pad, pad, cv.BORDER_CONSTANT, new cv.Scalar(value));
    return maskFromMat(dst);
  } finally {
    dst.delete();
    src.delete();
  }
}

/**
 * `cv2.distanceTransform(mask, cv.DIST_L2, maskSize)` on the given 0/255 mask:
 * the value of every pixel is its distance to the nearest **zero** pixel.
 */
export function distanceTransformL2(mask: MaskImage, maskSize = 3): FloatImage {
  const cv = getCv();
  const src = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.distanceTransform(src, dst, cv.DIST_L2, maskSize);
    const data = new Float32Array(dst.rows * dst.cols);
    data.set((dst.data32F as Float32Array).subarray(0, data.length));
    return { data, width: dst.cols, height: dst.rows };
  } finally {
    dst.delete();
    src.delete();
  }
}

/** Local mean of `mask > 0` with a `ksize` square stencil, as float32. */
export function boxFilterMean(mask: MaskImage, ksize: number): FloatImage {
  const cv = getCv();
  const src = new cv.Mat(mask.height, mask.width, cv.CV_32FC1);
  const dst = new cv.Mat();
  try {
    const values = src.data32F as Float32Array;
    for (let i = 0; i < mask.data.length; i++) values[i] = mask.data[i] > 0 ? 1 : 0;
    cv.boxFilter(src, dst, -1, new cv.Size(ksize, ksize));
    const data = new Float32Array(dst.rows * dst.cols);
    data.set((dst.data32F as Float32Array).subarray(0, data.length));
    return { data, width: dst.cols, height: dst.rows };
  } finally {
    dst.delete();
    src.delete();
  }
}

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

/** cv2.connectedComponentsWithStats (CV_32S labels, 4/8 connectivity). */
export function connectedComponentsWithStats(mask: MaskImage, connectivity: 4 | 8 = 8): ComponentsResult {
  const cv = getCv();
  const src = matFromMask(mask);
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  try {
    const count = cv.connectedComponentsWithStats(src, labels, stats, centroids, connectivity, cv.CV_32S);
    const labelData = new Int32Array(labels.rows * labels.cols);
    labelData.set((labels.data32S as Int32Array).subarray(0, labelData.length));
    const statData = stats.data32S as Int32Array;
    const centroidData = centroids.data32F as Float32Array;
    const statRows: ComponentStats[] = [];
    const centroidRows: { x: number; y: number }[] = [];
    for (let i = 0; i < count; i++) {
      statRows.push({
        x: statData[i * 5],
        y: statData[i * 5 + 1],
        width: statData[i * 5 + 2],
        height: statData[i * 5 + 3],
        area: statData[i * 5 + 4],
      });
      centroidRows.push({ x: centroidData[i * 2], y: centroidData[i * 2 + 1] });
    }
    return { width: mask.width, height: mask.height, count, labels: labelData, stats: statRows, centroids: centroidRows };
  } finally {
    centroids.delete();
    stats.delete();
    labels.delete();
    src.delete();
  }
}

/** cv2.connectedComponents labels only (CV_32S, 4/8 connectivity). */
export function connectedComponents(mask: MaskImage, connectivity: 4 | 8 = 4): {
  width: number;
  height: number;
  count: number;
  labels: Int32Array;
} {
  const cv = getCv();
  const src = matFromMask(mask);
  const labels = new cv.Mat();
  try {
    const count = cv.connectedComponents(src, labels, connectivity, cv.CV_32S);
    const labelData = new Int32Array(labels.rows * labels.cols);
    labelData.set((labels.data32S as Int32Array).subarray(0, labelData.length));
    return { width: mask.width, height: mask.height, count, labels: labelData };
  } finally {
    labels.delete();
    src.delete();
  }
}

// ---------------------------------------------------------------------------
// Contours
// ---------------------------------------------------------------------------

const CONTOUR_MODES: Record<ContourMode, string> = {
  external: 'RETR_EXTERNAL',
  list: 'RETR_LIST',
  ccomp: 'RETR_CCOMP',
  tree: 'RETR_TREE',
};

function contoursFromVector(vector: any): Contour[] {
  const result: Contour[] = [];
  for (let i = 0; i < vector.size(); i++) {
    const contour = vector.get(i);
    try {
      const values = contour.data32S as Int32Array;
      const points: number[][] = [];
      for (let p = 0; p < contour.rows; p++) points.push([values[p * 2], values[p * 2 + 1]]);
      if (points.length) result.push({ points });
    } finally {
      contour.delete();
    }
  }
  return result;
}

/** cv2.findContours with CHAIN_APPROX_SIMPLE; hierarchy is discarded. */
export function findContours(mask: MaskImage, mode: ContourMode = 'external'): Contour[] {
  return findContoursWithHierarchy(mask, mode).contours;
}

/** cv2.findContours where `hierarchy[i]` describes `contours[i]` (parallel arrays). */
export function findContoursWithHierarchy(mask: MaskImage, mode: ContourMode = 'external'): ContourHierarchy {
  const cv = getCv();
  const src = matFromMask(mask);
  const vector = new cv.MatVector();
  const hierarchyMat = new cv.Mat();
  try {
    cv.findContours(src, vector, hierarchyMat, cv[CONTOUR_MODES[mode]], cv.CHAIN_APPROX_SIMPLE);
    const contours = contoursFromVector(vector);
    const raw = hierarchyMat.data32S as Int32Array | undefined;
    const hierarchy: ContourNode[] = [];
    if (raw && raw.length >= contours.length * 4) {
      for (let i = 0; i < contours.length; i++) {
        hierarchy.push({ next: raw[i * 4], prev: raw[i * 4 + 1], firstChild: raw[i * 4 + 2], parent: raw[i * 4 + 3] });
      }
    } else {
      for (let i = 0; i < contours.length; i++) hierarchy.push({ next: -1, prev: -1, firstChild: -1, parent: -1 });
    }
    return { contours, hierarchy };
  } finally {
    hierarchyMat.delete();
    vector.delete();
    src.delete();
  }
}

function contourMat(contour: Contour | number[][]): any {
  const cv = getCv();
  const points = Array.isArray(contour) ? contour : contour.points;
  const flat: number[] = [];
  for (const point of points) flat.push(point[0], point[1]);
  return cv.matFromArray(points.length, 1, cv.CV_32SC2, flat);
}

/** cv2.contourArea (oriented=false). */
export function contourArea(contour: Contour | number[][]): number {
  const cv = getCv();
  const mat = contourMat(contour);
  try {
    return cv.contourArea(mat, false) as number;
  } finally {
    mat.delete();
  }
}

/** cv2.arcLength. */
export function arcLength(contour: Contour | number[][], closed = true): number {
  const cv = getCv();
  const mat = contourMat(contour);
  try {
    return cv.arcLength(mat, closed) as number;
  } finally {
    mat.delete();
  }
}

/** cv2.approxPolyDP. */
export function approxPolyDP(contour: Contour | number[][], epsilon: number, closed = true): Contour {
  const cv = getCv();
  const src = contourMat(contour);
  const dst = new cv.Mat();
  try {
    cv.approxPolyDP(src, dst, epsilon, closed);
    const values = dst.data32S as Int32Array;
    const points: number[][] = [];
    for (let i = 0; i < dst.rows; i++) points.push([values[i * 2], values[i * 2 + 1]]);
    return { points };
  } finally {
    dst.delete();
    src.delete();
  }
}

/** cv2.pointPolygonTest (integer point; `measureDist` matches cv2 semantics). */
export function pointPolygonTest(contour: Contour | number[][], point: [number, number], measureDist = false): number {
  const cv = getCv();
  const mat = contourMat(contour);
  try {
    return cv.pointPolygonTest(mat, new cv.Point(Math.round(point[0]), Math.round(point[1])), measureDist) as number;
  } finally {
    mat.delete();
  }
}

/** cv2.minEnclosingCircle. */
export function minEnclosingCircle(contour: Contour | number[][]): { center: [number, number]; radius: number } {
  const cv = getCv();
  const mat = contourMat(contour);
  try {
    const result = cv.minEnclosingCircle(mat);
    return { center: [result.center.x, result.center.y], radius: result.radius };
  } finally {
    mat.delete();
  }
}

/** cv2.minAreaRect over a polygon (points are converted to CV_32FC2). */
export function minAreaRect(points: number[][]): RotatedRectResult {
  const cv = getCv();
  const flat: number[] = [];
  for (const point of points) flat.push(point[0], point[1]);
  const mat = cv.matFromArray(points.length, 1, cv.CV_32FC2, flat);
  try {
    const rect = cv.minAreaRect(mat);
    return {
      center: [rect.center.x, rect.center.y],
      size: [rect.size.width, rect.size.height],
      angle: rect.angle,
    };
  } finally {
    mat.delete();
  }
}

/** cv2.moments + centroid convenience fields. */
export function contourMoments(contour: Contour | number[][]): MomentsResult {
  const cv = getCv();
  const mat = contourMat(contour);
  try {
    const moments = cv.moments(mat, false);
    const m00 = moments.m00 as number;
    return {
      m00,
      m10: moments.m10,
      m01: moments.m01,
      m20: moments.m20,
      m11: moments.m11,
      m02: moments.m02,
      m30: moments.m30,
      m21: moments.m21,
      m12: moments.m12,
      m03: moments.m03,
      cx: m00 !== 0 ? moments.m10 / m00 : 0,
      cy: m00 !== 0 ? moments.m01 / m00 : 0,
    };
  } finally {
    mat.delete();
  }
}

/** cv2.convexHull (returnPoints=true). */
export function convexHull(contour: Contour): Contour {
  const cv = getCv();
  const src = contourMat(contour);
  const hull = new cv.Mat();
  try {
    cv.convexHull(src, hull, false, true);
    const values = hull.data32S as Int32Array;
    const points: number[][] = [];
    for (let i = 0; i < hull.rows; i++) points.push([values[i * 2], values[i * 2 + 1]]);
    return { points };
  } finally {
    hull.delete();
    src.delete();
  }
}

/** cv2.isContourConvex. */
export function isContourConvex(contour: Contour): boolean {
  const cv = getCv();
  const mat = contourMat(contour);
  try {
    return cv.isContourConvex(mat) as boolean;
  } finally {
    mat.delete();
  }
}

function mutateMask(dst: MaskImage, draw: (mat: any) => void): void {
  const src = matFromMask(dst);
  try {
    draw(src);
    dst.data.set((src.data as Uint8Array).subarray(0, dst.data.length));
  } finally {
    src.delete();
  }
}

function buildContourVector(contours: Contour[]): { vector: any; mats: any[] } {
  const cv = getCv();
  const vector = new cv.MatVector();
  const mats: any[] = [];
  for (const contour of contours) {
    const mat = contourMat(contour);
    mats.push(mat);
    vector.push_back(mat);
  }
  return { vector, mats };
}

/** cv2.drawContours with thickness -1 (FILLED) writing into `dst.data`. */
export function fillContours(dst: MaskImage, contours: Contour[], value = 255): void {
  if (!contours.length) return;
  const cv = getCv();
  const { vector, mats } = buildContourVector(contours);
  try {
    mutateMask(dst, mat => cv.drawContours(mat, vector, -1, new cv.Scalar(value), -1));
  } finally {
    for (const mat of mats) mat.delete();
    vector.delete();
  }
}

/** cv2.drawContours outline with the given thickness writing into `dst.data`. */
export function drawContours(dst: MaskImage, contours: Contour[], thickness: number, value = 255): void {
  if (!contours.length) return;
  const cv = getCv();
  const { vector, mats } = buildContourVector(contours);
  try {
    mutateMask(dst, mat => cv.drawContours(mat, vector, -1, new cv.Scalar(value), thickness));
  } finally {
    for (const mat of mats) mat.delete();
    vector.delete();
  }
}

/** cv2.line writing into `dst.data`. */
export function line(dst: MaskImage, a: [number, number], b: [number, number], thickness = 1, value = 255): void {
  const cv = getCv();
  mutateMask(dst, mat => cv.line(mat, new cv.Point(Math.round(a[0]), Math.round(a[1])), new cv.Point(Math.round(b[0]), Math.round(b[1])), new cv.Scalar(value), thickness));
}

/** cv2.circle writing into `dst.data`; `thickness = -1` fills the disk. */
export function circle(dst: MaskImage, center: [number, number], radius: number, thickness = 1, value = 255): void {
  const cv = getCv();
  mutateMask(dst, mat => cv.circle(mat, new cv.Point(Math.round(center[0]), Math.round(center[1])), Math.max(0, Math.round(radius)), new cv.Scalar(value), thickness));
}

/** cv2.polylines writing into `dst.data`. */
export function polylines(dst: MaskImage, points: number[][], closed: boolean, thickness = 1, value = 255): void {
  if (points.length < 2) return;
  const cv = getCv();
  const { vector, mats } = buildContourVector([{ points }]);
  try {
    mutateMask(dst, mat => cv.polylines(mat, vector, closed, new cv.Scalar(value), thickness));
  } finally {
    for (const mat of mats) mat.delete();
    vector.delete();
  }
}

// ---------------------------------------------------------------------------
// Images / Hough / inpaint
// ---------------------------------------------------------------------------

/** cv2.inpaint(INPAINT_NS) of a BGR image; returns a fresh BGR image. */
export function inpaintBgr(image: BgrImage, mask: MaskImage, radius = 1): BgrImage {
  const cv = getCv();
  const src = matFromBgr(image);
  const paint = matFromMask(mask);
  const dst = new cv.Mat();
  try {
    cv.inpaint(src, paint, dst, radius, cv.INPAINT_NS);
    const data = new Uint8Array(dst.rows * dst.cols * 3);
    data.set((dst.data as Uint8Array).subarray(0, data.length));
    return { data, width: dst.cols, height: dst.rows };
  } finally {
    dst.delete();
    paint.delete();
    src.delete();
  }
}

/** cv2.inpaint(INPAINT_NS) of an RGBA image; returns a fresh RGBA image. */
export function inpaint(rgba: RgbaImage, mask: MaskImage, radius = 1): RgbaImage {
  return rgbaFromBgr(inpaintBgr(bgrFromRgba(rgba), mask, radius));
}

/** cv2.HoughCircles(HOUGH_GRADIENT). */
export function houghCircles(
  mask: MaskImage,
  options: { dp?: number; minDist: number; param1?: number; param2?: number; minRadius?: number; maxRadius?: number },
): HoughCircle[] {
  const cv = getCv();
  const src = matFromMask(mask);
  const circles = new cv.Mat();
  try {
    cv.HoughCircles(
      src,
      circles,
      cv.HOUGH_GRADIENT,
      options.dp ?? 1,
      options.minDist,
      options.param1 ?? 80,
      options.param2 ?? 13,
      options.minRadius ?? 0,
      options.maxRadius ?? 0,
    );
    const values = circles.data32F as Float32Array | undefined;
    if (!values || !values.length) return [];
    const result: HoughCircle[] = [];
    for (let i = 0; i + 2 < values.length; i += 3) result.push({ x: values[i], y: values[i + 1], radius: values[i + 2] });
    return result;
  } finally {
    circles.delete();
    src.delete();
  }
}

/** cv2.HoughLinesP (segments as x1,y1,x2,y2). */
export function houghLinesP(
  mask: MaskImage,
  options: { rho?: number; theta?: number; threshold: number; minLineLength: number; maxLineGap: number },
): HoughLine[] {
  const cv = getCv();
  const src = matFromMask(mask);
  const lines = new cv.Mat();
  try {
    cv.HoughLinesP(
      src,
      lines,
      options.rho ?? 1,
      options.theta ?? Math.PI / 180,
      options.threshold,
      options.minLineLength,
      options.maxLineGap,
    );
    const values = lines.data32S as Int32Array | undefined;
    if (!values || !values.length) return [];
    const result: HoughLine[] = [];
    for (let i = 0; i + 3 < values.length; i += 4) {
      result.push({ x1: values[i], y1: values[i + 1], x2: values[i + 2], y2: values[i + 3] });
    }
    return result;
  } finally {
    lines.delete();
    src.delete();
  }
}

/** cv2.remap with a caller-supplied CV_32FC2 map (advanced, caller-owned Mats). */
export function remap(src: any, mapX: any, mapY: any, interpolation?: number): any {
  const cv = getCv();
  const dst = new cv.Mat();
  cv.remap(src, dst, mapX, mapY, interpolation ?? cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0));
  return dst;
}
