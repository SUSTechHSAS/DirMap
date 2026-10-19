/** Optional course-number OCR for the browser pipeline: port of `dirmap/ocr.py`.
 *
 * Tesseract runs through `tesseract.js` (WASM) with CDN worker/core/language
 * assets by default; the whole module degrades gracefully — when the worker,
 * WASM core, language data or network are unavailable `labelControls` resolves
 * to `{available:false, recognized:0}` and the analysis keeps its `待核N`
 * labels, exactly like the Python no-OCR path (`dirmap/ocr.py:19-21`).
 *
 * The algorithm is the Python cascade: one global pass over the circle-erased
 * text ink, then per-control crop passes (fixed crops, an upper crop used to
 * isolate the number beside the ring, stroke-removed flat masks, `psm 7/11`
 * retries, and connected-component "glyph rows"), with the same eligibility,
 * stale-label clearing, nearest-control disambiguation, TSV word filtering
 * (digit whitelist, confidence >= 65) and duplicate-label ambiguity flag.
 *
 * Assets can be overridden for offline/self-hosted deployments through
 * `import.meta.env` (`VITE_TESSERACT_WORKER`, `VITE_TESSERACT_CORE`,
 * `VITE_TESSERACT_LANG`) or `globalThis.__DIRMAP_TESSERACT_{WORKER,CORE,LANG}__`.
 *
 * Deviations from the Python CLI reference, all behaviour-preserving:
 * - images are handed to tesseract.js as an uncompressed 8-bit grayscale PNG
 *   encoded here (the CLI wrote a PNG file; no DOM/canvas is used, so this
 *   module also runs in workers);
 * - with no controls (or when the worker cannot start) `available` is false
 *   instead of probing an executable;
 * - tesseract.js returns TSV from the in-memory API; a missing header line is
 *   tolerated.
 */
import type { MaskImage } from './cv.ts';
import { connectedComponentsWithStats, copyMakeBorder, circle, getCv, maskOf, maskNot, morphology, resizeByScale } from './cv.ts';
import type { AnalyzedControl, OcrResult } from './vision.ts';

export type { OcrResult } from './vision.ts';

const METHOD = 'tesseract_global_and_digit_crops';
const BORDER = 16;
const WORD_RE = /^(?:[1-9]-)?([1-9]\d{1,2})$/;

interface Word {
  label: string;
  confidence: number;
  center: [number, number];
  box: [number, number, number, number];
}

interface TesseractWorker {
  recognize(
    image: unknown,
    options?: Record<string, unknown>,
    output?: Record<string, unknown>,
  ): Promise<{ data: { tsv?: string; text?: string } }>;
  terminate(): Promise<unknown>;
}

let workerPromise: Promise<TesseractWorker> | null = null;
let workerUnavailable = false;

function workerOptions(): Record<string, unknown> {
  const env = ((import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {}) as Record<string, string | undefined>;
  const global = globalThis as Record<string, unknown>;
  const options: Record<string, unknown> = { logger: () => {} };
  const workerPath = (global.__DIRMAP_TESSERACT_WORKER__ as string | undefined) ?? env.VITE_TESSERACT_WORKER;
  const corePath = (global.__DIRMAP_TESSERACT_CORE__ as string | undefined) ?? env.VITE_TESSERACT_CORE;
  const langPath = (global.__DIRMAP_TESSERACT_LANG__ as string | undefined) ?? env.VITE_TESSERACT_LANG;
  const cachePath = (global.__DIRMAP_TESSERACT_CACHE_PATH__ as string | undefined) ?? env.VITE_TESSERACT_CACHE_PATH;
  if (workerPath) options.workerPath = workerPath;
  if (corePath) options.corePath = corePath;
  if (langPath) options.langPath = langPath;
  if (cachePath) options.cachePath = cachePath;
  return options;
}

/** Lazily create the shared tesseract.js worker (`eng`, LSTM only). */
async function getWorker(): Promise<TesseractWorker> {
  if (workerUnavailable) throw new Error('OCR 不可用');
  if (!workerPromise) {
    workerPromise = (async () => {
      const module = (await import('tesseract.js')) as unknown as {
        createWorker?: (langs: string, oem: number, options?: Record<string, unknown>) => Promise<TesseractWorker>;
        default?: { createWorker?: (langs: string, oem: number, options?: Record<string, unknown>) => Promise<TesseractWorker> };
      };
      const createWorker = module.createWorker ?? module.default?.createWorker;
      if (!createWorker) throw new Error('tesseract.js 无法加载');
      return createWorker('eng', 1, workerOptions());
    })().catch(error => {
      // Never retry a broken CDN/WASM setup on every analysis: mark unavailable.
      workerUnavailable = true;
      workerPromise = null;
      throw error;
    });
  }
  return workerPromise;
}

/** Release the OCR worker (e.g. on page teardown); a later call re-creates it. */
export async function terminateOcr(): Promise<void> {
  const pending = workerPromise;
  workerPromise = null;
  if (!pending) return;
  try {
    await (await pending).terminate();
  } catch {
    /* worker already gone */
  }
}

// ---------------------------------------------------------------------------
// Minimal PNG encoder (8-bit grayscale, stored deflate blocks)
// ---------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index++) {
    let value = index;
    for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    table[index] = value >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index++) crc = CRC_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function adler32(bytes: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (let index = 0; index < bytes.length; index++) {
    a = (a + bytes[index]) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}

function zlibStored(raw: Uint8Array): Uint8Array {
  const blocks = Math.max(1, Math.ceil(raw.length / 65535));
  const output = new Uint8Array(2 + blocks * 5 + raw.length + 4);
  let offset = 0;
  output[offset++] = 0x78; // CMF: deflate, 32K window
  output[offset++] = 0x01; // FLG: no dictionary, fastest
  let position = 0;
  for (let block = 0; block < blocks; block++) {
    const length = Math.min(65535, raw.length - position);
    const final = block === blocks - 1;
    output[offset++] = final ? 1 : 0;
    output[offset++] = length & 0xff;
    output[offset++] = (length >> 8) & 0xff;
    output[offset++] = ~length & 0xff;
    output[offset++] = (~length >> 8) & 0xff;
    output.set(raw.subarray(position, position + length), offset);
    offset += length;
    position += length;
  }
  const adler = adler32(raw);
  output[offset++] = (adler >>> 24) & 0xff;
  output[offset++] = (adler >>> 16) & 0xff;
  output[offset++] = (adler >>> 8) & 0xff;
  output[offset++] = adler & 0xff;
  return output;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length);
  for (let index = 0; index < 4; index++) chunk[4 + index] = type.charCodeAt(index);
  chunk.set(data, 8);
  view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)));
  return chunk;
}

/** Encode an 8-bit grayscale image as PNG bytes (no dependencies, no canvas). */
function encodeGrayPng(data: Uint8Array, width: number, height: number): Uint8Array {
  const header = new Uint8Array(13);
  const headerView = new DataView(header.buffer);
  headerView.setUint32(0, width);
  headerView.setUint32(4, height);
  header[8] = 8; // bit depth
  header[9] = 0; // grayscale
  header[10] = 0; // deflate
  header[11] = 0; // adaptive filtering
  header[12] = 0; // no interlace
  const raw = new Uint8Array((width + 1) * height);
  for (let row = 0; row < height; row++) {
    raw[row * (width + 1)] = 0; // filter type: none
    raw.set(data.subarray(row * width, row * width + width), row * (width + 1) + 1);
  }
  const idat = zlibStored(raw);
  const signature = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const parts = [signature, pngChunk('IHDR', header), pngChunk('IDAT', idat), pngChunk('IEND', new Uint8Array(0))];
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const png = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    png.set(part, offset);
    offset += part.length;
  }
  return png;
}

/** Browser: Blob (tesseract.js worker reads it with FileReader); Node: Buffer. */
function imagePayload(png: Uint8Array): unknown {
  const buffer = (globalThis as { Buffer?: { from?: (bytes: Uint8Array) => unknown } }).Buffer;
  if (buffer && typeof buffer.from === 'function') return buffer.from(png);
  return new Blob([png as unknown as BlobPart], { type: 'image/png' });
}

// ---------------------------------------------------------------------------
// TSV parsing
// ---------------------------------------------------------------------------

function parseTsv(tsv: string): Word[] {
  const words: Word[] = [];
  for (const line of tsv.split(/\r?\n/)) {
    if (!line) continue;
    const fields = line.split('\t');
    if (fields.length < 12) continue;
    if (!/^\d+$/.test(fields[0])) continue; // header line
    const text = fields[11].trim();
    const match = WORD_RE.exec(text);
    if (!match) continue;
    const confidence = Number.parseFloat(fields[10]);
    if (!Number.isFinite(confidence) || confidence < 65) continue;
    const left = Number.parseInt(fields[6], 10);
    const top = Number.parseInt(fields[7], 10);
    const wordWidth = Number.parseInt(fields[8], 10);
    const wordHeight = Number.parseInt(fields[9], 10);
    if (![left, top, wordWidth, wordHeight].every(Number.isFinite)) continue;
    words.push({
      label: match[1],
      confidence,
      // Offset is applied by the caller below (center/box are local here).
      center: [left + wordWidth / 2, top + wordHeight / 2],
      box: [left, top, left + wordWidth, top + wordHeight],
    });
  }
  return words;
}

// ---------------------------------------------------------------------------
// Mask helpers shared with the Python cascade
// ---------------------------------------------------------------------------

function copyMask(mask: MaskImage): MaskImage {
  return { data: new Uint8Array(mask.data), width: mask.width, height: mask.height };
}

function sliceMask(mask: MaskImage, x0: number, y0: number, x1: number, y1: number): MaskImage {
  const left = Math.max(0, Math.min(mask.width, x0));
  const top = Math.max(0, Math.min(mask.height, y0));
  const right = Math.max(left, Math.min(mask.width, x1));
  const bottom = Math.max(top, Math.min(mask.height, y1));
  const width = right - left;
  const height = bottom - top;
  const data = new Uint8Array(width * height);
  for (let row = 0; row < height; row++) {
    data.set(mask.data.subarray((top + row) * mask.width + left, (top + row) * mask.width + right), row * width);
  }
  return { data, width, height };
}

function masksEqual(left: MaskImage, right: MaskImage): boolean {
  if (left.width !== right.width || left.height !== right.height) return false;
  const a = left.data;
  const b = right.data;
  for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
  return true;
}

/** symbols.py::course_ink_mask (duplicated locally to avoid a vision.ts cycle). */
function courseInkStrokes(purple: MaskImage, controls: AnalyzedControl[]): Uint8Array {
  const strokes = maskOf(purple.width, purple.height);
  for (const control of controls) {
    const center: [number, number] = [Math.round(control.pixel[0]), Math.round(control.pixel[1])];
    const radius = control.radiusPx;
    const rings = [radius, ...(control.symbolGeometry?.ringRadii ?? [])];
    for (const ring of rings) circle(strokes, center, ring, Math.max(3, Math.round(radius * 0.15)), 255);
    if (control.symbolGeometry?.triangle) {
      // polylines is only needed for triangles; path drawing keeps the port small.
      const triangle = control.symbolGeometry.triangle;
      for (let index = 0; index < triangle.length; index++) {
        const next = triangle[(index + 1) % triangle.length];
        lineSegment(strokes, triangle[index], next, Math.max(3, Math.round(radius * 0.16)));
      }
    }
  }
  const result = new Uint8Array(purple.data.length);
  for (let index = 0; index < result.length; index++) result[index] = strokes.data[index] & purple.data[index];
  return result;
}

/** Draw a 1-pixel-value integer line into a mask (Bresenham, matches cv2.line for thickness 1). */
function lineSegment(dst: MaskImage, from: number[], to: number[], thickness: number): void {
  // cv2.line with a thick pen equals a dilated thin line for the shapes used
  // here (triangles at a few hundred pixels); keep an explicit line kernel so
  // the port stays dependency-free and deterministic.
  const size = Math.max(1, Math.round(thickness));
  const steps = Math.ceil(Math.max(Math.abs(to[0] - from[0]), Math.abs(to[1] - from[1])));
  const half = Math.floor(size / 2);
  for (let step = 0; step <= Math.max(1, steps); step++) {
    const fraction = steps > 0 ? step / steps : 0;
    const x = Math.round(from[0] + (to[0] - from[0]) * fraction);
    const y = Math.round(from[1] + (to[1] - from[1]) * fraction);
    for (let dy = -half; dy <= size - half - 1; dy++) {
      for (let dx = -half; dx <= size - half - 1; dx++) {
        const px = x + dx;
        const py = y + dy;
        if (px < 0 || py < 0 || px >= dst.width || py >= dst.height) continue;
        dst.data[py * dst.width + px] = 255;
      }
    }
  }
}

/** `vision.py`'s glyph-row grouping from `ocr.py::glyph_rows` (unused fallback kept out). */
function glyphRows(mask: MaskImage, point: [number, number], radius: number): [number, number, number, number][] {
  const x0 = Math.max(0, Math.floor(point[0] - radius * 5));
  const y0 = Math.max(0, Math.floor(point[1] - radius * 5));
  const x1 = Math.min(mask.width, Math.ceil(point[0] + radius * 5));
  const y1 = Math.min(mask.height, Math.ceil(point[1] + radius * 5));
  if (!(x1 > x0 && y1 > y0)) return [];
  const crop = sliceMask(mask, x0, y0, x1, y1);
  const components = connectedComponentsWithStats(crop, 8);
  const glyphs: [number, number, number, number][] = [];
  for (let label = 1; label < components.count; label++) {
    const stat = components.stats[label];
    if (radius * 0.25 <= stat.height && stat.height <= radius * 1.8
      && radius * 0.06 <= stat.width && stat.width <= radius * 2
      && stat.area >= Math.max(4, radius * radius * 0.035)) {
      glyphs.push([stat.x, stat.y, stat.width, stat.height]);
    }
  }
  const byX = glyphs.map((glyph, index) => ({ glyph, index })).sort((left, right) => left.glyph[0] - right.glyph[0]);
  const used = new Set<number>();
  const rows: [number, number, number, number][] = [];
  for (const { glyph: box, index } of byX) {
    if (used.has(index)) continue;
    const row: [number, number, number, number][] = [box];
    used.add(index);
    let right = box[0] + box[2];
    for (const { glyph: other, index: otherIndex } of byX) {
      if (used.has(otherIndex)) continue;
      const [xx, yy, width, height] = other;
      if (!(xx - right >= 0 && xx - right <= Math.max(box[3], height) * 0.7)) continue;
      if (Math.abs(yy + height / 2 - box[1] - box[3] / 2) > Math.max(box[3], height) * 0.35) continue;
      if (!(height / box[3] >= 0.55 && height / box[3] <= 1.8)) continue;
      row.push(other);
      used.add(otherIndex);
      right = xx + width;
    }
    let left = Number.POSITIVE_INFINITY;
    let top = Number.POSITIVE_INFINITY;
    let rowRight = Number.NEGATIVE_INFINITY;
    let bottom = Number.NEGATIVE_INFINITY;
    for (const [x, y, width, height] of row) {
      left = Math.min(left, x);
      top = Math.min(top, y);
      rowRight = Math.max(rowRight, x + width);
      bottom = Math.max(bottom, y + height);
    }
    if (rowRight > left && bottom > top) {
      const ratio = (rowRight - left) / Math.max(1, bottom - top);
      if (ratio >= 0.45 && ratio <= 4) rows.push([left + x0, top + y0, rowRight + x0, bottom + y0]);
    }
  }
  rows.sort((left, right) => {
    const leftDistance = Math.hypot((left[0] + left[2]) / 2 - point[0], (left[1] + left[3]) / 2 - point[1]);
    const rightDistance = Math.hypot((right[0] + right[2]) / 2 - point[0], (right[1] + right[3]) / 2 - point[1]);
    return leftDistance - rightDistance;
  });
  return rows.slice(0, 8);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Recognize control numbers on the purple course ink.
 *
 * Mutates `controls` in place (label/labelSource/labelConfidence/
 * labelPixelBox/labelAmbiguous) exactly like `dirmap/ocr.py::label_controls`,
 * and never throws: on any OCR failure it returns `available:false`.
 */
export async function labelControls(
  purple: MaskImage,
  controls: AnalyzedControl[],
  textInk?: MaskImage,
): Promise<OcrResult> {
  if (!controls.length) return { available: false, recognized: 0, method: METHOD };
  let worker: TesseractWorker;
  try {
    worker = await getWorker();
  } catch {
    return { available: false, recognized: 0, method: METHOD };
  }
  try {
    const eligible = new Set<string>();
    for (const control of controls) {
      if ((control.role ?? 'control') === 'control' && control.state !== 'accepted'
        && (control.labelSource === 'unread' || control.labelSource === 'tesseract')) {
        eligible.add(control.id);
      }
    }
    // A fresh OCR pass must not count labels left over from an earlier run;
    // explicit teacher labels and accepted targets are kept (ocr.py:24-30).
    controls.forEach((control, index) => {
      if (!eligible.has(control.id)) return;
      control.label = `待核${index + 1}`;
      control.labelSource = 'unread';
      delete control.labelConfidence;
      delete control.labelPixelBox;
      delete control.labelAmbiguous;
    });

    const width = purple.width;
    const height = purple.height;
    const text = textInk ?? purple;
    // Saturated course text separates bold numbers from pale restriction
    // hatching; the broad purple mask stays for terrain/circles (ocr.py:32-42).
    const strokes = courseInkStrokes(purple, controls);
    const clean = copyMask(text);
    const broadClean = copyMask(purple);
    const strokesRemoved: MaskImage = { data: new Uint8Array(text.data.length), width, height };
    const broadStrokesRemoved: MaskImage = { data: new Uint8Array(purple.data.length), width, height };
    for (let index = 0; index < strokesRemoved.data.length; index++) {
      strokesRemoved.data[index] = strokes[index] ? 0 : text.data[index];
      broadStrokesRemoved.data[index] = strokes[index] ? 0 : purple.data[index];
    }
    for (const control of controls) {
      circle(clean, [Math.round(control.pixel[0]), Math.round(control.pixel[1])], Math.round(control.radiusPx * 1.2), -1, 0);
      circle(broadClean, [Math.round(control.pixel[0]), Math.round(control.pixel[1])], Math.round(control.radiusPx * 1.2), -1, 0);
    }

    const recognize = async (
      mask: MaskImage,
      offset: [number, number] = [0, 0],
      scale = 1,
      psm = 11,
    ): Promise<Word[]> => {
      // ocr.py:46-71: white border, optional cubic upscale, digit whitelist.
      const padded = copyMakeBorder(maskNot(mask), BORDER, 255);
      let input = padded;
      if (scale !== 1) input = resizeByScale(padded, scale, scale, getCv().INTER_CUBIC);
      const png = encodeGrayPng(input.data, input.width, input.height);
      let tsv = '';
      try {
        const result = await worker.recognize(
          imagePayload(png),
          { tessedit_char_whitelist: '0123456789-', tessedit_pageseg_mode: String(psm) },
          { text: true, tsv: true },
        );
        tsv = result?.data?.tsv ?? '';
      } catch {
        return [];
      }
      const words = parseTsv(tsv);
      for (const word of words) {
        word.center = [offset[0] - BORDER + word.center[0] / scale, offset[1] - BORDER + word.center[1] / scale];
        word.box = [
          offset[0] - BORDER + word.box[0] / scale,
          offset[1] - BORDER + word.box[1] / scale,
          offset[0] - BORDER + word.box[2] / scale,
          offset[1] - BORDER + word.box[3] / scale,
        ];
      }
      return words;
    };

    let globalWords = await recognize(clean);
    if (!masksEqual(clean, broadClean)) globalWords = globalWords.concat(await recognize(broadClean));

    for (const control of controls) {
      if (!eligible.has(control.id)) continue;
      const point: [number, number] = [control.pixel[0], control.pixel[1]];
      const radius = control.radiusPx;
      let words = globalWords.slice();
      const nearby = words.filter(word => Math.hypot(word.center[0] - point[0], word.center[1] - point[1]) < radius * 4.8);
      if (!nearby.some(word => word.confidence > 90)) {
        for (const factor of [3, 4.5]) {
          const x0 = Math.max(0, Math.floor(point[0] - radius * factor));
          const y0 = Math.max(0, Math.floor(point[1] - radius * factor));
          const x1 = Math.min(width, Math.ceil(point[0] + radius * factor));
          const y1 = Math.min(height, Math.ceil(point[1] + radius * factor));
          if (!(x1 > x0 && y1 > y0)) continue;
          words = words.concat(await recognize(sliceMask(clean, x0, y0, x1, y1), [x0, y0], 2));
        }
        const x0 = Math.max(0, Math.floor(point[0] + radius * 0.35));
        const y0 = Math.max(0, Math.floor(point[1] + radius * -3.6));
        const x1 = Math.min(width, Math.ceil(point[0] + radius * 3.2));
        const y1 = Math.min(height, Math.ceil(point[1] + radius * -0.35));
        if (x1 > x0 && y1 > y0) {
          words = words.concat(await recognize(sliceMask(clean, x0, y0, x1, y1), [x0, y0], 3, 7));
        }
        // A numeral often touches a circle. Erasing the whole disk destroys
        // that numeral, so retry actual glyph rows from the original ink and
        // stroke-only masks with full geometry (ocr.py:119-131).
        for (const mask of [text, strokesRemoved, purple, broadStrokesRemoved]) {
          if (x1 > x0 && y1 > y0) {
            const slice = sliceMask(mask, x0, y0, x1, y1);
            words = words.concat(await recognize(slice, [x0, y0], 4, 7));
            const opened = morphology(slice, 'open', 2);
            words = words.concat(await recognize(opened, [x0, y0], 4, 11));
          }
          for (const [left, top, right, bottom] of glyphRows(mask, point, radius)) {
            if (!(right > left && bottom > top)) continue;
            const scale = Math.max(2, 48 / (bottom - top));
            words = words.concat(await recognize(sliceMask(mask, left, top, right, bottom), [left, top], scale, 7));
          }
        }
      }
      interface Option {
        score: number;
        label: string;
        confidence: number;
        box: [number, number, number, number];
      }
      const options: Option[] = [];
      const numberControls = controls.filter(candidate => (candidate.role ?? 'control') === 'control');
      for (const word of words) {
        const distance = Math.hypot(word.center[0] - point[0], word.center[1] - point[1]);
        if (distance > radius * 4.8) continue;
        let closestId: string | null = null;
        let closestScore = Number.POSITIVE_INFINITY;
        for (const candidate of numberControls) {
          const score = Math.hypot(word.center[0] - candidate.pixel[0], word.center[1] - candidate.pixel[1]) / candidate.radiusPx;
          if (score < closestScore) {
            closestScore = score;
            closestId = candidate.id;
          }
        }
        if (closestId !== control.id) continue;
        options.push({ score: (100 - word.confidence) / 8 + distance / Math.max(1, radius), label: word.label, confidence: word.confidence, box: word.box });
      }
      if (options.length) {
        options.sort((left, right) => left.score - right.score || left.label.localeCompare(right.label));
        const best = options[0];
        control.label = best.label;
        control.labelSource = 'tesseract';
        control.labelConfidence = Math.round(best.confidence) / 100;
        control.labelPixelBox = best.box;
      }
    }

    // ocr.py:146-154 — duplicate labels are ambiguous and stay reviewable.
    const counts = new Map<string, number>();
    for (const control of controls) counts.set(control.label, (counts.get(control.label) ?? 0) + 1);
    for (const control of controls) if ((counts.get(control.label) ?? 0) > 1) control.labelAmbiguous = true;

    let recognized = 0;
    let preserved = 0;
    for (const control of controls) {
      if (control.labelSource === 'tesseract' && eligible.has(control.id)) recognized++;
      if ((control.role ?? 'control') === 'control' && !eligible.has(control.id)) preserved++;
    }
    return {
      available: true,
      recognized,
      preserved,
      method: METHOD,
    };
  } catch {
    // Any unexpected tesseract.js/TSV shape problem must not fail the analysis.
    return { available: false, recognized: 0, method: METHOD };
  }
}

function truncateNum(value: string | number): number {
  return typeof value === 'number' ? Math.trunc(value) : 0;
}
