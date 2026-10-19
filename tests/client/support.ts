/** Shared Node helpers for the client-side processing goldens.
 *
 * The browser pipeline runs the same TypeScript modules as these tests; the
 * only Node-specific parts are PNG decoding (pngjs instead of
 * `createImageBitmap`/`OffscreenCanvas`, which keeps the pixels identical) and
 * file access under the repository root.
 *
 * Golden priority (docs/static-architecture.md Addendum C.1/C.5):
 * 1. `tests/fixtures/static/client-processing.json` — `referenceImplementation`
 *    is the exact CURRENT Python `dirmap.vision.analyze` output on
 *    `original.png` (primary golden); `artifact` holds the older bake outputs
 *    (tolerance-only golden).
 * 2. Workspace fallback (gitignored files only, for a partially baked tree):
 *    `data/projects/<id>/state.json` controls and
 *    `derived/<hash>/blocked.png` masks.
 *
 * Dependency policy: Node builtins + `pngjs` only (dev dependency). Nothing here
 * imports the DOM or touches the network.
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import type { MaskImage, RgbaImage } from '../../web/src/cv.ts';

/** Repository root (two levels above `tests/client/`). */
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

export function testPath(relativePath: string): string {
  return isAbsolute(relativePath) ? relativePath : resolve(ROOT, relativePath);
}

export function fileExists(relativePath: string): boolean {
  return existsSync(testPath(relativePath));
}

export function readJson<T>(relativePath: string): T {
  return JSON.parse(readFileSync(testPath(relativePath), 'utf8')) as T;
}

/** Decode a PNG into the RGBA image shape the browser pipeline consumes. */
export function decodePng(path: string): RgbaImage {
  const png = PNG.sync.read(readFileSync(testPath(path)));
  return { data: new Uint8ClampedArray(png.data), width: png.width, height: png.height };
}

/** Decode a grayscale/bitmap PNG (e.g. `blocked.png`) into a 0/255 mask. */
export function decodeMaskPng(path: string): MaskImage {
  const png = PNG.sync.read(readFileSync(testPath(path)));
  const count = png.width * png.height;
  const data = new Uint8Array(count);
  for (let index = 0; index < count; index++) data[index] = png.data[index * 4] > 127 ? 255 : 0;
  return { data, width: png.width, height: png.height };
}

/**
 * Native scan path: the committed lossless `original.png` made by the baker,
 * then a developer-local `.cache/cv-port/<id>-original.png` fallback (produced
 * by `cv2.imdecode` of the repo JPEG for local iteration before the bake).
 */
export function findMapImagePath(assetId: string): string | null {
  for (const candidate of [
    `web/public/data/maps/${assetId}/original.png`,
    `.cache/cv-port/${assetId}-original.png`,
  ]) {
    if (fileExists(candidate)) return candidate;
  }
  return null;
}

export function loadMapImage(assetId: string): RgbaImage | null {
  const path = findMapImagePath(assetId);
  return path ? decodePng(path) : null;
}

// ---------------------------------------------------------------------------
// Frozen golden fixture (task-9 bake)
// ---------------------------------------------------------------------------

export interface GoldenControl {
  id: string;
  label: string;
  role?: 'control' | 'start' | 'finish' | 'start_finish';
  pixel: [number, number];
  radiusPx: number;
  state: string;
  labelSource?: string;
}

export interface GoldenSamples {
  blocked: { x: number; y: number }[];
  costs: { x: number; y: number }[];
}

export interface ReferenceImplementation {
  generatedBy?: string;
  sourceCodeHash?: string;
  controls: GoldenControl[];
  featureCounts: Record<string, number>;
  statistics: Record<string, unknown>;
  forbiddenPixelFraction: number;
  blockedSampleValues: number[];
  costsSampleValues: number[];
  note?: string;
}

export interface ArtifactGolden {
  featureCounts: Record<string, number>;
  statistics: Record<string, unknown>;
  forbiddenPixelFraction: number;
  blockedSampleValues: number[];
  costsSampleValues: number[];
  note?: string;
}

export interface GoldenMap {
  assetId: string;
  width: number;
  height: number;
  controls: GoldenControl[];
  samples: GoldenSamples;
  referenceImplementation: ReferenceImplementation;
  artifact?: ArtifactGolden;
}

export interface ClientProcessingFixture {
  sampling?: { blockedSamples?: string; costsSamples?: string; lcg?: string };
  note?: string;
  maps: GoldenMap[];
}

const FIXTURE_PATH = 'tests/fixtures/static/client-processing.json';

export function loadClientProcessingFixture(): ClientProcessingFixture | null {
  if (!fileExists(FIXTURE_PATH)) return null;
  return readJson<ClientProcessingFixture>(FIXTURE_PATH);
}

export function fixtureMap(fixture: ClientProcessingFixture | null, assetId: string): GoldenMap | undefined {
  return fixture?.maps.find(entry => entry.assetId === assetId);
}

// ---------------------------------------------------------------------------
// Workspace fallback goldens (gitignored, only used when the frozen fixture is
// missing; keeps the test meaningful in a partially baked working tree)
// ---------------------------------------------------------------------------

export function loadBakedState(assetId: string): Record<string, unknown> | null {
  for (const candidate of [
    `web/public/data/maps/${assetId}/state.json`,
    `data/projects/${assetId}/state.json`,
  ]) {
    if (fileExists(candidate)) return readJson<Record<string, unknown>>(candidate);
  }
  return null;
}

/**
 * Baked `blocked.png` from the workspace artefacts (top-level or derived/).
 *
 * Simplification vs the task brief: `terrain.npz` is not parsed here — the
 * committed fixture already carries the baked cost sample values, and a
 * dependency-free NPZ reader (zip + npy) is not justified for a fallback that
 * only exists for a partially baked tree. The fallback therefore checks blocked
 * pixels only; cost parity is covered by `client-processing.json`.
 */
export function loadArtifactBlockedMask(assetId: string): MaskImage | null {
  const direct = `data/projects/${assetId}/blocked.png`;
  if (fileExists(direct)) return decodeMaskPng(direct);
  const derivedRoot = `data/projects/${assetId}/derived`;
  if (!fileExists(derivedRoot)) return null;
  for (const entry of readdirSync(testPath(derivedRoot))) {
    const candidate = `${derivedRoot}/${entry}/blocked.png`;
    if (fileExists(candidate)) return decodeMaskPng(candidate);
  }
  return null;
}

/** Campus reference features in local metres (registration tests). */
export function loadReferenceFeatures(): {
  frame: { crs: string; origin: [number, number] };
  features: { id: string; kind: string; center: [number, number]; area: number; aspect: number; points: [number, number][] }[];
} | null {
  const path = 'web/public/data/reference-features.json';
  return fileExists(path) ? readJson(path) : null;
}

// ---------------------------------------------------------------------------
// Deterministic sampling (same LCG as the baker's fixture generator)
// ---------------------------------------------------------------------------

/** `x_{n+1} = (1664525*x_n + 1013904223) mod 2^32`, then flat = x mod count. */
export function samplePoints(width: number, height: number, count: number, seed = 0x9e3779b9): [number, number][] {
  const total = width * height;
  const points: [number, number][] = [];
  let state = seed >>> 0;
  for (let index = 0; index < count; index++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const flat = state % total;
    points.push([flat % width, Math.floor(flat / width)]);
  }
  return points;
}

/** Deterministic mulberry32 generator (readable PRNG for test data). */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Compare a client mask with golden sample values at the shared coordinates. */
export function maskSampleAgreement(
  clientMask: Uint8Array,
  width: number,
  points: { x: number; y: number }[],
  expectedValues: number[],
): number {
  let agree = 0;
  for (let index = 0; index < points.length; index++) {
    const { x, y } = points[index];
    const value = clientMask[y * width + x];
    const expected = expectedValues[index] > 127 ? 255 : 0;
    if (value === expected) agree++;
  }
  return points.length ? agree / points.length : 1;
}

export interface NearestControl<T> {
  control: T | null;
  distance: number;
}

/** Nearest client control to a golden pixel (native pixels, Euclidean). */
export function nearestControl<T extends { pixel: [number, number] }>(controls: T[], pixel: [number, number]): NearestControl<T> {
  let best: T | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const control of controls) {
    const dx = control.pixel[0] - pixel[0];
    const dy = control.pixel[1] - pixel[1];
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = control;
    }
  }
  return { control: best, distance: bestDistance };
}
