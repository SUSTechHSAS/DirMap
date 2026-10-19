/** Golden tests for the browser image-registration pipeline (`dirmap/registration.py` port).
 *
 * Coverage:
 *  - manual anchor calibration: exact-affine recovery, validation holdout,
 *    Python error messages for conflicting/collinear/duplicate/outside anchors;
 *  - warp overlay: EPSG:3857 extent identical to the baked artefact, <=1700 px
 *    long side (bilinear) / <=1200 px (nearest), non-empty alpha, timings;
 *  - deterministic custom RANSAC affine fit;
 *  - approxPolyDP / outline IoU / principal angle parity probes against cv2;
 *  - Delaunay + regularized Laplacian registration mesh: round-trip, fold,
 *    duplicate and invalid-matrix rejection;
 *  - `fitObservedLandmarks` object-grouped holdout (direct port of the Python
 *    registration test);
 *  - `extendTransferBoundary` observed-boundary extension (Python test port);
 *  - `registerToMap` cross-map feature transfer (synthetic base map);
 *  - `autoRegister` on a synthetic map (no vision) and on the real szu-1 sheet
 *    against the baked state goldens;
 *  - heavy `DIRMAP_HEAVY_TESTS=1`: szu-2 inherits from szu-1 or falls back to a
 *    direct registration (documented limitation of the SIFT-free transfer).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadOpenCV } from '../../web/src/cv.ts';
import { analyze } from '../../web/src/vision.ts';
import { Mesh, frameLngLat, localScales } from '../../web/src/geometry.ts';
import type { RegistrationData } from '../../web/src/geometry.ts';
import {
  approxPolyDPClosed, autoRegister, buildRegistration, corners, delaunayTriangles,
  extendTransferBoundary, fitAffine, fitObservedLandmarks, manualRegister, Matcher,
  outlineIou, principalAngle, projectControls, registerToMap, warpOverlay,
} from '../../web/src/image-registration.ts';
import type {
  ReferenceFeature, RegistrationQuality, RgbaImage, SourceFeature,
} from '../../web/src/image-registration.ts';
import type { Anchor, Control, Pixel } from '../../web/src/types.ts';
import {
  decodePng, loadBakedState, loadMapImage, loadReferenceFeatures,
} from './support.ts';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

interface BakedState {
  assetId: string;
  width: number;
  height: number;
  frame: { crs: string; origin: Pixel };
  extent: number[];
  controls: Control[];
  quality: RegistrationQuality;
  registration: RegistrationData;
}

const ORIGIN: Pixel = [113.9304, 22.535];
/** Affine used by the synthetic manual-calibration tests (pixel -> local metres). */
const KNOWN_AFFINE: number[][] = [[0.6, 0.02, 300], [-0.01, -0.6, 900]];

function applyAffine(matrix: number[][], point: Pixel): Pixel {
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2],
    matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2],
  ];
}

function metresBetween(first: Pixel, second: Pixel, origin: Pixel = ORIGIN): number {
  const scales = localScales(origin);
  return Math.hypot((first[0] - second[0]) * scales[0], (first[1] - second[1]) * scales[1]);
}

function bakedState(assetId: string): BakedState {
  const state = loadBakedState(assetId);
  assert.ok(state, `baked state for ${assetId} must exist (web/public/data/maps/${assetId}/state.json)`);
  return state as unknown as BakedState;
}

function opaqueImage(width: number, height: number, colour: [number, number, number] = [30, 60, 90]): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index++) {
    data[index * 4] = colour[0];
    data[index * 4 + 1] = colour[1];
    data[index * 4 + 2] = colour[2];
    data[index * 4 + 3] = 255;
  }
  return { data, width, height };
}

let openCvReady: Promise<unknown> | null = null;
async function ensureOpenCv(): Promise<void> {
  if (!openCvReady) openCvReady = loadOpenCV();
  await openCvReady;
}

/** Analyze each asset once per process (Node tests share it within this file). */
const analysisCache = new Map<string, Awaited<ReturnType<typeof analyze>>>();
async function analysisFor(assetId: string): Promise<Awaited<ReturnType<typeof analyze>>> {
  const cached = analysisCache.get(assetId);
  if (cached) return cached;
  const image = loadMapImage(assetId);
  assert.ok(image, `original.png for ${assetId} must exist`);
  await ensureOpenCv();
  const analysis = await analyze(image as RgbaImage, { labeler: null });
  analysisCache.set(assetId, analysis);
  return analysis;
}

function referenceFeatures(): ReferenceFeature[] {
  const reference = loadReferenceFeatures();
  assert.ok(reference, 'web/public/data/reference-features.json must exist');
  return reference.features as ReferenceFeature[];
}

function syntheticAnchor(pixel: Pixel, purpose: 'fit' | 'validation' = 'fit'): Anchor {
  const local = applyAffine(KNOWN_AFFINE, pixel);
  return { pixel, lnglat: frameLngLat(local, ORIGIN), purpose };
}

function expectRejects(promise: Promise<unknown>, pattern: RegExp): Promise<void> {
  return assert.rejects(promise as Promise<unknown>, (error: unknown) => {
    assert.ok(error instanceof Error, 'expected an Error instance');
    assert.match(error.message, pattern);
    return true;
  });
}

const close = (actual: number, expected: number, tolerance: number, message?: string): void => {
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${message ?? 'value'} ${actual} vs ${expected} (tolerance ${tolerance})`);
};

const angleModPi = (angle: number): number => {
  const wrapped = ((angle % Math.PI) + Math.PI) % Math.PI;
  return Math.min(wrapped, Math.PI - wrapped);
};

// ---------------------------------------------------------------------------
// manualRegister
// ---------------------------------------------------------------------------

test('manualRegister recovers a known affine and honours the validation holdout', async () => {
  const fitPixels: Pixel[] = [[200, 300], [1200, 250], [900, 1500], [350, 1700], [1400, 1100]];
  const anchors = fitPixels.map(pixel => syntheticAnchor(pixel));
  // Validation points strictly inside the convex hull of the fit anchors.
  const validationPixels: Pixel[] = [[700, 800], [1000, 950]];
  const result = await manualRegister([...anchors, ...validationPixels.map(pixel => syntheticAnchor(pixel, 'validation'))]);
  const quality = result.quality;
  assert.equal(quality.method, 'reviewed_landmarks_mesh');
  assert.equal(quality.fitCount, fitPixels.length);
  assert.equal(quality.validationCount, validationPixels.length);
  assert.equal(quality.foldedTriangles, 0);
  assert.ok((quality.validationScope ?? '').includes('教师录入的独立核查点'));
  assert.ok(quality.p95ErrorM !== null && quality.p95ErrorM! < 1e-6, `exact affine should reproduce anchors (p95 ${quality.p95ErrorM})`);
  const mesh = new Mesh(result.registration);
  assert.ok(mesh.hasMesh);
  assert.equal(result.registration.source.length, fitPixels.length);
  for (let index = 0; index < fitPixels.length; index++) {
    const mapped = mesh.forward(fitPixels[index]);
    const expected = applyAffine(KNOWN_AFFINE, fitPixels[index]);
    assert.ok(mapped.inside, `fit anchor ${index} must be inside the measured hull`);
    close(metresBetween(mapped.local, expected), 0, 0.5, `fit anchor ${index} error`);
  }
  // Validation-only calibration reports null accuracy statistics, like Python.
  const validationOnly = await manualRegister(anchors);
  assert.equal(validationOnly.quality.validationCount, 0);
  assert.equal(validationOnly.quality.medianErrorM, null);
  assert.equal(validationOnly.quality.p95ErrorM, null);
  // Controls are projected through the mesh (calibrated true inside the hull).
  const controls = fitPixels.map((pixel, index): Control => ({
    id: `c${index + 1}`, label: String(index + 1), pixel, radiusPx: 10, required: true,
    state: 'candidate', serviceBaseSec: 0, servicePerPersonSec: 0,
  }));
  projectControls(controls, result.registration);
  assert.ok(controls.every(control => control.calibrated === true));
  // A null registration (no calibration yet) must leave controls untouched (Python parity).
  const uncalibrated: Control = {
    id: 'c0', label: '0', pixel: [10, 10], radiusPx: 5, required: false,
    state: 'candidate', calibrated: false, lnglat: null, serviceBaseSec: 0, servicePerPersonSec: 0,
  };
  projectControls([uncalibrated], null);
  assert.equal(uncalibrated.calibrated, false);
  assert.equal(uncalibrated.lnglat, null);
});

test('manualRegister rejects invalid anchors with the Python Chinese messages', async () => {
  // Fewer than three correspondences.
  await expectRejects(manualRegister([syntheticAnchor([100, 100]), syntheticAnchor([400, 300])]), /至少需要三组有效的同名点/);
  // Collinear pixel anchors.
  await expectRejects(
    manualRegister([syntheticAnchor([100, 100]), syntheticAnchor([300, 300]), syntheticAnchor([500, 500])]),
    /配准点不能共线/);
  // One anchor conflicts with the others.
  const conflicting = [syntheticAnchor([200, 300]), syntheticAnchor([1200, 250]), syntheticAnchor([900, 1500])];
  conflicting.push({ pixel: [350, 1700], lnglat: frameLngLat([applyAffine(KNOWN_AFFINE, [350, 1700])[0] + 150,
    applyAffine(KNOWN_AFFINE, [350, 1700])[1]], ORIGIN), purpose: 'fit' });
  await expectRejects(manualRegister(conflicting), /手动同名点中存在冲突，请检查坐标和对应关系/);
  // Duplicate measured points cannot form a mesh.
  const duplicate = syntheticAnchor([200, 300]);
  await expectRejects(
    manualRegister([duplicate, { ...duplicate, pixel: [200, 300] }, syntheticAnchor([1200, 250]), syntheticAnchor([900, 1500])]),
    /配准点不能重复/);
  // Validation point outside the measured hull.
  await expectRejects(
    manualRegister([...fitAnchors(), syntheticAnchor([5000, 5000], 'validation')]),
    /核查点超出配准点包围范围/);
  // Invalid geographic coordinates.
  await expectRejects(
    manualRegister([{ pixel: [100, 100], lnglat: [200, 100], purpose: 'fit' },
      syntheticAnchor([400, 300]), syntheticAnchor([700, 200])]),
    /经纬度无效/);
});

function fitAnchors(): Anchor[] {
  return [[200, 300], [1200, 250], [900, 1500], [350, 1700], [1400, 1100]].map(pixel => syntheticAnchor(pixel));
}

// ---------------------------------------------------------------------------
// warpOverlay
// ---------------------------------------------------------------------------

test('warpOverlay matches the baked szu-1 EPSG:3857 extent and size contract', async (t) => {
  const state = bakedState('szu-1');
  const image = opaqueImage(state.width, state.height);
  let started = performance.now();
  const overlay = warpOverlay(image, state.registration);
  const bilinearMs = performance.now() - started;
  assert.ok(Math.max(overlay.width, overlay.height) <= 1700,
    `bilinear long side ${Math.max(overlay.width, overlay.height)} must stay <= 1700`);
  assert.ok(overlay.extent.every(Number.isFinite));
  const extentTolerance = 1e-6;
  for (let index = 0; index < 4; index++) {
    const reference = state.extent[index];
    assert.ok(Math.abs(overlay.extent[index] - reference) <= extentTolerance * Math.max(1, Math.abs(reference)),
      `extent[${index}] ${overlay.extent[index]} vs baked ${reference}`);
  }
  // The synthetic image is opaque, so calibrated pixels must carry alpha 255 and its colour.
  let alphaPixels = 0;
  let colourPixels = 0;
  for (let index = 0; index < overlay.width * overlay.height; index++) {
    if (overlay.rgba[index * 4 + 3] === 255) {
      alphaPixels++;
      if (overlay.rgba[index * 4] === 30 && overlay.rgba[index * 4 + 1] === 60 && overlay.rgba[index * 4 + 2] === 90) colourPixels++;
    }
  }
  assert.ok(alphaPixels > overlay.width * overlay.height * 0.1,
    `warped overlay must have a substantial calibrated hull (${alphaPixels} alpha pixels)`);
  assert.ok(colourPixels > alphaPixels * 0.9, 'calibrated pixels keep the sampled colour');
  // Nearest/terrain variant: 1200 px long side, same geographic extent.
  started = performance.now();
  const terrain = warpOverlay(image, state.registration, { maxSide: 1200, nearest: true });
  const nearestMs = performance.now() - started;
  assert.ok(Math.max(terrain.width, terrain.height) <= 1200,
    `nearest long side ${Math.max(terrain.width, terrain.height)} must stay <= 1200`);
  for (let index = 0; index < 4; index++) {
    const reference = state.extent[index];
    assert.ok(Math.abs(terrain.extent[index] - reference) <= extentTolerance * Math.max(1, Math.abs(reference)));
  }
  let terrainAlpha = 0;
  for (let index = 3; index < terrain.rgba.length; index += 4) if (terrain.rgba[index] === 255) terrainAlpha++;
  assert.ok(terrainAlpha > 1000, 'nearest terrain warp has calibrated pixels');
  t.diagnostic(`warpOverlay sizes: bilinear ${overlay.width}x${overlay.height} ${bilinearMs.toFixed(0)} ms, `
    + `nearest ${terrain.width}x${terrain.height} ${nearestMs.toFixed(0)} ms (node)`);
  t.diagnostic(`warp extent ${overlay.extent.map(value => value.toFixed(3)).join(', ')} `
    + `(baked ${state.extent.map(value => value.toFixed(3)).join(', ')})`);
  // Generous guard rail for regressions (browser is measured separately by e2e);
  // node timing variance on a loaded CI machine is large, the contract is the output.
  assert.ok(bilinearMs < 30000, `bilinear warp ${bilinearMs.toFixed(0)} ms must stay < 30000 ms`);
});

// ---------------------------------------------------------------------------
// fitAffine (deterministic RANSAC)
// ---------------------------------------------------------------------------

test('fitAffine is a deterministic seeded RANSAC with least-squares refinement', () => {
  const trueAffine = [[0.6, 0.05, 100], [-0.02, -0.58, 900]];
  const random = (() => {
    let state = 1234567 >>> 0;
    return () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x100000000;
    };
  })();
  const source: Pixel[] = [];
  const target: Pixel[] = [];
  for (let index = 0; index < 120; index++) {
    const point: Pixel = [random() * 1500, random() * 1800];
    const expected = applyAffine(trueAffine, point);
    source.push(point);
    target.push([expected[0] + (random() - 0.5) * 1.5, expected[1] + (random() - 0.5) * 1.5]);
  }
  for (let index = 0; index < 15; index++) {
    source.push([random() * 1500, random() * 1800]);
    target.push([random() * 1000, random() * 1200]);
  }
  const fit = fitAffine(source, target, 5);
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 3; column++) {
      // Linear terms are tightly constrained; the translation absorbs the uniform
      // +/-0.75 px measurement noise of 120 points (~0.05 m standard error).
      const tolerance = column < 2 ? 0.02 : 0.2;
      close(fit.matrix[row][column], trueAffine[row][column], tolerance, `matrix[${row}][${column}]`);
    }
  }
  const inlierCount = fit.inliers.filter(Boolean).length;
  assert.ok(inlierCount >= 118, `RANSAC should keep the true correspondences (${inlierCount}/120)`);
  assert.equal(fit.inliers.length, source.length);
  // Same seed -> same result (repeatable, unlike opencv.js RNG).
  const again = fitAffine(source, target, 5);
  assert.deepEqual(again.matrix, fit.matrix);
  assert.deepEqual(again.inliers, fit.inliers);
  // Python error messages.
  assert.throws(() => fitAffine(source.slice(0, 2), target.slice(0, 2)), /至少需要三组有效的同名点/);
  assert.throws(() => fitAffine([[0, 0], [10, 10], [20, 20]], [[0, 0], [10, 10], [20, 20]]), /配准点不能共线/);
});

// ---------------------------------------------------------------------------
// OpenCV routine parity probes
// ---------------------------------------------------------------------------

test('approxPolyDP, corners and outline IoU match the OpenCV reference behaviour', () => {
  // cv2.approxPolyDP(closed=True, eps=0.5) on a rectangle sampled along its edges.
  const rectangleSamples: Pixel[] = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2],
    [3, 2.3], [2, 2], [1, 2], [0, 2], [0, 1]];
  assert.deepEqual(approxPolyDPClosed(rectangleSamples, 0.5), [[0, 0], [4, 0], [4, 2], [0, 2]]);
  // Only real corners survive the cosine filter.
  assert.deepEqual(corners([[0, 0], [3, 0], [6, 0], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]], 1.3),
    [[0, 0], [10, 0], [10, 10], [0, 10]]);
  // Rasterised IoU probes; the native scanline rasteriser matches cv2.fillPoly
  // within a couple of boundary pixels (deviation documented in the module doc).
  assert.ok(Math.abs(outlineIou([[1, 1], [8, 2], [3, 9]], [[3, 1], [10, 2], [5, 9]]) - 0.403509) < 0.05);
  assert.ok(Math.abs(outlineIou([[2, 2], [9, 2], [9, 7], [2, 7]], [[4, 3], [11, 3], [11, 8], [4, 8]]) - 0.454545) < 0.05);
  assert.ok(Math.abs(outlineIou([[1, 1], [9, 3], [7, 8], [3, 9], [0, 5]], [[2, 2], [9, 4], [6, 8], [3, 10], [0, 5]]) - 0.753846) < 0.05);
  assert.equal(outlineIou([[0, 0], [20, 0], [20, 5], [0, 5]], [[0, 0], [20, 0], [20, 5], [0, 5]]), 1);
  assert.equal(outlineIou([[0, 0], [10, 0], [10, 10]], [[100, 100], [110, 100], [110, 110]]), 0);
  // Principal angle: rotated rectangle axis (mod pi, cv2 moments-based).
  const angle = Math.PI / 6;
  const rotated = [[0, 0], [20, 0], [20, 5], [0, 5]].map(([x, y]) =>
    [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)] as Pixel);
  close(angleModPi(principalAngle(rotated)), angleModPi(angle), 1e-9);
  assert.equal(principalAngle([[0, 0], [1, 0]]), 0);
});

// ---------------------------------------------------------------------------
// Delaunay / registration mesh construction
// ---------------------------------------------------------------------------

test('registration mesh is valid, invertible and rejects folded/duplicate/invalid inputs', () => {
  const random = (() => {
    let state = 82 >>> 0;
    return () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x100000000;
    };
  })();
  const source: Pixel[] = [];
  for (let y = 0; y < 9; y++) for (let x = 0; x < 11; x++) source.push([x * 30, y * 30]);
  const affine = [[1.1, 0.07, 10], [0.04, -0.95, 220]];
  const target = source.map(point => applyAffine(affine, point));
  for (let index = 0; index < source.length; index++) {
    const x = source[index][0];
    const y = source[index][1];
    if (x > 0 && x < 300 && y > 0 && y < 240) {
      target[index][0] += (random() * 2 - 1) * 6;
      target[index][1] += (random() * 2 - 1) * 6;
    }
  }
  const built = buildRegistration(affine, source, target, 0);
  assert.equal(built.folded, 0);
  const mesh = new Mesh(built.registration);
  assert.ok(mesh.hasMesh);
  for (let index = 0; index < source.length; index++) {
    const forward = mesh.forward(source[index]);
    assert.ok(forward.inside, `grid vertex ${index} triangulated`);
    close(forward.local[0], target[index][0], 1e-9);
    close(forward.local[1], target[index][1], 1e-9);
    const inverse = mesh.inverse(target[index]);
    assert.ok(inverse !== null);
    close(inverse![0], source[index][0], 1e-9);
    close(inverse![1], source[index][1], 1e-9);
  }
  // Delaunay triangles partition the convex hull without overlap.
  const triangles = delaunayTriangles(source);
  let areaSum = 0;
  for (const [a, b, c] of triangles) {
    areaSum += Math.abs((source[b][0] - source[a][0]) * (source[c][1] - source[a][1])
      - (source[b][1] - source[a][1]) * (source[c][0] - source[a][0])) / 2;
  }
  close(areaSum, 300 * 240, 1e-6, 'triangulated hull area');
  // Fold detection (Python message).
  const foldedTarget = target.map(point => [point[0], point[1]] as Pixel);
  const interior = source.findIndex(point => point[0] === 150 && point[1] === 120);
  foldedTarget[interior] = [foldedTarget[interior][0] + 400, foldedTarget[interior][1]];
  assert.throws(() => buildRegistration(affine, source, foldedTarget, 0), /局部配准出现折叠或边界重叠，请修正同名点/);
  // Duplicate measured points / invalid matrix (Python messages).
  assert.throws(() => buildRegistration(affine, [source[0], source[0], source[1]], [[0, 0], [1, 0], [2, 0]], 1),
    /配准点不能重复/);
  assert.throws(() => buildRegistration([[0, 0, 0], [0, 0, 0]], source.slice(0, 3), target.slice(0, 3), 1),
    /配准矩阵无效/);
  assert.throws(() => buildRegistration(affine, source.slice(0, 3), target.slice(0, 2), 1),
    /配准网格点无效/);
});

// ---------------------------------------------------------------------------
// fitObservedLandmarks (port of tests/test_registration.py)
// ---------------------------------------------------------------------------

test('fitObservedLandmarks keeps whole objects in one fit/holdout split', () => {
  const pairs = [];
  const extra = [];
  for (let row = 0; row < 5; row++) {
    for (let column = 0; column < 5; column++) {
      const key = `building-${row}-${column}`;
      const point: Pixel = [column * 30, row * 30];
      pairs.push({
        referenceId: key, kind: 'building' as const, pixel: point, local: [point[0], -point[1]] as Pixel,
        errorM: 0, shapeError: 0, sourceIndex: pairs.length,
      });
      for (const [dx, dy] of [[-4, -4], [4, -4], [4, 4], [-4, 4]]) {
        extra.push({
          referenceId: key, kind: 'building_corner', pixel: [point[0] + dx, point[1] + dy] as Pixel,
          local: [point[0] + dx, -(point[1] + dy)] as Pixel,
        });
      }
    }
  }
  const { registration, quality } = fitObservedLandmarks(pairs, extra);
  const fitIds = new Set((quality.landmarks ?? []).filter(landmark => landmark.purpose === 'fit').map(landmark => landmark.referenceId));
  const holdoutIds = new Set((quality.landmarks ?? []).filter(landmark => landmark.purpose === 'validation').map(landmark => landmark.referenceId));
  assert.ok(fitIds.size > 0 && holdoutIds.size > 0);
  assert.equal([...fitIds].filter(key => holdoutIds.has(key)).length, 0, 'objects never leak across the split');
  assert.equal(quality.validationInsideHull, quality.validationCount);
  assert.ok((quality.p95ErrorM ?? Infinity) < 1e-5);
  assert.equal(quality.affineFallback, false);
  const observed = new Set(extra.map(landmark => landmark.pixel!.join(',')));
  assert.ok(registration.source.every(point => observed.has(point.join(','))), 'fit sources are observed corners');
  const mesh = new Mesh(registration);
  assert.ok(mesh.forward([-4, -4]).inside);
  assert.ok(mesh.forward([124, 124]).inside);
  assert.equal(mesh.forward([-4.1, -4]).inside, false, 'outside the measured hull stays uncalibrated');
});

// ---------------------------------------------------------------------------
// extendTransferBoundary (port of tests/test_registration.py)
// ---------------------------------------------------------------------------

test('extendTransferBoundary extends only with observed fitting corners', async () => {
  const matrix = [[1, 0, 0], [0, -1, 100]];
  const source: Pixel[] = [[0, 0], [80, 0], [80, 80], [0, 80]];
  const target = source.map(point => applyAffine(matrix, point));
  const built = buildRegistration(matrix, source, target, 1.5);
  const landmarks = source.map((point, index) => ({ pixel: point, local: target[index], purpose: 'fit' as const }));
  landmarks.push({ pixel: [40, 40], local: [40, 60], purpose: 'validation' as const });
  const extra = [[100, 0], [100, 100], [0, 100]].map(point => ({
    pixel: point as Pixel, local: [point[0], 100 - point[1]] as Pixel,
    purpose: 'fit' as const, kind: 'building_corner',
  }));
  extra.push({ pixel: [-20, 0] as Pixel, local: [-20, 100] as Pixel, purpose: 'validation' as const, kind: 'building_corner' });
  const transferQuality = {
    state: 'needs_review', method: 'feature_cross_map_ransac_mesh', fitCount: 4, validationCount: 1,
    medianErrorM: 0, p95ErrorM: 0, validationScope: 'x', landmarks,
  };
  const extended = await extendTransferBoundary(built.registration, transferQuality as RegistrationQuality,
    { ...transferQuality, landmarks: extra } as RegistrationQuality);
  const mesh = new Mesh(extended.registration);
  assert.equal(extended.quality.geographicBoundaryAnchors, 3);
  assert.equal(extended.quality.method, 'feature_cross_map_observed_boundary_mesh');
  assert.ok(mesh.forward([95, 95]).inside, 'observed geographic corners extend the hull');
  assert.equal(mesh.forward([-1, 40]).inside, false, 'validation-only corners never extend the hull');
  assert.ok((extended.quality.p95ErrorM ?? Infinity) < 1e-6);
});

// ---------------------------------------------------------------------------
// registerToMap (SIFT-free cross-map transfer, synthetic base map)
// ---------------------------------------------------------------------------

test('registerToMap transfers a base registration through feature matching', async () => {
  const reference = referenceFeatures();
  // Fictitious base map: pixel == local for a grid covering the campus extent.
  const grid: Pixel[] = [];
  for (let x = -450; x <= 700; x += 50) for (let y = -650; y <= 750; y += 50) grid.push([x, y]);
  const baseBuilt = buildRegistration([[1, 0, 0], [0, 1, 0]], grid, grid, 0);
  const campusFeatures = reference.filter(feature =>
    feature.center[0] > -450 && feature.center[0] < 700 && feature.center[1] > -650 && feature.center[1] < 750);
  const baseFeatures: SourceFeature[] = campusFeatures.map(feature => ({
    kind: feature.kind, center: [...feature.center] as Pixel, area: feature.area, aspect: feature.aspect,
    points: feature.points.map(point => [...point] as Pixel),
  }));
  // Source sheet pixels differ by a rotation/reflection + scale and translation.
  const sourceAffine = [[-1.2, 0.02, 2600], [0.01, 1.2, 300]];
  const sourcePixel = (local: Pixel): Pixel => applyAffine(sourceAffine, local);
  const sourceFeatures: SourceFeature[] = baseFeatures.map(feature => ({
    kind: feature.kind, center: sourcePixel(feature.center), area: feature.area * 1.44,
    aspect: feature.aspect, points: feature.points.map(sourcePixel),
  }));
  const result = await registerToMap(sourceFeatures, {
    features: baseFeatures, registration: baseBuilt.registration, quality: {
      state: 'needs_review', method: 'water_seed_observed_outline_ransac_mesh', fitCount: 100,
      validationCount: 10, medianErrorM: 1, p95ErrorM: 2.5, validationScope: 'base',
    } as RegistrationQuality, id: 'base-test', sourceHash: 'base-hash',
  });
  const quality = result.quality;
  assert.equal(quality.baseAssetId, 'base-test');
  assert.equal(quality.inheritedGeographicP95M, 2.5);
  assert.ok((quality.method ?? '').includes('cross_map'));
  assert.ok((quality.validationScope ?? '').includes('继承基准图'));
  assert.ok((quality.fitCount ?? 0) >= 20, `transfer mesh uses many observed landmarks (${quality.fitCount})`);
  assert.equal(quality.foldedTriangles, 0);
  const determinant = sourceAffine[0][0] * sourceAffine[1][1] - sourceAffine[0][1] * sourceAffine[1][0];
  const inverseSource: number[][] = [
    [sourceAffine[1][1] / determinant, -sourceAffine[0][1] / determinant,
      -(sourceAffine[1][1] / determinant * sourceAffine[0][2] - sourceAffine[0][1] / determinant * sourceAffine[1][2])],
    [-sourceAffine[1][0] / determinant, sourceAffine[0][0] / determinant,
      -(-sourceAffine[1][0] / determinant * sourceAffine[0][2] + sourceAffine[0][0] / determinant * sourceAffine[1][2])],
  ];
  const mesh = new Mesh(result.registration);
  let maxError = 0;
  for (const feature of sourceFeatures) {
    const mapped = mesh.forward(feature.center);
    if (!mapped.inside) continue;
    const expected = applyAffine(inverseSource, feature.center);
    maxError = Math.max(maxError, Math.hypot(mapped.local[0] - expected[0], mapped.local[1] - expected[1]));
  }
  assert.ok(maxError < 1, `source pixels must map back to local metres (max error ${maxError.toFixed(3)} m)`);
});

// ---------------------------------------------------------------------------
// autoRegister without vision (synthetic features from the baked registration)
// ---------------------------------------------------------------------------

test('autoRegister recovers the baked szu-1 transform from synthetic features', async (t) => {
  const state = bakedState('szu-1');
  const reference = referenceFeatures();
  const pixelFromLocal = (local: Pixel): Pixel => {
    const affine = state.registration.affine;
    const determinant = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0];
    const inverse = [[affine[1][1] / determinant, -affine[0][1] / determinant],
      [-affine[1][0] / determinant, affine[0][0] / determinant]];
    const shift = [-(inverse[0][0] * affine[0][2] + inverse[0][1] * affine[1][2]),
      -(inverse[1][0] * affine[0][2] + inverse[1][1] * affine[1][2])];
    return [inverse[0][0] * local[0] + inverse[0][1] * local[1] + shift[0],
      inverse[1][0] * local[0] + inverse[1][1] * local[1] + shift[1]];
  };
  const random = (() => {
    let seed = 42 >>> 0;
    return () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 0x100000000;
    };
  })();
  const noise = (scale: number): number => (random() * 2 - 1) * scale;
  const scaleSquared = 1 / (Math.abs(state.registration.affine[0][0] * state.registration.affine[1][1]
    - state.registration.affine[0][1] * state.registration.affine[1][0]));
  const features: SourceFeature[] = [];
  for (const feature of reference) {
    const center = pixelFromLocal(feature.center);
    if (!(center[0] > 20 && center[0] < state.width - 20 && center[1] > 20 && center[1] < state.height - 20)) continue;
    features.push({
      kind: feature.kind,
      center: [center[0] + noise(0.5), center[1] + noise(0.5)],
      area: feature.area * scaleSquared * (1 + noise(0.02)),
      aspect: feature.aspect * (1 + noise(0.01)),
      points: feature.points.map(point => {
        const mapped = pixelFromLocal(point);
        return [mapped[0] + noise(0.7), mapped[1] + noise(0.7)];
      }),
    });
  }
  assert.ok(features.filter(feature => feature.kind === 'water').length >= 4, 'synthetic map has water seeds');
  const started = performance.now();
  const result = await autoRegister(features, reference, state.width, state.height);
  const elapsed = performance.now() - started;
  const quality = result.quality;
  assert.equal(quality.foldedTriangles, 0);
  assert.equal(quality.affineFallback, false);
  assert.ok((quality.fitCount ?? 0) > 20, `synthetic registration fits many landmarks (${quality.fitCount})`);
  assert.ok((quality.p95ErrorM ?? Infinity) < 2, `synthetic holdout error ${quality.p95ErrorM} m`);
  // Recovered affine is close to the baked one.
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 3; column++) {
      const expected = state.registration.affine[row][column];
      close(result.registration.affine[row][column], expected, Math.abs(expected) * 0.02 + 0.2,
        `affine[${row}][${column}]`);
    }
  }
  const controls = state.controls.map(control => ({ ...control, pixel: [...control.pixel] as Pixel }));
  projectControls(controls, result.registration);
  assert.ok(controls.every(control => control.calibrated), 'every baked control is calibrated');
  const maxControlError = Math.max(...controls.map((control, index) =>
    metresBetween(control.lnglat!, state.controls[index].lnglat!, state.frame.origin)));
  assert.ok(maxControlError < 5, `synthetic map control max error ${maxControlError.toFixed(2)} m`);
  t.diagnostic(`synthetic szu-1 autoRegister: ${elapsed.toFixed(0)} ms, fitCount ${quality.fitCount}, `
    + `control max ${maxControlError.toFixed(2)} m`);
});

// ---------------------------------------------------------------------------
// Golden: szu-1 against the baked state
// ---------------------------------------------------------------------------

test('autoRegister reproduces the baked szu-1 registration', async (t) => {
  const state = bakedState('szu-1');
  const reference = referenceFeatures();
  const image = loadMapImage('szu-1');
  assert.ok(image, 'web/public/data/maps/szu-1/original.png must exist');
  await ensureOpenCv();
  const started = performance.now();
  const analysis = await analysisFor('szu-1');
  const analyzeMs = performance.now() - started;
  assert.equal(analysis.width, state.width);
  assert.equal(analysis.height, state.height);
  assert.ok(analysis.features.length > 100, 'vision analysis finds region features');
  const registerStarted = performance.now();
  const result = await autoRegister(analysis.features as SourceFeature[], reference, analysis.width, analysis.height);
  const autoMs = performance.now() - registerStarted;
  const quality = result.quality;
  const bakedQuality = state.quality;
  // Mesh integrity.
  assert.equal(quality.foldedTriangles, 0);
  assert.equal(quality.affineFallback, false);
  assert.equal(quality.state, 'needs_review');
  assert.equal(quality.method, 'water_seed_observed_outline_ransac_mesh');
  assert.ok((quality.candidateTransforms ?? 0) > 0);
  assert.ok((quality.observedCornerCount ?? 0) > 0);
  assert.ok((quality.validationScope ?? '').length > 0);
  assert.equal(result.registration.source.length, quality.fitCount);
  assert.equal(result.registration.triangles.length > 0, true);
  assert.ok(new Mesh(result.registration).hasMesh);
  assert.equal((quality.landmarks ?? []).length, (quality.fitCount ?? 0) + (quality.validationCount ?? 0));
  // Golden tolerances from the static architecture contract (C.2).
  const bakedFit = bakedQuality.fitCount;
  assert.ok(Math.abs((quality.fitCount ?? 0) - bakedFit) <= bakedFit * 0.25,
    `fitCount ${quality.fitCount} vs baked ${bakedFit} must stay within ±25%`);
  assert.ok(quality.p95ErrorM !== null && bakedQuality.p95ErrorM !== null);
  assert.ok(Math.abs(quality.p95ErrorM! - bakedQuality.p95ErrorM!) <= 2.5,
    `p95 ${quality.p95ErrorM} m vs baked ${bakedQuality.p95ErrorM} m must stay within ±2.5 m`);
  // Every baked control is calibrated by the client registration; positions agree.
  const controls = state.controls.map(control => ({ ...control, pixel: [...control.pixel] as Pixel }));
  projectControls(controls, result.registration);
  const uncalibrated = controls.filter(control => control.calibrated !== true).map(control => control.id);
  assert.deepEqual(uncalibrated, [], `all controls calibrated (${uncalibrated.join(',')})`);
  let maxControlError = 0;
  const controlErrors: string[] = [];
  for (let index = 0; index < controls.length; index++) {
    const expected = state.controls[index].lnglat;
    assert.ok(expected, `baked control ${state.controls[index].id} has a baked lnglat`);
    const error = metresBetween(controls[index].lnglat!, expected!, state.frame.origin);
    maxControlError = Math.max(maxControlError, error);
    controlErrors.push(`${state.controls[index].id}:${error.toFixed(2)}`);
  }
  assert.ok(maxControlError <= 6,
    `per-control lnglat max distance ${maxControlError.toFixed(2)} m must stay <= 6 m`);
  t.diagnostic('autoRegister szu-1 (client) vs baked state.json: '
    + `fitCount ${quality.fitCount}/${bakedFit} (${((quality.fitCount! - bakedFit) / bakedFit * 100).toFixed(1)}%), `
    + `p95 ${quality.p95ErrorM}/${bakedQuality.p95ErrorM} m, median ${quality.medianErrorM}/${bakedQuality.medianErrorM} m, `
    + `max ${quality.maxErrorM}/${bakedQuality.maxErrorM} m, candidateTransforms ${quality.candidateTransforms}/${bakedQuality.candidateTransforms}, `
    + `observedCorners ${quality.observedCornerCount}/${bakedQuality.observedCornerCount}`);
  t.diagnostic(`per-control lnglat error vs baked (m): ${controlErrors.join(' ')} (max ${maxControlError.toFixed(2)} m)`);
  t.diagnostic(`timings: analyze ${analyzeMs.toFixed(0)} ms, autoRegister ${autoMs.toFixed(0)} ms (node, opencv.js WASM)`);
  assert.ok(analyzeMs + autoMs < 180000, 'default client pipeline budget stays under 180 s');
});

// ---------------------------------------------------------------------------
// Heavy: szu-2 cross-map transfer
// ---------------------------------------------------------------------------

const heavy = process.env.DIRMAP_HEAVY_TESTS === '1';

test('heavy: szu-2 cross-map transfer or documented direct-registration fallback',
  { skip: heavy ? false : 'set DIRMAP_HEAVY_TESTS=1 to run the szu-2 cross-map test' }, async (t) => {
    const baseState = bakedState('szu-1');
    const sourceState = bakedState('szu-2');
    const reference = referenceFeatures();
    const started = performance.now();
    const [baseAnalysis, sourceAnalysis] = await Promise.all([analysisFor('szu-1'), analysisFor('szu-2')]);
    const analyzeMs = performance.now() - started;
    assert.ok(analysisCache.has('szu-1') && analysisCache.has('szu-2'));
    // Cross-map transfer through the already registered szu-1 sheet.
    const transfer = await registerToMap(sourceAnalysis.features as SourceFeature[], {
      features: baseAnalysis.features as SourceFeature[],
      registration: baseState.registration,
      quality: baseState.quality,
      id: 'szu-1',
      sourceHash: (baseState as unknown as { sourceHash?: string }).sourceHash,
    });
    const transferQuality = transfer.quality;
    assert.equal(transferQuality.baseAssetId, 'szu-1');
    assert.ok(typeof transferQuality.inheritedGeographicP95M === 'number',
      'transfer quality inherits the base geographic uncertainty');
    assert.ok((transferQuality.validationScope ?? '').includes('继承基准图'));
    assert.ok((transferQuality.method ?? '').includes('cross_map'));
    assert.equal(transferQuality.foldedTriangles, 0);
    // Direct automatic registration on szu-2 is the documented fallback when the
    // low-resolution sheet has too few common region features for a complete hull.
    let direct: Awaited<ReturnType<typeof autoRegister>> | null = null;
    let directError: string | null = null;
    try {
      direct = await autoRegister(sourceAnalysis.features as SourceFeature[], reference,
        sourceAnalysis.width, sourceAnalysis.height);
    } catch (error) {
      directError = error instanceof Error ? error.message : String(error);
    }
    const measure = (registration: RegistrationData) => {
      const controls = sourceState.controls.map(control => ({ ...control, pixel: [...control.pixel] as Pixel }));
      projectControls(controls, registration);
      const calibrated = controls.filter(control => control.calibrated === true).length;
      let maxError = 0;
      for (let index = 0; index < controls.length; index++) {
        const expected = sourceState.controls[index].lnglat;
        if (!controls[index].calibrated || !expected) continue;
        maxError = Math.max(maxError, metresBetween(controls[index].lnglat!, expected, sourceState.frame.origin));
      }
      return { calibrated, maxError, total: controls.length };
    };
    const transferCoverage = measure(transfer.registration);
    const directCoverage = direct ? measure(direct.registration) : { calibrated: 0, maxError: Infinity, total: sourceState.controls.length };
    t.diagnostic(`szu-2 transfer: method ${transferQuality.method}, fitCount ${transferQuality.fitCount}, `
      + `validationCount ${transferQuality.validationCount}, p95 ${transferQuality.p95ErrorM} m, `
      + `controls ${transferCoverage.calibrated}/${transferCoverage.total} (max ${transferCoverage.maxError.toFixed(2)} m), `
      + `inherited ${transferQuality.inheritedGeographicP95M} m`);
    t.diagnostic(`szu-2 direct autoRegister: ${direct ? `fitCount ${direct.quality.fitCount}, p95 ${direct.quality.p95ErrorM} m, `
      + `controls ${directCoverage.calibrated}/${directCoverage.total} (max ${directCoverage.maxError.toFixed(2)} m)` : `failed: ${directError}`}`);
    // C.2 acceptance: transfer p95 <= 5 m OR direct-registration fallback.
    if (transferCoverage.calibrated === sourceState.controls.length
      && transferQuality.p95ErrorM !== null && transferQuality.p95ErrorM <= 5) {
      assert.ok(transferCoverage.maxError <= 6,
        `transfer coverage complete; control max error ${transferCoverage.maxError.toFixed(2)} m must stay <= 6 m`);
    } else {
      assert.ok(direct !== null,
        'cross-map transfer leaves part of the sheet uncalibrated and direct autoRegister failed: '
        + `transfer ${transferCoverage.calibrated}/${transferCoverage.total}, error ${directError}`);
      assert.equal(directCoverage.calibrated, sourceState.controls.length,
        `direct fallback calibrates every baked szu-2 control (${directCoverage.calibrated}/${directCoverage.total})`);
      assert.ok(directCoverage.maxError <= 7,
        `direct fallback per-control lnglat max distance ${directCoverage.maxError.toFixed(2)} m must stay <= 7 m`);
    }
    // Observed-boundary extension keeps the transfer bookkeeping consistent.
    if (direct) {
      const extended = await extendTransferBoundary(transfer.registration, transferQuality, direct.quality);
      assert.ok((extended.quality.fitCount ?? 0) >= (transferQuality.fitCount ?? 0));
      t.diagnostic(`szu-2 extended transfer: method ${extended.quality.method}, fitCount ${extended.quality.fitCount}, `
        + `geographicBoundaryAnchors ${extended.quality.geographicBoundaryAnchors}, p95 ${extended.quality.p95ErrorM} m`);
    }
    t.diagnostic(`szu-2 heavy timings: analyze ${analyzeMs.toFixed(0)} ms (node, opencv.js WASM), `
      + `total ${(performance.now() - started).toFixed(0)} ms`);
    assert.ok(performance.now() - started < 300000, 'heavy cross-map test stays under 5 minutes');
  });
