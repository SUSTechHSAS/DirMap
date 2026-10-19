/** Golden tests for the browser legal-approach resolver (`dirmap/access.py` port).
 *
 * Loads the committed bake artefacts (state.json, graph.json + graph.bin.gz,
 * semantics.bin.gz, approaches.json) through the same `loader` seam the browser
 * data layer uses, resolves every baked control and compares the result with
 * the Python-built goldens:
 *  - `ok` approaches: client returns ok, identical node (or local position
 *    within 2 m), same method/automatic flag;
 *  - `ok:false` approaches: the exact Python Chinese message (szu-2 c2/c7);
 *  - szu-1 c6: the compound point-symbol path resolves automatically, offset <= 8 m;
 *  - a teacher-closed position is refused with the Python message.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import {
  decodeSemantics,
  prepareAccess,
  resolveApproach,
  resolveApproaches,
} from '../../web/src/access.ts';
import type { AccessContext, AccessSemantics, ApproachResolution, SemanticsBundle } from '../../web/src/access.ts';
import { circle, line, maskOf } from '../../web/src/cv.ts';
import type { MaskImage } from '../../web/src/cv.ts';
import { Mesh } from '../../web/src/geometry.ts';
import type { RegistrationData } from '../../web/src/geometry.ts';
import { RasterGraph, buildGraphFromRaster } from '../../web/src/raster-graph.ts';
import type { GraphMetadata } from '../../web/src/raster-graph.ts';
import type {
  BakedApproach, BakedApproaches, Control, MapState, Pixel, TerrainEdit,
} from '../../web/src/types.ts';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const ASSETS = ['szu-1', 'szu-2', 'szu-3'] as const;
type AssetId = (typeof ASSETS)[number];

interface BakedApproachesDocument extends BakedApproaches {
  startPixel: Pixel;
  startNode: number;
  startComponent: number;
}

function repoPath(...parts: string[]): string {
  return resolve(ROOT, ...parts);
}

function readJSON<T>(relative: string): T {
  return JSON.parse(readFileSync(repoPath(relative), 'utf8')) as T;
}

function mapPath(assetId: string, name: string): string {
  return repoPath('web/public/data/maps', assetId, name);
}

function arrayBufferOf(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

interface MapAssets {
  assetId: AssetId;
  state: MapState;
  metadata: GraphMetadata;
  approaches: BakedApproachesDocument;
  semantics: SemanticsBundle;
  buffer: ArrayBuffer;
}

function loadAssets(assetId: AssetId): MapAssets {
  const state = readJSON<MapState>(`web/public/data/maps/${assetId}/state.json`);
  const metadata = readJSON<GraphMetadata>(`web/public/data/maps/${assetId}/graph.json`);
  const approaches = readJSON<BakedApproachesDocument>(`web/public/data/maps/${assetId}/approaches.json`);
  const inflated = gunzipSync(readFileSync(mapPath(assetId, 'semantics.bin.gz')));
  const semantics = decodeSemantics(arrayBufferOf(inflated));
  assert.equal(semantics.width, state.width, `${assetId} semantics width`);
  assert.equal(semantics.height, state.height, `${assetId} semantics height`);
  const bundle = gunzipSync(readFileSync(mapPath(assetId, 'graph.bin.gz')));
  return { assetId, state, metadata, approaches, semantics, buffer: arrayBufferOf(bundle) };
}

const assetsCache = new Map<AssetId, MapAssets>();
const graphCache = new Map<AssetId, Promise<RasterGraph>>();
let cvReady: Promise<void> | null = null;

function assets(assetId: AssetId): MapAssets {
  let cached = assetsCache.get(assetId);
  if (!cached) {
    cached = loadAssets(assetId);
    assetsCache.set(assetId, cached);
  }
  return cached;
}

function realGraph(assetId: AssetId): Promise<RasterGraph> {
  let pending = graphCache.get(assetId);
  if (!pending) {
    const map = assets(assetId);
    pending = RasterGraph.load(assetId, map.state, {
      loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
    });
    graphCache.set(assetId, pending);
  }
  return pending;
}

async function ensureCv(): Promise<void> {
  if (!cvReady) cvReady = prepareAccess();
  await cvReady;
}

function contextFor(map: MapAssets, graph: RasterGraph): AccessContext {
  assert.ok(map.state.registration, `${map.assetId} registration`);
  return {
    graph,
    mesh: new Mesh(map.state.registration),
    semantics: map.semantics.masks,
    registration: map.state.registration,
  };
}

/** Python `registration.forward([target])` validity for `targetLnglat`. */
function localError(left: Pixel, right: Pixel): number {
  return Math.hypot(left[0] - right[0], left[1] - right[1]);
}

function expectedStartComponent(map: MapAssets, graph: RasterGraph): number {
  const start = map.approaches.startPixel ?? map.approaches.startNode;
  if (typeof map.approaches.startComponent === 'number') return map.approaches.startComponent;
  const node = graph.snap(start as Pixel);
  return graph.components[node];
}

interface MapReport {
  assetId: AssetId;
  bakedOk: number;
  resolvedOk: number;
  nodeIdentical: number;
  /** Mismatches explained by two numerically equidistant legal candidates. */
  equidistantTies: number;
  maxLocalErrorM: number;
  methodIdentical: number;
  automaticIdentical: number;
  failuresMatched: number;
  failuresExpected: number;
  elapsedMs: number;
}

const reports: MapReport[] = [];

test('semantics bundles decode to the committed mask layout', () => {
  for (const assetId of ASSETS) {
    const map = assets(assetId);
    for (const kind of ['building', 'water', 'point_ink', 'vegetation', 'restricted'] as const) {
      const mask = map.semantics.masks[kind];
      assert.ok(mask instanceof Uint8Array, `${assetId} ${kind} mask missing`);
      assert.equal(mask.length, map.state.width * map.state.height, `${assetId} ${kind} mask length`);
      let count = 0;
      for (let i = 0; i < mask.length; i++) if (mask[i]) count++;
      assert.ok(count > 0, `${assetId} ${kind} mask is empty`);
    }
  }
});

test('baked approaches reproduce on all three maps', { timeout: 300_000 }, async (t) => {
  await ensureCv();
  for (const assetId of ASSETS) {
    const map = assets(assetId);
    const graph = await realGraph(assetId);
    const context = contextFor(map, graph);
    const startComponent = expectedStartComponent(map, graph);
    const started = performance.now();
    const resolved = resolveApproaches(context, map.state.controls, { startComponent });
    const elapsedMs = performance.now() - started;
    const report: MapReport = {
      assetId,
      bakedOk: 0,
      resolvedOk: 0,
      nodeIdentical: 0,
      equidistantTies: 0,
      maxLocalErrorM: 0,
      methodIdentical: 0,
      automaticIdentical: 0,
      failuresMatched: 0,
      failuresExpected: 0,
      elapsedMs,
    };
    const bakedEntries = map.approaches.controls as Record<string, BakedApproach>;
    assert.deepEqual(Object.keys(resolved).sort(), Object.keys(bakedEntries).sort(),
      `${assetId} resolved control id set`);
    for (const [controlId, baked] of Object.entries(bakedEntries)) {
      const mine: ApproachResolution = resolved[controlId];
      if (baked.ok) {
        report.bakedOk++;
        assert.ok(mine.ok, `${assetId} ${controlId}: expected ok, got ${!mine.ok ? mine.message : ''}`);
        if (!mine.ok) continue;
        report.resolvedOk++;
        const localDelta = localError(mine.local, baked.local);
        report.maxLocalErrorM = Math.max(report.maxLocalErrorM, localDelta);
        if (mine.node === baked.node) report.nodeIdentical++;
        // Two legal candidate pixels can be exactly equidistant from a control
        // that sits on a stroke between them: the baked node is then decided by
        // Python's last-ulp mesh rounding (szu-2 c3: offsets agree to 1e-13 m,
        // candidates are antipodal). Accept that numerical tie explicitly; any
        // other node difference must still satisfy the 2 m position contract.
        const equidistantTie = mine.node !== baked.node && mine.automatic && baked.automatic
          && mine.offsetM !== 0 && Math.abs(mine.offsetM - baked.offsetM) <= 1e-6;
        if (equidistantTie) report.equidistantTies++;
        assert.ok(mine.node === baked.node || localDelta <= 2 || equidistantTie,
          `${assetId} ${controlId}: node ${mine.node} != ${baked.node} and local error ${localDelta.toFixed(3)} m > 2 m`);
        if (mine.method === baked.method) report.methodIdentical++;
        assert.equal(mine.method, baked.method, `${assetId} ${controlId} method`);
        if (mine.automatic === baked.automatic) report.automaticIdentical++;
        assert.equal(mine.automatic, baked.automatic, `${assetId} ${controlId} automatic`);
        assert.equal(mine.component, baked.component ?? mine.component, `${assetId} ${controlId} component`);
        assert.ok(Number.isFinite(mine.lnglat[0]) && Number.isFinite(mine.lnglat[1]),
          `${assetId} ${controlId} lnglat finite`);
        assert.equal(mine.targetPixel[0], baked.targetPixel?.[0], `${assetId} ${controlId} targetPixel x`);
        assert.equal(mine.targetPixel[1], baked.targetPixel?.[1], `${assetId} ${controlId} targetPixel y`);
        // Exact snaps must stay zero-offset and non-automatic; feature paths
        // must carry reviewRequired, matching the baker.
        assert.equal(mine.offsetM === 0, baked.offsetM === 0, `${assetId} ${controlId} zero offset`);
        if (baked.reviewRequired) assert.equal(mine.reviewRequired, true, `${assetId} ${controlId} reviewRequired`);
      } else {
        report.failuresExpected++;
        assert.ok(!mine.ok, `${assetId} ${controlId}: expected failure "${baked.message}", got ok`);
        if (!mine.ok) {
          if (mine.message === baked.message) report.failuresMatched++;
          assert.equal(mine.message, baked.message, `${assetId} ${controlId} message`);
        }
      }
    }
    assert.equal(report.resolvedOk, report.bakedOk, `${assetId} resolved ok count`);
    assert.equal(report.failuresMatched, report.failuresExpected, `${assetId} reproduced failures`);
    assert.equal(report.nodeIdentical + report.equidistantTies, report.bakedOk,
      `${assetId}: every node difference must be an equidistant tie`);
    reports.push(report);
  }
  const header = ['map', 'bakedOk', 'resolvedOk', 'node=', 'ties', 'maxLocalM', 'method=', 'auto=', 'fails', 'resolveMs'];
  const rows = reports.map(report => [
    report.assetId,
    `${report.bakedOk}`,
    `${report.resolvedOk}`,
    `${report.nodeIdentical}/${report.bakedOk}`,
    `${report.equidistantTies}`,
    report.maxLocalErrorM.toFixed(4),
    `${report.methodIdentical}/${report.bakedOk}`,
    `${report.automaticIdentical}/${report.bakedOk}`,
    `${report.failuresMatched}/${report.failuresExpected}`,
    report.elapsedMs.toFixed(0),
  ]);
  const widths = header.map((cell, column) => Math.max(cell.length, ...rows.map(row => row[column].length)));
  const line = (row: string[]): string => row.map((cell, column) => cell.padEnd(widths[column])).join('  ');
  t.diagnostic(line(header));
  for (const row of rows) t.diagnostic(line(row));
  // The port must be faithful to Python on the baked maps: identical methods,
  // automatic flags and offsets, and identical nodes except for documented
  // floating-point ties between equidistant legal candidates.
  for (const report of reports) {
    assert.equal(report.methodIdentical, report.bakedOk, `${report.assetId} method identical ratio`);
    assert.equal(report.automaticIdentical, report.bakedOk, `${report.assetId} automatic identical ratio`);
    assert.ok(report.nodeIdentical + report.equidistantTies === report.bakedOk,
      `${report.assetId} node identity or tie`);
    assert.ok(report.maxLocalErrorM <= 2.5, `${report.assetId} max local error`);
  }
  // Only szu-2 c3 (a stroke between two symmetric candidate pixels) may tie;
  // every other golden approach must land on the baked node exactly.
  const totalTies = reports.reduce((sum, report) => sum + report.equidistantTies, 0);
  assert.equal(totalTies, 1, 'documented equidistant-tie count');
});

test('szu-2 unresolved controls keep the exact Python messages', async () => {
  await ensureCv();
  const map = assets('szu-2');
  const graph = await realGraph('szu-2');
  const context = contextFor(map, graph);
  const resolved = resolveApproaches(context, map.state.controls, {
    startComponent: expectedStartComponent(map, graph),
  });
  assert.deepEqual(resolved.c2, { ok: false, message: '圈心处的地物或通行侧不明确，请核对实际到访位置' });
  assert.deepEqual(resolved.c7, { ok: false, message: '目标地物被禁入区域包围，请核对点位；不能跨禁区自动接入道路' });
});

test('szu-1 c6 compound point symbol resolves automatically within 8 m', async () => {
  await ensureCv();
  const map = assets('szu-1');
  const graph = await realGraph('szu-1');
  const context = contextFor(map, graph);
  const control = map.state.controls.find(candidate => candidate.id === 'c6');
  assert.ok(control, 'szu-1 c6 control');
  const resolved = resolveApproach(context, control, {
    startComponent: expectedStartComponent(map, graph),
  });
  assert.ok(resolved.ok, `c6 should resolve: ${!resolved.ok ? resolved.message : ''}`);
  if (!resolved.ok) return;
  assert.equal(resolved.automatic, true, 'c6 is a feature-based approach');
  assert.equal(resolved.method, 'compound_point_symbol_edge', 'c6 compound point symbol path');
  assert.equal(resolved.featureKind, 'building', 'c6 stands on the grey sculpture footprint');
  assert.ok(resolved.offsetM <= 8, `c6 offsetM ${resolved.offsetM} > 8`);
  const baked = map.approaches.controls.c6;
  assert.ok(baked && baked.ok, 'szu-1 c6 baked entry');
  if (baked.ok) {
    assert.equal(resolved.node, baked.node, 'c6 node');
    assert.ok(Math.abs(resolved.offsetM - baked.offsetM) < 1e-9, 'c6 offset');
  }
});

/** Rasterize `TerrainEdit` closes like `dirmap.graph.build_graph` (circle fill). */
function closedEditMask(width: number, height: number, edits: TerrainEdit[]): Uint8Array {
  const closed = new Uint8Array(width * height);
  for (const edit of edits) {
    if (edit.kind !== 'close') continue;
    const radius = Math.trunc(edit.radiusPx);
    for (const point of edit.pixels) {
      const centreX = Math.round(point[0]);
      const centreY = Math.round(point[1]);
      for (let y = Math.max(0, centreY - radius); y <= Math.min(height - 1, centreY + radius); y++) {
        for (let x = Math.max(0, centreX - radius); x <= Math.min(width - 1, centreX + radius); x++) {
          const dx = x - centreX;
          const dy = y - centreY;
          if (dx * dx + dy * dy <= radius * radius) closed[y * width + x] = 1;
        }
      }
    }
  }
  return closed;
}

test('a teacher-closed approach position is refused', async () => {
  await ensureCv();
  const map = assets('szu-2');
  const graph = await realGraph('szu-2');
  const context = contextFor(map, graph);
  const control = map.state.controls.find(candidate => candidate.id === 'c5');
  assert.ok(control, 'szu-2 c5 control');
  const centre: Pixel = [Math.round(control.pixel[0]), Math.round(control.pixel[1])];
  const edits: TerrainEdit[] = [{ kind: 'close', pixels: [centre], radiusPx: 3 }];
  const closed = closedEditMask(map.state.width, map.state.height, edits);
  const closedContext: AccessContext = { ...context, edits, closedEdits: closed };
  const resolved = resolveApproach(closedContext, control);
  assert.ok(!resolved.ok, 'closed approach position must be refused');
  if (!resolved.ok) assert.equal(resolved.message, '教师已封闭该位置，不能自动改用地物外沿');
});

test('an edited graph exposes teacher closures through graph.closedEdits', { timeout: 120_000 }, async () => {
  await ensureCv();
  const map = assets('szu-2');
  const control = map.state.controls.find(candidate => candidate.id === 'c5');
  assert.ok(control, 'szu-2 c5 control');
  const edits: TerrainEdit[] = [{
    kind: 'close',
    pixels: [[Math.round(control.pixel[0]), Math.round(control.pixel[1])]],
    radiusPx: 3,
  }];
  // RasterGraph.load rebuilds the edited graph exactly like the client pipeline
  // and must expose the teacher closures to the resolver.
  const graph = await RasterGraph.load('szu-2', { ...map.state, edits }, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
  });
  assert.ok(graph.closedEdits, 'edited graph closedEdits mask');
  const centre = [Math.round(control.pixel[0]), Math.round(control.pixel[1])] as Pixel;
  assert.equal(graph.closedEdits[centre[1] * graph.width + centre[0]], 1, 'closed edit pixel');
  const context = contextFor(map, graph);
  const resolved = resolveApproach(context, control);
  assert.ok(!resolved.ok, 'edited-closed position must be refused');
  if (!resolved.ok) assert.equal(resolved.message, '教师已封闭该位置，不能自动改用地物外沿');
});


// ---------------------------------------------------------------------------
// Synthetic scenes mirroring tests/test_access.py, on a tiny Python-compatible
// registration + raster graph. These pin the feature rules that the baked maps
// exercise only partially (vegetation vs wall, small vs large water, compound
// ring symbols, teacher closure on a drawn symbol).
// ---------------------------------------------------------------------------

const SCENE_WIDTH = 100;
const SCENE_HEIGHT = 90;

/** 100x90 pixel scene registered with the Python test fixture's affine. */
const SCENE_REGISTRATION: RegistrationData = {
  affine: [[0.5, 0, 0], [0, -0.5, 44.5]],
  source: [[0, 0], [99, 0], [99, 89], [0, 89]],
  target: [[0, 44.5], [49.5, 44.5], [49.5, 0], [0, 0]],
  triangles: [[0, 1, 2], [0, 2, 3]],
};

type SceneMasks = Record<'building' | 'water' | 'point_ink' | 'vegetation' | 'restricted', MaskImage>;

function emptyScene(): SceneMasks {
  return {
    building: maskOf(SCENE_WIDTH, SCENE_HEIGHT),
    water: maskOf(SCENE_WIDTH, SCENE_HEIGHT),
    point_ink: maskOf(SCENE_WIDTH, SCENE_HEIGHT),
    vegetation: maskOf(SCENE_WIDTH, SCENE_HEIGHT),
    restricted: maskOf(SCENE_WIDTH, SCENE_HEIGHT),
  };
}

function sceneControl(id: string, pixel: Pixel, radiusPx = 15): Control {
  return {
    id, label: id, pixel, radiusPx, required: true, role: 'control', state: 'candidate',
    serviceBaseSec: 0, servicePerPersonSec: 0,
  };
}

function syntheticContext(masks: SceneMasks, edits: TerrainEdit[] = []): AccessContext {
  const costs = new Float32Array(SCENE_WIDTH * SCENE_HEIGHT).fill(1);
  const semantics: AccessSemantics = {};
  for (const [kind, mask] of Object.entries(masks) as [keyof SceneMasks, MaskImage][]) {
    for (let index = 0; index < mask.data.length; index++) if (mask.data[index]) costs[index] = 0;
    semantics[kind] = mask.data;
  }
  const graph = buildGraphFromRaster(costs, SCENE_REGISTRATION, 'access-fixture', edits, 4,
    { width: SCENE_WIDTH, height: SCENE_HEIGHT });
  return { graph, mesh: new Mesh(SCENE_REGISTRATION), semantics, registration: SCENE_REGISTRATION };
}

test('Python access unit-test scenes reproduce on a synthetic raster', async () => {
  await ensureCv();
  // A cross drawn with point ink becomes a point_symbol_edge approach.
  {
    const masks = emptyScene();
    line(masks.point_ink, [45, 40], [49, 44], 1);
    line(masks.point_ink, [45, 44], [49, 40], 1);
    const resolved = resolveApproach(syntheticContext(masks), sceneControl('31', [47, 42]));
    assert.ok(resolved.ok, `point symbol: ${!resolved.ok ? resolved.message : ''}`);
    if (resolved.ok) {
      assert.equal(resolved.method, 'point_symbol_edge');
      assert.equal(resolved.automatic, true);
      assert.ok(resolved.offsetM <= 3, `point symbol offset ${resolved.offsetM}`);
    }
  }
  // A building facade approach lands outside the roof, within 3 m.
  {
    const masks = emptyScene();
    for (let y = 20; y < 65; y++) for (let x = 40; x < 80; x++) masks.building.data[y * SCENE_WIDTH + x] = 255;
    const resolved = resolveApproach(syntheticContext(masks), sceneControl('b', [41, 42]));
    assert.ok(resolved.ok, `building: ${!resolved.ok ? resolved.message : ''}`);
    if (resolved.ok) {
      assert.equal(resolved.method, 'building_edge');
      assert.ok(resolved.pixel[0] < 40, `building approach x ${resolved.pixel[0]}`);
      assert.ok(resolved.offsetM <= 3, `building offset ${resolved.offsetM}`);
    }
  }
  // A long wall is never promoted to a point symbol.
  {
    const masks = emptyScene();
    for (let y = 8; y < 80; y++) for (let x = 48; x < 50; x++) masks.point_ink.data[y * SCENE_WIDTH + x] = 255;
    const resolved = resolveApproach(syntheticContext(masks), sceneControl('wall', [48, 42], 16));
    assert.ok(!resolved.ok, 'long wall must not resolve');
    if (!resolved.ok) assert.equal(resolved.message, '圈心处的地物或通行侧不明确，请核对实际到访位置');
  }
  // A tree behind a wall cannot be approached through the wall.
  {
    const masks = emptyScene();
    circle(masks.vegetation, [40, 44], 2, -1);
    for (let y = 0; y < SCENE_HEIGHT; y++) masks.point_ink.data[y * SCENE_WIDTH + 43] = 255;
    const context = syntheticContext(masks);
    const right = context.graph.snap([60, 44]);
    const resolved = resolveApproach(context, sceneControl('tree', [40, 44]),
      { startComponent: context.graph.components[right] });
    assert.ok(!resolved.ok, 'tree behind wall must not resolve');
    if (!resolved.ok) assert.equal(resolved.message, '合法到访侧与起点不连通，请核对通道');
  }
  // A forbidden area is not replaced by a nearby road.
  {
    const masks = emptyScene();
    for (let y = 35; y < 50; y++) for (let x = 35; x < 50; x++) masks.restricted.data[y * SCENE_WIDTH + x] = 255;
    for (let y = 41; y < 44; y++) for (let x = 41; x < 44; x++) masks.point_ink.data[y * SCENE_WIDTH + x] = 255;
    const resolved = resolveApproach(syntheticContext(masks), sceneControl('closed', [42, 42]));
    assert.ok(!resolved.ok, 'restricted centre must not resolve');
    if (!resolved.ok) assert.equal(resolved.message, '圈心位于禁入区域，不能自动移到外部道路');
  }
  // A small round fountain has a perimeter approach; a lake does not.
  {
    const masks = emptyScene();
    circle(masks.water, [50, 45], 10, -1);
    const fountain = resolveApproach(syntheticContext(masks), sceneControl('fountain', [50, 45]));
    assert.ok(fountain.ok, `fountain: ${!fountain.ok ? fountain.message : ''}`);
    if (fountain.ok) {
      assert.equal(fountain.method, 'small_water_feature_edge');
      assert.ok(fountain.offsetM > 5 && fountain.offsetM < 7, `fountain offset ${fountain.offsetM}`);
    }
    const lakeMasks = emptyScene();
    circle(lakeMasks.water, [50, 45], 32, -1);
    const lake = resolveApproach(syntheticContext(lakeMasks), sceneControl('lake', [50, 45]));
    assert.ok(!lake.ok, 'interior lake point must not resolve');
  }
  // A bank target gets a water_edge approach within 3 m.
  {
    const masks = emptyScene();
    const angle = 25 * Math.PI / 180;
    for (let y = 0; y < SCENE_HEIGHT; y++) {
      for (let x = 0; x < SCENE_WIDTH; x++) {
        const dx = (x - 65) / 20;
        const dy = (y - 40) / 11;
        const rx = dx * Math.cos(angle) + dy * Math.sin(angle);
        const ry = -dx * Math.sin(angle) + dy * Math.cos(angle);
        if (rx * rx + ry * ry <= 1) masks.water.data[y * SCENE_WIDTH + x] = 255;
      }
    }
    const resolved = resolveApproach(syntheticContext(masks), sceneControl('bank', [48, 36]));
    assert.ok(resolved.ok, `bank: ${!resolved.ok ? resolved.message : ''}`);
    if (resolved.ok) {
      assert.equal(resolved.method, 'water_edge');
      assert.ok(resolved.offsetM <= 3, `bank offset ${resolved.offsetM}`);
    }
  }
  // Compound ring symbols resolve to their own rim with the start component.
  {
    const masks = emptyScene();
    for (const [cx, cy] of [[35, 35], [45, 45], [55, 55]] as [number, number][]) {
      circle(masks.building, [cx, cy], 7, 2);
    }
    const context = syntheticContext(masks);
    const main = context.graph.components[context.graph.snap([15, 45])];
    const resolved = resolveApproach(context, sceneControl('sculpture', [45, 45], 30),
      { startComponent: main });
    assert.ok(resolved.ok, `compound: ${!resolved.ok ? resolved.message : ''}`);
    if (resolved.ok) {
      assert.equal(resolved.method, 'compound_point_symbol_edge');
      assert.ok(resolved.offsetM <= 8, `compound offset ${resolved.offsetM}`);
      assert.ok(context.graph.costs[45 * SCENE_WIDTH + 45] > 0, 'symbol interior stays walkable');
    }
  }
  // A teacher closure of the drawn symbol is not bypassed.
  {
    const masks = emptyScene();
    for (let y = 40; y < 43; y++) for (let x = 45; x < 48; x++) masks.point_ink.data[y * SCENE_WIDTH + x] = 255;
    const context = syntheticContext(masks, [{ kind: 'close', pixels: [[46, 41]], radiusPx: 1 }]);
    assert.ok(context.graph.closedEdits, 'synthetic graph closes');
    const resolved = resolveApproach(context, sceneControl('closed-point', [46, 41]));
    assert.ok(!resolved.ok, 'closed point symbol must not resolve');
    if (!resolved.ok) assert.equal(resolved.message, '教师已封闭该位置，不能自动改用地物外沿');
  }
});
