/** Plan-matrix goldens (Python plan_matrix), course planning (optimizer golden)
 * and navigation re-planning against the baked release. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Mesh, frameLngLat } from '../../web/src/geometry.ts';
import { planMatrix, getPlanner, clearPlannerCache } from '../../web/src/planner.ts';
import type { PlanMatrixBundle, Planner } from '../../web/src/planner.ts';
import { RasterGraph } from '../../web/src/raster-graph.ts';
import { solve } from '../../web/src/optimizer.ts';
import type {
  BakedApproaches, Control, MapState, NavigationRequest, Pixel, PlanRequest, StaticRelease,
} from '../../web/src/types.ts';
import { assertCloseRelative, buildRealGraph, loadRealMap, readJSON } from './helpers.ts';

interface MatrixFixture {
  assetId: string;
  graphHash: string;
  metric: 'time' | 'distance';
  scenario: { controlIds: string[]; startPixel: Pixel; finishPixel: Pixel; metric: string };
  request: { version: number };
  startPixel: Pixel;
  finishPixel: Pixel;
  startNode: number;
  nodes: number[];
  connectors: number[];
  connectorWeights: number[];
  endpoints: { id: string; node: number; pixel: Pixel; approachPixel?: Pixel }[];
  matrix: number[][] | null;
  unresolved: { id: string; message: string }[];
  approaches: {
    controlId: string; node: number; pixel: Pixel; local: Pixel; method: string; automatic: boolean;
    offsetM: number; lnglat: Pixel; targetLnglat: Pixel | null;
  }[];
}

interface OptimizerGolden {
  assetId: string;
  graphHash: string;
  controlIds: string[];
  startPixel: Pixel;
  finishPixel: Pixel;
  mode: string;
  objective: string;
  matrix: number[][];
  request: {
    assetId: string; version: number; controlIds: string[]; startPixel: Pixel; finishPixel: Pixel;
    mode: string; objective: string; mandatoryControlIds?: string[]; mandatoryOrder?: boolean;
    teams: { id: string; people: number; speedMps: number; departureSec?: number }[];
    timeLimitSec: number;
  };
  expected: {
    status: string;
    routes: { teamId: string; controlIds: string[]; order: number[] }[];
    proof: { algorithm: string; globalOptimalityProven: boolean; objectiveValue: number[] };
  };
}

const matrixAssets = ['szu-1', 'szu-2', 'szu-3'];

async function graphAndMesh(assetId: string): Promise<{ graph: RasterGraph; mesh: Mesh; map: ReturnType<typeof loadRealMap>; controls: Control[]; approaches: BakedApproaches }> {
  const { graph, map } = await buildRealGraph(assetId);
  const mesh = new Mesh(map.state.registration);
  return { graph, mesh, map, controls: map.state.controls, approaches: map.approaches };
}

test('planMatrix reproduces the Python plan_matrix goldens', async () => {
  for (const assetId of matrixAssets) {
    const { graph, mesh, map, controls, approaches } = await graphAndMesh(assetId);
    const fixture = readJSON<MatrixFixture>(`tests/fixtures/static/${assetId}.matrix.json`);
    const scenario = fixture.scenario ?? {
      controlIds: fixture.request.controlIds, startPixel: fixture.startPixel,
      finishPixel: fixture.finishPixel, metric: fixture.metric,
    };
    const selected = scenario.controlIds.map(id => {
      const control = controls.find(candidate => candidate.id === id);
      assert.ok(control, `${assetId}: missing control ${id}`);
      return control;
    });
    const bundle: PlanMatrixBundle = planMatrix(graph, mesh, map.state.frame, selected,
      scenario.startPixel, scenario.finishPixel, approaches, { metric: 'time' });
    assert.deepEqual(bundle.unresolved, fixture.unresolved, `${assetId} unresolved`);
    if (fixture.matrix === null) {
      assert.ok(bundle.unresolved.length > 0, `${assetId}: expected unresolved matrix`);
      assert.equal(bundle.matrix.length, 0);
      continue;
    }
    assert.deepEqual(bundle.nodes, fixture.nodes, `${assetId} endpoint nodes`);
    assert.equal(bundle.endpoints.length, fixture.endpoints.length);
    for (let index = 0; index < bundle.connectors.length; index++) {
      assertCloseRelative(bundle.connectors[index], fixture.connectors[index], 1e-6, `${assetId} connector ${index}`);
      assertCloseRelative(bundle.connectorWeights[index], fixture.connectorWeights[index], 1e-6, `${assetId} connectorWeight ${index}`);
    }
    const size = fixture.matrix.length;
    assert.equal(bundle.matrix.length, size * size);
    for (let row = 0; row < size; row++) {
      for (let column = 0; column < size; column++) {
        assertCloseRelative(bundle.matrix[row * size + column], fixture.matrix[row][column], 1e-6,
          `${assetId} matrix[${row}][${column}]`);
      }
    }
    assert.equal(bundle.approaches.length, fixture.approaches.length, `${assetId} approach count`);
    for (const approach of bundle.approaches) {
      const golden = fixture.approaches.find(item => item.controlId === approach.controlId);
      assert.ok(golden, `${assetId}: unexpected approach ${approach.controlId}`);
      assert.equal(approach.method, golden.method, `${assetId} ${approach.controlId} method`);
      assert.equal(approach.automatic, golden.automatic, `${assetId} ${approach.controlId} automatic`);
      assert.deepEqual(approach.pixel.map(value => Math.round(value * 1e6) / 1e6), golden.pixel.map(value => Math.round(value * 1e6) / 1e6));
      assert.ok(Math.abs(approach.lnglat[0] - golden.lnglat[0]) < 1e-9);
      assert.ok(Math.abs(approach.lnglat[1] - golden.lnglat[1]) < 1e-9);
    }
  }
});

test('solve reproduces the Python exact-branch optimizer golden', () => {
  const golden = readJSON<OptimizerGolden>('tests/fixtures/static/optimizer.json');
  const state = readJSON<MapState>(`web/public/data/maps/${golden.assetId}/state.json`);
  const controls = golden.controlIds.map(id => {
    const control = state.controls.find(candidate => candidate.id === id);
    assert.ok(control, `missing control ${id}`);
    return control;
  });
  const solved = solve(golden.matrix, controls, golden.request.teams, golden.request.mode, golden.request.objective, golden.request.timeLimitSec);
  assert.equal(solved.status, golden.expected.status);
  assert.equal(solved.proof.algorithm, golden.expected.proof.algorithm);
  assert.equal(solved.proof.globalOptimalityProven, golden.expected.proof.globalOptimalityProven);
  assert.equal(solved.routes.length, golden.expected.routes.length);
  solved.routes.forEach((route, index) => {
    const expected = golden.expected.routes[index];
    assert.equal(route.teamId, expected.teamId);
    assert.deepEqual(route.order, expected.order, `team ${index} order`);
    assert.deepEqual(route.controlIds, expected.controlIds, `team ${index} controlIds`);
  });
  assert.equal(solved.proof.objectiveValue.length, golden.expected.proof.objectiveValue.length);
  solved.proof.objectiveValue.forEach((value, index) => {
    assertCloseRelative(value, golden.expected.proof.objectiveValue[index], 1e-12, `objectiveValue ${index}`);
  });
});

test('getPlanner + plan reproduces the optimizer golden course', async () => {
  const golden = readJSON<OptimizerGolden>('tests/fixtures/static/optimizer.json');
  const map = loadRealMap(golden.assetId);
  clearPlannerCache();
  const assets = { metadata: map.metadata, buffer: map.buffer, approaches: map.approaches };
  const planner = await getPlanner(map.state, undefined, assets);
  assert.equal(planner.graphHash, map.metadata.graphHash);
  const request = golden.request as unknown as PlanRequest;
  const plan = await planner.plan(request, map.state.controls);
  assert.equal(plan.status, 'OPTIMAL');
  assert.equal(plan.version, map.state.version);
  assert.equal(plan.graphHash, golden.graphHash);
  assert.equal(plan.routingSchema, map.metadata.schema);
  assert.equal(plan.draft, true);
  assert.match(plan.id ?? '', /^[0-9a-f]{16}$/);
  assert.equal(plan.routes.length, golden.expected.routes.length);
  plan.routes.forEach((route, index) => {
    const expected = golden.expected.routes[index];
    assert.equal(route.teamId, expected.teamId);
    assert.deepEqual(route.controlIds, expected.controlIds, `plan team ${index} controlIds`);
    assert.ok(route.geometry.length > 2, 'route has geometry');
    assert.equal(route.localGeometry.length, route.geometry.length);
    assert.ok(route.distanceM > 100, 'route distance');
    assert.ok(route.legs && route.legs.length === route.controlIds.length + 1, 'route legs');
    const legDistance = (route.legs ?? []).reduce((total, leg) => total + leg.distanceM, 0);
    assert.ok(Math.abs(legDistance - route.distanceM) < 1e-6, 'leg distances sum to total');
    assert.ok(route.nodeIds && route.nodeIds.length > 0);
    assert.ok(Math.abs((route.travelWeightM ?? 0) / 1.4 - route.travelSec) < 1e-6);
  });
  assert.equal(plan.proof?.algorithm, 'held_karp');
  assert.equal(plan.proof?.globalOptimalityProven, true);
  assert.equal(plan.proof?.graphHash, map.metadata.graphHash);
  assert.ok((plan.preparationSec ?? -1) >= 0);
  assert.ok((plan.planningElapsedSec ?? -1) >= (plan.preparationSec ?? 0));
  assert.ok(Math.abs((plan.createdAt ?? 0) - Date.now() / 1000) < 120);
  assert.equal(plan.graphStatistics?.nodes, map.metadata.nodeCount);
  assert.equal(plan.graphStatistics?.edges, map.metadata.directedEdgeCount);
  assert.equal(plan.graphStatistics?.components, map.metadata.componentCount);
  assert.equal(plan.graphStatistics?.gridStepPx, 4);
  assert.equal(plan.graphStatistics?.fineGridStepPx, 1);
  // Planner cache is keyed by `${assetId}:${version}`.
  const cached = await getPlanner(map.state, undefined, assets);
  assert.equal(cached, planner);
  clearPlannerCache();
  const fresh = await getPlanner(map.state, undefined, assets);
  assert.notEqual(fresh, planner);
});

test('plan() returns NEEDS_REVIEW with the baked unresolved messages', async () => {
  const map2 = loadRealMap('szu-2');
  clearPlannerCache();
  const planner = await getPlanner(map2.state, undefined, { metadata: map2.metadata, buffer: map2.buffer, approaches: map2.approaches });
  const controls = map2.state.controls.filter(control => (control.role ?? 'control') === 'control' && control.state !== 'rejected');
  const start = map2.state.controls.find(control => control.role === 'start' || control.role === 'start_finish');
  assert.ok(start);
  const startPixel: Pixel = start.approachPixel ?? start.pixel;
  const fixture = readJSON<MatrixFixture>('tests/fixtures/static/szu-2.matrix.json');
  const request: PlanRequest = {
    assetId: 'szu-2', version: map2.state.version, controlIds: controls.map(control => control.id),
    startPixel, finishPixel: startPixel, mode: 'all_visit', objective: 'min_max_finish_then_sum',
    teams: [{ id: '第 1 队', people: 4, speedMps: 1.4, departureSec: 0 }], timeLimitSec: 5,
  };
  const plan = await planner.plan(request, map2.state.controls);
  assert.equal(plan.status, 'NEEDS_REVIEW');
  assert.deepEqual(plan.unresolved, fixture.unresolved);
  assert.deepEqual(plan.routes, []);
  assert.equal(plan.graphHash, map2.metadata.graphHash);
  assert.equal(plan.version, map2.state.version);
});

test('plan() validates the request exactly like plan_course', async () => {
  const map2 = loadRealMap('szu-2');
  clearPlannerCache();
  const planner = await getPlanner(map2.state, undefined, { metadata: map2.metadata, buffer: map2.buffer, approaches: map2.approaches });
  const start = map2.state.controls.find(control => control.role === 'start' || control.role === 'start_finish');
  assert.ok(start);
  const control = map2.state.controls.find(candidate => (candidate.role ?? 'control') === 'control');
  assert.ok(control);
  const base = (overrides: Partial<PlanRequest>): PlanRequest => ({
    assetId: 'szu-2', version: map2.state.version, controlIds: [control.id],
    startPixel: start.pixel, finishPixel: start.pixel, mode: 'all_visit', objective: 'min_max_finish_then_sum',
    teams: [{ id: '第 1 队', people: 2, speedMps: 1.4, departureSec: 0 }], timeLimitSec: 5, ...overrides,
  });
  await assert.rejects(planner.plan(base({ version: map2.state.version + 1 }), map2.state.controls), /地图已更新，请刷新后重新规划/);
  await assert.rejects(planner.plan(base({ controlIds: [control.id, control.id] }), map2.state.controls), /目标点重复或不存在/);
  await assert.rejects(planner.plan(base({ controlIds: ['missing-id'] }), map2.state.controls), /目标点重复或不存在/);
  await assert.rejects(planner.plan(base({ controlIds: [] }), map2.state.controls), /请至少选择一个目标/);
  await assert.rejects(planner.plan(base({ controlIds: [start.id] }), map2.state.controls), /路线目标不能包含起终点符号或已排除的候选/);
  const rejectedControls = map2.state.controls.map(item => item.id === control.id ? { ...item, state: 'rejected' as const } : item);
  await assert.rejects(planner.plan(base({}), rejectedControls), /路线目标不能包含起终点符号或已排除的候选/);
});

function releaseFor(assetId: string, state: MapState, graphHash: string): StaticRelease {
  const start = state.controls.find(control => control.role === 'start' || control.role === 'start_finish' || control.role === 'finish');
  assert.ok(start);
  const pixel: Pixel = start.approachPixel ?? start.pixel;
  const team = { id: '第 1 队', people: 7, speedMps: 1.4, departureSec: 0 };
  const route = {
    teamId: team.id, people: team.people,
    controlIds: state.controls.filter(control => (control.role ?? 'control') === 'control' && control.state !== 'rejected').map(control => control.id),
    geometry: [], localGeometry: [], distanceM: 0, travelSec: 0, serviceSec: 0, durationSec: 0, departureSec: 0, finishSec: 0,
  };
  return {
    id: '0'.repeat(32), title: '测试课程', assetId, createdAt: Date.now() / 1000,
    plan: {
      id: '1'.repeat(16), assetId, status: 'OPTIMAL', routes: [route], version: state.version,
      graphHash, draft: false, routingSchema: 'native-quadtree-v4-roof-palette',
      request: {
        assetId, version: state.version, controlIds: route.controlIds, startPixel: pixel, finishPixel: pixel,
        mode: 'all_visit', objective: 'min_max_finish_then_sum', teams: [team], timeLimitSec: 5,
      },
      proof: { algorithm: 'held_karp', globalOptimalityProven: true, scope: '', elapsedSec: 0, objectiveValue: [0, 0] },
    },
    controls: state.controls, frame: state.frame, extent: [], quality: state.quality, review: null,
    version: state.version, routingSchema: 'native-quadtree-v4-roof-palette', superseded: false, static: true,
  };
}

test('navigate re-plans a baked static course from the current location', async () => {
  const map = loadRealMap('szu-3');
  clearPlannerCache();
  const planner = await getPlanner(map.state, undefined, { metadata: map.metadata, buffer: map.buffer, approaches: map.approaches });
  const release = releaseFor('szu-3', map.state, map.metadata.graphHash);
  const mesh = new Mesh(map.state.registration);
  const startPointer = release.plan.request as PlanRequest;
  const mapped = mesh.forward(startPointer.startPixel);
  assert.ok(mapped.inside);
  const lnglat = frameLngLat(mapped.local, map.state.frame.origin);
  const route = release.plan.routes[0];
  const request: NavigationRequest = {
    teamId: route.teamId, lnglat, accuracyM: 5, timestampMs: Date.now(), completedControlIds: [],
  };
  const response = await planner.navigate(release, request, map.state.controls);
  assert.equal(response.status, 'READY');
  assert.equal(response.version, map.state.version);
  assert.equal(response.graphHash, map.metadata.graphHash);
  assert.deepEqual(response.completedControlIds, []);
  assert.ok((response.generatedAt ?? 0) > 0);
  assert.ok(response.route);
  assert.deepEqual(response.route.controlIds, route.controlIds);
  assert.equal(response.route.teamId, route.teamId);
  assert.ok(response.route.geometry.length > 2);
  assert.equal(response.route.localGeometry.length, response.route.geometry.length);
  assert.ok(response.route.distanceM > 0);
  assert.ok(response.route.travelSec >= 0);
  assert.ok(response.route.serviceSec > 0);
  assert.equal(response.route.departureSec, 0);
  assert.equal(response.route.finishSec, response.route.durationSec);
  assert.match(String(response.scope), /保持发布时的点位分配和访问顺序/);
  // Partial progress keeps the published order.
  const second = await planner.navigate(release, { ...request, completedControlIds: [route.controlIds[0]] }, map.state.controls);
  assert.equal(second.status, 'READY');
  assert.deepEqual(second.route?.controlIds, route.controlIds.slice(1));
});

test('navigate validates accuracy, freshness, team, progress and course staleness', async () => {
  const map = loadRealMap('szu-3');
  clearPlannerCache();
  const planner = await getPlanner(map.state, undefined, { metadata: map.metadata, buffer: map.buffer, approaches: map.approaches });
  const release = releaseFor('szu-3', map.state, map.metadata.graphHash);
  const route = release.plan.routes[0];
  const request = release.plan.request as PlanRequest;
  const mesh = new Mesh(map.state.registration);
  const mapped = mesh.forward(request.startPixel);
  assert.ok(mapped.inside);
  const lnglat = frameLngLat(mapped.local, map.state.frame.origin);
  const base: NavigationRequest = { teamId: route.teamId, lnglat, accuracyM: 5, timestampMs: Date.now(), completedControlIds: [] };
  await assert.rejects(planner.navigate(release, { ...base, accuracyM: 25 }, map.state.controls), /定位精度不足，请移至开阔处后重试/);
  await assert.rejects(planner.navigate(release, { ...base, timestampMs: Date.now() - 60000 }, map.state.controls), /定位已过期，请等待新的定位样本/);
  await assert.rejects(planner.navigate(release, { ...base, timestampMs: Date.now() + 30000 }, map.state.controls), /定位已过期，请等待新的定位样本/);
  await assert.rejects(planner.navigate(release, { ...base, teamId: '不存在的队' }, map.state.controls), /队伍不属于这份课程/);
  await assert.rejects(planner.navigate(release, { ...base, completedControlIds: [route.controlIds[1]] }, map.state.controls), /已完成点必须按本队课程顺序记录，不能跳过必到点/);
  await assert.rejects(
    planner.navigate({ ...release, superseded: true }, base, map.state.controls),
    /课程已更新，请使用教师提供的新课程链接/,
  );
  await assert.rejects(
    planner.navigate({ ...release, routingSchema: 'stale-schema' }, base, map.state.controls),
    /课程已更新，请使用教师提供的新课程链接/,
  );
  await assert.rejects(
    planner.navigate({ ...release, version: release.version + 1 }, base, map.state.controls),
    /课程已更新，请使用教师提供的新课程链接/,
  );
  await assert.rejects(
    planner.navigate({ ...release, plan: { ...release.plan, graphHash: 'deadbeefdeadbeefdead' } }, base, map.state.controls),
    /课程通行图与当前静态构建不一致，无法重新规划（本机通行修正不会随课程分发）/,
  );
  await assert.rejects(
    planner.navigate(release, { ...base, lnglat: [0, 0] }, map.state.controls),
    /当前位置超出已校准的课程范围/,
  );
});

test('navigate returns NEEDS_REVIEW when a remaining approach is unresolved', async () => {
  const map2 = loadRealMap('szu-2');
  clearPlannerCache();
  const planner = await getPlanner(map2.state, undefined, { metadata: map2.metadata, buffer: map2.buffer, approaches: map2.approaches });
  const start = map2.state.controls.find(control => control.role === 'start' || control.role === 'start_finish');
  const unresolvedControl = map2.state.controls.find(control => control.id === 'c2');
  assert.ok(start && unresolvedControl);
  const pixel: Pixel = start.approachPixel ?? start.pixel;
  const release = releaseFor('szu-2', map2.state, map2.metadata.graphHash);
  const route = release.plan.routes[0];
  route.controlIds = ['c2'];
  (release.plan.request as PlanRequest).controlIds = ['c2'];
  const mesh = new Mesh(map2.state.registration);
  const mapped = mesh.forward(pixel);
  assert.ok(mapped.inside);
  const lnglat = frameLngLat(mapped.local, map2.state.frame.origin);
  const response = await planner.navigate(release, {
    teamId: route.teamId, lnglat, accuracyM: 5, timestampMs: Date.now(), completedControlIds: [],
  }, map2.state.controls);
  assert.equal(response.status, 'NEEDS_REVIEW');
  assert.equal(response.route, null);
  assert.deepEqual(response.unresolved, [{
    id: 'c2', message: '圈心处的地物或通行侧不明确，请核对实际到访位置',
  }]);
  assert.equal(response.graphHash, map2.metadata.graphHash);
});

test('teacher-confirmed approachPixel uses the manual snap path', async () => {
  const map2 = loadRealMap('szu-2');
  const document = map2.approaches as unknown as {
    startComponent: number;
    controls: Record<string, { ok: boolean; pixel?: Pixel; component?: number }>;
  };
  const found = Object.entries(document.controls)
    .find(([, value]) => value.ok && value.component === document.startComponent && value.pixel);
  assert.ok(found, 'a baked ok approach in the start component');
  const [controlId, baked] = found;
  const controls = map2.state.controls.map(control => control.id === controlId
    ? { ...control, approachPixel: baked.pixel as Pixel }
    : control);
  const state: MapState = { ...map2.state, version: map2.state.version + 1, controls };
  const start = state.controls.find(control => control.role === 'start' || control.role === 'start_finish');
  assert.ok(start);
  clearPlannerCache();
  const planner = await getPlanner(state, undefined, { metadata: map2.metadata, buffer: map2.buffer, approaches: map2.approaches });
  const startPixel: Pixel = start.approachPixel ?? start.pixel;
  const request: PlanRequest = {
    assetId: 'szu-2', version: state.version, controlIds: [controlId], startPixel, finishPixel: startPixel,
    mode: 'all_visit', objective: 'min_max_finish_then_sum',
    teams: [{ id: '第 1 队', people: 2, speedMps: 1.4, departureSec: 0 }], timeLimitSec: 5,
  };
  const plan = await planner.plan(request, controls);
  assert.equal(plan.status, 'OPTIMAL');
  const approach = plan.approaches?.find(item => item.controlId === controlId);
  assert.ok(approach);
  assert.equal(approach.method, 'manual');
  assert.equal(approach.automatic, false);
  assert.deepEqual(approach.pixel, baked.pixel);
});

test('edited planner rebuilds the graph and re-resolves approaches', async () => {
  const map2 = loadRealMap('szu-2');
  const { graph: baseGraph } = await buildRealGraph('szu-2');
  const picked = Math.floor(baseGraph.nodeCount / 2);
  const editPixel: Pixel = [baseGraph.pixels[picked * 2], baseGraph.pixels[picked * 2 + 1]];
  const edits = [{ kind: 'close' as const, pixels: [editPixel], radiusPx: 4 }];
  const editedState: MapState = { ...map2.state, version: map2.state.version + 1, edits };
  clearPlannerCache();
  const planner = await getPlanner(editedState, undefined, { metadata: map2.metadata, buffer: map2.buffer, approaches: map2.approaches });
  assert.match(planner.graphHash, new RegExp(`^${map2.metadata.graphHash}-e[0-9a-f]{16}$`));
  const start = editedState.controls.find(control => control.role === 'start' || control.role === 'start_finish');
  const control = editedState.controls.find(candidate => (candidate.role ?? 'control') === 'control' && candidate.id !== 'c2' && candidate.id !== 'c7');
  assert.ok(start && control);
  const startPixel: Pixel = start.approachPixel ?? start.pixel;
  const request: PlanRequest = {
    assetId: 'szu-2', version: editedState.version, controlIds: [control.id],
    startPixel, finishPixel: startPixel, mode: 'all_visit', objective: 'min_max_finish_then_sum',
    teams: [{ id: '第 1 队', people: 2, speedMps: 1.4, departureSec: 0 }], timeLimitSec: 5,
  };
  const plan = await planner.plan(request, editedState.controls);
  assert.equal(plan.graphHash, planner.graphHash);
  assert.equal(plan.status, 'OPTIMAL');
  assert.equal(plan.routes.length, 1);
  const approach = plan.approaches?.find(item => item.controlId === control.id);
  assert.ok(approach, 'edited planner still returns a re-resolved approach');
  assert.ok(Number.isFinite(approach.pixel[0]) && Number.isFinite(approach.pixel[1]));
  assert.equal(approach.controlId, control.id);
  assert.ok(typeof approach.method === 'string' && approach.method.length > 0);
  assert.ok(Number.isFinite(approach.lnglat[0]) && Number.isFinite(approach.lnglat[1]));
});
