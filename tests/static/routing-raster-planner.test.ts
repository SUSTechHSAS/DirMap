/** Planner factory for client-built graphs: plan + navigate without baked
 * bundle fetches, using approaches resolved against the supplied graph. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Mesh, frameLngLat } from '../../web/src/geometry.ts';
import { buildGraphFromRaster, loadGraphBundle } from '../../web/src/raster-graph.ts';
import { createPlannerFromGraph } from '../../web/src/planner.ts';
import { loadRealMap } from './helpers.ts';
import type { Pixel, PlanRequest, StaticRelease } from '../../web/src/types.ts';

interface BakedCosts {
  width: number;
  height: number;
  costs: Float32Array;
}

async function bakedCosts(assetId: string): Promise<BakedCosts> {
  const map = loadRealMap(assetId);
  const bundle = loadGraphBundle(map.buffer);
  const costs = new Float32Array(bundle.width * bundle.height);
  for (let index = 0; index < costs.length; index++) costs[index] = bundle.palette[bundle.costs[index]];
  return { width: bundle.width, height: bundle.height, costs };
}

function buildRaster(assetId: string, map: ReturnType<typeof loadRealMap>, costs: BakedCosts) {
  return buildGraphFromRaster(costs.costs, map.state.registration!, map.metadata.sourceHash, [], 4, {
    width: costs.width,
    height: costs.height,
    assetId,
    schema: map.metadata.schema,
  });
}
test('createPlannerFromGraph plans and navigates a client-built graph', async () => {
  const map = loadRealMap('szu-2');
  const costs = await bakedCosts('szu-2');
  // The rebuilt graph is node-identical to the baked bundle, so the baked
  // approaches.json entries are valid for it.
  const graph = buildRaster('szu-2', map, costs);
  const planner = createPlannerFromGraph(graph, map.approaches, map.state);
  assert.equal(planner.graphHash, graph.metadata.graphHash);
  assert.equal(createPlannerFromGraph(graph, map.approaches, map.state), planner, 'planner cached by graph hash');
  const start = map.state.controls.find(control => control.role === 'start' || control.role === 'start_finish');
  assert.ok(start);
  const startPixel: Pixel = start.approachPixel ?? start.pixel;
  const request: PlanRequest = {
    assetId: 'szu-2', version: map.state.version, controlIds: ['c3'], startPixel, finishPixel: startPixel,
    mode: 'all_visit', objective: 'min_max_finish_then_sum',
    teams: [{ id: '第 1 队', people: 2, speedMps: 1.4, departureSec: 0 }], timeLimitSec: 5,
  };
  const plan = await planner.plan(request, map.state.controls);
  assert.equal(plan.status, 'OPTIMAL');
  assert.equal(plan.graphHash, graph.metadata.graphHash);
  assert.match(plan.graphHash, /^client-[0-9a-f]{16}$/);
  assert.equal(plan.routes.length, 1);
  assert.deepEqual(plan.routes[0].controlIds, ['c3']);
  // Navigation over a frozen client release with the same graph hash.
  const mesh = new Mesh(map.state.registration);
  const mapped = mesh.forward(startPixel);
  assert.ok(mapped.inside);
  const lnglat = frameLngLat(mapped.local, map.state.frame.origin);
  const team = request.teams[0];
  const release: StaticRelease = {
    id: '0'.repeat(32), title: '客户端课程', assetId: 'szu-2', createdAt: Date.now() / 1000,
    plan: {
      id: '1'.repeat(16), assetId: 'szu-2', status: 'OPTIMAL', version: map.state.version,
      graphHash: graph.metadata.graphHash, draft: false, routingSchema: graph.metadata.schema,
      routes: [{
        teamId: team.id, people: team.people, controlIds: ['c3'], geometry: [], localGeometry: [],
        distanceM: 0, travelSec: 0, serviceSec: 0, durationSec: 0, departureSec: 0, finishSec: 0,
      }],
      request: { ...request, controlIds: ['c3'] },
      proof: { algorithm: 'held_karp', globalOptimalityProven: true, scope: '', elapsedSec: 0, objectiveValue: [0, 0] },
    },
    controls: map.state.controls, frame: map.state.frame, extent: [], quality: map.state.quality, review: null,
    version: map.state.version, routingSchema: graph.metadata.schema, superseded: false, static: true,
  };
  const response = await planner.navigate(release, {
    teamId: team.id, lnglat, accuracyM: 5, timestampMs: Date.now(), completedControlIds: [],
  }, map.state.controls);
  assert.equal(response.status, 'READY');
  assert.equal(response.graphHash, graph.metadata.graphHash);
  assert.ok(response.route);
  assert.deepEqual(response.route.controlIds, ['c3']);
});
