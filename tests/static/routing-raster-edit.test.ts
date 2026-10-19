/** Client-side raster builds: modified rasters, input validation and the
 * node-query helpers the browser approach resolver relies on. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Mesh } from '../../web/src/geometry.ts';
import { RasterGraph, buildGraphFromRaster, loadGraphBundle } from '../../web/src/raster-graph.ts';
import { loadRealMap } from './helpers.ts';
import type { Pixel, TerrainEdit } from '../../web/src/types.ts';

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
test('buildGraphFromRaster handles arbitrary modified rasters without opening blocked pixels', async () => {
  const map = loadRealMap('szu-2');
  const costs = await bakedCosts('szu-2');
  const modified = costs.costs.slice();
  const edits: TerrainEdit[] = [{ kind: 'close', pixels: [[300, 200]], radiusPx: 6 }];
  // Simulate a client raster with a closed rectangle.
  for (let y = 190; y <= 210; y++) for (let x = 290; x <= 310; x++) modified[y * costs.width + x] = 0;
  const graph = buildGraphFromRaster(modified, map.state.registration!, 'client-source', [], 4, {
    width: costs.width, height: costs.height, assetId: 'uploaded',
  });
  assert.match(graph.metadata.graphHash, /^client-[0-9a-f]{16}$/);
  assert.equal(graph.metadata.assetId, 'uploaded');
  assert.ok(graph.nodeCount > 1000);
  // Every node centre is inside the registration mesh.
  const mesh = new Mesh(map.state.registration);
  const inside = mesh.forwardMany(graph.pixels).inside;
  for (let node = 0; node < graph.nodeCount; node++) assert.equal(inside[node], 1, `node ${node} outside registration`);
  // No blocked closed pixel is owned by a leaf.
  for (let y = 190; y <= 210; y++) {
    for (let x = 290; x <= 310; x++) {
      assert.equal(graph.costs[y * costs.width + x], 0);
      assert.equal(graph.owner[y * costs.width + x], 0xffffffff, `blocked ${x},${y} must not be owned`);
    }
  }
  // Editing the same raster through `edits` changes the hash and fills closedEdits.
  const edited = buildGraphFromRaster(costs.costs, map.state.registration!, 'client-source', edits, 4, {
    width: costs.width, height: costs.height, assetId: 'uploaded',
  });
  assert.notEqual(edited.metadata.graphHash, graph.metadata.graphHash);
  assert.notEqual(edited.metadata.graphHash, buildGraphFromRaster(costs.costs, map.state.registration!, map.metadata.sourceHash, [], 4,
    { width: costs.width, height: costs.height }).metadata.graphHash);
  assert.ok(edited.closedEdits);
  assert.equal(edited.closedEdits![200 * costs.width + 300], 1);
  assert.equal(edited.costs[200 * costs.width + 300], 0);
});

test('buildGraphFromRaster validates its inputs like Python build_graph', async () => {
  const map = loadRealMap('szu-2');
  const costs = await bakedCosts('szu-2');
  const registrationWithoutSize = { ...map.state.registration! };
  delete (registrationWithoutSize as { width?: number }).width;
  delete (registrationWithoutSize as { height?: number }).height;
  assert.throws(() => buildGraphFromRaster(costs.costs, registrationWithoutSize, 'x', [], 4),
    /缺少通行图尺寸/);
  assert.throws(() => buildGraphFromRaster(costs.costs.slice(0, 100), map.state.registration!, 'x', [], 4,
    { width: costs.width, height: costs.height }), /通行图尺寸无效/);
  assert.throws(() => buildGraphFromRaster(costs.costs, map.state.registration!, 'x', [], 0,
    { width: costs.width, height: costs.height }), /通行网格步长无效/);
  assert.throws(() => buildGraphFromRaster(costs.costs, map.state.registration!, 'x', [], 17,
    { width: costs.width, height: costs.height }), /通行网格步长无效/);
  assert.throws(() => buildGraphFromRaster(costs.costs, map.state.registration!, 'x', [], 3,
    { width: costs.width, height: costs.height }), /自适应网格步长必须是 1、2、4、8 或 16/);
  assert.throws(() => buildGraphFromRaster(costs.costs, map.state.registration!, 'x',
    [{ kind: 'close', pixels: [[-1, 0]], radiusPx: 4 }], 4,
    { width: costs.width, height: costs.height }), /修正区域超出地图/);
  assert.throws(() => buildGraphFromRaster(costs.costs, map.state.registration!, 'x',
    [{ kind: 'open', pixels: [[5, 5]], radiusPx: 0 }], 4,
    { width: costs.width, height: costs.height }), /通行修正无效/);
});


test('RasterGraph query helpers used by the approach resolver', async () => {
  const map = loadRealMap('szu-2');
  const costs = await bakedCosts('szu-2');
  const graph = buildRaster('szu-2', map, costs);
  const node = Math.floor(graph.nodeCount / 3);
  const centre: Pixel = [graph.pixels[node * 2], graph.pixels[node * 2 + 1]];
  assert.equal(graph.nearestNode(centre), node);
  assert.equal(graph.nodeAtPixel(centre), node);
  assert.ok(graph.queryNodesNear(centre, 0).includes(node));
  const near = graph.queryNodesNear(centre, 12);
  assert.ok(near.length >= 1);
  let previous = -1;
  for (const candidate of near) {
    const dx = graph.pixels[candidate * 2] - centre[0];
    const dy = graph.pixels[candidate * 2 + 1] - centre[1];
    const distance = Math.hypot(dx, dy);
    assert.ok(distance <= 12 + 1e-9);
    assert.ok(distance >= previous - 1e-9, 'queryNodesNear must be nearest-first');
    previous = distance;
  }
  assert.equal(graph.nearestNode([-10000, -10000]), null);
  assert.equal(graph.nodeAtPixel([-10, 0]), null);
  assert.equal(graph.nodeAtPixel([Number.NaN, 0]), null);
  // closedEdits is null on a graph built without close edits and set otherwise.
  assert.equal(graph.closedEdits, null);
  const edited = buildGraphFromRaster(costs.costs, map.state.registration!, map.metadata.sourceHash,
    [{ kind: 'close', pixels: [[300, 200]], radiusPx: 5 }], 4, { width: costs.width, height: costs.height });
  assert.ok(edited.closedEdits);
  assert.equal(edited.closedEdits![200 * costs.width + 300], 1);
});

test('RasterGraph.fromArrays rebuilds an identical graph and keeps the graphHash override', async () => {
  const map = loadRealMap('szu-2');
  const costs = await bakedCosts('szu-2');
  const graph = buildRaster('szu-2', map, costs);
  const rebuilt = RasterGraph.fromArrays({
    width: graph.width,
    height: graph.height,
    step: graph.step,
    costs: graph.costs,
    pixels: graph.pixels,
    sizes: graph.sizes,
    terrain: graph.terrain,
    registration: map.state.registration!,
    metadata: { graphHash: 'client-0123456789abcdef', assetId: 'from-arrays', schema: graph.metadata.schema },
  });
  assert.equal(rebuilt.nodeCount, graph.nodeCount);
  assert.equal(rebuilt.edgeCount, graph.edgeCount);
  assert.equal(rebuilt.componentCount, graph.componentCount);
  assert.equal(rebuilt.metadata.graphHash, 'client-0123456789abcdef');
  assert.equal(rebuilt.metadata.assetId, 'from-arrays');
  assert.equal(rebuilt.metadata.nodeCount, graph.nodeCount);
  for (let index = 0; index < graph.nodeCount; index++) {
    assert.equal(rebuilt.pixels[index * 2], graph.pixels[index * 2]);
    assert.equal(rebuilt.pixels[index * 2 + 1], graph.pixels[index * 2 + 1]);
  }
  assert.deepEqual(rebuilt.owner, graph.owner);
});
