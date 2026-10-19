/** Client-side `buildGraphFromRaster` golden: rebuilding the navigation graph
 * from the baked costs + registration must reproduce the Python `build_graph`
 * quadtree/domain result exactly (nodes, CSR, components) and the baked
 * Dijkstra samples within spec tolerance. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Mesh, frameLngLat } from '../../web/src/geometry.ts';
import { RasterGraph, buildGraphFromRaster, loadGraphBundle } from '../../web/src/raster-graph.ts';
interface DijkstraGolden {
  samples: { from: number; to: number; time: number; distance: number }[];
}
import { readJSON, loadRealMap, assertCloseRelative } from './helpers.ts';
import type { Pixel, PlanRequest, StaticRelease, TerrainEdit } from '../../web/src/types.ts';


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

const assets = ['szu-1', 'szu-2', 'szu-3'];
test('buildGraphFromRaster reproduces the baked quadtree graph exactly', async (t) => {
  for (const assetId of assets) {
    const map = loadRealMap(assetId);
    const bundle = loadGraphBundle(map.buffer);
    const costs = new Float32Array(bundle.width * bundle.height);
    for (let index = 0; index < costs.length; index++) costs[index] = bundle.palette[bundle.costs[index]];
    const started = performance.now();
    const graph = buildGraphFromRaster(costs, map.state.registration!, map.metadata.sourceHash, [], 4, {
      width: bundle.width,
      height: bundle.height,
      assetId,
      schema: map.metadata.schema,
    });
    const buildMs = performance.now() - started;
    const metadata = map.metadata;
    assert.equal(graph.nodeCount, metadata.nodeCount, `${assetId} nodeCount`);
    assert.equal(graph.edgeCount, metadata.directedEdgeCount, `${assetId} directedEdgeCount`);
    assert.equal(graph.componentCount, metadata.componentCount, `${assetId} componentCount`);
    assert.equal(graph.width, metadata.width);
    assert.equal(graph.height, metadata.height);
    // Node order, leaf size and terrain must match the bundle exactly.
    let pixelMismatch = 0;
    let sizeMismatch = 0;
    let terrainMismatch = 0;
    for (let index = 0; index < graph.nodeCount; index++) {
      if (graph.pixels[index * 2] !== bundle.pixels[index * 2]
        || graph.pixels[index * 2 + 1] !== bundle.pixels[index * 2 + 1]) pixelMismatch++;
      if (graph.sizes[index] !== bundle.sizes[index]) sizeMismatch++;
      if (graph.terrain[index] !== bundle.palette[bundle.terrain[index]]) terrainMismatch++;
    }
    assert.equal(pixelMismatch, 0, `${assetId} node centres`);
    assert.equal(sizeMismatch, 0, `${assetId} node cell sizes`);
    assert.equal(terrainMismatch, 0, `${assetId} node terrain values`);
    // Canonical undirected sums prove float64 local positions match Python.
    let lengthSum = 0;
    let weightSum = 0;
    for (let node = 0; node < graph.nodeCount; node++) {
      for (let edge = graph.indptr[node]; edge < graph.indptr[node + 1]; edge++) {
        if (graph.indices[edge] > node) {
          lengthSum += graph.lengths[edge];
          weightSum += graph.weights[edge];
        }
      }
    }
    assertCloseRelative(lengthSum, metadata.expected.lengthSum, 1e-6, `${assetId} lengthSum`);
    assertCloseRelative(weightSum, metadata.expected.weightSum, 1e-6, `${assetId} weightSum`);
    const golden = readJSON<DijkstraGolden>(`tests/fixtures/static/${assetId}.dijkstra.json`);
    for (const sample of golden.samples) {
      const time = graph.distanceToTargets([sample.from], [sample.to], 'time')[0];
      const distance = graph.distanceToTargets([sample.from], [sample.to], 'distance')[0];
      assertCloseRelative(time, sample.time, 1e-6, `${assetId} rebuilt time ${sample.from}->${sample.to}`);
      assertCloseRelative(distance, sample.distance, 1e-6, `${assetId} rebuilt distance ${sample.from}->${sample.to}`);
    }
    assert.match(graph.metadata.graphHash, /^client-[0-9a-f]{16}$/);
    assert.equal(graph.closedEdits, null);
    if (assetId === 'szu-3') {
      t.diagnostic(`szu-3 buildGraphFromRaster ${buildMs.toFixed(0)} ms`);
      assert.ok(buildMs < 8000, `buildGraphFromRaster szu-3 ${buildMs.toFixed(0)} ms must stay under 8000 ms`);
    }
    // Same inputs must produce the same client hash on a reload.
    if (assetId === 'szu-2') {
      const again = buildGraphFromRaster(costs, map.state.registration!, map.metadata.sourceHash, [], 4,
        { width: bundle.width, height: bundle.height });
      assert.equal(again.metadata.graphHash, graph.metadata.graphHash, `${assetId} client hash stability`);
    }
  }
});

