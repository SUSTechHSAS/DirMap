/** Graph-construction invariants, snapping and shortest paths against the
 * baked bundle and Python-generated reference fixtures. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { RasterGraph, loadGraphBundle } from '../../web/src/raster-graph.ts';
import type { GraphMetadata } from '../../web/src/raster-graph.ts';
import type { Pixel, TerrainEdit } from '../../web/src/types.ts';
import { assertCloseRelative, buildRealGraph, loadRealMap, missingData, readJSON } from './helpers.ts';

interface SegmentGolden { a: Pixel; b: Pixel; pixels: Pixel[]; clear: boolean }
interface SnapGolden { controlId: string; pixel: Pixel; node?: number; exact?: number | null; error?: string }
interface PythonReference {
  width: number; height: number; nodeCount: number; graphHash: string;
  meshSamples: unknown[];
  nodeLocal: { node: number; pixel: Pixel; local: Pixel }[];
  segments: SegmentGolden[];
  snaps: SnapGolden[];
}
interface DijkstraGolden {
  assetId: string; graphHash: string;
  samples: { from: number; to: number; fromPixel: Pixel; toPixel: Pixel; time: number; distance: number }[];
}

const reference = readJSON<Record<string, PythonReference>>('tests/static/fixtures/python-reference.json');
const assets = ['szu-1', 'szu-2', 'szu-3'];
const graphCache = new Map<string, Promise<{ graph: RasterGraph; map: ReturnType<typeof loadRealMap> }>>();

function realGraph(assetId: string) {
  let pending = graphCache.get(assetId);
  if (!pending) {
    pending = buildRealGraph(assetId);
    graphCache.set(assetId, pending);
  }
  return pending;
}

test('real graph bundles reproduce every baked invariant', async () => {
  for (const assetId of assets) {
    const missing = missingData(assetId);
    assert.equal(missing, false, String(missing));
    const progress: { percent: number; message: string }[] = [];
    const map = loadRealMap(assetId);
    const graph = await RasterGraph.load(assetId, map.state, {
      loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
      progress: (percent, message) => progress.push({ percent, message }),
    });
    const metadata = map.metadata;
    assert.equal(graph.nodeCount, metadata.nodeCount, `${assetId} nodeCount`);
    assert.equal(graph.edgeCount, metadata.directedEdgeCount, `${assetId} directedEdgeCount`);
    assert.equal(graph.componentCount, metadata.componentCount, `${assetId} componentCount`);
    assert.equal(graph.width, metadata.width);
    assert.equal(graph.height, metadata.height);
    assert.equal(graph.metadata.graphHash, metadata.graphHash);
    assert.ok(progress.some(event => event.message.includes('载入通行图')));
    assert.ok(progress.some(event => event.message.includes('建立保留细墙的通行图')));
    assert.ok(progress.some(event => event.message.includes('完成通行图')));
    // expected.* are float64 undirected-pair sums in canonical order.
    let lengthSum = 0;
    let weightSum = 0;
    let pairCount = 0;
    for (let node = 0; node < graph.nodeCount; node++) {
      for (let edge = graph.indptr[node]; edge < graph.indptr[node + 1]; edge++) {
        if (graph.indices[edge] > node) {
          lengthSum += graph.lengths[edge];
          weightSum += graph.weights[edge];
          pairCount++;
        }
      }
    }
    assert.equal(pairCount, metadata.expected.edgeCount, `${assetId} undirected pair count`);
    assertCloseRelative(lengthSum, metadata.expected.lengthSum, 1e-6, `${assetId} lengthSum`);
    assertCloseRelative(weightSum, metadata.expected.weightSum, 1e-6, `${assetId} weightSum`);
  }
});

test('CSR rows are canonical, sorted and symmetric', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    for (let node = 0; node < graph.nodeCount; node++) {
      const start = graph.indptr[node];
      const end = graph.indptr[node + 1];
      for (let edge = start; edge < end; edge++) {
        assert.ok(Number.isFinite(graph.lengths[edge]) && graph.lengths[edge] >= 0, `${assetId} length ${edge}`);
        assert.ok(Number.isFinite(graph.weights[edge]) && graph.weights[edge] >= 0, `${assetId} weight ${edge}`);
        if (edge > start) {
          assert.ok(graph.indices[edge - 1] < graph.indices[edge], `${assetId} row ${node} columns not increasing at ${edge}`);
        }
      }
    }
    // Sample reverse edges for symmetry.
    for (let node = 0; node < graph.nodeCount; node += 97) {
      const edge = graph.indptr[node];
      if (edge >= graph.indptr[node + 1]) continue;
      const to = graph.indices[edge];
      const reverse = graph.edgeMetrics(to, node);
      assert.ok(reverse, `${assetId} missing reverse edge ${node}->${to}`);
      assert.equal(reverse.length, graph.lengths[edge]);
      assert.equal(reverse.weight, graph.weights[edge]);
    }
  }
});

test('mesh local positions match Python for every sampled node', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    for (const sample of reference[assetId].nodeLocal) {
      const node = sample.node;
      assert.equal(graph.pixels[node * 2], sample.pixel[0], `${assetId} node ${node} pixel x`);
      assert.equal(graph.pixels[node * 2 + 1], sample.pixel[1], `${assetId} node ${node} pixel y`);
      assert.ok(Math.abs(graph.local[node * 2] - sample.local[0]) < 1e-9, `${assetId} node ${node} local x`);
      assert.ok(Math.abs(graph.local[node * 2 + 1] - sample.local[1]) < 1e-9, `${assetId} node ${node} local y`);
    }
  }
});

test('Dijkstra reproduces the Python float64 samples within 1e-6 relative', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    const golden = readJSON<DijkstraGolden>(`tests/fixtures/static/${assetId}.dijkstra.json`);
    assert.equal(golden.samples.length, 40, `${assetId} sample count`);
    for (const sample of golden.samples) {
      const time = graph.dijkstra([sample.from], 'time', [sample.to]).distances[sample.to];
      const distance = graph.dijkstra([sample.from], 'distance', [sample.to]).distances[sample.to];
      assert.ok(Number.isFinite(time), `${assetId} ${sample.from}->${sample.to} time unreachable`);
      assert.ok(Number.isFinite(distance), `${assetId} ${sample.from}->${sample.to} distance unreachable`);
      assertCloseRelative(time, sample.time, 1e-6, `${assetId} time ${sample.from}->${sample.to}`);
      assertCloseRelative(distance, sample.distance, 1e-6, `${assetId} distance ${sample.from}->${sample.to}`);
    }
  }
});

test('full predecessor rows reconstruct the same distances', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    const golden = readJSON<DijkstraGolden>(`tests/fixtures/static/${assetId}.dijkstra.json`);
    const sample = golden.samples[0];
    const { distances, predecessors } = graph.dijkstra([sample.from], 'time');
    assertCloseRelative(distances[sample.to], sample.time, 1e-6, `${assetId} full-row distance`);
    assert.equal(predecessors[sample.from], -9999);
    let node = sample.to;
    let steps = 0;
    let sum = 0;
    while (node !== sample.from) {
      const previous = predecessors[node];
      assert.ok(previous >= 0, `${assetId} predecessor chain broken at ${node}`);
      const metrics = graph.edgeMetrics(previous, node);
      assert.ok(metrics);
      sum += metrics.weight;
      node = previous;
      steps++;
      assert.ok(steps < graph.nodeCount);
    }
    assertCloseRelative(sum, sample.time, 1e-6, `${assetId} predecessor chain sum`);
  }
});

test('A* shortestPath matches Dijkstra and returns inclusive endpoints', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    const golden = readJSON<DijkstraGolden>(`tests/fixtures/static/${assetId}.dijkstra.json`);
    for (const sample of golden.samples.slice(0, 3)) {
      for (const metric of ['time', 'distance'] as const) {
        const path = graph.shortestPath(sample.from, sample.to, metric);
        assert.equal(path[0], sample.from);
        assert.equal(path[path.length - 1], sample.to);
        let weight = 0;
        let length = 0;
        for (let index = 0; index + 1 < path.length; index++) {
          const metrics = graph.edgeMetrics(path[index], path[index + 1]);
          assert.ok(metrics, `${assetId} missing path edge ${path[index]}->${path[index + 1]}`);
          weight += metrics.weight;
          length += metrics.length;
        }
        const expected = metric === 'time' ? sample.time : sample.distance;
        assertCloseRelative(metric === 'time' ? weight : length, expected, 1e-6, `${assetId} A* ${metric} ${sample.from}->${sample.to}`);
      }
    }
  }
});

test('segmentPixels and clearSegment reproduce the Python rasterization', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    for (const segment of reference[assetId].segments) {
      const pixels = graph.segmentPixels(segment.a, segment.b);
      assert.deepEqual(pixels, segment.pixels, `${assetId} segment ${segment.a}->${segment.b}`);
      assert.equal(graph.clearSegment(segment.a, segment.b), segment.clear, `${assetId} clear ${segment.a}->${segment.b}`);
    }
    assert.throws(() => graph.segmentPixels([0, 0], [Number.NaN, 1]), /线段坐标无效/);
  }
});

test('snap/node_at reproduce the Python owner-grid behaviour', async () => {
  for (const assetId of assets) {
    const { graph } = await realGraph(assetId);
    for (const golden of reference[assetId].snaps) {
      if (golden.error) {
        assert.throws(() => graph.snap(golden.pixel), (error: Error) => error.message === golden.error, `${assetId} ${golden.controlId} message`);
        continue;
      }
      assert.equal(graph.snap(golden.pixel), golden.node, `${assetId} snap ${golden.controlId}`);
      assert.equal(graph.node_at(golden.pixel, 1), golden.exact, `${assetId} node_at ${golden.controlId}`);
    }
    assert.throws(() => graph.snap([Number.NaN, 0]), /到访坐标无效/);
  }
});

test('edited graphs rebuild locally without opening blocked pixels', async () => {
  const map = loadRealMap('szu-2');
  const { graph: baseGraph } = await realGraph('szu-2');
  const picked = Math.floor(baseGraph.nodeCount / 2);
  const editPixel: Pixel = [baseGraph.pixels[picked * 2], baseGraph.pixels[picked * 2 + 1]];
  const edits: TerrainEdit[] = [{ kind: 'close', pixels: [editPixel], radiusPx: 4 }];
  const editedState = { ...map.state, version: map.state.version + 1, edits };
  const progress: number[] = [];
  const progressMessages: string[] = [];
  const graph = await RasterGraph.load('szu-2', editedState, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
    edits,
    progress: (percent, message) => { progress.push(percent); progressMessages.push(message); },
  });
  assert.match(graph.metadata.graphHash, new RegExp(`^${map.metadata.graphHash}-e[0-9a-f]{16}$`));
  assert.ok(progress.includes(20) && progress.includes(60) && progress.includes(90));
  assert.ok(progressMessages.some(message => message.includes('建立保留细墙的通行图')));
  assert.notEqual(graph.nodeCount, baseGraph.nodeCount, 'closing a free area must change the quadtree leaves');
  const x = Math.round(editPixel[0]);
  const y = Math.round(editPixel[1]);
  assert.equal(graph.costs[y * graph.width + x], 0, 'closed edit pixel must be blocked');
  // Rebuilt node ids still own exactly their cells.
  let occupied = 0;
  for (let node = 0; node < graph.nodeCount; node++) {
    const size = graph.sizes[node];
    const x0 = Math.round(graph.pixels[node * 2] - (size - 1) / 2);
    const y0 = Math.round(graph.pixels[node * 2 + 1] - (size - 1) / 2);
    assert.ok(x0 >= 0 && y0 >= 0 && x0 + size <= graph.width && y0 + size <= graph.height);
    assert.equal(graph.owner[y0 * graph.width + x0], node);
    occupied++;
  }
  assert.equal(occupied, graph.nodeCount);
  await assert.rejects(async () => await RasterGraph.load('szu-2', { ...editedState, edits: [{ kind: 'open', pixels: [[-1, 2]], radiusPx: 3 }] }, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
    edits: [{ kind: 'open', pixels: [[-1, 2]], radiusPx: 3 }],
  }), /修正区域超出地图/);
  await assert.rejects(async () => await RasterGraph.load('szu-2', { ...editedState, edits: [{ kind: 'open', pixels: [[5, 5]], radiusPx: 0 }] }, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
    edits: [{ kind: 'open', pixels: [[5, 5]], radiusPx: 0 }],
  }), /通行修正无效/);
});

test('graph bundle parser rejects corrupt payloads', () => {
  assert.throws(() => loadGraphBundle(new ArrayBuffer(10)), /通行图数据格式无效/);
  const valid = loadRealMap('szu-2').buffer;
  const corrupted = valid.slice(0);
  new Uint8Array(corrupted)[0] = 0;
  assert.throws(() => loadGraphBundle(corrupted), /通行图数据格式无效/);
  const truncated = valid.slice(0, valid.byteLength - 2);
  assert.throws(() => loadGraphBundle(truncated), /通行图数据长度不足|通行图数据格式无效/);
});

test('graph metadata is exposed unmodified for unedited graphs', async () => {
  const { graph, map } = await realGraph('szu-2');
  const metadata: GraphMetadata = graph.metadata;
  assert.equal(metadata.assetId, map.metadata.assetId);
  assert.equal(metadata.schema, map.metadata.schema);
  assert.deepEqual(metadata.costsPalette, map.metadata.costsPalette);
});
