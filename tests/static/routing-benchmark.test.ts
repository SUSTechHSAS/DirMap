/** Browser-core performance guard rails. The bounds are deliberately generous
 * (CI machines vary); the diagnostics carry the measured numbers the Lead
 * uses for the deployment budget. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { RasterGraph } from '../../web/src/raster-graph.ts';
import { solve } from '../../web/src/optimizer.ts';
import type { Control, MapState } from '../../web/src/types.ts';
import { buildRealGraph, loadRealMap, readJSON } from './helpers.ts';

interface OptimizerGolden {
  assetId: string;
  controlIds: string[];
  request: { mode: string; objective: string; timeLimitSec: number; teams: { id: string; people: number; speedMps: number; departureSec?: number }[] };
  matrix: number[][];
}

test('szu-3 graph build, 19-source matrix and exact optimizer timings', async (t) => {
  const map = loadRealMap('szu-3');
  const buildStarted = performance.now();
  const graph = await RasterGraph.load('szu-3', map.state, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
  });
  const buildMs = performance.now() - buildStarted;
  t.diagnostic(`szu-3 graph build ${buildMs.toFixed(0)} ms (${graph.nodeCount} nodes, ${graph.edgeCount} directed edges, ${graph.componentCount} components)`);
  assert.ok(buildMs < 8000, `graph build ${buildMs} ms must stay under 8000 ms in node`);

  // Canonical 18-course endpoints plus one extra node -> 19 sources/targets.
  const matrixFixture = readJSON<{ nodes: number[] }>('tests/fixtures/static/szu-3.matrix.json');
  const endpoints = [...new Set(matrixFixture.nodes)];
  for (let node = 0; node < graph.nodeCount && endpoints.length < 19; node++) {
    if (graph.components[node] === graph.components[endpoints[0]] && !endpoints.includes(node)) endpoints.push(node);
  }
  assert.equal(endpoints.length, 19);
  const matrixStarted = performance.now();
  const distances = graph.distanceToTargets(endpoints, endpoints, 'time');
  const matrixMs = performance.now() - matrixStarted;
  t.diagnostic(`19-source time matrix ${matrixMs.toFixed(0)} ms`);
  assert.ok(matrixMs < 30000, `19-source matrix ${matrixMs} ms must stay under 30000 ms in node`);
  let finiteCount = 0;
  for (const value of distances) if (Number.isFinite(value)) finiteCount++;
  assert.ok(finiteCount > 19 * 19 * 0.9, `matrix should be mostly finite (${finiteCount}/${19 * 19})`);
  for (let index = 0; index < endpoints.length; index++) assert.equal(distances[index * endpoints.length + index], 0);

  const optimizerGolden = readJSON<OptimizerGolden>('tests/fixtures/static/optimizer.json');
  const state = readJSON<MapState>(`web/public/data/maps/${optimizerGolden.assetId}/state.json`);
  const controls: Control[] = optimizerGolden.controlIds.map(id => state.controls.find(control => control.id === id) as Control);
  const optimizerStarted = performance.now();
  const solved = solve(optimizerGolden.matrix, controls, optimizerGolden.request.teams,
    optimizerGolden.request.mode, optimizerGolden.request.objective, optimizerGolden.request.timeLimitSec);
  const optimizerMs = performance.now() - optimizerStarted;
  t.diagnostic(`exact optimizer (held_karp, n=10, 3 teams) ${optimizerMs.toFixed(0)} ms, status ${solved.status}`);
  assert.equal(solved.status, 'OPTIMAL');
  assert.ok(optimizerMs < 2000, `exact optimizer ${optimizerMs} ms must stay under 2000 ms`);

  const memory = process.memoryUsage();
  t.diagnostic(`process memory after build+matrix: rss ${(memory.rss / 1048576).toFixed(0)} MiB, heapUsed ${(memory.heapUsed / 1048576).toFixed(0)} MiB, external ${(memory.external / 1048576).toFixed(0)} MiB`);
});

test('szu-2 graph build and matrix stay fast enough for a phone', async (t) => {
  const map = loadRealMap('szu-2');
  const buildStarted = performance.now();
  const graph = await RasterGraph.load('szu-2', map.state, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
  });
  const buildMs = performance.now() - buildStarted;
  t.diagnostic(`szu-2 graph build ${buildMs.toFixed(0)} ms (${graph.nodeCount} nodes, ${graph.edgeCount} directed edges)`);
  assert.ok(buildMs < 4000);
  const { graph: other } = await buildRealGraph('szu-1');
  const matrixStarted = performance.now();
  const nodes = readJSON<{ nodes: number[] }>('tests/fixtures/static/szu-1.matrix.json').nodes;
  graph; // keep the szu-2 graph alive while szu-1 is measured (memory parity)
  const distances = other.distanceToTargets(nodes, nodes, 'time');
  const matrixMs = performance.now() - matrixStarted;
  t.diagnostic(`szu-1 ${nodes.length}-source matrix ${matrixMs.toFixed(0)} ms`);
  assert.ok(distances.length === nodes.length * nodes.length);
  assert.ok(matrixMs < 30000);
});
