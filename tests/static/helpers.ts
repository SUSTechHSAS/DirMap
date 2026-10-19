/** Shared fixtures for the static routing-core golden tests. */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { RasterGraph } from '../../web/src/raster-graph.ts';
import type { GraphMetadata } from '../../web/src/raster-graph.ts';
import type { BakedApproaches, MapState } from '../../web/src/types.ts';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

export function repoPath(...parts: string[]): string {
  return resolve(ROOT, ...parts);
}

export function readJSON<T>(relative: string): T {
  return JSON.parse(readFileSync(repoPath(relative), 'utf8')) as T;
}

export function exists(relative: string): boolean {
  return existsSync(repoPath(relative));
}

export interface RealMap {
  assetId: string;
  state: MapState;
  metadata: GraphMetadata;
  approaches: BakedApproaches;
  buffer: ArrayBuffer;
}

export function dataPaths(assetId: string) {
  return {
    state: repoPath('web/public/data/maps', assetId, 'state.json'),
    metadata: repoPath('web/public/data/maps', assetId, 'graph.json'),
    approaches: repoPath('web/public/data/maps', assetId, 'approaches.json'),
    bundle: repoPath('web/public/data/maps', assetId, 'graph.bin.gz'),
  };
}

export function missingData(assetId: string): string | false {
  const paths = dataPaths(assetId);
  const missing = Object.values(paths).filter(path => !existsSync(path));
  if (missing.length) return `missing baked data: ${missing.join(', ')}`;
  return false;
}

export function loadRealMap(assetId: string): RealMap {
  const paths = dataPaths(assetId);
  const state = JSON.parse(readFileSync(paths.state, 'utf8')) as MapState;
  const metadata = JSON.parse(readFileSync(paths.metadata, 'utf8')) as GraphMetadata;
  const approaches = JSON.parse(readFileSync(paths.approaches, 'utf8')) as BakedApproaches;
  const compressed = readFileSync(paths.bundle);
  const inflated = gunzipSync(compressed);
  const buffer = inflated.buffer.slice(inflated.byteOffset, inflated.byteOffset + inflated.byteLength) as ArrayBuffer;
  return { assetId, state, metadata, approaches, buffer };
}

export async function buildRealGraph(assetId: string): Promise<{ graph: RasterGraph; map: RealMap }> {
  const map = loadRealMap(assetId);
  const graph = await RasterGraph.load(assetId, map.state, {
    loader: { metadata: async () => map.metadata, buffer: async () => map.buffer },
  });
  return { graph, map };
}

/** |actual - expected| <= tolerance * max(1, |expected|). */
export function assertCloseRelative(actual: number, expected: number, tolerance: number, label: string): void {
  const difference = Math.abs(actual - expected);
  if (!Number.isFinite(actual) || difference > tolerance * Math.max(1, Math.abs(expected))) {
    throw new Error(`${label}: ${actual} vs expected ${expected} (difference ${difference}, tolerance ${tolerance}*max(1,|expected|))`);
  }
}
