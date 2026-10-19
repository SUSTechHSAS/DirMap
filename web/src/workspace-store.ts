/** Browser-local workspace for client-processed maps (IndexedDB + object URLs).
 *
 * The baked data under `web/public/data/` stays the immutable default. This
 * module stores maps the browser produced itself (re-processed bundled sheets or
 * uploaded images) together with the heavy rasters the pipeline needs for
 * planning and re-registration. Small catalog metadata lives in localStorage so
 * the map list can be rendered synchronously; everything else is in IndexedDB.
 */
import type { AnalysisResult, RegionFeature } from './vision';
import type { BakedApproaches, MapState, Pixel } from './types';
import { packSemantics, unpackSemantics, type SemanticsMaskMap } from './semantics';

export type LocalMapKind = 'processed' | 'upload' | 'terrain';

export interface LocalMapSummary {
  assetId: string;
  name: string;
  filename: string;
  sha256: string;
  kind: LocalMapKind;
  createdAt: number;
  width: number;
  height: number;
}

/** Persisted analysis payload: everything planning/terrain refresh needs later. */
export interface StoredAnalysis {
  width: number;
  height: number;
  controls: AnalysisResult['controls'];
  features: RegionFeature[];
  costsIndex: Uint8Array;
  costsPalette: number[];
  semantics: Uint8Array; // packed DMSEM1 bytes
  statistics: AnalysisResult['statistics'];
  warnings: string[];
  sourceKind: LocalMapSummary['kind'];
}

export interface LocalMapRecord {
  summary: LocalMapSummary;
  state: MapState;
  analysis: StoredAnalysis;
  approaches: BakedApproaches;
}

const DB_NAME = 'dirmap-workspace-v1';
const DB_VERSION = 1;
const STORE_MAPS = 'maps';
const STORE_FILES = 'files';
const SUMMARY_KEY = 'dirmap-local-maps-v1';
const FILE_KINDS = ['original', 'overlay', 'terrain', 'terrain-warp'] as const;
export type LocalFileKind = typeof FILE_KINDS[number];

let dbPromise: Promise<IDBDatabase> | null = null;
const objectUrls = new Map<string, string>();
const localAssetIds = new Set<string>();

function openDatabase(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('当前浏览器不支持本地工作区（IndexedDB）')); return; }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_MAPS)) database.createObjectStore(STORE_MAPS, { keyPath: 'summary.assetId' });
      if (!database.objectStoreNames.contains(STORE_FILES)) database.createObjectStore(STORE_FILES);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('无法打开本地工作区'));
  });
  return dbPromise;
}

function request<T>(store: IDBObjectStore, operation: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    operation.onsuccess = () => resolve(operation.result);
    operation.onerror = () => reject(operation.error ?? new Error('本地工作区读写失败'));
  });
}

async function withStore<T>(name: string, mode: IDBTransactionMode, action: (store: IDBObjectStore) => Promise<T>): Promise<T> {
  const database = await openDatabase();
  const transaction = database.transaction(name, mode);
  const result = await action(transaction.objectStore(name));
  await new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('本地工作区事务失败'));
    transaction.onabort = () => reject(transaction.error ?? new Error('本地工作区事务中止'));
  });
  return result;
}

function fileKey(assetId: string, file: LocalFileKind): string { return `${assetId}/${file}`; }

function readSummaries(): LocalMapSummary[] {
  try {
    const value = localStorage.getItem(SUMMARY_KEY);
    const list = value ? JSON.parse(value) as LocalMapSummary[] : [];
    return Array.isArray(list) ? list.filter(item => item && typeof item.assetId === 'string') : [];
  } catch { return []; }
}

function writeSummaries(list: LocalMapSummary[]): void {
  try { localStorage.setItem(SUMMARY_KEY, JSON.stringify(list)); } catch { /* private mode: this session keeps the record in memory only */ }
}

export function listLocalMaps(): LocalMapSummary[] {
  return readSummaries().sort((a, b) => b.createdAt - a.createdAt);
}

export function localMapSummary(assetId: string): LocalMapSummary | null {
  return readSummaries().find(item => item.assetId === assetId) ?? null;
}

export function hasLocalMap(assetId: string): boolean {
  return localAssetIds.has(assetId) || readSummaries().some(item => item.assetId === assetId);
}

/** Load the summary list synchronously into the object-URL/lookup cache. */
export function rememberLocalMaps(): LocalMapSummary[] {
  const list = listLocalMaps();
  for (const summary of list) localAssetIds.add(summary.assetId);
  return list;
}

export async function loadLocalMap(assetId: string): Promise<LocalMapRecord | null> {
  try {
    const record = await withStore<LocalMapRecord | undefined>(STORE_MAPS, 'readonly', store => request(store, store.get(assetId)));
    if (!record) return null;
    localAssetIds.add(assetId);
    return record;
  } catch { return null; }
}

/** Persist a processed map and its image asset blobs. */
export async function saveLocalMap(record: LocalMapRecord, files: Partial<Record<LocalFileKind, Blob>> = {}): Promise<void> {
  await withStore(STORE_MAPS, 'readwrite', async store => { await request(store, store.put(record)); });
  for (const [kind, blob] of Object.entries(files)) {
    if (blob) await putLocalFile(record.summary.assetId, kind as LocalFileKind, blob);
  }
  const summaries = readSummaries().filter(item => item.assetId !== record.summary.assetId);
  summaries.push(record.summary);
  writeSummaries(summaries);
  localAssetIds.add(record.summary.assetId);
  // Only replaced files lose their object URL. A state-only save (teacher
  // review/edits) must not revoke the overlay/original URLs still in use.
  for (const kind of Object.keys(files)) clearObjectUrls(record.summary.assetId, kind as LocalFileKind);
}

/** Update only the map state of an existing record (teacher edits, review). */
export async function updateLocalMapState(assetId: string, state: MapState): Promise<void> {
  const record = await loadLocalMap(assetId);
  if (!record) return;
  record.state = state;
  await saveLocalMap(record, {});
}

export async function putLocalFile(assetId: string, kind: LocalFileKind, blob: Blob): Promise<void> {
  await withStore(STORE_FILES, 'readwrite', async store => { await request(store, store.put(blob, fileKey(assetId, kind))); });
  clearObjectUrls(assetId, kind);
}

export async function hasLocalFile(assetId: string, kind: LocalFileKind): Promise<boolean> {
  try {
    const value = await withStore(STORE_FILES, 'readonly', store => request(store, store.getKey(fileKey(assetId, kind))));
    return value !== undefined;
  } catch { return false; }
}

export async function getLocalFile(assetId: string, kind: LocalFileKind): Promise<Blob | null> {
  try {
    const value = await withStore<Blob | undefined>(STORE_FILES, 'readonly', store => request(store, store.get(fileKey(assetId, kind))));
    return value ?? null;
  } catch { return null; }
}

/** Object URL for a locally stored asset; cached until the file is replaced. */
export async function localFileUrl(assetId: string, kind: LocalFileKind): Promise<string | null> {
  const key = fileKey(assetId, kind);
  const existing = objectUrls.get(key);
  if (existing) return existing;
  const blob = await getLocalFile(assetId, kind);
  if (!blob) return null;
  const url = URL.createObjectURL(blob);
  objectUrls.set(key, url);
  return url;
}

function clearObjectUrls(assetId: string, kind?: LocalFileKind): void {
  if (kind) {
    const key = fileKey(assetId, kind);
    const url = objectUrls.get(key);
    if (url) { URL.revokeObjectURL(url); objectUrls.delete(key); }
    return;
  }
  for (const current of FILE_KINDS) clearObjectUrls(assetId, current);
}

export async function deleteLocalMap(assetId: string): Promise<void> {
  try {
    await withStore(STORE_MAPS, 'readwrite', async store => { await request(store, store.delete(assetId)); });
    await withStore(STORE_FILES, 'readwrite', async store => {
      for (const kind of FILE_KINDS) await request(store, store.delete(fileKey(assetId, kind)));
    });
  } catch { /* best effort */ }
  clearObjectUrls(assetId);
  localAssetIds.delete(assetId);
  writeSummaries(readSummaries().filter(item => item.assetId !== assetId));
}

/** Decode packed semantics from a stored analysis record. */
export function storedSemantics(analysis: StoredAnalysis): SemanticsMaskMap {
  return unpackSemantics(analysis.semantics).masks;
}

export function makeStoredAnalysis(result: AnalysisResult, sourceKind: LocalMapSummary['kind']): StoredAnalysis {
  return {
    width: result.width,
    height: result.height,
    controls: result.controls,
    features: result.features,
    costsIndex: result.costsIndex,
    costsPalette: [...result.costsPalette],
    semantics: packSemantics(result.width, result.height, result.semantics as unknown as SemanticsMaskMap,
      ['building', 'water', 'point_ink', 'vegetation', 'restricted']),
    statistics: result.statistics,
    warnings: result.warnings,
    sourceKind,
  };
}

/** Convert palette-indexed costs into the float costs the graph builder uses. */
export function floatCosts(analysis: StoredAnalysis): Float32Array {
  const palette = analysis.costsPalette;
  const costs = new Float32Array(analysis.costsIndex.length);
  for (let index = 0; index < costs.length; index++) costs[index] = palette[analysis.costsIndex[index]] ?? 0;
  return costs;
}

export type { Pixel, MapState };
