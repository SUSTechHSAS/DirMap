/** Static data access for the GitHub Pages build.
 *
 * Every URL is resolved against the Vite base so the application works both at
 * `/` and at a repository sub-path such as `/DirMap/`. There is no server: all
 * reads are ordinary static-file fetches, and every write is local (see
 * `static-store.ts`).
 */
import { escapeHtml, type Approach, type BakedApproaches, type BakedCatalog, type Control, type Health, type MapState, type Plan, type ReleaseSummary, type StaticRelease } from './types';
import { hasLocalMap, loadLocalMap, localFileUrl, rememberLocalMaps, type LocalFileKind, type LocalMapSummary } from './workspace-store';

/** Reference polygon used by the in-browser auto-registration matcher. */
export interface ReferenceFeatureRecord {
  id: string; kind: 'building' | 'water' | 'pitch';
  center: [number, number]; area: number; aspect: number; points: [number, number][];
}
export interface ReferenceFeatures {
  frame?: { crs?: string; origin?: [number, number] };
  features: ReferenceFeatureRecord[];
}

/** Base URL of the application, always ending in '/'. */
export const BASE: string = import.meta.env.BASE_URL;

/** Resolve a path relative to the application base. */
export function dataUrl(path: string): string {
  return `${BASE}${path.replace(/^\.?\//, '')}`;
}

/** URL of a file inside the baked data directory (`web/public/data/`). */
export function staticUrl(path: string): string {
  return dataUrl(`data/${path.replace(/^\.?\//, '')}`);
}

export function mapDataUrl(assetId: string, file: string): string {
  return staticUrl(`maps/${encodeURIComponent(assetId)}/${file}`);
}

/** URL of an image belonging to a map sheet (original scan, overlay, terrain). */
export function mapAssetUrl(assetId: string, file: string, version?: number): string {
  const query = version === undefined ? '' : `?v=${version}`;
  return mapDataUrl(assetId, file) + query;
}

/** URL of a legend file registered in the catalog (index based). */
export function legendUrl(url: string): string {
  return dataUrl(url);
}

/** URL of a release overlay: either frozen in the bundle or the map overlay. */
export function releaseOverlayUrl(release: Pick<StaticRelease, 'assetId' | 'overlay'>): string {
  return release.overlay ? dataUrl(release.overlay) : assetUrl(release.assetId, 'overlay.png');
}

/** Map filenames to the locally stored asset kinds. */
const LOCAL_FILE_KINDS: Record<string, LocalFileKind> = {
  'original.png': 'original', 'original.jpg': 'original', 'overlay.png': 'overlay',
  'terrain.png': 'terrain', 'terrain-warp.png': 'terrain-warp',
};
const localAssetUrls = new Map<string, string>();
const preparedAssets = new Set<string>();

/** Resolve the object URLs of locally processed/uploaded assets for a map. */
export async function prepareLocalAssets(assetId: string): Promise<void> {
  if (preparedAssets.has(assetId) || !hasLocalMap(assetId)) return;
  const record = await loadLocalMap(assetId);
  if (!record) return;
  for (const kind of ['original', 'overlay', 'terrain', 'terrain-warp'] as LocalFileKind[]) {
    const url = await localFileUrl(assetId, kind);
    if (url) localAssetUrls.set(`${assetId}/${kind}`, url);
  }
  preparedAssets.add(assetId);
}

/** Invalidate cached object URLs (e.g. after re-processing a map). */
export function invalidateLocalAssets(assetId: string): void {
  preparedAssets.delete(assetId);
  for (const key of localAssetUrls.keys()) if (key.startsWith(`${assetId}/`)) localAssetUrls.delete(key);
}

/** Synchronous asset URL: local object URL when available, baked URL otherwise. */
export function assetUrl(assetId: string, file: string, version?: number): string {
  const kind = LOCAL_FILE_KINDS[file];
  if (kind) {
    const local = localAssetUrls.get(`${assetId}/${kind}`);
    if (local) return local;
  }
  return mapAssetUrl(assetId, file, version);
}

/** Merge client-processed/uploaded maps into the baked catalog. */
export function mergeCatalog(catalog: BakedCatalog, localMaps: LocalMapSummary[]): BakedCatalog {
  const overridden = new Set(localMaps.map(map => map.assetId));
  const maps = catalog.maps.map(map => ({ ...map, local: overridden.has(map.id) }));
  for (const local of localMaps) {
    if (maps.some(map => map.id === local.assetId)) continue;
    maps.push({
      id: local.assetId, name: local.name, filename: local.filename, sha256: local.sha256,
      processed: true, controls: null,
      courseDefaults: null, hasGraph: false, hasApproaches: false, local: true, sourceKind: local.kind,
    });
  }
  return { ...catalog, maps };
}

/** Whether a map id belongs to a locally stored map/processing result. */
export function isLocalMap(assetId: string): boolean {
  return hasLocalMap(assetId);
}

/** Decode the lossless native scan the browser pipeline uses as input. */
export async function loadOriginalPng(assetId: string): Promise<Blob> {
  const local = await localFileUrl(assetId, 'original');
  if (local) {
    const response = await fetch(local);
    if (response.ok) return await response.blob();
  }
  const response = await fetch(mapDataUrl(assetId, 'original.png'));
  if (!response.ok) throw new Error(`无法载入原图（${response.status}）：${assetId}`);
  return await response.blob();
}

/** Add local maps to the object-URL registry at startup. */
export function rememberLocalCatalog(): LocalMapSummary[] {
  return rememberLocalMaps();
}

async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url, { credentials: 'same-origin' });
  if (!response.ok) throw new Error(`无法载入静态数据（${response.status}）：${url}`);
  return await response.json() as T;
}

export async function loadCatalog(): Promise<BakedCatalog> {
  const catalog = await fetchJSON<BakedCatalog>(staticUrl('catalog.json'));
  return { ...catalog, static: true };
}

export async function loadHealth(): Promise<Health | null> {
  try { return await fetchJSON<Health>(staticUrl('health.json')); }
  catch { return null; }
}

export async function loadReference(): Promise<object> {
  return await fetchJSON<object>(staticUrl('reference.json'));
}

export async function loadMapState(assetId: string): Promise<MapState | null> {
  try { return await fetchJSON<MapState>(mapDataUrl(assetId, 'state.json')); }
  catch { return null; }
}

export async function loadReferenceFeatures(): Promise<ReferenceFeatures> {
  return await fetchJSON<ReferenceFeatures>(staticUrl('reference-features.json'));
}

export async function loadApproaches(assetId: string): Promise<BakedApproaches | null> {
  try { return await fetchJSON<BakedApproaches>(mapDataUrl(assetId, 'approaches.json')); }
  catch { return null; }
}

export async function loadGraphMetadata(assetId: string): Promise<import('./raster-graph').GraphMetadata> {
  return await fetchJSON(mapDataUrl(assetId, 'graph.json'));
}

/** Fetch a graph bundle and inflate it. Cached per asset for the session. */
const bundleCache = new Map<string, Promise<ArrayBuffer>>();

export async function loadGraphBuffer(assetId: string): Promise<ArrayBuffer> {
  let pending = bundleCache.get(assetId);
  if (pending) return await pending;
  pending = (async () => {
    const url = mapDataUrl(assetId, 'graph.bin.gz');
    const response = await fetch(url, { credentials: 'same-origin' });
    if (!response.ok) throw new Error(`无法载入通行图（${response.status}）：${url}`);
    const compressed = await response.arrayBuffer();
    const magic = new Uint8Array(compressed, 0, 2);
    if (magic[0] !== 0x1f || magic[1] !== 0x8b) return compressed; // already inflated (dev servers)
    if (typeof DecompressionStream === 'function') {
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
      return await new Response(stream).arrayBuffer();
    }
    throw new Error('当前浏览器不支持 DecompressionStream，无法载入通行图');
  })();
  bundleCache.set(assetId, pending);
  try { return await pending; } catch (error) { bundleCache.delete(assetId); throw error; }
}

const releaseIndexCache: { value: Promise<ReleaseSummary[]> | null } = { value: null };

export async function loadReleaseIndex(): Promise<ReleaseSummary[]> {
  releaseIndexCache.value ??= fetchJSON<{ releases?: ReleaseSummary[] }>(staticUrl('releases.json'))
    .then(value => value.releases ?? [])
    .catch(() => []);
  return await releaseIndexCache.value;
}

export async function loadRelease(id: string): Promise<StaticRelease | null> {
  if (!/^[0-9a-f]{32}$/.test(id)) return null;
  try { return await fetchJSON<StaticRelease>(staticUrl(`releases/${id}.json`)); }
  catch { return null; }
}

/** Download a locally created release as a JSON file the teacher can commit. */
export function downloadRelease(release: StaticRelease): void {
  downloadJSON(`release-${release.id}.json`, release);
}

export function downloadJSON(name: string, data: unknown): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Interpolate the baked WMM declination table for a date. */
export function declinationFor(health: Health | null, now = new Date()): number | null {
  const table = health?.declinationTable;
  if (!table?.length) return null;
  const target = now.getTime(), first = Date.parse(table[0].date), last = Date.parse(table[table.length - 1].date);
  if (!Number.isFinite(target) || !Number.isFinite(first) || target < first || target > last) return null;
  let previous = table[0];
  for (const entry of table) {
    const time = Date.parse(entry.date);
    if (time >= target) {
      if (time === target || entry === previous) return entry.deg;
      const ratio = (target - Date.parse(previous.date)) / (time - Date.parse(previous.date));
      return previous.deg + (entry.deg - previous.deg) * ratio;
    }
    previous = entry;
  }
  return table[table.length - 1].deg;
}

/** Re-export the HTML escaper so data-adjacent views share one implementation. */
export { escapeHtml };

export type { Approach, Control, Plan };
