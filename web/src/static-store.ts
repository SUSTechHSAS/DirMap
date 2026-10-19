/** Browser-local persistence for the static build.
 *
 * The baked data is immutable; teachers can still adjust targets, terrain
 * corrections and publishing metadata. Those edits live in localStorage,
 * keyed per map sheet, and are merged over the baked state on load.
 */
import type { Control, MapState, ReleaseSummary, StaticRelease, TerrainEdit } from './types';

const STATE_PREFIX = 'dirmap-static-state-v1:';
const RELEASE_KEY = 'dirmap-static-releases-v1';
const RULE_PREFIX = 'dirmap-course-rules-v1:';

export interface StateOverride {
  version: number;
  controls: Control[];
  edits: TerrainEdit[];
  reviewed?: boolean;
  review?: unknown;
  updatedAt: number;
  /** Monotonic local revision used to derive the effective map version. */
  revision: number;
}

function readJSON<T>(key: string): T | null {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) as T : null;
  } catch { return null; }
}

function writeJSON(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: edits stay in memory */ }
}

export function loadStateOverride(assetId: string): StateOverride | null {
  const saved = readJSON<StateOverride>(STATE_PREFIX + assetId);
  if (!saved || !Array.isArray(saved.controls) || !Array.isArray(saved.edits)) return null;
  if (!Number.isFinite(saved.version) || !Number.isFinite(saved.revision)) return null;
  return saved;
}

export function saveStateOverride(assetId: string, base: MapState, controls: Control[], edits: TerrainEdit[]): StateOverride {
  const existing = loadStateOverride(assetId);
  const revision = (existing?.revision ?? 0) + 1;
  const override: StateOverride = {
    version: base.version + revision,
    controls, edits, reviewed: base.reviewed,
    review: (base as MapState & { review?: unknown }).review,
    updatedAt: Date.now(), revision,
  };
  writeJSON(STATE_PREFIX + assetId, override);
  return override;
}

export function clearStateOverride(assetId: string): void {
  try { localStorage.removeItem(STATE_PREFIX + assetId); } catch { /* nothing to clear */ }
}

/** Merge a saved local override over the immutable baked state. */
export function applyStateOverride(base: MapState, override: StateOverride | null): MapState {
  if (!override) return base;
  return {
    ...base,
    version: override.version,
    controls: override.controls,
    edits: override.edits,
    reviewed: override.reviewed ?? base.reviewed,
    ...(override.review !== undefined ? { review: override.review } : {}),
  } as MapState;
}

export function hasStateOverride(assetId: string): boolean {
  return loadStateOverride(assetId) !== null;
}

/** Local courses created by the teacher in this browser. */
export function loadLocalReleases(): StaticRelease[] {
  const saved = readJSON<StaticRelease[]>(RELEASE_KEY);
  return Array.isArray(saved) ? saved.filter(release => release && typeof release.id === 'string') : [];
}

export function saveLocalRelease(release: StaticRelease): void {
  const releases = loadLocalReleases().filter(item => item.id !== release.id);
  releases.unshift(release);
  writeJSON(RELEASE_KEY, releases.slice(0, 10));
}

export function findLocalRelease(id: string): StaticRelease | null {
  return loadLocalReleases().find(release => release.id === id) ?? null;
}

export function localReleaseSummaries(): ReleaseSummary[] {
  return loadLocalReleases().map(({ id, title, assetId, createdAt }) => ({ id, title, assetId, createdAt }));
}

export function removeLocalRelease(id: string): void {
  writeJSON(RELEASE_KEY, loadLocalReleases().filter(release => release.id !== id));
}

export function loadRules(key: string): unknown {
  return readJSON(RULE_PREFIX + key);
}

export function saveRules(key: string, value: unknown): void {
  writeJSON(RULE_PREFIX + key, value);
}

export function clearRules(key: string): void {
  try { localStorage.removeItem(RULE_PREFIX + key); } catch { /* nothing to clear */ }
}

/** Copy text in a way that also works on insecure local test servers. */
export async function copyText(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; }
  catch { return false; }
}
