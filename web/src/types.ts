export type Pixel = [number, number];
export interface Control {
  id: string; label: string; pixel: Pixel; radiusPx: number; required: boolean;
  role?: 'control' | 'start' | 'finish' | 'start_finish';
  state: 'candidate' | 'accepted' | 'rejected'; calibrated?: boolean; lnglat?: Pixel | null;
  approachPixel?: Pixel | null; serviceBaseSec: number; servicePerPersonSec: number;
  confidence?: number;
}
export interface TerrainEdit { kind: 'open' | 'close'; pixels: Pixel[]; radiusPx: number; cost?: number }
export interface Anchor { pixel: Pixel; lnglat: Pixel; purpose: 'fit' | 'validation' }
export interface MapState {
  assetId: string; processed?: boolean; name: string; width: number; height: number; version: number;
  registrationStale?: boolean;
  terrainSchema?: string; routingSchema?: string; graphHash?: string | null; review?: unknown; static?: true;
  controls: Control[]; edits: TerrainEdit[]; extent?: number[]; reviewed: boolean;
  registration: { affine: number[][]; source: Pixel[]; target: Pixel[]; triangles: number[][] } | null;
  quality: { state: string; method: string; fitCount: number; validationCount: number; medianErrorM: number | null; p95ErrorM: number | null; inheritedGeographicP95M?: number; validationScope: string; baseAssetId?: string } | null;
  warnings: string[]; statistics: { buildingCandidates: number; waterCandidates: number; controlCandidates: number; forbiddenPixelFraction: number };
  frame: { crs: string; origin: Pixel };
}
export interface Route {
  teamId: string; people: number; controlIds: string[]; geometry: Pixel[]; localGeometry: Pixel[];
  distanceM: number; travelSec: number; serviceSec: number; durationSec: number; departureSec: number; finishSec: number;
  travelWeightM?: number; nodeIds?: number[]; edgeIds?: number[][];
  legs?: { fromId: string; toId: string; startIndex: number; endIndex: number; distanceM: number }[];
}
export interface Plan {
  id: string; status: string; routes: Route[]; version: number; graphHash: string; draft: boolean;
  assetId?: string; routingSchema?: string; request?: PlanRequest; createdAt?: number;
  preparationSec?: number; planningElapsedSec?: number;
  graphStatistics?: { nodes: number; edges: number; components: number; gridStepPx: number; fineGridStepPx: number };
  unresolved?: { id: string; message: string }[];
  approaches?: Approach[];
  coverage?: { mode: string; mandatoryControlIds: string[]; mandatoryOrder: boolean; sharedControlIds: string[] };
  proof?: {
    algorithm: string; globalOptimalityProven: boolean; scope: string; elapsedSec: number; objectiveValue: number[];
    objective?: string; bestBound?: number[] | null; relativeGap?: number | null; graphHash?: string; solverBudgetSec?: number;
  };
}
export interface PlanRequest {
  assetId: string; version: number; controlIds: string[]; startPixel: Pixel; finishPixel: Pixel;
  mode: string; objective: string; mandatoryControlIds?: string[]; mandatoryOrder?: boolean;
  teams: { id: string; people: number; speedMps: number; departureSec?: number }[];
  timeLimitSec: number;
}
export interface Job<T = unknown> {
  id: string; kind: string; assetId: string; state: 'queued' | 'running' | 'succeeded' | 'failed';
  progress: number; message?: string; createdAt: number; recovered?: boolean; attempts: number; result?: T;
}
export interface Approach {
  controlId: string; targetPixel: Pixel; pixel: Pixel; method: string;
  automatic: boolean; offsetM: number; lnglat: Pixel; targetLnglat: Pixel;
}
export interface Catalog { maps: { id: string; name: string; filename: string; sha256?: string; processed: boolean; controls: number | null;
  courseDefaults?: { mode: string; mandatoryLabels: string[]; mandatoryOrder: boolean; description?: string } | null
  hasGraph?: boolean; hasApproaches?: boolean }[]; legends: { name: string; url: string }[] }

/** Static build metadata returned with the catalog. */
export interface BakedCatalog extends Catalog {
  static: true;
  generatedAt?: string;
  routingSchema?: string;
  terrainSchema?: string;
  reference?: { crs: string; name: string; attribution: string; licenseUrl: string; frame?: { crs: string; origin: Pixel } };
}

export interface Health {
  status: string; version: string;
  frame: { crs: string; origin: Pixel; unit?: string; axes?: string };
  static?: boolean; adminTokenRequired?: boolean;
  modelDate?: string; magneticDeclinationDeg?: number | null;
  declinationTable?: { date: string; deg: number }[];
  routingSchema?: string; terrainSchema?: string;
}

/** Legal approach position resolved at bake time (see docs/static-architecture.md). */
export type BakedApproach =
  | { ok: true; node: number; pixel: Pixel; local: Pixel; method: string; automatic: boolean; offsetM: number;
      lnglat: Pixel; targetLnglat: Pixel | null; featureKind?: string; reviewRequired?: boolean; targetPixel?: Pixel }
  | { ok: false; message: string };

export interface BakedApproaches {
  assetId: string; routingSchema?: string; graphHash?: string;
  controls: Record<string, BakedApproach>;
}

/** A course frozen into the static data bundle (or created locally by a teacher). */
export interface StaticRelease {
  id: string; title: string; assetId: string; createdAt: number;
  plan: Plan; controls: Control[]; frame: MapState['frame']; extent: number[];
  quality: MapState['quality']; review: unknown; version: number;
  routingSchema: string;
  superseded: boolean;
  static?: true;
  offline?: boolean;
  overlay?: string;
}

export interface ReleaseSummary { id: string; title: string; assetId: string; createdAt: number }

export interface NavigationRequest {
  teamId: string; lnglat: Pixel; accuracyM: number; timestampMs: number; completedControlIds: string[];
}

export interface NavigateResponse {
  status: 'READY' | 'NEEDS_REVIEW';
  version: number; graphHash: string;
  completedControlIds?: string[];
  generatedAt?: number;
  route: Route | null;
  unresolved?: { id: string; message: string }[];
  scope?: string;
}

export type PlannerProgress = (percent: number, message: string) => void;

export function localScales(origin: Pixel): Pixel {
  const phi = origin[1] * Math.PI / 180, e = .00669437999014;
  const n = 6378137 / Math.sqrt(1 - e * Math.sin(phi) ** 2);
  const m = 6378137 * (1 - e) / (1 - e * Math.sin(phi) ** 2) ** 1.5;
  return [n * Math.cos(phi) * Math.PI / 180, m * Math.PI / 180];
}

export function interpolate(point: Pixel, source: Pixel[], target: Pixel[], triangles: number[][]): Pixel | null {
  for (const ids of triangles) {
    const [a, b, c] = ids.map(i => source[i]);
    const determinant = (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1]);
    if (Math.abs(determinant) < 1e-10) continue;
    const u = ((point[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (point[1] - c[1])) / determinant;
    const v = ((a[0] - c[0]) * (point[1] - c[1]) - (point[0] - c[0]) * (a[1] - c[1])) / determinant;
    if (u < -1e-7 || v < -1e-7 || u + v > 1 + 1e-7) continue;
    const [p, q, r] = ids.map(i => target[i]);
    return [u * p[0] + v * q[0] + (1 - u - v) * r[0], u * p[1] + v * q[1] + (1 - u - v) * r[1]];
  }
  return null;
}

export function pixelToLngLat(pixel: Pixel, state: MapState): Pixel | null {
  if (!state.registration) return null;
  const r = state.registration, xy = interpolate(pixel, r.source, r.target, r.triangles);
  const scale = localScales(state.frame.origin);
  return xy ? [state.frame.origin[0] + xy[0] / scale[0], state.frame.origin[1] + xy[1] / scale[1]] : null;
}

export function lngLatToPixel(lnglat: Pixel, state: MapState): Pixel | null {
  if (!state.registration) return null;
  const scale = localScales(state.frame.origin);
  const xy: Pixel = [(lnglat[0] - state.frame.origin[0]) * scale[0], (lnglat[1] - state.frame.origin[1]) * scale[1]];
  const r = state.registration;
  return interpolate(xy, r.target, r.source, r.triangles);
}

export const escapeHtml = (text: unknown) => String(text ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]!));
