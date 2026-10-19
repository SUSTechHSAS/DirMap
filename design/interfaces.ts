/** Design contract, not a running API. Runtime validators must enforce the
 * invariants in implementation-plan.zh-CN.md; TypeScript alone cannot do so.
 * GCJ02 data deliberately does not masquerade as RFC 7946 WGS84 GeoJSON.
 */

export type Id = string;
export type LonLat =
  | { crs: "WGS84"; lng: number; lat: number }
  | { crs: "GCJ02"; lng: number; lat: number };
export interface XY { frameId: Id; xM: number; yM: number }
export interface Pixel { assetId: Id; u: number; v: number }
export type ReviewState = "candidate" | "accepted" | "needs_review" | "rejected";

export interface ProjectFrame {
  id: Id;
  kind: "LOCAL_GCJ02_ELLIPSOIDAL_LINEAR_V1";
  origin: Extract<LonLat, { crs: "GCJ02" }>;
  axis: "east_north";
  unit: "metre";
  maxRadiusM: number;
  coordinateConversionProvider: "amap";
}

export interface SourceAsset {
  id: Id;
  sha256: string;
  filename: string;
  mimeType: "image/jpeg" | "image/png" | "application/pdf";
  // PDF originals have no raster dimensions until a derived image exists.
  widthPx: number | null;
  heightPx: number | null;
  role: "map" | "legend";
  roleVerified: boolean;
  symbolSpecification: string | null;
  sourceDate: string | null;
  // Cropping, EXIF orientation and photo rectification must remain invertible.
  normalizationTransformId: Id | null;
}

export interface Landmark {
  id: Id;
  pixel: Pixel;
  reference: Extract<LonLat, { crs: "GCJ02" }>;
  local: XY;
  source: "campus_gis" | "converted_osm" | "survey" | "map_click";
  feature: "building_corner" | "road_junction" | "water_corner" | "other";
  referenceAccuracyM: number | null;
  purpose: "fit" | "validation";
  state: ReviewState;
}

export interface Registration {
  id: Id;
  assetId: Id;
  frameId: Id;
  version: number;
  model: "affine" | "regularized_piecewise_affine";
  fitLandmarkIds: Id[];
  independentValidationIds: Id[];
  validHull: XY[];
  transformArtifact: string;
  metrics: {
    validationCount: number;
    medianErrorM: number | null;
    p95ErrorM: number | null;
    maxErrorM: number | null;
    foldedTriangles: number;
    overlappingTriangles: number;
  };
  state: ReviewState;
}

export interface Control {
  id: Id;
  label: string;
  pixel: Pixel;
  registrationId: Id;
  exactPosition: XY;
  displayPosition: Extract<LonLat, { crs: "GCJ02" }>;
  // Approach nodes may differ from the cartographic centre; preserve both.
  approachNodeIds: Id[];
  level: number | null;
  required: boolean;
  serviceBaseSec: number;
  servicePerPersonSec: number;
  state: ReviewState;
}

export interface Edge {
  id: Id;
  from: Id;
  to: Id;
  geometry: XY[];
  level: number;
  direction: "both" | "forward";
  access: "allowed" | "forbidden" | "unknown";
  surface: "paved" | "path" | "open_land" | "forest" | "stairs" | "portal";
  lengthM: number;
  speedFactor: number;
  crossingDelaySec: number;
  minWidthM: number | null;
  confidence: number;
  sourceAssetIds: Id[];
  sourceSymbolIds: string[];
  fieldCheckedAt: string | null;
  validFrom: string | null;
  validUntil: string | null;
}

export interface GraphSnapshot {
  id: Id;
  frameId: Id;
  registrationIds: Id[];
  nodes: Array<{ id: Id; point: XY; level: number }>;
  edges: Edge[];
  portals: Array<{ id: Id; from: Id; to: Id; transitionSec: number }>;
  contentHash: string;
  state: "draft" | "validated" | "released" | "superseded";
}

export interface Team {
  id: Id;
  people: number;
  speedMps: number;
  startNodeId: Id;
  finishNodeId: Id;
  earliestStartSec: number;
  latestFinishSec: number | null;
  maxDistanceM: number | null;
}

export interface PlanRequest {
  graphSnapshotId: Id;
  controlIds: Id[];
  // Default all_visit. Shared targets are replicated PER TEAM in that model.
  mode: "all_visit" | "split_coverage" | "fixed_order";
  fixedOrder?: Id[];
  teams: Team[];
  objective: "min_max_finish_then_sum" | "min_total_time" | "min_distance";
  departurePolicy: "simultaneous" | "staggered";
  maxStartDelaySec: number;
  useResourceScheduling: boolean;
  solvePolicy: "exact" | "quick";
  timeLimitSec: number;
}

export type PlanStatus = "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" |
  "TIMEOUT_NO_SOLUTION" | "NEEDS_REVIEW" | "INVALID_INPUT";

export interface PlanResult {
  id: Id;
  request: PlanRequest;
  status: PlanStatus;
  routes: Array<{
    teamId: Id;
    controlIds: Id[];
    nodeIds: Id[];
    edgeIds: Id[];
    localGeometry: XY[];
    displayGeometry: Array<Extract<LonLat, { crs: "GCJ02" }>>;
    departureSec: number;
    durationSec: number;
    distanceM: number;
  }>;
  proof: {
    algorithm: "held_karp" | "subset_dp" | "cp_sat" | "routing_local_search";
    // Proof is scoped to this exact model, discretization, and frozen graph.
    modelHash: string;
    graphHash: string;
    stage: "primary" | "secondary" | "finished";
    objectiveValue: number | null;
    bestBound: number | null;
    relativeGap: number | null;
    globalOptimalityProven: boolean;
    elapsedSec: number;
  };
  unresolvedControlIds: Id[];
  reasons: Array<{ code: string; entityIds: Id[]; message: string }>;
}

export interface HeadingSample {
  degrees: number;
  reference: "true_north" | "magnetic_north" | "relative" | "unknown";
  source: "ios_compass" | "absolute_orientation" | "gnss_course";
  accuracyDeg: number | null;
  // Unix epoch milliseconds; convert DOM event timeStamp using timeOrigin.
  timestampMs: number;
  screenAngleDeg: number;
}

export interface LocationSample {
  raw: LonLat;
  display: Extract<LonLat, { crs: "GCJ02" }> | null;
  accuracyM: number;
  speedMps: number | null;
  courseDeg: number | null;
  // Unix epoch milliseconds, preserving the actual fix time.
  timestampMs: number;
  provider: "browser_wgs84" | "amap_gcj02";
  converted: boolean;
}
