/** Static course planning and navigation re-planning: a browser port of
 * `dirmap/service.py::plan_course` / `navigate_course`, `graph.plan_matrix`
 * and `graph.expand_route`, backed by the baked multi-resolution graph and
 * baked legal approach positions.
 */
import { frameLngLat, frameLocal, Mesh } from './geometry.ts';
import type { FrameInfo } from './geometry.ts';
import { solve } from './optimizer.ts';
import type { OptimizerResult, OptimizerTeam } from './optimizer.ts';
import { RasterGraph } from './raster-graph.ts';
import type { GraphMetadata, Metric } from './raster-graph.ts';
import type {
  Approach, BakedApproach, BakedApproaches, Control, MapState, NavigateResponse, NavigationRequest,
  Pixel, Plan, PlanRequest, PlannerProgress, Route, StaticRelease,
} from './types.ts';

/** Legal approach position as baked by the static data pipeline. */
interface BakedApproachEntry {
  ok: boolean;
  message?: string;
  node?: number;
  pixel?: Pixel;
  local?: Pixel;
  method?: string;
  automatic?: boolean;
  offsetM?: number;
  lnglat?: Pixel;
  targetLnglat?: Pixel | null;
  component?: number;
}

/** The baked approach document, including the addendum fields from the baker. */
export interface BakedApproachesDocument extends BakedApproaches {
  startPixel?: Pixel;
  finishPixel?: Pixel;
  startNode?: number;
  startComponent?: number;
  note?: string;
  controls: Record<string, BakedApproach>;
}

export interface PlannerAssets {
  metadata: GraphMetadata;
  buffer: ArrayBuffer;
  approaches: BakedApproaches | null;
  mesh?: Mesh;
}

/** Node-level description of the (n+2)-endpoint shortest-path problem. */
export interface MatrixEndpoint {
  id: string;
  pixel: Pixel;
  approachPixel: Pixel;
}

export interface PlanMatrixBundle {
  metric: Metric;
  matrix: Float64Array;
  nodes: number[];
  endpoints: MatrixEndpoint[];
  world: Float64Array;
  connectors: Float64Array;
  connectorWeights: Float64Array;
  unresolved: { id: string; message: string }[];
  approaches: Approach[];
}

export interface PlanMatrixOptions {
  metric?: Metric;
  /** True when the graph was rebuilt from local teacher edits. */
  edited?: boolean;
}

export interface ExpandedRoute {
  geometry: Pixel[];
  localGeometry: Pixel[];
  distanceM: number;
  travelWeightM: number;
  nodeIds: number[];
  edgeIds: number[][];
  legs: { fromId: string; toId: string; startIndex: number; endIndex: number; distanceM: number }[];
}

export interface Planner {
  readonly graphHash: string;
  plan(request: PlanRequest, controls: Control[], progress?: PlannerProgress): Promise<Plan>;
  navigate(release: StaticRelease, request: NavigationRequest, controls: Control[]): Promise<NavigateResponse>;
}

const noProgress: PlannerProgress = () => undefined;

/** NumPy `np.rint`: round half to even. */
function rintEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function tick(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function nowSeconds(): number {
  return performance.now() / 1000;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function randomHex(bytes: number): string {
  const buffer = new Uint8Array(bytes);
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.getRandomValues === 'function') cryptoApi.getRandomValues(buffer);
  else for (let i = 0; i < bytes; i++) buffer[i] = Math.floor(Math.random() * 256);
  let text = '';
  for (const byte of buffer) text += byte.toString(16).padStart(2, '0');
  return text;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

interface ResolvedControl {
  node: number;
  pixel: Pixel;
  local: Pixel;
  method: string;
  automatic: boolean;
  offsetM: number;
  lnglat: Pixel;
  targetLnglat: Pixel | null;
}

function lngLatAt(local: Pixel, frame: FrameInfo): Pixel {
  return frameLngLat(local, frame.origin);
}

function approachEntry(control: Control, resolved: ResolvedControl): Approach {
  return {
    controlId: control.id,
    targetPixel: [control.pixel[0], control.pixel[1]],
    pixel: [resolved.pixel[0], resolved.pixel[1]],
    method: resolved.method,
    automatic: resolved.automatic,
    offsetM: resolved.offsetM,
    lnglat: resolved.lnglat,
    targetLnglat: resolved.targetLnglat as Pixel,
  };
}

function targetLngLatOf(control: Control, mesh: Mesh, frame: FrameInfo): Pixel | null {
  const mapped = mesh.forward(control.pixel);
  return mapped.inside ? lngLatAt(mapped.local, frame) : null;
}

function resolveManualApproach(
  control: Control, point: Pixel, graph: RasterGraph, mesh: Mesh, frame: FrameInfo,
  startComponent: number | undefined,
): ResolvedControl {
  const mapped = mesh.forward(point);
  if (!mapped.inside) throw new Error('目标超出有效配准范围，需要补充参照');
  const x = rintEven(point[0]);
  const y = rintEven(point[1]);
  if (x < 0 || y < 0 || x >= graph.width || y >= graph.height) throw new Error('到访位置超出原图');
  let node: number;
  try {
    node = graph.snap(point, 4);
  } catch {
    throw new Error('指定到访位置不可达，请核对门洞和到访侧');
  }
  if (startComponent !== undefined && graph.components[node] !== startComponent) {
    throw new Error('指定到访位置不可达，请核对门洞和到访侧');
  }
  return {
    node,
    pixel: [point[0], point[1]],
    local: mapped.local,
    method: 'manual',
    automatic: false,
    offsetM: 0,
    lnglat: lngLatAt(mapped.local, frame),
    targetLnglat: targetLngLatOf(control, mesh, frame),
  };
}

function resolveEditedApproach(
  control: Control, entry: BakedApproachEntry | undefined, graph: RasterGraph, mesh: Mesh, frame: FrameInfo,
  startComponent: number | undefined,
): ResolvedControl {
  const manual = control.approachPixel != null;
  const point: Pixel = manual
    ? [control.approachPixel![0], control.approachPixel![1]]
    : entry && entry.ok && entry.pixel
      ? [entry.pixel[0], entry.pixel[1]]
      : [control.pixel[0], control.pixel[1]];
  const mapped = mesh.forward(point);
  if (!mapped.inside) throw new Error('目标超出有效配准范围，需要补充参照');
  let node: number;
  try {
    node = graph.snap(point, 4);
  } catch {
    throw new Error(manual ? '指定到访位置不可达，请核对门洞和到访侧' : '合法到访侧与起点不连通，请核对通道');
  }
  if (startComponent !== undefined && graph.components[node] !== startComponent) {
    throw new Error(manual ? '指定到访位置不可达，请核对门洞和到访侧' : '合法到访侧与起点不连通，请核对通道');
  }
  return {
    node,
    pixel: [point[0], point[1]],
    local: mapped.local,
    method: manual ? 'manual' : entry && entry.ok ? entry.method ?? 'exact' : 'exact',
    automatic: manual ? false : Boolean(entry && entry.ok && entry.automatic),
    offsetM: manual ? 0 : entry && entry.ok ? entry.offsetM ?? 0 : 0,
    lnglat: lngLatAt(mapped.local, frame),
    targetLnglat: targetLngLatOf(control, mesh, frame),
  };
}

function resolveBakedApproach(
  control: Control, entry: BakedApproachEntry | undefined, graph: RasterGraph, mesh: Mesh, frame: FrameInfo,
  startComponent: number | undefined,
): ResolvedControl {
  if (control.approachPixel != null) {
    return resolveManualApproach(control, [control.approachPixel[0], control.approachPixel[1]], graph, mesh, frame, startComponent);
  }
  if (entry) {
    if (!entry.ok) throw new Error(String(entry.message ?? '目标地物或通行侧不明确，请核对实际到访位置'));
    if (!entry.pixel || !entry.local || typeof entry.node !== 'number') throw new Error('到访位置数据无效');
    const node = entry.node;
    if (!Number.isInteger(node) || node < 0 || node >= graph.nodeCount) throw new Error('到访位置数据无效');
    const component = typeof entry.component === 'number' ? entry.component : graph.components[node];
    if (startComponent !== undefined && component !== startComponent) throw new Error('合法到访侧与起点不连通，请核对通道');
    return {
      node,
      pixel: [entry.pixel[0], entry.pixel[1]],
      local: [entry.local[0], entry.local[1]],
      method: String(entry.method ?? 'exact'),
      automatic: Boolean(entry.automatic),
      offsetM: Number(entry.offsetM ?? 0),
      lnglat: entry.lnglat ? [entry.lnglat[0], entry.lnglat[1]] : lngLatAt(entry.local, frame),
      targetLnglat: entry.targetLnglat ? [entry.targetLnglat[0], entry.targetLnglat[1]] : null,
    };
  }
  // No baked entry and no teacher-confirmed pixel: fall back to a direct snap
  // (the browser cannot re-run the full cartographic feature resolver).
  const mapped = mesh.forward(control.pixel);
  if (!mapped.inside) throw new Error('目标超出有效配准范围，需要补充参照');
  const node = graph.snap(control.pixel);
  return {
    node,
    pixel: [control.pixel[0], control.pixel[1]],
    local: mapped.local,
    method: 'exact',
    automatic: false,
    offsetM: 0,
    lnglat: lngLatAt(mapped.local, frame),
    targetLnglat: lngLatAt(mapped.local, frame),
  };
}

function componentCounts(graph: RasterGraph): Uint32Array {
  const counts = new Uint32Array(graph.componentCount);
  for (const component of graph.components) counts[component]++;
  return counts;
}

interface MatrixEndpointInternal {
  id: string;
  pixel: Pixel;
  approachPixel: Pixel;
  control: Control | null;
}

/**
 * Port of `dirmap/graph.py::plan_matrix` using the baked approach positions.
 */
export function planMatrix(
  graph: RasterGraph, mesh: Mesh, frame: FrameInfo, controls: readonly Control[],
  startPixel: Pixel, finishPixel: Pixel, approaches: BakedApproaches | null,
  options: PlanMatrixOptions = {},
): PlanMatrixBundle {
  const metric = options.metric ?? 'time';
  const edited = options.edited === true;
  const document = approaches as BakedApproachesDocument | null;
  const entries = document && document.controls ? document.controls as unknown as Record<string, BakedApproachEntry> : null;
  const endpointCount = controls.length + 2;
  const endpoints: MatrixEndpointInternal[] = [{ id: 'start', pixel: startPixel, approachPixel: startPixel, control: null }];
  for (const control of controls) {
    endpoints.push({ id: control.id, pixel: control.pixel, approachPixel: control.pixel, control });
  }
  endpoints.push({ id: 'finish', pixel: finishPixel, approachPixel: finishPixel, control: null });
  const nodes: number[] = [];
  const world = new Float64Array(endpointCount * 2);
  const approachList: Approach[] = [];
  const unresolved: { id: string; message: string }[] = [];
  let startComponent: number | undefined;
  for (const endpoint of endpoints) {
    try {
      if (endpoint.control === null) {
        const mapped = mesh.forward(endpoint.pixel);
        if (!mapped.inside) throw new Error('点位超出配准有效范围');
        const node = graph.snap(endpoint.pixel);
        if (endpoint.id === 'start') startComponent = graph.components[node];
        const position = nodes.length;
        nodes.push(node);
        world[position * 2] = mapped.local[0];
        world[position * 2 + 1] = mapped.local[1];
        endpoint.approachPixel = [endpoint.pixel[0], endpoint.pixel[1]];
        continue;
      }
      const control = endpoint.control;
      const entry = entries ? entries[control.id] : undefined;
      const resolved = edited
        ? resolveEditedApproach(control, entry, graph, mesh, frame, startComponent)
        : resolveBakedApproach(control, entry, graph, mesh, frame, startComponent);
      if (approaches !== null || control.approachPixel != null) approachList.push(approachEntry(control, resolved));
      const position = nodes.length;
      nodes.push(resolved.node);
      world[position * 2] = resolved.local[0];
      world[position * 2 + 1] = resolved.local[1];
      endpoint.approachPixel = resolved.pixel;
    } catch (error) {
      unresolved.push({ id: endpoint.id, message: errorMessage(error) });
    }
  }
  if (unresolved.length) {
    return {
      metric, matrix: new Float64Array(0), nodes: [], endpoints: endpoints.map(endpoint => ({
        id: endpoint.id, pixel: endpoint.pixel, approachPixel: endpoint.approachPixel,
      })), world, connectors: new Float64Array(0), connectorWeights: new Float64Array(0), unresolved, approaches: approachList,
    };
  }
  const components = nodes.map(node => graph.components[node]);
  if (components.some(component => component !== components[0])) {
    const counts = componentCounts(graph);
    return {
      metric, matrix: new Float64Array(0), nodes: [], endpoints: endpoints.map(endpoint => ({
        id: endpoint.id, pixel: endpoint.pixel, approachPixel: endpoint.approachPixel,
      })), world, connectors: new Float64Array(0), connectorWeights: new Float64Array(0),
      unresolved: endpoints.flatMap((endpoint, index) => components[index] === components[0] ? [] : [{
        id: endpoint.id,
        message: `位于独立通行区域（${counts[components[index]]} 个节点），检查门洞或目标的到访侧`,
      }]),
      approaches: approachList,
    };
  }
  const distances = graph.distanceToTargets(nodes, nodes, metric);
  const connectors = new Float64Array(endpointCount);
  for (let index = 0; index < endpointCount; index++) {
    const dx = world[index * 2] - graph.local[nodes[index] * 2];
    const dy = world[index * 2 + 1] - graph.local[nodes[index] * 2 + 1];
    connectors[index] = Math.sqrt(dx * dx + dy * dy);
  }
  const factors = new Float64Array(endpointCount);
  for (let index = 0; index < endpointCount; index++) {
    const pixel = endpoints[index].approachPixel;
    const x = Math.min(graph.width - 1, Math.max(0, rintEven(pixel[0])));
    const y = Math.min(graph.height - 1, Math.max(0, rintEven(pixel[1])));
    factors[index] = graph.costs[y * graph.width + x];
  }
  const connectorWeights = new Float64Array(endpointCount);
  for (let index = 0; index < endpointCount; index++) {
    connectorWeights[index] = metric === 'distance' ? connectors[index] : connectors[index] * factors[index];
  }
  const matrix = new Float64Array(endpointCount * endpointCount);
  for (let row = 0; row < endpointCount; row++) {
    for (let column = 0; column < endpointCount; column++) {
      matrix[row * endpointCount + column] = (distances[row * endpointCount + column] + connectorWeights[row]) + connectorWeights[column];
    }
    matrix[row * endpointCount + row] = 0;
  }
  return {
    metric, matrix, nodes,
    endpoints: endpoints.map(endpoint => ({ id: endpoint.id, pixel: endpoint.pixel, approachPixel: endpoint.approachPixel })),
    world, connectors, connectorWeights, unresolved: [], approaches: approachList,
  };
}

/**
 * Port of `dirmap/graph.py::expand_route`: A* per leg, then accumulate native
 * distances plus connector metres exactly like the Python helper.
 */
export function expandRoute(
  graph: RasterGraph, bundle: PlanMatrixBundle, order: readonly number[], frame: FrameInfo,
  pathCache?: Map<string, number[]>,
): ExpandedRoute {
  const positions: Pixel[] = [];
  const nodeIds: number[] = [];
  const edgeIds: number[][] = [];
  const legs: ExpandedRoute['legs'] = [];
  let distance = 0;
  let travelWeight = 0;
  for (let index = 0; index + 1 < order.length; index++) {
    const a = order[index];
    const b = order[index + 1];
    const startIndex = positions.length;
    const beforeDistance = distance;
    const beforeWeight = travelWeight;
    const start = bundle.nodes[a];
    const finish = bundle.nodes[b];
    const cacheKey = `${start}:${finish}`;
    let path = pathCache?.get(cacheKey);
    if (!path) {
      path = graph.shortestPath(start, finish, bundle.metric);
      pathCache?.set(cacheKey, path);
    }
    positions.push([bundle.world[a * 2], bundle.world[a * 2 + 1]]);
    for (const node of path) positions.push([graph.local[node * 2], graph.local[node * 2 + 1]]);
    positions.push([bundle.world[b * 2], bundle.world[b * 2 + 1]]);
    nodeIds.push(...path);
    for (let step = 0; step + 1 < path.length; step++) {
      const from = path[step];
      const to = path[step + 1];
      edgeIds.push([from, to]);
      const metrics = graph.edgeMetrics(from, to);
      if (metrics === null) throw new Error('必到点不连通');
      distance += metrics.length;
      travelWeight += metrics.weight;
    }
    distance += bundle.connectors[a] + bundle.connectors[b];
    travelWeight += bundle.connectorWeights[a] + bundle.connectorWeights[b];
    legs.push({
      fromId: bundle.endpoints[a].id,
      toId: bundle.endpoints[b].id,
      startIndex,
      endIndex: positions.length - 1,
      distanceM: distance - beforeDistance,
    });
  }
  return {
    geometry: positions.map(point => frameLngLat(point, frame.origin)),
    localGeometry: positions,
    distanceM: distance,
    travelWeightM: travelWeight,
    nodeIds,
    edgeIds,
    legs,
  };
}

class StaticPlanner implements Planner {
  private readonly state: MapState;
  private readonly graph: RasterGraph;
  private readonly mesh: Mesh;
  private readonly approaches: BakedApproaches | null;
  private readonly edited: boolean;
  private navigationLock: Promise<void> = Promise.resolve();

  constructor(state: MapState, graph: RasterGraph, mesh: Mesh, approaches: BakedApproaches | null, edited: boolean) {
    this.state = state;
    this.graph = graph;
    this.mesh = mesh;
    this.approaches = approaches;
    this.edited = edited;
  }

  get graphHash(): string {
    return this.graph.metadata.graphHash;
  }

  private get routingSchema(): string {
    return this.state.routingSchema ?? this.graph.metadata.schema;
  }

  private get frame(): FrameInfo {
    return { crs: this.state.frame.crs, origin: this.state.frame.origin };
  }

  /** Port of `dirmap/service.py::plan_course`. */
  async plan(request: PlanRequest, controls: Control[], progress: PlannerProgress = noProgress): Promise<Plan> {
    const started = nowSeconds();
    if (request.version !== this.state.version) throw new Error('地图已更新，请刷新后重新规划');
    const lookup = new Map<string, Control>();
    for (const control of controls) lookup.set(control.id, control);
    const ids = request.controlIds;
    if (new Set(ids).size !== ids.length || ids.some(id => !lookup.has(id))) throw new Error('目标点重复或不存在');
    const selected = ids.map(id => lookup.get(id) as Control);
    if (!selected.length) throw new Error('请至少选择一个目标');
    if (selected.some(control => (control.role ?? 'control') !== 'control' || control.state === 'rejected')) {
      throw new Error('路线目标不能包含起终点符号或已排除的候选');
    }
    const planId = randomHex(8);
    progress(10, '建立保留细墙的通行图');
    await tick();
    progress(40, '计算必到点之间的合法最短路');
    await tick();
    const metric: Metric = request.objective === 'min_distance' ? 'distance' : 'time';
    const bundle = planMatrix(this.graph, this.mesh, this.frame, selected, request.startPixel, request.finishPixel,
      this.approaches, { metric, edited: this.edited });
    if (bundle.unresolved.length) {
      return {
        id: planId,
        assetId: this.state.assetId,
        request,
        routingSchema: this.routingSchema,
        status: 'NEEDS_REVIEW',
        unresolved: bundle.unresolved,
        routes: [],
        graphHash: this.graphHash,
        version: this.state.version,
        approaches: bundle.approaches,
        draft: !this.state.reviewed,
      };
    }
    progress(65, '优化队伍分配与访问顺序');
    await tick();
    const preparationElapsed = nowSeconds() - started;
    const solverBudget = Math.max(0.01, (request.timeLimitSec ?? 5) - preparationElapsed - 0.65);
    const solved: OptimizerResult = solve(
      bundle.matrix, selected, request.teams as OptimizerTeam[], request.mode, request.objective,
      solverBudget, request.mandatoryControlIds, request.mandatoryOrder ?? false,
    );
    progress(75, '展开队伍路线');
    await tick();
    const pathCache = new Map<string, number[]>();
    const routes: Route[] = solved.routes.map(route => {
      const team = request.teams.find(candidate => candidate.id === route.teamId);
      if (!team) throw new Error('队伍不属于这份课程');
      const expanded = expandRoute(this.graph, bundle, route.order, this.frame, pathCache);
      const travelSec = expanded.travelWeightM / team.speedMps;
      return {
        teamId: route.teamId,
        people: route.people,
        controlIds: route.controlIds,
        geometry: expanded.geometry,
        localGeometry: expanded.localGeometry,
        distanceM: expanded.distanceM,
        travelWeightM: expanded.travelWeightM,
        nodeIds: expanded.nodeIds,
        edgeIds: expanded.edgeIds,
        legs: expanded.legs,
        travelSec,
        serviceSec: route.serviceSec,
        durationSec: travelSec + route.serviceSec,
        departureSec: route.departureSec,
        finishSec: travelSec + route.serviceSec + route.departureSec,
      };
    });
    const proof = {
      ...solved.proof,
      graphHash: this.graphHash,
      solverBudgetSec: round3(solverBudget),
      scope: solved.proof.scope,
    };
    const draft = !this.state.reviewed || bundle.approaches.some(approach => approach.automatic);
    if (bundle.approaches.some(approach => approach.automatic)) {
      proof.scope = `${proof.scope} 自动建议的到访位置尚待核验；最优性针对当前选定位置。`;
    }
    progress(100, '完成路线规划');
    const plan: Plan = {
      id: planId,
      assetId: this.state.assetId,
      request,
      routingSchema: this.routingSchema,
      status: solved.status,
      routes,
      graphHash: this.graphHash,
      version: this.state.version,
      approaches: bundle.approaches,
      draft,
      createdAt: Date.now() / 1000,
      coverage: solved.coverage,
      proof,
      graphStatistics: {
        nodes: this.graph.nodeCount,
        edges: this.graph.edgeCount,
        components: this.graph.componentCount,
        gridStepPx: this.graph.step,
        fineGridStepPx: this.edited ? this.graph.step : 1,
      },
      preparationSec: round3(preparationElapsed),
      planningElapsedSec: round3(nowSeconds() - started),
    };
    return plan;
  }

  /** Port of `dirmap/service.py::navigate_course`. */
  async navigate(release: StaticRelease, request: NavigationRequest, controls: Control[]): Promise<NavigateResponse> {
    return await this.withNavigationLock(async () => {
      if (request.accuracyM > 20) throw new Error('定位精度不足，请移至开阔处后重试');
      const age = Date.now() - request.timestampMs;
      if (age < -5000 || age > 20000) throw new Error('定位已过期，请等待新的定位样本');
      const published = release.plan;
      if (!published || !Array.isArray(published.routes)) throw new Error('课程已更新，请使用教师提供的新课程链接');
      const route = published.routes.find(candidate => candidate.teamId === request.teamId);
      if (!route) throw new Error('队伍不属于这份课程');
      const completed = request.completedControlIds ?? [];
      const expectedPrefix = route.controlIds.slice(0, completed.length);
      if (completed.length !== expectedPrefix.length || completed.some((id, index) => id !== expectedPrefix[index])) {
        throw new Error('已完成点必须按本队课程顺序记录，不能跳过必到点');
      }
      const refreshed = Date.now() - request.timestampMs;
      if (refreshed > 20000) throw new Error('定位已过期，请等待新的定位样本');
      if (release.superseded) throw new Error('课程已更新，请使用教师提供的新课程链接');
      if (release.routingSchema && release.routingSchema !== this.routingSchema) {
        throw new Error('课程已更新，请使用教师提供的新课程链接');
      }
      if (this.state.version !== release.version) {
        throw new Error('课程已更新，请使用教师提供的新课程链接');
      }
      // A frozen static release is itself the reviewed artifact: the baked map
      // state stays `reviewed:false` (draft source data) but the published
      // course is immutable, exactly like a server release made after review.
      if (!this.state.reviewed && release.static !== true) {
        throw new Error('课程已更新，请使用教师提供的新课程链接');
      }
      if (published.graphHash !== this.graphHash) {
        throw new Error('课程通行图与当前静态构建不一致，无法重新规划（本机通行修正不会随课程分发）');
      }
      const world = frameLocal(request.lnglat, this.state.frame.origin);
      const pixel = this.mesh.inverse(world);
      if (pixel === null || !Number.isFinite(pixel[0]) || !Number.isFinite(pixel[1])) {
        throw new Error('当前位置超出已校准的课程范围');
      }
      if (!published.request || !Array.isArray(published.request.teams)) throw new Error('课程已更新，请使用教师提供的新课程链接');
      const team = published.request.teams.find(candidate => candidate.id === route.teamId);
      if (!team) throw new Error('队伍不属于这份课程');
      const remaining = route.controlIds.slice(completed.length);
      const publishedControls = release.controls && release.controls.length ? release.controls : controls;
      const lookup = new Map<string, Control>();
      for (const control of publishedControls) lookup.set(control.id, control);
      const remainingControls: Control[] = [];
      for (const id of remaining) {
        const control = lookup.get(id);
        if (!control) throw new Error('课程已更新，请使用教师提供的新课程链接');
        remainingControls.push(control);
      }
      const bundle = planMatrix(this.graph, this.mesh, this.frame, remainingControls,
        [pixel[0], pixel[1]], published.request.finishPixel, this.approaches, { metric: 'time', edited: this.edited });
      if (bundle.unresolved.length) {
        return {
          status: 'NEEDS_REVIEW',
          unresolved: bundle.unresolved,
          version: this.state.version,
          graphHash: this.graphHash,
          route: null,
        };
      }
      const order = Array.from({ length: remainingControls.length + 2 }, (_, index) => index);
      const expanded = expandRoute(this.graph, bundle, order, this.frame);
      const travelSec = expanded.travelWeightM / team.speedMps;
      let serviceSec = 0;
      for (const control of remainingControls) serviceSec += (control.serviceBaseSec ?? 5) + team.people * (control.servicePerPersonSec ?? 1);
      const navRoute: Route = {
        teamId: route.teamId,
        people: team.people,
        controlIds: remaining,
        geometry: expanded.geometry,
        localGeometry: expanded.localGeometry,
        distanceM: expanded.distanceM,
        travelWeightM: expanded.travelWeightM,
        nodeIds: expanded.nodeIds,
        edgeIds: expanded.edgeIds,
        legs: expanded.legs,
        travelSec,
        serviceSec,
        durationSec: travelSec + serviceSec,
        departureSec: 0,
        finishSec: travelSec + serviceSec,
      };
      return {
        status: 'READY',
        version: this.state.version,
        graphHash: this.graphHash,
        completedControlIds: completed,
        generatedAt: Date.now() / 1000,
        route: navRoute,
        scope: '从当前位置到本队剩余点及终点的合法最短路；保持发布时的点位分配和访问顺序。',
      };
    });
  }

  private async withNavigationLock<T>(work: () => Promise<T>): Promise<T> {
    const previous = this.navigationLock;
    let releaseLock: () => void = () => undefined;
    this.navigationLock = new Promise<void>(resolve => { releaseLock = resolve; });
    await previous;
    try {
      return await work();
    } finally {
      releaseLock();
    }
  }
}

const plannerCache = new Map<string, Promise<Planner>>();
const clientPlannerCache = new Map<string, Planner>();

/** Drop every cached planner (baked and client graphs). */
export function clearPlannerCache(): void {
  plannerCache.clear();
  clientPlannerCache.clear();
}

/**
 * Planner for a client-built graph (uploaded/processed map). `approaches` must
 * have been resolved against exactly this graph (e.g. by `web/src/access.ts`),
 * so `plan`/`navigate` never fetch baked bundles. Client graphs are large: the
 * cache keeps only the newest graph's planner; the caller owns the graph.
 */
export function createPlannerFromGraph(graph: RasterGraph, approaches: BakedApproaches, state: MapState): Planner {
  if (!state.registration) throw new Error('缺少有效配准范围');
  const key = `${state.assetId}:${state.version}:${graph.metadata.graphHash}`;
  const cached = clientPlannerCache.get(key);
  if (cached) return cached;
  const planner = new StaticPlanner(state, graph, new Mesh(state.registration), approaches, false);
  clientPlannerCache.clear();
  clientPlannerCache.set(key, planner);
  return planner;
}

async function buildPlanner(state: MapState, progress: PlannerProgress, assets: PlannerAssets | undefined): Promise<Planner> {
  let loader: { metadata: () => Promise<GraphMetadata>; buffer: () => Promise<ArrayBuffer> };
  let approachPromise: Promise<BakedApproaches | null>;
  if (assets) {
    loader = { metadata: async () => assets.metadata, buffer: async () => assets.buffer };
    approachPromise = Promise.resolve(assets.approaches);
  } else {
    const data = await import('./data.ts');
    loader = {
      metadata: () => data.loadGraphMetadata(state.assetId),
      buffer: () => data.loadGraphBuffer(state.assetId),
    };
    approachPromise = data.loadApproaches(state.assetId);
  }
  const edited = (state.edits?.length ?? 0) > 0;
  const graph = await RasterGraph.load(state.assetId, { ...state, edits: state.edits ?? [] }, {
    progress, loader, edits: state.edits ?? [],
  });
  progress(95, '载入到访位置');
  const approaches = await approachPromise;
  const mesh = assets?.mesh ?? new Mesh(state.registration);
  progress(100, '通行图已就绪');
  return new StaticPlanner(state, graph, mesh, approaches, edited);
}

/**
 * Load (or reuse) the planner for a map version: baked graph + approach data.
 */
export async function getPlanner(state: MapState, progress: PlannerProgress = noProgress, assets?: PlannerAssets): Promise<Planner> {
  if (!state || typeof state.assetId !== 'string') throw new Error('请先识别地图');
  const key = `${state.assetId}:${state.version}`;
  const cached = plannerCache.get(key);
  if (cached) return await cached;
  const pending = buildPlanner(state, progress, assets);
  plannerCache.set(key, pending);
  // A baked szu-3 graph is ~80-100 MiB of typed arrays; keep only the newest
  // few versions so repeated teacher edits cannot exhaust a phone's memory.
  while (plannerCache.size > 3) {
    const oldest = plannerCache.keys().next().value;
    if (oldest === undefined || oldest === key) break;
    plannerCache.delete(oldest);
  }
  try {
    return await pending;
  } catch (error) {
    plannerCache.delete(key);
    throw error;
  }
}
