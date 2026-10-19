/** Browser-side raster navigation graph: static bundle loading, the exact
 * quadtree-leaf CSR construction of `dirmap/graph.py`, snapping and shortest
 * paths.
 *
 * The baked `graph.bin.gz` already contains the filtered quadtree leaves in
 * Python's node order, so the fast path only rebuilds the owner grid and the
 * undirected edge set in canonical order. Teacher edits invalidate that node
 * set, so edited graphs re-run the quadtree decomposition locally.
 */
import { Mesh } from './geometry.ts';
import type { MapState, Pixel, PlannerProgress, TerrainEdit } from './types.ts';

export type Metric = 'time' | 'distance';

/** Node id used by the owner grid for "no quadtree leaf here". */
const NO_OWNER = 0xffffffff;
const UNREACHABLE = -9999;

export interface GraphMetadata {
  assetId: string;
  schema: string;
  sourceHash: string;
  graphHash: string;
  step: number;
  adaptive: boolean;
  width: number;
  height: number;
  nodeCount: number;
  directedEdgeCount: number;
  componentCount: number;
  costsPalette: number[];
  expected: { lengthSum: number; weightSum: number; edgeCount: number };
  samples: { from: number; to: number; time: number; distance: number }[];
}

/** Metadata-independent arrays decoded from the gzip-inflated graph bundle. */
export interface GraphBundle {
  width: number;
  height: number;
  nodeCount: number;
  palette: Float32Array;
  paletteCount: number;
  costs: Uint8Array;
  pixels: Float64Array;
  sizes: Uint8Array;
  terrain: Uint8Array;
}

export interface GraphLoadOptions {
  progress?: PlannerProgress;
  loader?: {
    metadata: () => Promise<GraphMetadata>;
    buffer: () => Promise<ArrayBuffer>;
  };
  /** Non-empty edits rebuild the graph from the baked costs. */
  edits?: TerrainEdit[];
}

interface GraphData {
  metadata: GraphMetadata;
  width: number;
  height: number;
  nodeCount: number;
  pixels: Float64Array;
  local: Float64Array;
  costs: Float32Array;
  owner: Uint32Array;
  sizes: Uint8Array;
  terrain: Float32Array;
  indptr: Uint32Array;
  indices: Uint32Array;
  weights: Float32Array;
  lengths: Float32Array;
  components: Uint32Array;
  componentCount: number;
}

interface EdgeArrays {
  indptr: Uint32Array;
  indices: Uint32Array;
  weights: Float32Array;
  lengths: Float32Array;
  pairCount: number;
}

/** NumPy `np.rint`: round half to even. */
function rintEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function yieldToUi(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/** Decode the raw (already inflated) `graph.bin.gz` payload. */
export function loadGraphBundle(buffer: ArrayBuffer): GraphBundle {
  if (buffer.byteLength < 24) throw new Error('通行图数据格式无效');
  const bytes = new Uint8Array(buffer, 0, 8);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
  if (magic !== 'DMGRAPH1') throw new Error('通行图数据格式无效');
  const view = new DataView(buffer);
  const version = view.getUint16(8, true);
  if (version !== 1) throw new Error(`通行图数据版本不受支持（${version}）`);
  const width = view.getUint32(10, true);
  const height = view.getUint32(14, true);
  const nodeCount = view.getUint32(18, true);
  const paletteCount = view.getUint16(22, true);
  let offset = 24;
  const palette = new Float32Array(paletteCount);
  for (let i = 0; i < paletteCount; i++) {
    palette[i] = view.getFloat32(offset, true);
    offset += 4;
  }
  if (offset + width * height + nodeCount * 10 > buffer.byteLength) throw new Error('通行图数据长度不足');
  const costs = new Uint8Array(buffer.slice(offset, offset + width * height));
  offset += width * height;
  const centreX = new Float32Array(buffer.slice(offset, offset + nodeCount * 4));
  offset += nodeCount * 4;
  const centreY = new Float32Array(buffer.slice(offset, offset + nodeCount * 4));
  offset += nodeCount * 4;
  const sizes = new Uint8Array(buffer.slice(offset, offset + nodeCount));
  offset += nodeCount;
  const terrain = new Uint8Array(buffer.slice(offset, offset + nodeCount));
  offset += nodeCount;
  if (offset !== buffer.byteLength) throw new Error('通行图数据格式无效');
  const pixels = new Float64Array(nodeCount * 2);
  for (let i = 0; i < nodeCount; i++) {
    pixels[i * 2] = centreX[i];
    pixels[i * 2 + 1] = centreY[i];
  }
  return { width, height, nodeCount, palette, paletteCount, costs, pixels, sizes, terrain };
}

class KeyCollector {
  private chunks: Float64Array[] = [];
  private current: Float64Array = new Float64Array(1 << 20);
  private used = 0;
  private total = 0;

  push(value: number): void {
    if (this.used === this.current.length) {
      this.chunks.push(this.current);
      this.current = new Float64Array(1 << 20);
      this.used = 0;
    }
    this.current[this.used++] = value;
    this.total++;
  }

  finish(): Float64Array {
    const keys = new Float64Array(this.total);
    let offset = 0;
    for (const chunk of this.chunks) {
      keys.set(chunk, offset);
      offset += chunk.length;
    }
    keys.set(this.current.subarray(0, this.used), offset);
    return keys;
  }
}

/** Build undirected edges from the owner grid exactly like `_quadtree_graph`. */
function buildEdgeArrays(
  owner: Uint32Array, nodeCount: number, width: number, height: number,
  local: Float64Array, sizes: Uint8Array, terrain: Float32Array,
): EdgeArrays {
  const collector = new KeyCollector();
  const count = nodeCount;
  const pushKey = (a: number, b: number): void => {
    collector.push(a < b ? a * count + b : b * count + a);
  };
  // (1, 0) horizontal neighbours
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width - 1; x++) {
      const a = owner[row + x];
      const b = owner[row + x + 1];
      if (a !== NO_OWNER && b !== NO_OWNER && a !== b) pushKey(a, b);
    }
  }
  // (0, 1) vertical neighbours
  if (height > 1) {
    for (let y = 0; y < height - 1; y++) {
      const row = y * width;
      const next = row + width;
      for (let x = 0; x < width; x++) {
        const a = owner[row + x];
        const b = owner[next + x];
        if (a !== NO_OWNER && b !== NO_OWNER && a !== b) pushKey(a, b);
      }
    }
  }
  // (1, 1) down-right diagonals, both axial neighbours must be valid
  if (height > 1) {
    for (let y = 0; y < height - 1; y++) {
      const row = y * width;
      const next = row + width;
      for (let x = 0; x < width - 1; x++) {
        const a = owner[row + x];
        const b = owner[next + x + 1];
        if (a === NO_OWNER || b === NO_OWNER || a === b) continue;
        if (owner[row + x + 1] === NO_OWNER || owner[next + x] === NO_OWNER) continue;
        pushKey(a, b);
      }
    }
  }
  // (-1, 1) down-left diagonals
  if (height > 1) {
    for (let y = 0; y < height - 1; y++) {
      const row = y * width;
      const next = row + width;
      for (let x = 1; x < width; x++) {
        const a = owner[row + x];
        const b = owner[next + x - 1];
        if (a === NO_OWNER || b === NO_OWNER || a === b) continue;
        if (owner[row + x - 1] === NO_OWNER || owner[next + x] === NO_OWNER) continue;
        pushKey(a, b);
      }
    }
  }
  const keys = collector.finish();
  keys.sort();
  let uniqueCount = 0;
  for (let i = 0; i < keys.length; i++) {
    if (i === 0 || keys[i] !== keys[i - 1]) keys[uniqueCount++] = keys[i];
  }
  const counts = new Uint32Array(nodeCount);
  for (let i = 0; i < uniqueCount; i++) {
    const key = keys[i];
    const a = Math.floor(key / count);
    const b = key - a * count;
    counts[a]++;
    counts[b]++;
  }
  const indptr = new Uint32Array(nodeCount + 1);
  for (let i = 0; i < nodeCount; i++) indptr[i + 1] = indptr[i] + counts[i];
  const total = indptr[nodeCount];
  const indices = new Uint32Array(total);
  const weights = new Float32Array(total);
  const lengths = new Float32Array(total);
  const cursor = indptr.slice(0, nodeCount);
  for (let i = 0; i < uniqueCount; i++) {
    const key = keys[i];
    const a = Math.floor(key / count);
    const b = key - a * count;
    const dx = local[a * 2] - local[b * 2];
    const dy = local[a * 2 + 1] - local[b * 2 + 1];
    const length = Math.sqrt(dx * dx + dy * dy);
    const sizeA = sizes[a];
    const sizeB = sizes[b];
    const numerator = Math.fround(Math.fround(terrain[a] * sizeA) + Math.fround(terrain[b] * sizeB));
    const ratio = Math.fround(numerator / Math.fround(sizeA + sizeB));
    const weight = length * ratio;
    let position = cursor[a]++;
    indices[position] = b;
    lengths[position] = length;
    weights[position] = weight;
    position = cursor[b]++;
    indices[position] = a;
    lengths[position] = length;
    weights[position] = weight;
  }
  return { indptr, indices, weights, lengths, pairCount: uniqueCount };
}

function buildComponents(indptr: Uint32Array, indices: Uint32Array, nodeCount: number): { components: Uint32Array; componentCount: number } {
  const parent = new Int32Array(nodeCount);
  const rank = new Uint8Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) parent[i] = i;
  const find = (start: number): number => {
    let node = start;
    while (parent[node] !== node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  };
  for (let node = 0; node < nodeCount; node++) {
    for (let edge = indptr[node]; edge < indptr[node + 1]; edge++) {
      let a = find(node);
      let b = find(indices[edge]);
      if (a === b) continue;
      if (rank[a] < rank[b]) {
        const swap = a;
        a = b;
        b = swap;
      }
      parent[b] = a;
      if (rank[a] === rank[b]) rank[a]++;
    }
  }
  const components = new Uint32Array(nodeCount);
  const rootLabel = new Int32Array(nodeCount).fill(-1);
  let componentCount = 0;
  for (let node = 0; node < nodeCount; node++) {
    const root = find(node);
    let label = rootLabel[root];
    if (label < 0) {
      label = componentCount++;
      rootLabel[root] = label;
    }
    components[node] = label;
  }
  return { components, componentCount };
}

/** Binary min-heap with decrease-key over `keys`. */
class MinHeap {
  private readonly heap: Int32Array;
  private readonly position: Int32Array;
  private readonly keys: Float64Array;
  size = 0;

  constructor(capacity: number, keys: Float64Array) {
    this.heap = new Int32Array(capacity);
    this.position = new Int32Array(capacity);
    this.keys = keys;
  }

  reset(): void {
    this.position.fill(-1);
    this.size = 0;
  }

  insertOrDecrease(node: number): void {
    let position = this.position[node];
    if (position < 0) {
      position = this.size++;
      this.heap[position] = node;
      this.position[node] = position;
    }
    this.siftUp(position);
  }

  pop(): number {
    const root = this.heap[0];
    this.position[root] = -1;
    const last = this.size - 1;
    this.size = last;
    if (last > 0) {
      const moved = this.heap[last];
      this.heap[0] = moved;
      this.position[moved] = 0;
      this.siftDown(0);
    }
    return root;
  }

  private siftUp(start: number): void {
    const node = this.heap[start];
    const key = this.keys[node];
    let position = start;
    while (position > 0) {
      const parentPosition = (position - 1) >> 1;
      const parentNode = this.heap[parentPosition];
      if (this.keys[parentNode] <= key) break;
      this.heap[position] = parentNode;
      this.position[parentNode] = position;
      position = parentPosition;
    }
    this.heap[position] = node;
    this.position[node] = position;
  }

  private siftDown(start: number): void {
    const node = this.heap[start];
    const key = this.keys[node];
    const half = this.size >> 1;
    let position = start;
    while (position < half) {
      let child = position * 2 + 1;
      const other = child + 1;
      if (other < this.size && this.keys[this.heap[other]] < this.keys[this.heap[child]]) child = other;
      const childNode = this.heap[child];
      if (key <= this.keys[childNode]) break;
      this.heap[position] = childNode;
      this.position[childNode] = position;
      position = child;
    }
    this.heap[position] = node;
    this.position[node] = position;
  }
}

interface NodeGridIndex {
  cols: number;
  rows: number;
  cell: number;
  offsets: Uint32Array;
  entries: Uint32Array;
}

/** Re-rasterize teacher edits onto the baked costs (`open` -> cost, `close` -> 0). */
function applyEdits(costs: Float32Array, width: number, height: number, edits: TerrainEdit[]): void {
  const fillCircle = (cx: number, cy: number, radius: number, value: number): void => {
    const x0 = Math.max(0, cx - radius);
    const x1 = Math.min(width - 1, cx + radius);
    const y0 = Math.max(0, cy - radius);
    const y1 = Math.min(height - 1, cy + radius);
    const radiusSquared = radius * radius;
    for (let y = y0; y <= y1; y++) {
      const row = y * width;
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy <= radiusSquared) costs[row + x] = value;
      }
    }
  };
  const fillSegment = (ax: number, ay: number, bx: number, by: number, radius: number, value: number): void => {
    const x0 = Math.max(0, Math.min(ax, bx) - radius);
    const x1 = Math.min(width - 1, Math.max(ax, bx) + radius);
    const y0 = Math.max(0, Math.min(ay, by) - radius);
    const y1 = Math.min(height - 1, Math.max(ay, by) + radius);
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSquared = dx * dx + dy * dy;
    const radiusSquared = radius * radius;
    for (let y = y0; y <= y1; y++) {
      const row = y * width;
      for (let x = x0; x <= x1; x++) {
        let t = 0;
        if (lengthSquared > 0) {
          t = ((x - ax) * dx + (y - ay) * dy) / lengthSquared;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
        }
        const px = x - (ax + t * dx);
        const py = y - (ay + t * dy);
        if (px * px + py * py <= radiusSquared) costs[row + x] = value;
      }
    }
  };
  for (const edit of edits) {
    const points: Pixel[] = [];
    for (const point of edit.pixels ?? []) {
      const x = rintEven(point[0]);
      const y = rintEven(point[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x >= width || y < 0 || y >= height) {
        throw new Error('修正区域超出地图');
      }
      points.push([x, y]);
    }
    if (!points.length) throw new Error('修正区域超出地图');
    const value = edit.kind === 'close' ? 0 : Number(edit.cost ?? 1);
    const radius = Math.trunc(Number(edit.radiusPx ?? 6));
    if (!(radius >= 1 && radius <= 100) || !(value >= 0) || !Number.isFinite(value)) throw new Error('通行修正无效');
    if (points.length === 1) {
      fillCircle(points[0][0], points[0][1], radius, value);
    } else {
      for (let i = 0; i + 1 < points.length; i++) {
        fillSegment(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], radius, value);
      }
      for (const point of [points[0], points[points.length - 1]]) fillCircle(point[0], point[1], radius, value);
    }
  }
}

/** Local quadtree re-decomposition used only when teacher edits change the costs. */
function quadtreeCells(
  costs: Float32Array, mesh: Mesh, width: number, height: number, step: number,
): { pixels: Float64Array; sizes: Uint8Array; terrain: Float32Array; owner: Int32Array; count: number } {
  if ((step & (step - 1)) !== 0 || step < 1 || step > 16) throw new Error('自适应网格步长必须是 1、2、4、8 或 16');
  const owners = new Int32Array(width * height).fill(-1);
  const centreXs: number[] = [];
  const centreYs: number[] = [];
  const cellSizes: number[] = [];
  const terrainValues: number[] = [];
  let count = 0;
  for (let size = step; size >= 1; size >>= 1) {
    const tileCols = Math.floor(width / size);
    const tileRows = Math.floor(height / size);
    for (let tileY = 0; tileY < tileRows; tileY++) {
      const y0 = tileY * size;
      for (let tileX = 0; tileX < tileCols; tileX++) {
        const x0 = tileX * size;
        const firstRow = y0 * width + x0;
        const first = costs[firstRow];
        if (!(first > 0) || owners[firstRow] >= 0) continue;
        let fresh = true;
        for (let yy = 0; yy < size && fresh; yy++) {
          const row = (y0 + yy) * width + x0;
          for (let xx = 0; xx < size; xx++) {
            if (owners[row + xx] >= 0) {
              fresh = false;
              break;
            }
            const value = costs[row + xx];
            if (!(value > 0) || Math.abs(value - first) >= 1e-6) {
              fresh = false;
              break;
            }
          }
        }
        if (!fresh) continue;
        const id = count++;
        for (let yy = 0; yy < size; yy++) owners.fill(id, (y0 + yy) * width + x0, (y0 + yy) * width + x0 + size);
        centreXs.push(x0 + (size - 1) / 2);
        centreYs.push(y0 + (size - 1) / 2);
        cellSizes.push(size);
        terrainValues.push(first);
      }
    }
  }
  if (count < 2) throw new Error('配准范围内没有足够可通行区域');
  let pixels = new Float64Array(count * 2);
  for (let i = 0; i < count; i++) {
    pixels[i * 2] = centreXs[i];
    pixels[i * 2 + 1] = centreYs[i];
  }
  const forward = mesh.forwardMany(pixels);
  let sizes = Uint8Array.from(cellSizes);
  let terrain = Float32Array.from(terrainValues);
  let validCount = 0;
  for (let i = 0; i < count; i++) if (forward.inside[i]) validCount++;
  if (validCount !== count) {
    const mapping = new Int32Array(count).fill(-1);
    let next = 0;
    for (let i = 0; i < count; i++) if (forward.inside[i]) mapping[i] = next++;
    if (next < 2) throw new Error('配准范围内没有足够可通行区域');
    for (let i = 0; i < owners.length; i++) if (owners[i] >= 0) owners[i] = mapping[owners[i]];
    const compactPixels = new Float64Array(next * 2);
    const compactSizes = new Uint8Array(next);
    const compactTerrain = new Float32Array(next);
    for (let i = 0; i < count; i++) {
      const mapped = mapping[i];
      if (mapped < 0) continue;
      compactPixels[mapped * 2] = pixels[i * 2];
      compactPixels[mapped * 2 + 1] = pixels[i * 2 + 1];
      compactSizes[mapped] = sizes[i];
      compactTerrain[mapped] = terrain[i];
    }
    pixels = compactPixels;
    sizes = compactSizes;
    terrain = compactTerrain;
    count = next;
  }
  return { pixels, sizes, terrain, owner: owners, count };
}

/** Two independent FNV-1a 32-bit passes -> 16 lowercase hex characters. */
function shortHash(text: string): string {
  const hash = (seed: number): string => {
    let value = seed >>> 0;
    for (let i = 0; i < text.length; i++) {
      value ^= text.charCodeAt(i);
      value = Math.imul(value, 0x01000193) >>> 0;
    }
    return value.toString(16).padStart(8, '0');
  };
  return hash(0x811c9dc5) + hash(0x7f4a7c15);
}

/**
 * Native-pixel navigation graph with metre-weighted CSR edges.
 */
export class RasterGraph {
  readonly metadata: GraphMetadata;
  readonly width: number;
  readonly height: number;
  readonly step: number;
  readonly nodeCount: number;
  readonly pixels: Float64Array;
  readonly local: Float64Array;
  readonly costs: Float32Array;
  readonly sizes: Uint8Array;
  readonly terrain: Float32Array;
  readonly indptr: Uint32Array;
  readonly indices: Uint32Array;
  readonly weights: Float32Array;
  readonly lengths: Float32Array;
  readonly components: Uint32Array;
  readonly componentCount: number;
  readonly owner: Uint32Array;

  private readonly scratchDistance: Float64Array;
  private readonly scratchParent: Int32Array;
  private readonly dijkstraHeap: MinHeap;
  private readonly targetMask: Uint8Array;
  private readonly aStarDistance: Float64Array;
  private readonly aStarScore: Float64Array;
  private readonly aStarHeuristic: Float64Array;
  private readonly aStarParent: Int32Array;
  private readonly aStarHeap: MinHeap;
  private readonly heuristicScale: number;
  private nodeGrid: NodeGridIndex | null = null;

  private constructor(data: GraphData) {
    this.metadata = data.metadata;
    this.width = data.width;
    this.height = data.height;
    this.step = Number(data.metadata.step) || 4;
    this.nodeCount = data.nodeCount;
    this.pixels = data.pixels;
    this.local = data.local;
    this.costs = data.costs;
    this.sizes = data.sizes;
    this.terrain = data.terrain;
    this.indptr = data.indptr;
    this.indices = data.indices;
    this.weights = data.weights;
    this.lengths = data.lengths;
    this.components = data.components;
    this.componentCount = data.componentCount;
    this.owner = data.owner;
    this.scratchDistance = new Float64Array(this.nodeCount);
    this.scratchParent = new Int32Array(this.nodeCount);
    this.dijkstraHeap = new MinHeap(this.nodeCount, this.scratchDistance);
    this.targetMask = new Uint8Array(this.nodeCount);
    this.aStarDistance = new Float64Array(this.nodeCount);
    this.aStarScore = new Float64Array(this.nodeCount);
    this.aStarHeuristic = new Float64Array(this.nodeCount);
    this.aStarParent = new Int32Array(this.nodeCount);
    this.aStarHeap = new MinHeap(this.nodeCount, this.aStarScore);
    let minimumCost = Infinity;
    for (const value of data.metadata.costsPalette ?? []) {
      if (Number.isFinite(value) && value > 0 && value < minimumCost) minimumCost = value;
    }
    this.heuristicScale = Number.isFinite(minimumCost) ? minimumCost : 1;
  }

  /** Load and build a graph from the static data bundle. */
  static async load(assetId: string, state: MapState, opts: GraphLoadOptions = {}): Promise<RasterGraph> {
    const progress = opts.progress ?? ((): void => undefined);
    const registration = state.registration;
    if (!registration) throw new Error('缺少有效配准范围');
    const mesh = new Mesh(registration);
    if (!mesh.hasMesh) throw new Error('缺少有效配准范围');
    progress(20, '载入通行图');
    let loader = opts.loader;
    if (!loader) {
      const data = await import('./data.ts');
      loader = {
        metadata: () => data.loadGraphMetadata(assetId),
        buffer: () => data.loadGraphBuffer(assetId),
      };
    }
    const metadata = await loader.metadata();
    const buffer = await loader.buffer();
    await yieldToUi();
    progress(60, '建立保留细墙的通行图');
    const bundle = loadGraphBundle(buffer);
    if (!metadata || typeof metadata !== 'object') throw new Error('通行图元数据无效');
    if (bundle.width !== metadata.width || bundle.height !== metadata.height) throw new Error('通行图数据与元数据不一致');
    const edits = opts.edits ?? state.edits ?? [];
    let data: GraphData;
    if (edits.length > 0) {
      data = RasterGraph.rebuildWithEdits(bundle, metadata, mesh, edits);
    } else {
      data = RasterGraph.buildFromBundle(bundle, metadata, mesh);
    }
    await yieldToUi();
    progress(90, '完成通行图');
    return new RasterGraph(data);
  }

  private static buildFromBundle(bundle: GraphBundle, metadata: GraphMetadata, mesh: Mesh): GraphData {
    const { width, height, nodeCount } = bundle;
    const costs = new Float32Array(width * height);
    for (let i = 0; i < costs.length; i++) costs[i] = bundle.palette[bundle.costs[i]];
    const forward = mesh.forwardMany(bundle.pixels);
    const owner = new Uint32Array(width * height).fill(NO_OWNER);
    for (let node = 0; node < nodeCount; node++) {
      const size = bundle.sizes[node];
      const x0 = rintEven(bundle.pixels[node * 2] - (size - 1) / 2);
      const y0 = rintEven(bundle.pixels[node * 2 + 1] - (size - 1) / 2);
      for (let row = 0; row < size; row++) {
        const start = (y0 + row) * width + x0;
        owner.fill(node, start, start + size);
      }
    }
    const terrain = new Float32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) terrain[node] = bundle.palette[bundle.terrain[node]];
    const edges = buildEdgeArrays(owner, nodeCount, width, height, forward.local, bundle.sizes, terrain);
    const components = buildComponents(edges.indptr, edges.indices, nodeCount);
    return {
      metadata, width, height, nodeCount, pixels: bundle.pixels, local: forward.local, costs,
      owner, sizes: bundle.sizes, terrain,
      indptr: edges.indptr, indices: edges.indices, weights: edges.weights, lengths: edges.lengths,
      components: components.components, componentCount: components.componentCount,
    };
  }

  private static rebuildWithEdits(
    bundle: GraphBundle, metadata: GraphMetadata, mesh: Mesh, edits: TerrainEdit[],
  ): GraphData {
    const { width, height } = bundle;
    const costs = new Float32Array(width * height);
    for (let i = 0; i < costs.length; i++) costs[i] = bundle.palette[bundle.costs[i]];
    applyEdits(costs, width, height, edits);
    const step = Number(metadata.step) || 4;
    const cells = quadtreeCells(costs, mesh, width, height, step);
    const nodeCount = cells.count;
    const owner = new Uint32Array(width * height).fill(NO_OWNER);
    for (let i = 0; i < cells.owner.length; i++) {
      if (cells.owner[i] >= 0) owner[i] = cells.owner[i];
    }
    const forward = mesh.forwardMany(cells.pixels);
    const edges = buildEdgeArrays(owner, nodeCount, width, height, forward.local, cells.sizes, cells.terrain);
    const components = buildComponents(edges.indptr, edges.indices, nodeCount);
    const hash = `${metadata.graphHash}-e${shortHash(JSON.stringify(edits))}`;
    const updated: GraphMetadata = {
      ...metadata,
      graphHash: hash,
      nodeCount,
      directedEdgeCount: edges.indices.length,
      componentCount: components.componentCount,
    };
    return {
      metadata: updated, width, height, nodeCount, pixels: cells.pixels, local: forward.local, costs,
      owner, sizes: cells.sizes, terrain: cells.terrain,
      indptr: edges.indptr, indices: edges.indices, weights: edges.weights, lengths: edges.lengths,
      components: components.components, componentCount: components.componentCount,
    };
  }

  /** Number of directed CSR entries (Python `weights.nnz`). */
  get edgeCount(): number {
    return this.indices.length;
  }

  /** NumPy `np.rint`-compatible nearest node owned by a free pixel. */
  node_at(pixel: Pixel, maxDistancePx = 3, near = false): number | null {
    const pointX = pixel[0];
    const pointY = pixel[1];
    if (!Number.isFinite(pointX) || !Number.isFinite(pointY)) throw new Error('到访坐标无效');
    const width = this.width;
    const height = this.height;
    const x = rintEven(pointX);
    const y = rintEven(pointY);
    if (x < 0 || x >= width || y < 0 || y >= height) return null;
    const limit = Math.trunc(Math.max(1, maxDistancePx));
    const owner = this.owner;
    let closest = -1;
    let closestDistance = Infinity;
    for (let radius = 0; radius <= limit; radius++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width, x + radius + 1);
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(height, y + radius + 1);
      const hitX: number[] = [];
      const hitY: number[] = [];
      const hitNode: number[] = [];
      const hitDistance: number[] = [];
      for (let row = y0; row < y1; row++) {
        const base = row * width;
        for (let column = x0; column < x1; column++) {
          const node = owner[base + column];
          if (node === NO_OWNER) continue;
          const dx = column - pointX;
          const dy = row - pointY;
          hitX.push(column);
          hitY.push(row);
          hitNode.push(node);
          hitDistance.push(Math.sqrt(dx * dx + dy * dy));
        }
      }
      if (!hitNode.length) continue;
      const order = hitDistance.map((_, index) => index).sort((left, right) => hitDistance[left] - hitDistance[right]);
      for (const index of order) {
        const node = hitNode[index];
        const distance = hitDistance[index];
        if (closest < 0 || distance < closestDistance) {
          closest = node;
          closestDistance = distance;
        }
        const owned: Pixel = [hitX[index], hitY[index]];
        if (this.clearSegment([pointX, pointY], owned) && this.clearSegment([pointX, pointY], this.nodePixel(node))) return node;
      }
    }
    return near && closest >= 0 ? closest : null;
  }

  /** Nearest node reachable by a clear segment from the pixel. */
  snap(pixel: Pixel, maxDistancePx = 24): number {
    const pointX = pixel[0];
    const pointY = pixel[1];
    if (!Number.isFinite(pointX) || !Number.isFinite(pointY)) throw new Error('到访坐标无效');
    const exact = this.node_at(pixel, 1);
    if (exact !== null) return exact;
    const candidates = this.nodesWithin([pointX, pointY], maxDistancePx);
    for (const node of candidates) {
      if (this.clearSegment([pointX, pointY], this.nodePixel(node))) return node;
    }
    throw new Error('点位没有无障碍连接，请在同一侧选择实际可到达的位置');
  }

  private nodePixel(node: number): Pixel {
    return [this.pixels[node * 2], this.pixels[node * 2 + 1]];
  }

  /** All node ids whose centre lies within `radiusPx`, nearest first. */
  private nodesWithin(point: Pixel, radiusPx: number): number[] {
    const index = this.ensureNodeGrid();
    const cell = index.cell;
    const x0 = Math.max(0, Math.floor((point[0] - radiusPx) / cell));
    const x1 = Math.min(index.cols - 1, Math.floor((point[0] + radiusPx) / cell));
    const y0 = Math.max(0, Math.floor((point[1] - radiusPx) / cell));
    const y1 = Math.min(index.rows - 1, Math.floor((point[1] + radiusPx) / cell));
    const candidates: { node: number; distance: number }[] = [];
    for (let row = y0; row <= y1; row++) {
      for (let column = x0; column <= x1; column++) {
        const cellIndex = row * index.cols + column;
        for (let e = index.offsets[cellIndex]; e < index.offsets[cellIndex + 1]; e++) {
          const node = index.entries[e];
          const dx = this.pixels[node * 2] - point[0];
          const dy = this.pixels[node * 2 + 1] - point[1];
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance <= radiusPx + 1e-9) candidates.push({ node, distance });
        }
      }
    }
    candidates.sort((left, right) => left.distance - right.distance || left.node - right.node);
    return candidates.map(entry => entry.node);
  }

  private ensureNodeGrid(): NodeGridIndex {
    if (this.nodeGrid) return this.nodeGrid;
    const cell = 16;
    const cols = Math.max(1, Math.ceil(this.width / cell));
    const rows = Math.max(1, Math.ceil(this.height / cell));
    const counts = new Uint32Array(cols * rows);
    const cellOf = (node: number): number => {
      let column = Math.floor(this.pixels[node * 2] / cell);
      let row = Math.floor(this.pixels[node * 2 + 1] / cell);
      if (column < 0) column = 0;
      if (column >= cols) column = cols - 1;
      if (row < 0) row = 0;
      if (row >= rows) row = rows - 1;
      return row * cols + column;
    };
    for (let node = 0; node < this.nodeCount; node++) counts[cellOf(node)]++;
    const offsets = new Uint32Array(cols * rows + 1);
    for (let i = 0; i < cols * rows; i++) offsets[i + 1] = offsets[i] + counts[i];
    const entries = new Uint32Array(offsets[cols * rows]);
    const cursor = offsets.slice(0, cols * rows);
    for (let node = 0; node < this.nodeCount; node++) entries[cursor[cellOf(node)]++] = node;
    this.nodeGrid = { cols, rows, cell, offsets, entries };
    return this.nodeGrid;
  }

  /** All native pixel squares touched by the segment, including corners. */
  segmentPixels(a: Pixel, b: Pixel): Pixel[] {
    const ax = a[0];
    const ay = a[1];
    const bx = b[0];
    const by = b[1];
    if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) {
      throw new Error('线段坐标无效');
    }
    const deltaX = bx - ax;
    const deltaY = by - ay;
    const times: number[] = [0, 1];
    for (let axis = 0; axis < 2; axis++) {
      const delta = axis === 0 ? deltaX : deltaY;
      if (Math.abs(delta) < 1e-12) continue;
      const from = axis === 0 ? ax : ay;
      const low = Math.min(from, axis === 0 ? bx : by);
      const high = Math.max(from, axis === 0 ? bx : by);
      const first = Math.ceil(low - 0.5);
      const last = Math.floor(high - 0.5);
      for (let boundary = first; boundary <= last; boundary++) times.push((boundary + 0.5 - from) / delta);
    }
    for (let i = 0; i < times.length; i++) times[i] = times[i] < 0 ? 0 : times[i] > 1 ? 1 : times[i];
    times.sort((left, right) => left - right);
    const uniqueTimes: number[] = [];
    for (const time of times) if (!uniqueTimes.length || time !== uniqueTimes[uniqueTimes.length - 1]) uniqueTimes.push(time);
    const sampleTimes: number[] = [];
    for (let i = 0; i < uniqueTimes.length; i++) {
      sampleTimes.push(uniqueTimes[i]);
      if (i + 1 < uniqueTimes.length) sampleTimes.push((uniqueTimes[i] + uniqueTimes[i + 1]) / 2);
    }
    const keys = new Set<string>();
    const touched: Pixel[] = [];
    const add = (x: number, y: number): void => {
      const key = `${x},${y}`;
      if (keys.has(key)) return;
      keys.add(key);
      touched.push([x, y]);
    };
    for (const time of sampleTimes) {
      const pointX = ax + time * deltaX;
      const pointY = ay + time * deltaY;
      const cellX = Math.floor(pointX + 0.5 + 1e-9);
      const cellY = Math.floor(pointY + 0.5 + 1e-9);
      const boundaryX = Math.abs(pointX + 0.5 - rintEven(pointX + 0.5)) <= 1e-8;
      const boundaryY = Math.abs(pointY + 0.5 - rintEven(pointY + 0.5)) <= 1e-8;
      add(cellX, cellY);
      if (boundaryX) add(cellX - 1, cellY);
      if (boundaryY) add(cellX, cellY - 1);
      if (boundaryX && boundaryY) add(cellX - 1, cellY - 1);
    }
    touched.sort((left, right) => left[0] - right[0] || left[1] - right[1]);
    return touched;
  }

  /** Whether the straight segment only crosses free pixels. */
  clearSegment(a: Pixel, b: Pixel): boolean {
    const pixels = this.segmentPixels(a, b);
    const width = this.width;
    for (const point of pixels) {
      const x = point[0];
      const y = point[1];
      if (x < 0 || y < 0 || x >= width || y >= this.height) return false;
      if (!(this.costs[y * width + x] > 0)) return false;
    }
    return true;
  }

  /** Single-source shortest paths for every requested source (full rows). */
  dijkstra(sources: number[], metric: Metric = 'time', targets?: number[]): { distances: Float64Array; predecessors: Int32Array } {
    const nodeCount = this.nodeCount;
    for (const source of sources) {
      if (!Number.isInteger(source) || source < 0 || source >= nodeCount) throw new Error('最短路节点越界');
    }
    const usingTargets = Array.isArray(targets) && targets.length > 0;
    let remaining = 0;
    if (usingTargets) {
      this.targetMask.fill(0);
      for (const target of targets as number[]) {
        if (!Number.isInteger(target) || target < 0 || target >= nodeCount) throw new Error('最短路节点越界');
        if (!this.targetMask[target]) {
          this.targetMask[target] = 1;
          remaining++;
        }
      }
    }
    const rows = sources.length;
    const distances = new Float64Array(rows * nodeCount);
    distances.fill(Infinity);
    const predecessors = new Int32Array(rows * nodeCount);
    predecessors.fill(UNREACHABLE);
    for (let row = 0; row < rows; row++) {
      this.scratchDistance.fill(Infinity);
      this.scratchParent.fill(UNREACHABLE);
      const settledTargets = this.runDijkstra(sources[row], metric, usingTargets, remaining);
      void settledTargets;
      distances.set(this.scratchDistance, row * nodeCount);
      predecessors.set(this.scratchParent, row * nodeCount);
    }
    return { distances, predecessors };
  }

  /** Distances between every source and target (no predecessor rows retained). */
  distanceToTargets(sources: number[], targets: number[], metric: Metric = 'time'): Float64Array {
    const nodeCount = this.nodeCount;
    for (const source of sources) {
      if (!Number.isInteger(source) || source < 0 || source >= nodeCount) throw new Error('最短路节点越界');
    }
    this.targetMask.fill(0);
    let remaining = 0;
    for (const target of targets) {
      if (!Number.isInteger(target) || target < 0 || target >= nodeCount) throw new Error('最短路节点越界');
      if (!this.targetMask[target]) {
        this.targetMask[target] = 1;
        remaining++;
      }
    }
    const rows = sources.length;
    const columns = targets.length;
    const matrix = new Float64Array(rows * columns);
    matrix.fill(Infinity);
    for (let row = 0; row < rows; row++) {
      this.scratchDistance.fill(Infinity);
      if (remaining > 0) this.runDijkstra(sources[row], metric, true, remaining);
      else this.runDijkstra(sources[row], metric, false, 0);
      for (let column = 0; column < columns; column++) matrix[row * columns + column] = this.scratchDistance[targets[column]];
    }
    return matrix;
  }

  /** Dijkstra into the scratch distance buffer; returns unsettled target count. */
  private runDijkstra(source: number, metric: Metric, useTargets: boolean, targetCount: number): number {
    const distance = this.scratchDistance;
    const parent = this.scratchParent;
    const indices = this.indices;
    const indptr = this.indptr;
    const weights = this.weights;
    const lengths = this.lengths;
    const mask = this.targetMask;
    const heap = this.dijkstraHeap;
    distance[source] = 0;
    heap.reset();
    heap.insertOrDecrease(source);
    let remaining = targetCount;
    while (heap.size > 0) {
      const node = heap.pop();
      const value = distance[node];
      if (!Number.isFinite(value)) break;
      if (useTargets && mask[node]) {
        remaining--;
        if (remaining === 0) break;
      }
      const start = indptr[node];
      const end = indptr[node + 1];
      for (let edge = start; edge < end; edge++) {
        const next = indices[edge];
        const candidate = value + (metric === 'time' ? weights[edge] : lengths[edge]);
        if (candidate < distance[next]) {
          distance[next] = candidate;
          parent[next] = node;
          heap.insertOrDecrease(next);
        }
      }
    }
    return remaining;
  }

  /** Edge length and weight for one CSR edge. */
  edgeMetrics(from: number, to: number): { length: number; weight: number } | null {
    const start = this.indptr[from];
    const end = this.indptr[from + 1];
    for (let edge = start; edge < end; edge++) {
      if (this.indices[edge] === to) return { length: this.lengths[edge], weight: this.weights[edge] };
    }
    return null;
  }

  /** A* node path (inclusive of both endpoints); throws when disconnected. */
  shortestPath(from: number, to: number, metric: Metric = 'time'): number[] {
    const nodeCount = this.nodeCount;
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0 || from >= nodeCount || to >= nodeCount) {
      throw new Error('必到点不连通');
    }
    if (from === to) return [from];
    const distance = this.aStarDistance;
    const score = this.aStarScore;
    const heuristicCache = this.aStarHeuristic;
    const parent = this.aStarParent;
    const indices = this.indices;
    const indptr = this.indptr;
    const weights = this.weights;
    const lengths = this.lengths;
    const local = this.local;
    distance.fill(Infinity);
    score.fill(Infinity);
    heuristicCache.fill(-1);
    parent.fill(-1);
    const targetX = local[to * 2];
    const targetY = local[to * 2 + 1];
    const scale = metric === 'time' ? this.heuristicScale : 1;
    const heuristic = (node: number): number => {
      let value = heuristicCache[node];
      if (value < 0) {
        const dx = local[node * 2] - targetX;
        const dy = local[node * 2 + 1] - targetY;
        value = scale * Math.sqrt(dx * dx + dy * dy);
        heuristicCache[node] = value;
      }
      return value;
    };
    const heap = this.aStarHeap;
    heap.reset();
    distance[from] = 0;
    score[from] = heuristic(from);
    heap.insertOrDecrease(from);
    while (heap.size > 0) {
      const node = heap.pop();
      if (node === to) {
        const path = [to];
        let current = to;
        while (current !== from) {
          current = parent[current];
          if (current < 0) throw new Error('必到点不连通');
          path.push(current);
        }
        path.reverse();
        return path;
      }
      const value = distance[node];
      const start = indptr[node];
      const end = indptr[node + 1];
      for (let edge = start; edge < end; edge++) {
        const next = indices[edge];
        const candidate = value + (metric === 'time' ? weights[edge] : lengths[edge]);
        if (candidate < distance[next]) {
          distance[next] = candidate;
          parent[next] = node;
          score[next] = candidate + heuristic(next);
          heap.insertOrDecrease(next);
        }
      }
    }
    throw new Error('必到点不连通');
  }
}
