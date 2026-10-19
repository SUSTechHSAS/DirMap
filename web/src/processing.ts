/** Client-side map processing pipeline.
 *
 * Browser counterpart of `dirmap/service.py` + `vision.py` + `registration.py` +
 * `access.py`: decode a scan, recognize controls/terrain, register it to the
 * campus reference (or reuse an existing fit), build the navigation graph,
 * resolve legal approaches and render the overlays. Everything runs locally;
 * results persist through `workspace-store.ts` and are planned with the
 * client-built graph.
 *
 * The heavy computer-vision modules stay DOM-free (`vision.ts`,
 * `image-registration.ts`, `access.ts`); only the bitmap decode/encode helpers in
 * this file touch canvas APIs, so the pipeline can later move into a worker.
 */
import { analyze, type AnalysisResult, type RgbaImage } from './vision';
import { autoRegister, manualRegister, projectControls, warpOverlay } from './image-registration';
import { prepareAccess, resolveApproaches, type AccessSemantics } from './access';
import { buildGraphFromRaster, type RasterGraph } from './raster-graph';
import { createPlannerFromGraph, type Planner } from './planner';
import { loadOriginalPng, loadReferenceFeatures } from './data';
import {
  floatCosts, loadLocalMap, localFileUrl, saveLocalMap, storedSemantics,
  type LocalFileKind, type LocalMapKind, type LocalMapRecord, type LocalMapSummary, type StoredAnalysis,
} from './workspace-store';
import type { Anchor, BakedApproach, BakedApproaches, Control, MapState, Pixel, PlannerProgress, TerrainEdit } from './types';
import { Mesh, type RegistrationData } from './geometry';

export const ROUTING_SCHEMA = 'native-quadtree-v4-roof-palette';
export const FRAME: { crs: string; origin: Pixel } = { crs: 'WGS84', origin: [113.9304, 22.535] };

export type ProgressCallback = PlannerProgress;

export interface WarpedRaster { rgba: Uint8ClampedArray; width: number; height: number; extent: number[] }

export interface ProcessOptions {
  assetId: string;
  name: string;
  filename: string;
  sourceHash: string;
  image: RgbaImage;
  existing?: MapState | null;
  mode: 'process' | 'terrain';
  progress?: ProgressCallback;
}

export interface ProcessResult {
  state: MapState;
  analysis: AnalysisResult;
  overlay: WarpedRaster;
  terrainWarp: WarpedRaster;
  approaches: BakedApproaches;
  graphHash: string;
  files: { overlay: Blob; terrain: Blob; terrainWarp: Blob };
}

function step(progress: ProgressCallback | undefined, from: number, to: number): ProgressCallback {
  return (percent: number, message: string) => progress?.(from + (to - from) * percent / 100, message);
}

/** Decode a PNG/JPEG blob into raw RGBA pixels without colour conversion. */
export async function decodeImageBlob(blob: Blob): Promise<RgbaImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' } as ImageBitmapOptions);
  } catch { bitmap = await createImageBitmap(blob); }
  try {
    const { width, height } = bitmap;
    const data = new Uint8ClampedArray(width * height * 4);
    if (typeof OffscreenCanvas !== 'undefined') {
      const canvas = new OffscreenCanvas(width, height);
      const context = canvas.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
      context.drawImage(bitmap, 0, 0);
      data.set(context.getImageData(0, 0, width, height).data);
    } else {
      const canvas = document.createElement('canvas');
      canvas.width = width; canvas.height = height;
      const context = canvas.getContext('2d', { willReadFrequently: true })!;
      context.drawImage(bitmap, 0, 0);
      data.set(context.getImageData(0, 0, width, height).data);
    }
    return { data, width, height };
  } finally { bitmap.close?.(); }
}

/** Encode RGBA pixels as a PNG blob (overlay / terrain visualization). */
export async function encodeImageRgba(rgba: Uint8ClampedArray, width: number, height: number): Promise<Blob> {
  if (!width || !height || rgba.length !== width * height * 4) throw new Error('图像尺寸无效');
  const pixels = new ImageData(new Uint8ClampedArray(rgba), width, height);
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    (canvas.getContext('2d') as OffscreenCanvasRenderingContext2D).putImageData(pixels, 0, 0);
    return await canvas.convertToBlob({ type: 'image/png' });
  }
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  canvas.getContext('2d')!.putImageData(pixels, 0, 0);
  return await new Promise<Blob>((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('无法编码图像')), 'image/png'));
}

export async function sha256Hex(data: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

export async function sha256Blob(blob: Blob): Promise<string> {
  return await sha256Hex(await blob.arrayBuffer());
}

/** Native scan for an asset: baked lossless PNG or the locally stored upload. */
export async function loadNativeImage(assetId: string): Promise<{ image: RgbaImage; sha256?: string }> {
  const local = await localFileUrl(assetId, 'original');
  if (local) {
    const response = await fetch(local);
    if (response.ok) {
      const blob = await response.blob();
      return { image: await decodeImageBlob(blob), sha256: await sha256Blob(blob) };
    }
  }
  const blob = await loadOriginalPng(assetId);
  return { image: await decodeImageBlob(blob), sha256: await sha256Blob(blob) };
}

function floatCostsFromAnalysis(analysis: AnalysisResult): Float32Array {
  const costs = new Float32Array(analysis.costsIndex.length);
  const palette = analysis.costsPalette;
  for (let index = 0; index < costs.length; index++) costs[index] = palette[analysis.costsIndex[index]] ?? 0;
  return costs;
}

function approachDocument(assetId: string, graphHash: string, controls: Record<string, BakedApproach>): BakedApproaches {
  return { assetId, routingSchema: ROUTING_SCHEMA, graphHash, controls } as BakedApproaches;
}

/** Component of the canonical start/finish symbol, used by the approach retry rule. */
function canonicalStartComponent(controls: Control[], graph: RasterGraph): number | undefined {
  const start = controls.find(control => ['start', 'start_finish'].includes(control.role ?? ''));
  if (!start) return undefined;
  try { return graph.components[graph.snap(start.approachPixel ?? start.pixel)]; }
  catch { return undefined; }
}

function semanticsFromAnalysis(analysis: AnalysisResult): AccessSemantics {
  const masks = analysis.semantics as unknown as Record<string, Uint8Array | undefined>;
  return {
    building: masks.building, water: masks.water, point_ink: masks.point_ink,
    vegetation: masks.vegetation, restricted: masks.restricted,
  };
}

function warp(image: RgbaImage, registration: RegistrationData, maxSide: number, nearest = false): WarpedRaster {
  return warpOverlay(image, registration, { maxSide, nearest }) as unknown as WarpedRaster;
}

/** Run the full in-browser pipeline for one map sheet. */
export async function processMap(options: ProcessOptions): Promise<ProcessResult> {
  const { assetId, image, existing, mode } = options;
  const progress = options.progress;
  const report = progress ?? (() => {});
  report(2, mode === 'terrain' ? '重新识别通行区域…' : '解析原图像素与图层…');
  const analysis = await analyze(image, {
    progress: step(progress, 2, mode === 'terrain' ? 45 : 50),
    labeler: mode === 'terrain' ? null : undefined,
  });

  let registration: RegistrationData | null = existing?.registration ?? null;
  let quality: MapState['quality'] = (existing?.quality ?? null) as MapState['quality'];
  if (mode === 'process' && !registration) {
    report(52, '匹配校园参照…');
    try {
      const reference = await loadReferenceFeatures();
      const result = await autoRegister(analysis.features as never, reference.features as never,
        analysis.width, analysis.height, { progress: step(progress, 50, 68) });
      registration = result.registration as RegistrationData;
      quality = result.quality as unknown as MapState['quality'];
    } catch (error) {
      analysis.warnings.push(`自动配准需要教师核对：${error instanceof Error ? error.message : String(error)}`);
    }
  } else {
    report(58, '复用已有配准…');
  }

  const controls: Control[] = (mode === 'terrain' && existing?.controls?.length
    ? existing.controls : analysis.controls as unknown as Control[]).map(control => ({ ...control }));
  if (registration) projectControls(controls, registration);

  let overlay: WarpedRaster = { rgba: new Uint8ClampedArray(0), width: 0, height: 0, extent: [] };
  let terrainWarp: WarpedRaster = { rgba: new Uint8ClampedArray(0), width: 0, height: 0, extent: [] };
  let approaches: BakedApproaches = approachDocument(assetId, '', {});
  let graphHash = '';
  const edits: TerrainEdit[] = existing?.edits ?? [];
  if (registration) {
    report(70, '生成统一坐标叠层…');
    overlay = warp(image, registration, 1700);
    const terrainSource: RgbaImage = { data: analysis.terrainRgba, width: analysis.width, height: analysis.height };
    terrainWarp = warp(terrainSource, registration, 1200, true);
    report(82, '建立通行图与合法到访位置…');
    const graph = buildGraphFromRaster(floatCostsFromAnalysis(analysis), registration, options.sourceHash, edits, 4,
      { width: analysis.width, height: analysis.height, assetId });
    graphHash = graph.metadata.graphHash;
    await prepareAccess();
    const resolved = resolveApproaches({
      graph, mesh: new Mesh(registration), semantics: semanticsFromAnalysis(analysis),
      edits, closedEdits: graph.closedEdits, registration,
    }, controls, { startComponent: canonicalStartComponent(controls, graph) });
    approaches = approachDocument(assetId, graphHash, resolved as Record<string, BakedApproach>);
  }

  report(90, '保存本机处理结果…');
  const statistics = analysis.statistics as unknown as MapState['statistics'];
  const warnings = [...new Set([...analysis.warnings, ...(existing?.warnings ?? [])])].slice(0, 12);
  const common = {
    assetId, name: options.name, filename: options.filename, sourceHash: options.sourceHash,
    width: analysis.width, height: analysis.height,
    version: (existing?.version ?? 0) + 1, controls, edits, reviewed: false,
    registration, quality, statistics, frame: FRAME,
    terrainSchema: 'course-occlusion-v2-roof-palette', graphHash: graphHash || null, static: false,
  };
  const state = (registration
    ? { ...common, extent: overlay.extent, warnings }
    : { ...common, extent: undefined, warnings: [...new Set([...warnings, '自动配准证据不足，请录入同名点后继续。'])] }) as unknown as MapState;
  const [overlayBlob, terrainBlob, terrainWarpBlob] = await Promise.all([
    overlay.width ? encodeImageRgba(overlay.rgba, overlay.width, overlay.height) : Promise.resolve(new Blob()),
    encodeImageRgba(analysis.terrainRgba, analysis.width, analysis.height),
    terrainWarp.width ? encodeImageRgba(terrainWarp.rgba, terrainWarp.width, terrainWarp.height) : Promise.resolve(new Blob()),
  ]);
  return {
    state, analysis, overlay, terrainWarp, approaches, graphHash,
    files: { overlay: overlayBlob, terrain: terrainBlob, terrainWarp: terrainWarpBlob },
  };
}

/** Re-run registration from teacher anchor pairs, keeping the stored analysis. */
export async function reregisterMap(options: {
  assetId: string; image: RgbaImage; existing: MapState; record: LocalMapRecord; anchors: Anchor[];
  progress?: ProgressCallback;
}): Promise<ProcessResult> {
  const { assetId, existing, record, anchors, image } = options;
  const progress = options.progress ?? (() => {});
  progress(4, '拟合同名点并生成局部校准网格…');
  const result = await manualRegister(anchors);
  const registration = result.registration as RegistrationData;
  const quality = result.quality as unknown as MapState['quality'];
  progress(40, '投影检查点与配准数据…');
  const controls = existing.controls.map(control => ({ ...control }));
  projectControls(controls, registration);
  progress(55, '反采样原图叠层…');
  const overlay = warp(image, registration, 1700);
  const terrainFile = await localFileUrl(assetId, 'terrain');
  const terrainBlob = terrainFile ? await (await fetch(terrainFile)).blob() : null;
  let terrainWarp: WarpedRaster = { rgba: new Uint8ClampedArray(0), width: 0, height: 0, extent: [] };
  if (terrainBlob) {
    const terrainImage = await decodeImageBlob(terrainBlob);
    terrainWarp = warp(terrainImage, registration, 1200, true);
  }
  progress(72, '重新生成通行图与到访位置…');
  const costs = floatCosts(record.analysis);
  const graph = buildGraphFromRaster(costs, registration, existing.sourceHash ?? record.summary.sha256, existing.edits, 4,
    { width: record.analysis.width, height: record.analysis.height, assetId });
  const semantics = storedSemantics(record.analysis);
  await prepareAccess();
  const resolved = resolveApproaches({ graph, mesh: new Mesh(registration), semantics, edits: existing.edits, closedEdits: graph.closedEdits, registration }, controls,
    { startComponent: canonicalStartComponent(controls, graph) });
  const graphHash = graph.metadata.graphHash;
  const approaches = approachDocument(assetId, graphHash, resolved as Record<string, BakedApproach>);
  const state = {
    ...existing, version: existing.version + 1, reviewed: false, controls,
    registration, quality, extent: overlay.extent, graphHash,
    warnings: [...new Set([...(existing.warnings ?? []), '已用同名点重新校准；旧路线已失效。'])].slice(0, 12),
    static: false,
  } as unknown as MapState;
  // Re-registration keeps the stored analysis (costs/features/statistics), but
  // persistResult calls makeStoredAnalysis(), which expects unpacked 0/1 mask
  // arrays rather than the packed DMSEM1 bytes kept in IndexedDB.
  const analysis = { ...record.analysis, semantics: storedSemantics(record.analysis) } as unknown as AnalysisResult;
  return {
    state, analysis, overlay, terrainWarp, approaches, graphHash,
    files: {
      overlay: await encodeImageRgba(overlay.rgba, overlay.width, overlay.height),
      terrain: terrainBlob ?? new Blob(),
      terrainWarp: terrainWarp.width ? await encodeImageRgba(terrainWarp.rgba, terrainWarp.width, terrainWarp.height) : new Blob(),
    },
  };
}

/** Recompute approaches after teacher edits to controls/terrain. */
export async function refreshLocalApproaches(record: LocalMapRecord, progress?: ProgressCallback): Promise<LocalMapRecord> {
  const registration = record.state.registration;
  if (!registration) return record;
  progress?.(40, '重新计算合法到访位置…');
  const costs = floatCosts(record.analysis);
  const graph = buildGraphFromRaster(costs, registration, record.state.sourceHash ?? record.summary.sha256, record.state.edits, 4,
    { width: record.analysis.width, height: record.analysis.height, assetId: record.summary.assetId });
  const semantics = storedSemantics(record.analysis);
  await prepareAccess();
  const resolved = resolveApproaches({ graph, mesh: new Mesh(registration), semantics, edits: record.state.edits, closedEdits: graph.closedEdits, registration }, record.state.controls,
    { startComponent: canonicalStartComponent(record.state.controls, graph) });
  const approaches = approachDocument(record.summary.assetId, graph.metadata.graphHash, resolved as Record<string, BakedApproach>);
  const next: LocalMapRecord = { ...record, state: { ...record.state, graphHash: graph.metadata.graphHash }, approaches };
  await saveLocalMap(next);
  clearClientPlannerCache();
  progress?.(90, '到访位置已更新');
  return next;
}

/** Build (or reuse) the browser graph + planner for a client-processed map. */
const clientGraphs = new Map<string, Promise<ClientPlannerBundle>>();

export interface ClientPlannerBundle { graph: RasterGraph; planner: Planner; graphHash: string }

export async function getClientPlannerBundle(record: LocalMapRecord, progress?: ProgressCallback): Promise<ClientPlannerBundle> {
  const versionKey = `${record.summary.assetId}:${record.state.version}:${record.state.graphHash ?? ''}`;
  const cached = clientGraphs.get(versionKey);
  if (cached) return await cached;
  const pending = (async () => {
    progress?.(20, '准备本地通行图…');
    const registration = record.state.registration;
    if (!registration) throw new Error('请先完成地图配准');
    progress?.(40, '建立保留细墙的通行图…');
    const graph = buildGraphFromRaster(floatCosts(record.analysis), registration,
      record.state.sourceHash ?? record.summary.sha256, record.state.edits, 4,
      { width: record.analysis.width, height: record.analysis.height, assetId: record.summary.assetId, graphHash: record.state.graphHash ?? undefined });
    progress?.(80, '准备路线规划…');
    const planner = createPlannerFromGraph(graph, record.approaches, record.state);
    for (const key of [...clientGraphs.keys()]) if (key !== versionKey) clientGraphs.delete(key);
    const bundle = { graph, planner, graphHash: graph.metadata.graphHash };
    clientGraphs.set(versionKey, Promise.resolve(bundle));
    return bundle;
  })();
  clientGraphs.set(versionKey, pending);
  try { return await pending; } catch (error) { clientGraphs.delete(versionKey); throw error; }
}

export async function getClientPlanner(record: LocalMapRecord, progress?: ProgressCallback): Promise<Planner> {
  return (await getClientPlannerBundle(record, progress)).planner;
}

export function clearClientPlannerCache(): void { clientGraphs.clear(); }

export type { LocalMapKind, LocalMapRecord, LocalMapSummary, StoredAnalysis, LocalFileKind, AnalysisResult };
