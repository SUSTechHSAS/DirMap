import 'ol/ol.css';
import './style.css';
import { CampusMap, teamColors } from './map';
import { NavigationSensors, type HeadingSample } from './heading';
import { PixelEditor } from './editor';
import { OffRouteMonitor, remainingPath } from './navigation';
import { installOffline, saveOfflineCourse } from './offline';
import {
  assetUrl, declinationFor, downloadJSON, downloadRelease, invalidateLocalAssets, legendUrl, loadCatalog, loadHealth, loadMapState,
  loadReference, loadRelease, prepareLocalAssets, rememberLocalCatalog,
} from './data';
import {
  applyStateOverride, clearStateOverride, findLocalRelease, hasStateOverride, loadStateOverride,
  saveLocalRelease, saveStateOverride,
} from './static-store';
import {
  clearClientPlannerCache, decodeImageBlob, getClientPlanner, loadNativeImage, processMap, reregisterMap,
  refreshLocalApproaches, sha256Hex, type ProcessResult,
} from './processing';
import {
  deleteLocalMap, hasLocalMap, loadLocalMap, localMapSummary, makeStoredAnalysis, putLocalFile,
  rememberLocalMaps, saveLocalMap, type LocalFileKind, type LocalMapRecord, type LocalMapSummary,
} from './workspace-store';
import { clearPlannerCache, getPlanner } from './planner';
import {
  escapeHtml as e, lngLatToPixel, pixelToLngLat,
  type Anchor, type BakedCatalog, type Control, type MapState, type NavigateResponse, type Pixel, type Plan,
  type PlanRequest, type Route, type StaticRelease, type TerrainEdit,
} from './types';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const number = (id: string) => Number($<HTMLInputElement>(id).value);
const value = (id: string) => $<HTMLInputElement>(id).value;
const minutes = (seconds: number) => `${Math.floor(Math.round(seconds) / 60)} 分 ${Math.round(seconds) % 60} 秒`;
const distance = (metres: number) => metres >= 1000 ? `${(metres / 1000).toFixed(2)} km` : `${Math.round(metres)} m`;
let catalog: BakedCatalog;
/** The immutable state loaded from the static bundle; local edits are merged over it. */
let bakedState: MapState | null = null;
/** Client-processed / uploaded record for the active map, when one exists. */
let localRecord: LocalMapRecord | null = null;
let state: MapState | null = null;
let activeId = 'szu-1';
let plan: Plan | null = null;
let planRevision = 0;
let mandatory = new Set<string>();
let missingPresetLabels: string[] = [];
let startPixel: Pixel | null = null, finishPixel: Pixel | null = null;
let mapAction: 'start' | 'finish' | 'anchor' | null = null;
let pendingAnchor: Pixel | null = null;
let anchors: Anchor[] = [];
let follow = false, northOnly = false;
let lastPosition: GeolocationPosition | null = null;
/** When the last accepted geolocation sample was received (ms). */
let lastPositionReceivedAt = 0;
let selectedTeam: string | null = null;
let busy = false;
let courseId = new URLSearchParams(location.search).get('course');
let completed = new Set<string>();
let studentControls: Control[] = [];
let courseSuperseded = false;
let currentRelease: StaticRelease | null = null;
let navigationRoute: Route | null = null;
let navigationRevision = 0;
let navigating = false;
let resumeSensors = false;
const offRoute = new OffRouteMonitor();

const APP_TEMPLATE = `
<header class="app-header"><a href="/" class="brand" aria-label="DirMap 首页"><span class="brand-mark"><svg viewBox="0 0 30 32" aria-hidden="true"><path d="M15 2 28 26 15 20 2 26Z" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M15 9v11" stroke="currentColor" stroke-width="2"/></svg></span><span>DirMap<small>深大定向</small></span></a><div class="header-context"><span class="context-dot"></span>深圳大学 · 粤海校区<span class="header-divider"></span><span id="header-mode">课程工作台</span></div><div class="header-actions"><span id="build-tag" class="status-tag" title="静态构建：识别、配准与通行图在构建时预计算，无后端服务">静态版 · 无需后端</span><span id="release-badge" class="status-tag">备课草稿</span><button id="open-help" class="text-button">使用说明</button></div></header>
<div class="workspace"><aside class="sidebar" id="teacher-panel"><div class="sidebar-intro"><p class="eyebrow">ORIENTEERING CLASSROOM</p><h1>让每一次出发<br>都有方向<span>。</span></h1><p>校准地图，安排路线，一起探索校园。</p></div>
<section class="source-section"><div class="section-title"><h2>课程地图</h2><button id="upload-button" class="text-button">＋ 导入</button><input id="upload-input" type="file" accept="image/jpeg,image/png" hidden></div><div id="map-cards" class="map-cards"></div><div class="source-actions"><button id="process-current" class="primary-button">自动识别与配准</button><button id="process-all" class="soft-button">处理三图</button></div><p id="static-source-note" class="muted compact">识别、配准与通行图在浏览器内计算；构建时数据作为快速默认值。</p><div id="job-status" class="job-status hidden" role="status"><div><span id="job-message">准备中</span><span id="job-percent">0%</span></div><progress id="job-progress" max="100" value="0"></progress></div><div id="map-summary" class="summary-strip">选择地图开始准备课程</div></section>
<nav class="tabs" aria-label="工作区域"><button data-tab="course" class="active">课程设置</button><button data-tab="registration">地图配准</button><button data-tab="terrain">通行核验</button></nav>
<section id="tab-course" class="tab-content"><div class="section-title"><h2>路线规则</h2><span class="step-number">01</span></div><label class="field-label" for="course-mode">访问方式</label><select id="course-mode"><option value="all_visit">每队访问全部目标</option><option value="split_coverage">全班分担全部目标</option><option value="fixed_order">按列表顺序访问</option></select><div class="form-grid"><label>总人数<input id="people" type="number" value="20" min="1" max="500"></label><label>队伍数<input id="teams" type="number" value="4" min="1" max="30"></label><label>步行 / 跑速 <span>m/s</span><input id="speed" type="number" value="1.4" step="0.1" min="0.2" max="10"></label><label>错峰间隔 <span>秒</span><input id="stagger" type="number" value="0" min="0" max="3600"></label></div><p id="group-preview" class="muted compact">每队 5 人 · 同时出发</p><label class="field-label" for="objective">优化目标</label><select id="objective"><option value="min_max_finish_then_sum">优先让最后一队更早完成</option><option value="min_total_time">全班总耗时最短</option></select>
<div class="section-title section-gap"><h2>起终点与目标</h2><span class="step-number">02</span></div><div class="endpoint-row"><button id="pick-start" class="endpoint-button"><i class="endpoint-dot start"></i><span>在地图选起点</span></button><button id="pick-finish" class="endpoint-button"><i class="endpoint-dot finish"></i><span>在地图选终点</span></button></div><label class="check-label"><input id="return-start" type="checkbox" checked>回到起点结束</label><div class="control-heading"><span id="control-count">尚未识别目标</span><button id="edit-controls" class="text-button">原图校核 ↗</button></div><p class="muted compact">勾选本次课程目标；固定顺序模式可上下调整。</p><div id="control-list" class="control-list"><div class="empty-small">识别后，控制圈会显示在这里。</div></div><button id="plan-button" class="primary-button wide plan-button">生成队伍路线 <span>→</span></button><div id="plan-errors" role="status"></div><div id="route-results"></div>
<div id="publish-section" class="publish-section hidden"><button id="review-button" class="soft-button wide">核验课程并发布</button><div id="share-result"></div></div></section>
<section id="tab-registration" class="tab-content hidden"><div class="section-title"><h2>地物匹配与局部校准</h2><span class="step-number">01</span></div><p class="muted">自动配准在浏览器内运行；同名点可修正纸图比例与局部偏差。</p><div id="registration-quality" class="quality-card">完成自动处理后显示匹配结果。</div><button id="add-anchor" class="soft-button wide">＋ 在原图选择同名点</button><label class="check-label section-gap"><input id="validation-anchor" type="checkbox">下一点用于独立核查，不参与拟合</label><div id="anchor-list" class="anchor-list"></div><button id="fit-anchors" class="primary-button wide" disabled>用同名点重新校准</button><p class="muted">至少 3 个分散的非共线拟合点。建议加入 8 个以上拟合点，并另设核查点。</p><a id="export-map" class="text-link" href="#">下载当前配准和目标数据 ↗</a></section>
<section id="tab-terrain" class="tab-content hidden"><div class="section-title"><h2>核对可通行区域</h2><span class="step-number">01</span></div><p class="muted">红色区域为识别出的障碍。重点核对围墙、门洞、水体、禁区和上下层通道。</p><label class="check-label"><input id="show-terrain" type="checkbox">在地图上显示障碍候选</label><button id="edit-terrain" class="primary-button wide section-gap">打开原图修正通行区域</button><div id="terrain-summary" class="quality-card"></div><button id="reset-static-data" class="text-button wide" disabled>恢复构建时数据</button><p class="muted">本机修正保存后需重新生成路线；修正只保存在本浏览器。二维相交不能证明桥上、桥下可以互通。</p><div class="section-title section-gap"><h2>随图图例</h2></div><div id="legend-links" class="legend-links"></div></section></aside>
<aside id="student-panel" class="sidebar student-panel hidden"><p class="eyebrow">TODAY'S ADVENTURE</p><h1 id="course-title">定向越野课</h1><p class="muted">选择队伍，打开定位，出发探索。</p><label class="field-label" for="student-team">我的队伍</label><select id="student-team"></select><div id="student-overview" class="quality-card"></div><div class="section-title"><h2>下一目标</h2><span id="student-progress" class="muted"></span></div><div id="next-control" class="next-control"></div><button id="check-in" class="primary-button wide">已完成这个目标</button><button id="undo-check-in" class="text-button wide">撤销上一个打点</button><div id="student-stops" class="student-stops"></div><p class="muted">打点由本人确认；当前位置精度不足时，请结合原图和现场地物判断。</p></aside>
<main class="map-panel"><div id="map" role="application" aria-label="校园标准地图与定向路线"></div><div class="map-topbar"><div class="map-title"><span class="map-kicker">CAMPUS EXPLORER</span><strong id="map-title">深圳大学 · 粤海校区</strong><span id="map-subtitle">标准地图 / WGS84</span></div><div class="layer-tools"><label class="check-label"><input id="standard-map" type="checkbox" checked>标准底图</label><label class="check-label teacher-only"><input id="stitch-maps" type="checkbox">叠合三图</label><label class="opacity-control">原图 <input id="overlay-opacity" type="range" min="0" max="100" value="70" aria-label="原图透明度"></label></div></div><div id="map-action-hint" class="map-action-hint hidden" role="status"></div><button id="north-button" class="compass-button" title="北向上" aria-label="切换北向上"><span>N</span><svg id="compass-arrow" viewBox="0 0 24 28" aria-hidden="true"><path d="M12 1 21 25 12 20 3 25Z" fill="#295c46"/><path d="M12 1v19l9 5Z" fill="#abbdb1"/></svg></button><div class="map-bottom"><div class="navigation-card"><button id="follow-button" class="follow-button"><span class="locate-icon">◎</span><span id="follow-label">开始定位与朝向跟随</span></button><div id="sensor-status" class="sensor-status" role="status">手机打开 HTTPS 链接即可启用</div></div><div class="map-key"><span><i class="key-control"></i>目标点</span><span><i class="key-route"></i>队伍路线</span><span id="location-status">位置尚未开启</span></div></div></main></div><div id="toast" class="toast hidden" role="status"></div>`;
document.querySelector('#app')!.innerHTML = APP_TEMPLATE;

const campusMap = new CampusMap($('map'));
const sensors = new NavigationSensors();
$('objective').insertAdjacentHTML('beforeend', '<option value="min_distance">全班总路程最短</option>');
$('course-mode').insertAdjacentHTML('beforeend', '<option value="mixed_coverage">全员必到 + 其余点全班分担</option>');
$('course-mode').insertAdjacentHTML('afterend', '<div id="mixed-rules" class="quality-card hidden"><strong>混合访问规则</strong><p id="mixed-rule-summary"></p><p class="muted compact">在下方勾选“每队必到”；其余选中点由队伍分担。整队同行，每名队员都会经过本队必到点。</p><label class="check-label"><input id="mandatory-order" type="checkbox">指定每队必到点的先后顺序</label><div id="mandatory-order-list"></div><p id="mandatory-rule-error" class="danger"></p></div>');
$<HTMLInputElement>('mandatory-order').onchange = () => { resetPlan(); persistRules(); renderState(); };
$('sensor-status').insertAdjacentHTML('afterend', '<button id="calibrate-compass" class="text-button compass-calibration">校准罗盘</button><button id="open-diagnostics" class="text-button">定位诊断</button>');
$('objective').insertAdjacentHTML('afterend', '<label class="field-label" for="solver-budget">规划计算预算</label><select id="solver-budget"><option value="5">快速规划 · 5 秒</option><option value="30">继续优化 · 30 秒</option></select>');
$<HTMLSelectElement>('solver-budget').onchange = resetPlan;
$('edit-terrain').insertAdjacentHTML('beforebegin', '<p class="muted">原图紫色连线只表示课程关系，不是实际道路。路线会按其下方的地形绕开水体、建筑和围墙。</p><button id="refresh-terrain" class="soft-button wide">更新通行识别（保留教师校核）</button>');
if (import.meta.env.PROD) {
  void installOffline();
  $('course-title').insertAdjacentHTML('afterend', '<p id="offline-status" class="muted" role="status"></p><button id="save-offline" class="text-button">保存本课供离线查看</button>');
  $('save-offline').onclick = () => { void cacheStudentCourse(); };
}
// Expose deterministic hooks in development and for static end-to-end checks.
// Production users only get these when a `?e2e` query parameter is present.
if (import.meta.env.DEV || new URLSearchParams(location.search).has('e2e')) {
  (window as Window & { dirmap?: unknown }).dirmap = {
    campusMap, sensors, getPlanner, getClientPlanner,
    getState: () => state, getRelease: () => currentRelease, getPlan: () => plan,
    getLocalRecord: () => localRecord, getCatalog: () => catalog,
    processCurrent: () => runProcessing('process'),
    uploadBlob: (blob: Blob, name: string) => uploadSheet(new File([blob], name)),
  };
}

const IN_APP_BROWSER = /MicroMessenger|WeChat|QQ\/|Weibo|AlipayClient/i.test(navigator.userAgent);
function maybeInAppBrowserHint() {
  if (!IN_APP_BROWSER) return;
  const message = '当前在微信/QQ 等内嵌浏览器中，定位可能被禁用：请点右上角 ···，选择“在浏览器打开”';
  $('sensor-status').textContent = message;
  $('location-status').textContent = '内嵌浏览器可能禁用定位，请用系统浏览器打开';
  notify(message);
}
function notify(message: string, error = false) {
  $('toast').textContent = message;
  $('toast').classList.remove('hidden');
  $('toast').classList.toggle('error', error);
  setTimeout(() => $('toast').classList.add('hidden'), 7000);
}
function progress(percent: number, message: string) {
  $('job-status').classList.remove('hidden');
  $('job-message').textContent = message;
  $('job-percent').textContent = `${Math.round(percent)}%`;
  $<HTMLProgressElement>('job-progress').value = percent;
}
function hideProgress() { $('job-status').classList.add('hidden'); }
async function task(action: () => Promise<void>) {
  if (busy) return;
  busy = true;
  const ids = ['process-current', 'process-all', 'upload-button', 'plan-button', 'review-button',
    'edit-terrain', 'edit-controls', 'refresh-terrain', 'add-anchor', 'fit-anchors', 'reset-static-data'];
  for (const id of ids) { const element = $(id) as HTMLButtonElement | null; if (element) element.disabled = true; }
  try { await action(); }
  catch (error) { notify(error instanceof Error ? error.message : String(error), true); }
  finally {
    busy = false;
    for (const id of ids) { const element = $(id) as HTMLButtonElement | null; if (element) element.disabled = false; }
    if (state) renderState();
    hideProgress();
  }
}
function randomId32(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
}
function updateStaticNotice() {
  const note = document.getElementById('static-source-note');
  if (note) note.textContent = '识别、配准与通行图在浏览器内计算；构建时数据作为快速默认值，也可导入新图后自动处理。';
}

/** Planner for the active map: client-built graph or build-time bundle. */
async function activePlanner(onProgress?: (percent: number, message: string) => void) {
  if (localRecord) return await getClientPlanner(localRecord, onProgress);
  if (!state) throw new Error('请先载入地图');
  return await getPlanner(state, onProgress);
}

function resetPlan() {
  planRevision++;
  plan = null; selectedTeam = null;
  campusMap.setRoutes([]); campusMap.setApproaches([]);
  $('route-results').innerHTML = ''; $('plan-errors').innerHTML = '';
  $('publish-section').classList.add('hidden');
}
function rulesKey() { return 'dirmap-course-rules-v1:' + (catalog.maps.find(map => map.id === activeId)?.sha256 ?? activeId); }
function persistRules() {
  try {
    localStorage.setItem(rulesKey(), JSON.stringify({
      mode: value('course-mode'),
      mandatory: [...mandatory],
      order: $<HTMLInputElement>('mandatory-order').checked,
      selected: state?.controls.filter(control => control.required).map(control => control.id),
    }));
  } catch { /* planning still works when private mode prevents local preferences */ }
}
function initializeRules() {
  mandatory = new Set(); missingPresetLabels = [];
  $<HTMLInputElement>('mandatory-order').checked = false;
  const allowed = ['all_visit', 'split_coverage', 'fixed_order', 'mixed_coverage'];
  try {
    const saved = JSON.parse(localStorage.getItem(rulesKey()) ?? 'null');
    if (saved && allowed.includes(saved.mode) && Array.isArray(saved.mandatory) && saved.mandatory.every((id: unknown) => typeof id === 'string')) {
      $<HTMLSelectElement>('course-mode').value = saved.mode;
      mandatory = new Set(saved.mandatory.filter((id: string) => state?.controls.some(control => control.id === id && (control.role ?? 'control') === 'control')));
      $<HTMLInputElement>('mandatory-order').checked = saved.order === true;
      if (Array.isArray(saved.selected) && state) for (const control of state.controls) control.required = (control.role ?? 'control') === 'control' && (saved.selected.includes(control.id) || mandatory.has(control.id));
      return;
    }
  } catch { /* use this sheet's explicit default rules */ }
  const defaults = catalog.maps.find(map => map.id === activeId)?.courseDefaults;
  $<HTMLSelectElement>('course-mode').value = defaults?.mode ?? 'all_visit';
  if (defaults && state) {
    for (const label of defaults.mandatoryLabels) {
      const matches = state.controls.filter(control => control.label === label && (control.role ?? 'control') === 'control' && control.state !== 'rejected');
      if (matches.length === 1) mandatory.add(matches[0].id); else missingPresetLabels.push(label);
    }
    for (const control of state.controls) control.required = (control.role ?? 'control') === 'control' && control.state !== 'rejected';
    $<HTMLInputElement>('mandatory-order').checked = defaults.mandatoryOrder;
  }
}
function rulePayload() {
  const mixed = value('course-mode') === 'mixed_coverage';
  const ids = [...mandatory].filter(id => state?.controls.some(control => control.id === id && control.required && control.state !== 'rejected'));
  return { mandatoryControlIds: mixed ? ids : [], mandatoryOrder: mixed && $<HTMLInputElement>('mandatory-order').checked };
}
function renderMixedRules() {
  const mixed = value('course-mode') === 'mixed_coverage';
  $('mixed-rules').classList.toggle('hidden', !mixed);
  if (!mixed) return;
  const rules = rulePayload();
  const labels = rules.mandatoryControlIds.map(id => state?.controls.find(control => control.id === id)?.label ?? id);
  const total = state?.controls.filter(control => control.required && (control.role ?? 'control') === 'control' && control.state !== 'rejected').length ?? 0;
  $('mixed-rule-summary').textContent = `每队必到：${labels.join('、') || '尚未选择'}；其余 ${total - labels.length} 点全班分担。${rules.mandatoryOrder ? '必到点按下列顺序访问，可在其间插入分担点。' : '必到点可按任意顺序访问。'}`;
  $('mandatory-order-list').innerHTML = rules.mandatoryOrder
    ? rules.mandatoryControlIds.map((id, index) => `<div class="mandatory-order-row"><span>${index + 1}. ${e(labels[index])}</span><button class="text-button" data-mandatory-up="${e(id)}" aria-label="上移必到点 ${e(labels[index])}">↑</button><button class="text-button" data-mandatory-down="${e(id)}" aria-label="下移必到点 ${e(labels[index])}">↓</button></div>`).join('')
    : '';
  $('mandatory-rule-error').textContent = missingPresetLabels.length ? `尚未识别或编号重复：${missingPresetLabels.join('、')}。请校核后明确勾选每队必到点。` : '';
}
function renderCatalog() {
  $('map-cards').innerHTML = catalog.maps.map((map, index) => `<button class="map-card ${map.id === activeId ? 'active' : ''}" data-map="${e(map.id)}"><span class="map-index">0${index + 1}</span><span class="map-card-name">${e(map.name)}<small>${map.processed ? `${map.controls ?? '—'} 个候选目标` : '原图待处理'}${map.local ? ' · 本机处理' : ''}</small></span><span class="map-card-state">${map.local ? '◆' : map.processed ? '✓' : '○'}</span></button>`).join('');
  $('legend-links').innerHTML = catalog.legends.map(legend => `<a href="${e(legendUrl(legend.url))}" target="_blank" rel="noreferrer">${e(legend.name.includes('aigc') ? '随图定向运动图例（含 AIGC 标记）' : legend.name)} <span>↗</span></a>`).join('');
}
async function loadState(id: string, fit = true) {
  activeId = id;
  // Uploaded maps have no build-time state; skip the (404) static fetch.
  bakedState = id.startsWith('upload-') ? null : await loadMapState(id);
  localRecord = await loadLocalMap(id);
  if (localRecord) {
    await prepareLocalAssets(id);
    state = { ...localRecord.state };
  } else {
    state = bakedState ? applyStateOverride(bakedState, loadStateOverride(id)) : null;
  }
  await prepareLocalAssets(id);
  anchors = []; renderAnchors();
  resetPlan(); startPixel = finishPixel = null; mapAction = null; pendingAnchor = null;
  $('map-action-hint').classList.add('hidden');
  campusMap.setEndpoint('start', null); campusMap.setEndpoint('finish', null);
  $('pick-start').querySelector('span')!.textContent = '在地图选起点';
  $('pick-finish').querySelector('span')!.textContent = '在地图选终点';
  $<HTMLInputElement>('return-start').checked = true;
  initializeRules();
  const starts = state?.controls.filter(control => ['start', 'start_finish'].includes(control.role ?? '') && control.calibrated && control.state !== 'rejected') ?? [];
  const finishes = state?.controls.filter(control => ['finish', 'start_finish'].includes(control.role ?? '') && control.calibrated && control.state !== 'rejected') ?? [];
  if (state && starts.length === 1) {
    startPixel = starts[0].approachPixel ?? starts[0].pixel;
    campusMap.setEndpoint('start', pixelToLngLat(startPixel, state));
    $('pick-start').querySelector('span')!.textContent = '已识别原图起点';
  }
  if (state && finishes.length === 1) {
    finishPixel = finishes[0].approachPixel ?? finishes[0].pixel;
    campusMap.setEndpoint('finish', pixelToLngLat(finishPixel, state));
    $('pick-finish').querySelector('span')!.textContent = '已识别原图终点';
    $<HTMLInputElement>('return-start').checked = JSON.stringify(startPixel) === JSON.stringify(finishPixel);
  }
  renderCatalog(); renderState(); await updateOverlays();
  if (fit && state?.extent) campusMap.fit(state.extent);
}
async function updateOverlays() {
  let states = state ? [state] : [];
  if ($<HTMLInputElement>('stitch-maps').checked) {
    const others = await Promise.all(catalog.maps.filter(map => map.processed && map.id !== activeId)
      .map(async map => {
        const local = await loadLocalMap(map.id);
        if (local) { await prepareLocalAssets(map.id); return local.state; }
        if (map.id.startsWith('upload-')) return null;
        const baked = await loadMapState(map.id);
        return baked ? applyStateOverride(baked, loadStateOverride(map.id)) : null;
      }));
    states = [...others.filter((value): value is MapState => value !== null), ...states];
  }
  campusMap.setOverlays(states, number('overlay-opacity') / 100, activeId);
  campusMap.setTerrain(state, $<HTMLInputElement>('show-terrain').checked);
}
function renderState() {
  renderMixedRules();
  campusMap.setControls(state?.controls ?? []);
  $('release-badge').textContent = state?.reviewed ? '已核验' : localRecord ? '本机处理' : (hasStateOverride(activeId) ? '本机草稿' : '备课草稿');
  const reset = document.getElementById('reset-static-data') as HTMLButtonElement | null;
  if (reset) {
    reset.disabled = !localRecord && !hasStateOverride(activeId);
    reset.textContent = localRecord ? '删除本机处理结果并恢复构建时数据' : '恢复构建时数据';
    reset.title = reset.disabled ? '当前显示构建时数据' : '清除本机处理/修改结果';
  }
  if (!state) {
    $('map-summary').textContent = '原图尚未处理';
    $('control-list').innerHTML = '<div class="empty-small">点击“自动识别与配准”在浏览器内处理这张图。</div>';
    $('control-count').textContent = '尚未识别目标';
    $('registration-quality').textContent = '没有可用的配准数据。';
    return;
  }
  $('map-title').textContent = state.name;
  $('map-subtitle').textContent = state.registrationStale ? '基准图已更新，请重新配准' : state.registration ? '已叠合至标准地图 · 待现场核对' : '地理参照待补充';
  $('map-summary').innerHTML = `<span><b>${state.controls.length}</b> 候选目标</span><span><b>${state.quality?.fitCount ?? 0}</b> 配准点</span><span class="review-dot">${localRecord ? '本机处理' : hasStateOverride(activeId) ? '本机修改' : '待复核'}</span>`;
  $('control-count').textContent = `${state.controls.filter(control => control.required).length} / ${state.controls.length} 个目标已选`;
  $('control-list').innerHTML = state.controls.map((control, index) => `<div class="control-row" data-id="${e(control.id)}"><label class="control-toggle"><input type="checkbox" data-required="${e(control.id)}" ${control.role && control.role !== 'control' ? 'disabled' : ''} ${control.required ? 'checked' : ''}><span class="control-number ${control.calibrated ? '' : 'uncalibrated'}">${e(control.label)}</span></label>${value('course-mode') === 'mixed_coverage' && (control.role ?? 'control') === 'control' ? `<label class="every-team-toggle"><input type="checkbox" data-mandatory="${e(control.id)}" aria-label="${e(control.label)} 每队必到" ${mandatory.has(control.id) ? 'checked' : ''}>每队必到</label>` : ''}<span class="control-state">${!control.calibrated ? '待补配准' : control.state === 'accepted' ? '已确认' : '待核对'}</span><button class="text-button" data-control-edit="${e(control.id)}" aria-label="编辑目标 ${e(control.label)}">↗</button>${value('course-mode') === 'fixed_order' ? `<button class="text-button" data-up="${index}" aria-label="上移 ${e(control.label)}">↑</button><button class="text-button" data-down="${index}" aria-label="下移 ${e(control.label)}">↓</button>` : ''}</div>`).join('');
  const quality = state.quality;
  $('registration-quality').innerHTML = quality
    ? `<span class="quality-label">${quality.baseAssetId ? '共同地物对齐' : '标准地图地物匹配'}</span><div class="quality-values"><div><strong>${quality.fitCount}</strong><span>拟合同名点</span></div><div><strong>${quality.p95ErrorM?.toFixed(1) ?? '—'}<small> m</small></strong><span>留出误差 P95</span></div></div><p>${e(quality.validationScope)}</p>${quality.inheritedGeographicP95M ? `<p>基准图地理匹配 P95：${quality.inheritedGeographicP95M.toFixed(1)} m</p>` : ''}`
    : '自动匹配证据不足；可在“同名点”中手动校准，或直接在原图核验后规划。';
  $('terrain-summary').innerHTML = `<strong>${Math.round(state.statistics.forbiddenPixelFraction * 100)}%</strong> 像素被识别为障碍候选<p>${state.edits.length} 处本机通行修正</p><p>路线会检查每条边经过的原图像素。未知层级及模糊的门洞仍需核验。</p>${localRecord ? '<p id="local-edit-note">正在使用本机识别/处理结果；修改只保存在本浏览器。</p>' : hasStateOverride(activeId) ? '<p id="local-edit-note">正在使用本机修改；本机修改可规划路线，但不会随静态站点分发给其他设备。</p>' : ''}`;
  $('export-map').setAttribute('download', `${activeId}.json`);
}
async function saveEdits(controls: Control[], edits: TerrainEdit[]) {
  if (!state) return;
  if (localRecord) {
    // Client-processed maps persist their full state locally; the browser graph
    // and approach suggestions must be rebuilt for the edited costs/positions.
    progress(15, '保存本机修改并重新计算通行图…');
    const next: MapState = { ...state, controls, edits, version: state.version + 1, reviewed: false };
    localRecord = { ...localRecord, state: next };
    await saveLocalMap(localRecord);
    clearClientPlannerCache(); clearPlannerCache();
    try { await refreshLocalApproaches(localRecord, progress); }
    catch (error) { notify(`到访位置稍后重新计算：${error instanceof Error ? error.message : String(error)}`, true); }
    localRecord = await loadLocalMap(activeId) ?? localRecord;
    state = localRecord.state;
    anchors = [];
    resetPlan(); renderState(); await updateOverlays();
    notify('修改已保存到本机，请重新生成路线');
    return;
  }
  if (!bakedState) return;
  // Baked maps keep teacher edits in a small override; build data stays untouched.
  const saved = saveStateOverride(activeId, bakedState, controls, edits);
  state = applyStateOverride(bakedState, saved);
  clearPlannerCache();
  resetPlan(); renderState(); await updateOverlays();
  notify('修改已保存到本机，请重新生成路线');
}
function renderAnchors() {
  $('anchor-list').innerHTML = anchors.map((anchor, index) => `<div><span>${index + 1}. ${anchor.purpose === 'fit' ? '拟合点' : '核查点'}</span><small>${anchor.lnglat[0].toFixed(6)}, ${anchor.lnglat[1].toFixed(6)}</small><button data-remove-anchor="${index}" class="text-button" aria-label="删除同名点 ${index + 1}">×</button></div>`).join('');
  const fitButton = document.getElementById('fit-anchors') as HTMLButtonElement | null;
  if (fitButton) fitButton.disabled = anchors.filter(anchor => anchor.purpose === 'fit').length < 3;
}
function openEditor(mode: 'inspect' | 'close' | 'open' | 'anchor' = 'inspect', id: string | null = null) {
  if (!state) { notify('请先载入地图'); return; }
  new PixelEditor(state, saveEdits, (pixel) => {
    pendingAnchor = pixel; mapAction = 'anchor';
    $('map-action-hint').textContent = '在标准地图上点击同一个建筑角或路口';
    $('map-action-hint').classList.remove('hidden');
    campusMap.setOpacity(.1);
  }, plan?.approaches ?? []).open(mode, id);
}

/** Persist a processing result and refresh local records. */
async function persistResult(assetId: string, result: ProcessResult, kind: LocalMapSummary['kind']) {
  const summary: LocalMapSummary = {
    assetId, name: result.state.name, filename: result.state.filename ?? '',
    sha256: result.state.sourceHash ?? '', kind, createdAt: Date.now(),
    width: result.state.width, height: result.state.height,
  };
  const record: LocalMapRecord = {
    summary, state: result.state,
    analysis: makeStoredAnalysis(result.analysis, kind),
    approaches: result.approaches,
  };
  const files: Partial<Record<LocalFileKind, Blob>> = {
    overlay: result.files.overlay, terrain: result.files.terrain, 'terrain-warp': result.files.terrainWarp,
  };
  await saveLocalMap(record, files);
  invalidateLocalAssets(assetId);
  localRecord = await loadLocalMap(assetId);
  await prepareLocalAssets(assetId);
  clearPlannerCache();
}
/** Full in-browser recognition + registration + terrain pass for the active map. */
async function runProcessing(mode: 'process' | 'terrain') {
  if (!state && !bakedState) throw new Error('请先导入或选择地图');
  const map = catalog.maps.find(candidate => candidate.id === activeId);
  progress(2, mode === 'terrain' ? '更新通行识别…' : '自动识别与配准…');
  const { image, sha256 } = await loadNativeImage(activeId);
  const result = await processMap({
    assetId: activeId,
    name: state?.name ?? map?.name ?? activeId,
    filename: state?.filename ?? map?.filename ?? '',
    sourceHash: state?.sourceHash ?? sha256 ?? map?.sha256 ?? '',
    image, existing: state, mode, progress,
  });
  await persistResult(activeId, result, localRecord?.summary.kind ?? 'processed');
  catalog = { ...catalog, maps: catalog.maps.map(entry => entry.id === activeId
    ? { ...entry, processed: true, controls: result.state.controls.length, local: true, version: result.state.version } : entry) };
  await loadState(activeId);
  hideProgress();
  notify(mode === 'terrain' ? '通行识别已更新；旧路线已失效，请重新生成' : '识别与配准完成，请核对候选点与通行区域');
}
/** Process all bundled sheets in the browser (keeps each local result). */
async function processAllMaps() {
  const maps = catalog.maps.filter(map => !map.local);
  for (const [index, map] of maps.entries()) {
    progress(index / maps.length * 100, `处理 ${map.name}（${index + 1}/${maps.length}）`);
    const baked = await loadMapState(map.id);
    const existingLocal = await loadLocalMap(map.id);
    const existing = existingLocal?.state ?? baked;
    const { image, sha256 } = await loadNativeImage(map.id);
    const result = await processMap({
      assetId: map.id, name: existing?.name ?? map.name, filename: existing?.filename ?? map.filename,
      sourceHash: existing?.sourceHash ?? sha256 ?? map.sha256 ?? '', image, existing, mode: 'process',
      progress: (percent: number, message: string) => progress((index + percent / 100) / maps.length * 100, `${map.name}：${message}`),
    });
    await persistResult(map.id, result, 'processed');
  }
  $<HTMLInputElement>('stitch-maps').checked = true;
  await loadState(activeId);
  hideProgress();
  notify(`已在本机处理 ${maps.length} 张地图，可叠合查看`);
}
/** Import a new scan, process it in the browser and add it to the local catalog. */
async function uploadSheet(file: File) {
  const bytes = await file.arrayBuffer();
  const sha256 = await sha256Hex(bytes);
  const assetId = `upload-${sha256.slice(0, 12)}`;
  const blob = new Blob([bytes], { type: file.type || 'image/png' });
  await putLocalFile(assetId, 'original', blob);
  progress(2, `解析 ${file.name}…`);
  const image = await decodeImageBlob(blob);
  const name = file.name.replace(/\.[^.]+$/, '').slice(0, 80);
  const result = await processMap({ assetId, name, filename: file.name, sourceHash: sha256, image, mode: 'process', progress });
  await persistResult(assetId, result, 'upload');
  catalog = { ...catalog, maps: [...catalog.maps.filter(map => map.id !== assetId), {
    id: assetId, name, filename: file.name, sha256, processed: true, controls: result.state.controls.length,
    version: result.state.version, courseDefaults: null, hasGraph: false, hasApproaches: false, local: true, sourceKind: 'upload',
  }] };
  await loadState(assetId);
  hideProgress();
  notify(result.state.registration ? '导入并处理完成，请核对目标与通行区域' : '已导入；自动配准证据不足，请用同名点校准');
}

function groupPreview() {
  const people = number('people'), teams = number('teams');
  $('group-preview').textContent = Number.isInteger(people) && Number.isInteger(teams) && teams > 0 && teams <= people
    ? `${teams} 队，每队 ${Math.floor(people / teams)}${people % teams ? `–${Math.ceil(people / teams)}` : ''} 人 · ${number('stagger') ? `间隔 ${number('stagger')} 秒出发` : '同时出发'}`
    : '队数不能超过人数，均须为正整数';
}
function showPlan(result: Plan) {
  plan = result;
  campusMap.setApproaches(result.approaches ?? []);
  if (result.status === 'NEEDS_REVIEW') {
    $('plan-errors').innerHTML = `<div class="notice-card"><strong>这些位置需要核对后再规划</strong>${result.unresolved!.map(item => `<p>${e(state?.controls.find(control => control.id === item.id)?.label ?? ({ start: '起点', finish: '终点' }[item.id] ?? item.id))}：${e(item.message)}${state?.controls.some(control => control.id === item.id) ? ` <button data-control-edit="${e(item.id)}" class="text-button">原图修正 ↗</button>` : ''}</p>`).join('')}</div>`;
    return;
  }
  campusMap.setRoutes(result.routes);
  $('route-results').innerHTML = `<div class="result-heading"><h2>队伍路线</h2><span class="status-tag ${result.status === 'OPTIMAL' ? 'success' : ''}">${result.status === 'OPTIMAL' ? '模型内已证明最优' : '当前最佳可行'}</span></div><p class="muted compact">基于构建时通行图；${result.draft ? '核验后可发布给学生。' : '可以发布给学生。'}${result.status === 'OPTIMAL' ? '' : '静态版不运行 CP-SAT，大规模采用限时可行解。'}</p>${result.routes.map((route, index) => `<button class="route-card" data-team="${e(route.teamId)}" style="--team-color:${teamColors[index % teamColors.length]}"><span class="route-name"><i></i>${e(route.teamId)}<small>${route.people} 人</small></span><span class="route-metrics"><strong>${distance(route.distanceM)}</strong><span>${minutes(route.durationSec)}</span></span><span class="route-stops">${route.controlIds.map(id => e(state?.controls.find(control => control.id === id)?.label ?? id)).join(' → ')}</span><span class="route-detail">${route.departureSec ? `${Math.round(route.departureSec)} 秒后出发 · ` : ''}打点 ${Math.round(route.serviceSec)} 秒</span></button>`).join('')}<button id="export-route" class="text-button wide">下载路线 GeoJSON</button>`;
  $('publish-section').classList.remove('hidden');
  if (result.coverage?.mode === 'mixed_coverage') {
    const ids = result.coverage.mandatoryControlIds;
    $('route-results').insertAdjacentHTML('afterbegin', `<div id="mixed-coverage-result" class="notice-card"><strong>已验证混合覆盖</strong><p>每队均到访 ${ids.map(id => e(state?.controls.find(control => control.id === id)?.label ?? id)).join('、')}；其余 ${result.coverage.sharedControlIds.length} 个点均已分配。${result.coverage.mandatoryOrder ? '必到点按指定顺序访问。' : '必到点顺序自由。'}</p></div>`);
  }
  if (result.proof?.scope) $('route-results').insertAdjacentHTML('afterbegin', `<p class="muted compact">${e(result.proof.scope)}</p>`);
  const suggestions = result.approaches?.filter(approach => approach.automatic) ?? [];
  if (suggestions.length) {
    $('route-results').insertAdjacentHTML('afterbegin', `<div class="notice-card"><strong>${suggestions.length} 处到访位置建议</strong><p>路线停在可通行的地物外沿；紫色圈心保持原位。请核对橙色到访标记。</p>${suggestions.map(approach => `<p>${e(state?.controls.find(control => control.id === approach.controlId)?.label)}：距圈心 ${approach.offsetM.toFixed(1)} m <button class="text-button" data-approach-review="${e(approach.controlId)}">在原图核对 ↗</button></p>`).join('')}</div>`);
    document.querySelectorAll<HTMLButtonElement>('[data-approach-review]').forEach(button => { button.onclick = () => openEditor('inspect', button.dataset.approachReview!); });
  }
  $('export-route').onclick = () => download('dirmap-routes.geojson', { type: 'FeatureCollection', features: result.routes.map(route => ({ type: 'Feature', properties: { team: route.teamId, distanceM: route.distanceM, durationSec: route.durationSec, graphHash: result.graphHash }, geometry: { type: 'LineString', coordinates: route.geometry } })) });
}
function download(name: string, data: unknown) { downloadJSON(name, data); }

document.querySelectorAll<HTMLButtonElement>('[data-tab]').forEach(button => button.onclick = () => {
  document.querySelectorAll('[data-tab]').forEach(other => other.classList.toggle('active', other === button));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.toggle('hidden', tab.id !== `tab-${button.dataset.tab}`));
});
$('map-cards').onclick = event => { const button = (event.target as HTMLElement).closest<HTMLElement>('[data-map]'); if (button && !busy) task(() => loadState(button.dataset.map!)); };
$('process-current').onclick = () => task(() => runProcessing('process'));
$('process-all').onclick = () => task(() => processAllMaps());
$('refresh-terrain').onclick = () => task(() => runProcessing('terrain'));
$('upload-button').onclick = () => $<HTMLInputElement>('upload-input').click();
$<HTMLInputElement>('upload-input').onchange = event => {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  void task(() => uploadSheet(file));
};
for (const id of ['people', 'teams', 'stagger', 'speed']) $<HTMLInputElement>(id).oninput = () => { groupPreview(); resetPlan(); };
$<HTMLSelectElement>('objective').onchange = resetPlan;
$<HTMLSelectElement>('course-mode').onchange = () => { missingPresetLabels = []; resetPlan(); persistRules(); renderState(); };
for (const id of ['pick-start', 'pick-finish']) $(id).onclick = () => {
  if (!state?.registration) { notify('请先完成配准'); return; }
  mapAction = id === 'pick-start' ? 'start' : 'finish';
  $('map-action-hint').textContent = `点击地图中可通行的位置，设置${mapAction === 'start' ? '起点' : '终点'}`;
  $('map-action-hint').classList.remove('hidden');
};
$<HTMLInputElement>('return-start').onchange = () => {
  if ($<HTMLInputElement>('return-start').checked) { finishPixel = startPixel; if (startPixel && state) campusMap.setEndpoint('finish', pixelToLngLat(startPixel, state)); }
  resetPlan();
};
campusMap.onClick = (lnglat) => {
  if (mapAction === 'anchor' && pendingAnchor) {
    anchors.push({ pixel: pendingAnchor, lnglat, purpose: $<HTMLInputElement>('validation-anchor').checked ? 'validation' : 'fit' });
    pendingAnchor = null; mapAction = null; renderAnchors();
    $('map-action-hint').classList.add('hidden');
    campusMap.setOpacity(number('overlay-opacity') / 100);
    return;
  }
  if (!mapAction || !state) return;
  const pixel = lngLatToPixel(lnglat, state);
  if (!pixel) { notify('此位置超出已校准范围，请检查数据'); return; }
  if (mapAction === 'start') {
    startPixel = pixel; $('pick-start').querySelector('span')!.textContent = '起点已设置';
    campusMap.setEndpoint('start', lnglat); if ($<HTMLInputElement>('return-start').checked) finishPixel = pixel;
  } else {
    finishPixel = pixel; $<HTMLInputElement>('return-start').checked = false;
    $('pick-finish').querySelector('span')!.textContent = '终点已设置'; campusMap.setEndpoint('finish', lnglat);
  }
  mapAction = null; $('map-action-hint').classList.add('hidden'); resetPlan();
};
campusMap.onDrag = () => { follow = false; if (sensors.active) $('follow-label').textContent = '继续跟随'; };
document.body.addEventListener('click', event => {
  const target = event.target as HTMLElement;
  const reorder = target.closest<HTMLElement>('[data-mandatory-up],[data-mandatory-down]');
  if (reorder) {
    const ids = [...mandatory], id = reorder.dataset.mandatoryUp ?? reorder.dataset.mandatoryDown!;
    const from = ids.indexOf(id), to = from + (reorder.dataset.mandatoryUp !== undefined ? -1 : 1);
    if (from >= 0 && to >= 0 && to < ids.length) { [ids[from], ids[to]] = [ids[to], ids[from]]; mandatory = new Set(ids); resetPlan(); persistRules(); renderMixedRules(); }
    return;
  }
  const element = target.closest<HTMLElement>('[data-control-edit],[data-up],[data-down],[data-team],[data-remove-anchor]');
  if (!element) return;
  if (element.dataset.controlEdit) openEditor('inspect', element.dataset.controlEdit);
  else if (element.dataset.removeAnchor !== undefined) { anchors.splice(Number(element.dataset.removeAnchor), 1); renderAnchors(); }
  else if (element.dataset.team) {
    selectedTeam = selectedTeam === element.dataset.team ? null : element.dataset.team;
    campusMap.setRoutes(plan?.routes ?? [], selectedTeam);
    document.querySelectorAll('[data-team]').forEach(card => card.classList.toggle('selected', (card as HTMLElement).dataset.team === selectedTeam));
  } else if (state) {
    const from = Number(element.dataset.up ?? element.dataset.down), to = from + (element.dataset.up !== undefined ? -1 : 1);
    if (to >= 0 && to < state.controls.length) { [state.controls[from], state.controls[to]] = [state.controls[to], state.controls[from]]; resetPlan(); renderState(); }
  }
});
$('control-list').onchange = event => {
  const input = event.target as HTMLInputElement, id = input.dataset.mandatory ?? input.dataset.required;
  const control = state?.controls.find(item => item.id === id); if (!control) return;
  if (input.dataset.mandatory) { if (input.checked) { mandatory.add(control.id); control.required = true; } else mandatory.delete(control.id); missingPresetLabels = []; }
  else { control.required = input.checked; if (!control.required) mandatory.delete(control.id); }
  resetPlan(); persistRules(); renderState();
};
$('edit-controls').onclick = () => openEditor();
$('edit-terrain').onclick = () => openEditor('close');
$('add-anchor').onclick = () => openEditor('anchor');
$('fit-anchors').onclick = () => task(async () => {
  if (!state) return;
  if (!localRecord) {
    // The stored analysis is needed before anchors can rebuild costs/approaches.
    await runProcessing('process');
    if (!localRecord) throw new Error('本机处理结果不可用');
  }
  const { image } = await loadNativeImage(activeId);
  const result = await reregisterMap({ assetId: activeId, image, existing: state!, record: localRecord!, anchors, progress });
  await persistResult(activeId, result, localRecord!.summary.kind);
  catalog = { ...catalog, maps: catalog.maps.map(entry => entry.id === activeId
    ? { ...entry, local: true, processed: true, controls: result.state.controls.length, version: result.state.version } : entry) };
  anchors = []; renderAnchors();
  await loadState(activeId);
  notify('已用同名点重新校准；旧路线已失效');
});
$('export-map').onclick = event => { event.preventDefault(); if (state) download(`${activeId}.json`, state); };
$('reset-static-data').onclick = () => task(async () => {
  if (localRecord) {
    const wasUpload = !(await loadMapState(activeId));
    await deleteLocalMap(activeId);
    localRecord = null;
    invalidateLocalAssets(activeId);
    clearClientPlannerCache(); clearPlannerCache();
    catalog = await loadCatalog();
    rememberLocalCatalog();
    for (const summary of rememberLocalMaps()) {
      const existing = catalog.maps.find(entry => entry.id === summary.assetId);
      if (existing) existing.local = true;
      else if (summary.assetId !== activeId) catalog.maps.push({ id: summary.assetId, name: summary.name, filename: summary.filename, sha256: summary.sha256,
        processed: true, controls: null, courseDefaults: null, hasGraph: false, hasApproaches: false, local: true, sourceKind: summary.kind });
    }
    if (wasUpload) activeId = catalog.maps[0]?.id ?? activeId;
    anchors = [];
    await loadState(activeId);
    notify('已删除本机处理结果，恢复构建时数据');
    return;
  }
  clearStateOverride(activeId);
  clearPlannerCache();
  await loadState(activeId);
  notify('已恢复构建时数据');
});
$<HTMLInputElement>('standard-map').onchange = event => campusMap.setStandardMap((event.target as HTMLInputElement).checked);
$<HTMLInputElement>('overlay-opacity').oninput = () => campusMap.setOpacity(number('overlay-opacity') / 100);
$<HTMLInputElement>('stitch-maps').onchange = () => task(updateOverlays);
$<HTMLInputElement>('show-terrain').onchange = () => campusMap.setTerrain(state, $<HTMLInputElement>('show-terrain').checked);
$('plan-button').onclick = () => task(async () => {
  if (!state?.registration) throw new Error('请先识别并校准地图');
  if (!startPixel) throw new Error('请在地图上选择起点');
  const finish = $<HTMLInputElement>('return-start').checked ? startPixel : finishPixel;
  if (!finish) throw new Error('请在地图上选择终点');
  const people = number('people'), teams = number('teams'), speed = number('speed'), stagger = number('stagger');
  if (![people, teams].every(Number.isInteger) || people < 1 || teams < 1 || teams > people || teams > 30) throw new Error('请填写有效人数与队数，队数不能超过人数');
  if (!Number.isFinite(speed) || speed < .2 || speed > 10 || !Number.isFinite(stagger) || stagger < 0) throw new Error('请填写有效速度和出发间隔');
  const controlIds = state.controls.filter(control => control.required && control.state !== 'rejected').map(control => control.id);
  if (!controlIds.length) throw new Error('请至少选择一个目标点');
  const rules = rulePayload();
  if (value('course-mode') === 'mixed_coverage' && (missingPresetLabels.length || !rules.mandatoryControlIds.length)) throw new Error('请明确选择所有每队必到点后再规划');
  resetPlan();
  const revision = planRevision;
  const request: PlanRequest = {
    assetId: activeId, version: state.version, controlIds, startPixel, finishPixel: finish,
    mode: value('course-mode'), objective: value('objective'), ...rules,
    teams: Array.from({ length: teams }, (_, index) => ({ id: `第 ${index + 1} 队`, people: Math.floor(people / teams) + (index < people % teams ? 1 : 0), speedMps: speed, departureSec: index * stagger })),
    timeLimitSec: number('solver-budget'),
  };
  const planner = await activePlanner(progress);
  const result = await planner.plan(request, state.controls, progress);
  if (revision !== planRevision) throw new Error('计算期间规则已改变，请按新规则重新生成路线');
  showPlan(result);
});

$('follow-button').onclick = async () => {
  if (courseSuperseded) { notify('旧路线已失效，请使用教师提供的新课程链接', true); return; }
  northOnly = false; follow = true;
  if (IN_APP_BROWSER) maybeInAppBrowserHint();
  // Restart so a previous permission error / stalled watch can be retried.
  if (sensors.active) sensors.stop();
  $('location-status').textContent = '正在请求定位权限…';
  $('follow-label').textContent = '正在开启定位…';
  await sensors.start();
  if (sensors.active) {
    $('follow-label').textContent = '跟随已开启';
    if (sensors.active && $('location-status').textContent === '正在请求定位权限…') $('location-status').textContent = '等待首次定位…';
  } else $('follow-label').textContent = '定位未开启，点此重试';
};
$('north-button').onclick = () => { northOnly = true; campusMap.northUp(); $('compass-arrow').style.transform = 'rotate(0deg)'; $('sensor-status').textContent = '北向上 · 点击跟随按钮恢复朝向跟随'; };
sensors.onHeading = (sample: HeadingSample | null) => {
  if (!sample) { if (sensors.active && !northOnly) $('sensor-status').textContent = '等待可靠朝向 · 可缓慢走动或校准罗盘'; return; }
  if (follow && !northOnly) { campusMap.setHeading(sample.degrees); $('compass-arrow').style.transform = `rotate(${-sample.degrees}deg)`; }
  if (!northOnly) $('sensor-status').textContent = `${sample.source === 'compass' ? sample.reference === 'true' ? '罗盘方向' : '罗盘方向（北向待校准）' : '行进方向'} ${Math.round(sample.degrees)}°${follow ? ' · 跟随中' : ' · 已暂停跟随'}`;
};
sensors.onPosition = position => {
  lastPosition = position; lastPositionReceivedAt = Date.now();
  campusMap.setLocation(position, follow);
  const accuracy = position.coords.accuracy;
  $('location-status').textContent = Number.isFinite(accuracy) ? `定位精度 ±${Math.round(accuracy)} m` : '已定位（精度未知）';
  $('follow-label').textContent = '跟随已开启';
  if (courseId) { renderStudent(); void updateNavigation(position); }
};
function browserHint(message: string): string {
  return IN_APP_BROWSER && !message.includes('系统浏览器') ? `${message}（微信内嵌浏览器请点右上角 ··· 选择“在浏览器打开”）` : message;
}
let lastSensorNotice = '';
sensors.onStatus = message => {
  const text = browserHint(message);
  $('sensor-status').textContent = text;
  // Surface permission/unsupported states instead of only updating small text.
  if (text !== lastSensorNotice) { lastSensorNotice = text; notify(text); }
};
sensors.onLocationStatus = message => { $('location-status').textContent = browserHint(message); };
window.addEventListener('pagehide', () => { resumeSensors = sensors.active; sensors.stop(); });
window.addEventListener('pageshow', event => { if (event.persisted && resumeSensors && !courseSuperseded) void sensors.start(); });

function simpleDialog(title: string, body: string): HTMLDialogElement {
  const dialog = document.createElement('dialog'); dialog.className = 'info-dialog';
  dialog.innerHTML = `<div class="dialog-heading"><h2>${e(title)}</h2><button class="icon-button" data-close aria-label="关闭">×</button></div>${body}`;
  document.body.append(dialog);
  dialog.querySelector<HTMLButtonElement>('[data-close]')!.onclick = () => { dialog.close(); dialog.remove(); };
  dialog.addEventListener('cancel', () => dialog.remove());
  dialog.showModal();
  return dialog;
}
$('open-diagnostics').onclick = () => {
  const dialog = document.createElement('dialog');
  dialog.className = 'info-dialog diagnostics-dialog';
  dialog.innerHTML = `<div class="dialog-heading"><h2>定位与朝向诊断</h2><button class="icon-button" data-close aria-label="关闭">×</button></div>
    <p class="muted compact">构建 ${e(__DIRMAP_COMMIT__)} · 若线下回报问题请截图本页或点“复制诊断信息”。</p>
    <pre id="diagnostics-body" class="diagnostics-body"></pre>
    <button id="copy-diagnostics" class="soft-button wide">复制诊断信息</button>
    <button id="force-update" class="primary-button wide">强制更新到最新版本</button>
    <p class="muted compact">“强制更新”会清除本站离线缓存并重新加载；网络恢复后离线缓存会重新建立。</p>`;
  document.body.append(dialog);
  const pre = dialog.querySelector<HTMLPreElement>('#diagnostics-body')!;
  let permission = '查询中…';
  const permissionPromise = navigator.permissions?.query({ name: 'geolocation' as PermissionName })
    .then(state => { permission = state.state; }).catch(() => { permission = '不可查询'; });
  const update = () => {
    const d = sensors.diagnostics();
    const age = (time: number) => time ? `${Date.now() - time} ms` : '无';
    pre.textContent = [
      `构建: ${__DIRMAP_COMMIT__}`,
      `页面: ${location.href}`,
      `安全上下文: ${window.isSecureContext ? '是' : '否'}`,
      `内嵌浏览器: ${IN_APP_BROWSER ? '是（微信/QQ 等，定位可能被禁用）' : '否'}`,
      `定位权限: ${permission}`,
      `定位接口: ${navigator.geolocation ? '有' : '无'}`,
      `定位已启动: ${d.active ? '是' : '否'}   定时器/监看: ${d.watchId ?? 'null'}`,
      `原始定位回调次数: ${d.rawPositionCount}   通过校验次数: ${d.acceptedPositionCount}`,
      `最后原始时间戳距今: ${age(d.lastRawTimestamp)}`,
      `最后通过校验距今: ${age(d.lastAcceptedReceivedAt)}`,
      `最近定位错误: ${d.lastGeoError ?? '无'}`,
      `朝向接口: ${d.orientationSupported ? '有' : '无'}   朝向权限: ${d.orientationPermissionGranted ? '已允许' : '未允许/未请求'}`,
      `朝向事件: ${d.orientationEventCount}（其中绝对方向 ${d.absoluteEventCount}）   有效朝向样本: ${d.headingSampleCount}`,
      `当前朝向: ${d.heading === null ? '无' : `${Math.round(d.heading)}°`}`,
      `UA: ${navigator.userAgent}`,
    ].join('\n');
  };
  void permissionPromise; update();
  const timer = setInterval(update, 1000);
  const close = () => { clearInterval(timer); dialog.close(); dialog.remove(); };
  dialog.querySelector<HTMLButtonElement>('[data-close]')!.onclick = close;
  dialog.addEventListener('cancel', close);
  dialog.querySelector<HTMLButtonElement>('#copy-diagnostics')!.onclick = async (event) => {
    const button = event.currentTarget as HTMLButtonElement;
    try { await navigator.clipboard.writeText(pre.textContent ?? ''); button.textContent = '已复制'; }
    catch { button.textContent = '复制失败，请截图'; }
  };
  dialog.querySelector<HTMLButtonElement>('#force-update')!.onclick = () => {
    void (async () => {
      try {
        for (const name of await caches.keys()) await caches.delete(name);
        const registrations = await navigator.serviceWorker?.getRegistrations?.() ?? [];
        await Promise.all(registrations.map(registration => registration.unregister()));
      } finally { location.reload(); }
    })();
  };
  dialog.showModal();
};
$('calibrate-compass').onclick = () => {
  const dialog = simpleDialog('校准手机朝向', `<p>在开阔处平放手机，将屏幕上沿对准已知的真北方向。默认真北为 0°；也可以输入已知地物的方位角。</p><label class="field-label">已知方位角（顺时针，度）<input id="known-bearing" type="number" min="0" max="359" value="0"></label><p id="compass-calibration-error" class="danger" role="status"></p><button id="confirm-compass" class="primary-button wide">已对准，校准当前方向</button><button id="clear-compass-calibration" class="text-button wide">恢复设备原始方向</button><p class="muted">校准只影响罗盘；行进方向仍由定位数据计算。远离金属和磁铁后可重新校准。</p>`);
  dialog.querySelector<HTMLButtonElement>('#confirm-compass')!.onclick = () => {
    try { sensors.calibrate(number('known-bearing')); dialog.close(); dialog.remove(); notify('已校准罗盘方向'); }
    catch (error) { $('compass-calibration-error').textContent = error instanceof Error ? error.message : String(error); }
  };
  dialog.querySelector<HTMLButtonElement>('#clear-compass-calibration')!.onclick = () => { sensors.clearCalibration(); dialog.close(); dialog.remove(); notify('已恢复设备方向'); };
};
$('open-help').onclick = () => simpleDialog(`使用 DirMap · 构建 ${__DIRMAP_COMMIT__}`, `<ol class="help-list"><li>在“课程地图”选择内置地图或导入自己的扫描图；点击“自动识别与配准”，识别、配准与通行图全部在浏览器内计算，构建时数据作为快速默认值。</li><li>在“原图校核”中核对目标编号、类型和到访位置；修改保存在本机浏览器，可随时删除本机处理结果恢复构建时数据。</li><li>核对围墙、门洞、禁区与桥隧；“更新通行识别”可在保留教师校核的前提下重算地形；“同名点”可修正局部配准偏差。</li><li>填写人数、队数与课程规则，选择起终点并生成路线；小规模课程使用精确动态规划，大规模使用时间预算内的可行解。</li><li>发布前确认课程；课程 JSON 与叠图保存在本机并下载。把课程与地图数据提交到站点数据目录后重新部署，其他设备即可打开课程链接。</li></ol><p class="muted">浏览器内处理依赖 WebAssembly 识别引擎与大图内存，手机处理大图可能较慢；离线时使用已缓存数据。标准地图及校园参照 © OpenStreetMap contributors。自动匹配误差不等价于现场精度。手机传感器需 HTTPS 和浏览器权限。</p>`);
$('review-button').onclick = () => {
  if (!state || !plan?.routes.length) return;
  const suggested = plan.approaches?.filter(approach => approach.automatic) ?? [];
  const localOnly = !!localRecord;
  const dialog = simpleDialog('发布前核验与生成静态课程', `<p class="muted">核验结果与课程保存在本机浏览器，同时下载课程 JSON 供你提交到站点数据目录。${localOnly ? '本机处理的地图仅能在本浏览器预览，跨设备分享需要把处理数据纳入构建数据后重新部署。' : ''}</p><label class="check-label"><input id="review-registration" type="checkbox">配准位置与实际地物一致</label><label class="check-label"><input id="review-controls" type="checkbox">本次目标编号、圈心与到访位置已核对</label><label class="check-label"><input id="review-passages" type="checkbox">路线不穿越禁区、围墙及不可通行的层级</label><label class="field-label">核验备注<textarea id="review-notes" rows="3" placeholder="现场核验日期、临时封闭路段等"></textarea></label><p id="review-error" class="danger" role="status"></p><button id="publish-confirm" class="primary-button wide">生成学生课程链接</button><p class="muted compact">让其他设备访问：把下载的 <code>release-&lt;id&gt;.json</code> 放入 <code>web/public/data/releases/</code>，在 <code>web/public/data/releases.json</code> 的 <code>releases</code> 数组加入 <code>{"id":"…","title":"…","assetId":"…","createdAt":…}</code>，重新构建部署。</p>`);
  if (suggested.length) dialog.querySelector('#review-error')!.insertAdjacentHTML('beforebegin', `<label class="check-label"><input id="review-approaches" type="checkbox">已核对并采用 ${suggested.length} 处橙色标记的建议到访位置</label>`);
  dialog.querySelector<HTMLButtonElement>('#publish-confirm')!.onclick = async () => {
    const button = dialog.querySelector<HTMLButtonElement>('#publish-confirm')!; button.disabled = true;
    try {
      if (!['registration', 'controls', 'passages'].every(name => $<HTMLInputElement>(`review-${name}`).checked)) throw new Error('请完成三项核验后发布');
      if (suggested.length && !$<HTMLInputElement>('review-approaches').checked) throw new Error('请先在原图核对并确认建议的到访位置');
      const previousPlan = plan!;
      const controls = state!.controls.map(control => {
        if (!control.required) return control;
        const approach = suggested.find(item => item.controlId === control.id);
        return { ...control, state: 'accepted' as const, ...(approach ? { approachPixel: approach.pixel } : {}) };
      });
      // Accepting targets/approaches is a review action: costs, registration and
      // the graph hash do not change, so the reviewed plan remains valid. Keep
      // the map version stable so the frozen course keeps matching local state
      // and do not build a second planner graph (that can exhaust a phone tab).
      if (localRecord) {
        localRecord = { ...localRecord, state: { ...localRecord.state, controls } };
        await saveLocalMap(localRecord);
        state = localRecord.state;
      } else if (bakedState) {
        const saved = saveStateOverride(activeId, bakedState, controls, state!.edits, { bump: false });
        state = applyStateOverride(bakedState, saved);
      }
      const result = previousPlan;
      const releaseId = randomId32();
      const release: StaticRelease = {
        id: releaseId, title: '深大定向越野课', assetId: activeId, createdAt: Date.now() / 1000,
        plan: result, controls: state!.controls, frame: state!.frame, extent: state!.extent ?? [],
        quality: state!.quality, review: { at: Date.now() / 1000, static: true, notes: value('review-notes') },
        version: state!.version, routingSchema: __DIRMAP_ROUTING_SCHEMA__, superseded: false,
        static: true, overlay: localOnly ? undefined : `data/maps/${activeId}/overlay.png`,
      };
      saveLocalRelease(release);
      downloadRelease(release);
      const link = new URL(location.href); link.search = `?course=${releaseId}`; link.hash = '';
      $('share-result').innerHTML = `<label class="field-label">学生课程链接<input readonly value="${e(link.href)}" aria-label="学生课程链接"></label><a href="${e(link.href)}" target="_blank" rel="noreferrer" class="text-link">本机预览学生页面 ↗</a><p class="muted compact">已下载 <code>release-${releaseId}.json</code>。${localOnly ? '本机处理的地图仅能在本浏览器预览；提交处理数据到构建目录后重新部署即可跨设备分享。' : state!.edits.length ? '注意：本课程的通行修正只在本机存在，其他设备需要重新构建站点数据。' : '按 DEPLOY.md 提交后重新部署，链接即可在其他设备打开。'}</p>`;
      dialog.close(); dialog.remove(); renderState(); notify('已生成静态课程并下载 JSON');
    } catch (error) { $('review-error').textContent = error instanceof Error ? error.message : String(error); button.disabled = false; }
  };
};

function studentRoute(): Route | undefined { return plan?.routes.find(route => route.teamId === value('student-team')); }
function progressKey() { return `dirmap-progress:${courseId}:${value('student-team')}`; }
function renderStudent() {
  const route = studentRoute(); if (!route) return;
  $('student-overview').innerHTML = `<div class="quality-values"><div><strong>${distance(route.distanceM)}</strong><span>全程距离</span></div><div><strong>${Math.round(route.durationSec / 60)}<small> 分钟</small></strong><span>预计用时（含打点）</span></div></div><p>${route.people} 人同行${route.departureSec ? ` · 开课后 ${route.departureSec} 秒出发` : ' · 统一出发'}</p>`;
  const requiredByAll = new Set(plan?.coverage?.mandatoryControlIds ?? []);
  if (plan?.coverage?.mode === 'mixed_coverage') {
    const labels = [...requiredByAll].map(id => e(studentControls.find(control => control.id === id)?.label ?? id));
    $('student-overview').insertAdjacentHTML('beforeend', `<p id="student-coverage">全员必到：${labels.join('、')}；其余为本队分担点。${plan.coverage.mandatoryOrder ? '必到点按教师指定顺序经过。' : '必到点无指定先后，已按本队路线安排。'}</p>`);
  }
  const index = route.controlIds.findIndex(id => !completed.has(id));
  const next = index < 0 ? null : studentControls.find(control => control.id === route.controlIds[index]);
  $('student-progress').textContent = `${route.controlIds.filter(id => completed.has(id)).length} / ${route.controlIds.length}`;
  let metres: number | null = null;
  if (next?.lnglat && lastPosition && Date.now() - lastPositionReceivedAt < 15000) { const lat = next.lnglat[1] * Math.PI / 180; metres = Math.hypot((next.lnglat[0] - lastPosition.coords.longitude) * 111320 * Math.cos(lat), (next.lnglat[1] - lastPosition.coords.latitude) * 110700); }
  $('next-control').innerHTML = next
    ? `<span class="next-label">目标 ${e(next.label)}</span><strong>${metres === null ? '等待定位' : `直线 ${distance(metres)}`}</strong><p>沿地图中的队伍路线前往</p>`
    : '<span class="next-label">所有目标已完成</span><strong>前往课程终点</strong><p>仍需按路线返回终点。</p>';
  $<HTMLButtonElement>('check-in').disabled = !next || courseSuperseded;
  $('student-stops').innerHTML = route.controlIds.map((id, index) => `<div class="student-stop ${completed.has(id) ? 'done' : ''}"><i>${completed.has(id) ? '✓' : index + 1}</i><span>目标 ${e(studentControls.find(control => control.id === id)?.label ?? id)}${plan?.coverage?.mode === 'mixed_coverage' ? `<small> · ${requiredByAll.has(id) ? '全员必到' : '本队分担'}</small>` : ''}</span><small>${completed.has(id) ? '已完成' : '待到访'}</small></div>`).join('');
}
function resetNavigation() {
  navigationRevision++; navigationRoute = null; offRoute.reset();
  campusMap.setRoutes(plan?.routes ?? [], value('student-team'));
}
async function updateNavigation(position: GeolocationPosition) {
  const route = studentRoute();
  if (!route || !courseId || courseSuperseded || !follow || navigating) return;
  const lnglat: Pixel = [position.coords.longitude, position.coords.latitude];
  if (!offRoute.update({ lnglat, accuracy: position.coords.accuracy, timestamp: position.timestamp }, remainingPath(navigationRoute ?? route, completed), Date.now())) return;
  if (!state || !currentRelease) return;
  const revision = navigationRevision, teamId = route.teamId;
  const completedControlIds = route.controlIds.filter(id => completed.has(id));
  navigating = true; $('location-status').textContent = '已偏离路线，正在计算返回路径…';
  try {
    const planner = await getPlanner(state, progress);
    const result: NavigateResponse = await planner.navigate(currentRelease, {
      teamId, lnglat, accuracyM: position.coords.accuracy, timestampMs: position.timestamp, completedControlIds,
    }, studentControls);
    if (revision !== navigationRevision || courseSuperseded || teamId !== studentRoute()?.teamId) return;
    if (result.status !== 'READY' || !result.route) throw new Error(result.unresolved?.[0]?.message ?? '当前位置没有可确认的合法连接');
    navigationRoute = result.route;
    campusMap.setRoutes(plan!.routes.map(route => route.teamId === teamId ? navigationRoute! : route), teamId);
    $('location-status').textContent = '已更新返回路线 · 保持本队剩余点顺序';
  } catch (error) {
    if (revision === navigationRevision) $('location-status').textContent = navigator.onLine
      ? `暂不能重规划：${error instanceof Error ? error.message : String(error)}`
      : '当前离线，保留已下载路线';
  } finally { navigating = false; hideProgress(); }
}
$<HTMLSelectElement>('student-team').onchange = () => {
  try { completed = new Set(JSON.parse(localStorage.getItem(progressKey()) ?? '[]')); } catch { completed = new Set(); }
  // Discard malformed/outdated local progress instead of skipping targets.
  const ids = studentRoute()?.controlIds ?? [];
  const count = ids.findIndex(id => !completed.has(id));
  completed = new Set(ids.slice(0, count < 0 ? ids.length : count));
  resetNavigation(); renderStudent();
};
$('check-in').onclick = () => { if (courseSuperseded) return; const next = studentRoute()?.controlIds.find(id => !completed.has(id)); if (next) { completed.add(next); localStorage.setItem(progressKey(), JSON.stringify([...completed])); resetNavigation(); renderStudent(); } };
$('undo-check-in').onclick = () => { if (courseSuperseded) return; const last = [...completed].at(-1); if (last) { completed.delete(last); localStorage.setItem(progressKey(), JSON.stringify([...completed])); resetNavigation(); renderStudent(); } };

async function boot() {
  maybeInAppBrowserHint();
  const health = await loadHealth();
  sensors.declination = declinationFor(health);
  loadReference().then(data => campusMap.setReference(data)).catch(() => notify('校园参考暂不可用，正在显示在线底图'));
  if (courseId) {
    $('teacher-panel').classList.add('hidden'); $('student-panel').classList.remove('hidden');
    document.querySelectorAll('.teacher-only').forEach(element => element.classList.add('hidden'));
    $('header-mode').textContent = '学生导航';
    const course = findLocalRelease(courseId) ?? await loadRelease(courseId);
    if (!course) throw new Error('课程不存在，请向教师获取最新课程链接；静态站点的课程需要先提交到数据目录');
    currentRelease = course;
    if (course.superseded || course.routingSchema !== __DIRMAP_ROUTING_SCHEMA__) {
      courseSuperseded = true; $('release-badge').textContent = '课程已更新';
      $('sensor-status').textContent = '旧路线已失效，请使用教师提供的新链接';
      $<HTMLButtonElement>('check-in').disabled = true; $<HTMLButtonElement>('undo-check-in').disabled = true;
      throw new Error('这份课程的地图或通行状态已更新，请向教师获取新链接');
    }
    localRecord = await loadLocalMap(course.assetId);
    if (localRecord) await prepareLocalAssets(course.assetId);
    state = localRecord?.state ?? await loadMapState(course.assetId);
    plan = course.plan; studentControls = course.controls; $('course-title').textContent = course.title;
    $('release-badge').textContent = course.offline ? '离线 · 状态未更新' : localRecord ? '本机课程' : '课堂导航';
    $('student-team').innerHTML = plan.routes.map(route => `<option value="${e(route.teamId)}">${e(route.teamId)} · ${route.people} 人</option>`).join('');
    campusMap.setReleaseOverlay(course, .65); campusMap.setControls(course.controls); campusMap.setApproaches(plan.approaches ?? []);
    $<HTMLSelectElement>('student-team').dispatchEvent(new Event('change'));
    if (import.meta.env.PROD) {
      if (course.offline) $('offline-status').textContent = '正在使用已保存课程；联网后检查教师更新。';
      else if (localRecord) $('offline-status').textContent = '本机处理课程：地图与路线数据保存在本机浏览器，刷新后仍可用。';
      else void cacheStudentCourse();
    }
  } else {
    catalog = await loadCatalog();
    rememberLocalCatalog();
    for (const entry of catalog.maps) if (hasLocalMap(entry.id)) entry.local = true;
    for (const summary of rememberLocalMaps()) {
      if (!catalog.maps.some(entry => entry.id === summary.assetId)) {
        catalog.maps.push({ id: summary.assetId, name: summary.name, filename: summary.filename, sha256: summary.sha256,
          processed: true, controls: null, courseDefaults: null, hasGraph: false, hasApproaches: false,
          local: true, sourceKind: summary.kind });
      }
    }
    activeId = catalog.maps[0]?.id ?? activeId;
    updateStaticNotice(); renderCatalog();
    await loadState(activeId);
  }
}
async function cacheStudentCourse() {
  if (!courseId || courseSuperseded) return;
  if (localRecord) { $('offline-status').textContent = '本机处理课程：地图数据保存在本机浏览器，无需额外离线缓存。'; return; }
  $('offline-status').textContent = '正在保存原图、路线与目标…';
  try { await saveOfflineCourse(courseId); $('offline-status').textContent = '本课已保存；断网后可查看原图、路线并记录打点。'; }
  catch (error) { $('offline-status').textContent = error instanceof Error ? error.message : String(error); }
}
boot().catch(error => notify(error instanceof Error ? error.message : String(error), true));
