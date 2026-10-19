import 'ol/ol.css';
import './style.css';
import { CampusMap, teamColors } from './map';
import { NavigationSensors, type HeadingSample } from './heading';
import { PixelEditor } from './editor';
import { OffRouteMonitor, remainingPath } from './navigation';
import { installOffline, saveOfflineCourse } from './offline';
import {
  declinationFor, downloadJSON, downloadRelease, legendUrl, loadCatalog, loadHealth, loadMapState,
  loadReference, loadRelease,
} from './data';
import {
  applyStateOverride, clearStateOverride, findLocalRelease, hasStateOverride, loadStateOverride,
  saveLocalRelease, saveStateOverride,
} from './static-store';
import { clearPlannerCache, getPlanner } from './planner';
import {
  escapeHtml as e, lngLatToPixel, pixelToLngLat,
  type BakedCatalog, type Control, type MapState, type NavigateResponse, type Pixel, type Plan,
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
let state: MapState | null = null;
let activeId = 'szu-1';
let plan: Plan | null = null;
let planRevision = 0;
let mandatory = new Set<string>();
let missingPresetLabels: string[] = [];
let startPixel: Pixel | null = null, finishPixel: Pixel | null = null;
let mapAction: 'start' | 'finish' | null = null;
let follow = false, northOnly = false;
let lastPosition: GeolocationPosition | null = null;
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
<section class="source-section"><div class="section-title"><h2>课程地图</h2><span class="status-tag">构建时数据</span></div><div id="map-cards" class="map-cards"></div><p id="static-source-note" class="muted compact">静态版：识别、配准与通行图已在构建时完成；可直接校核目标并生成路线。</p><div id="job-status" class="job-status hidden" role="status"><div><span id="job-message">准备中</span><span id="job-percent">0%</span></div><progress id="job-progress" max="100" value="0"></progress></div><div id="map-summary" class="summary-strip">选择地图开始准备课程</div></section>
<nav class="tabs" aria-label="工作区域"><button data-tab="course" class="active">课程设置</button><button data-tab="registration">地图配准</button><button data-tab="terrain">通行核验</button></nav>
<section id="tab-course" class="tab-content"><div class="section-title"><h2>路线规则</h2><span class="step-number">01</span></div><label class="field-label" for="course-mode">访问方式</label><select id="course-mode"><option value="all_visit">每队访问全部目标</option><option value="split_coverage">全班分担全部目标</option><option value="fixed_order">按列表顺序访问</option></select><div class="form-grid"><label>总人数<input id="people" type="number" value="20" min="1" max="500"></label><label>队伍数<input id="teams" type="number" value="4" min="1" max="30"></label><label>步行 / 跑速 <span>m/s</span><input id="speed" type="number" value="1.4" step="0.1" min="0.2" max="10"></label><label>错峰间隔 <span>秒</span><input id="stagger" type="number" value="0" min="0" max="3600"></label></div><p id="group-preview" class="muted compact">每队 5 人 · 同时出发</p><label class="field-label" for="objective">优化目标</label><select id="objective"><option value="min_max_finish_then_sum">优先让最后一队更早完成</option><option value="min_total_time">全班总耗时最短</option></select>
<div class="section-title section-gap"><h2>起终点与目标</h2><span class="step-number">02</span></div><div class="endpoint-row"><button id="pick-start" class="endpoint-button"><i class="endpoint-dot start"></i><span>在地图选起点</span></button><button id="pick-finish" class="endpoint-button"><i class="endpoint-dot finish"></i><span>在地图选终点</span></button></div><label class="check-label"><input id="return-start" type="checkbox" checked>回到起点结束</label><div class="control-heading"><span id="control-count">尚未识别目标</span><button id="edit-controls" class="text-button">原图校核 ↗</button></div><p class="muted compact">勾选本次课程目标；固定顺序模式可上下调整。</p><div id="control-list" class="control-list"><div class="empty-small">识别后，控制圈会显示在这里。</div></div><button id="plan-button" class="primary-button wide plan-button">生成队伍路线 <span>→</span></button><div id="plan-errors" role="status"></div><div id="route-results"></div>
<div id="publish-section" class="publish-section hidden"><button id="review-button" class="soft-button wide">核验课程并发布</button><div id="share-result"></div></div></section>
<section id="tab-registration" class="tab-content hidden"><div class="section-title"><h2>地物匹配与局部校准</h2><span class="step-number">01</span></div><p class="muted">自动配准在构建时完成；静态版不重新运行地物匹配。教师可在本机核对目标并将修改保存在浏览器。</p><div id="registration-quality" class="quality-card">完成自动处理后显示匹配结果。</div><p class="muted section-gap">重新识别、OCR 或同名点重校准需要在本地运行构建数据脚本（见 DEPLOY.md），然后重新部署静态站点。</p><a id="export-map" class="text-link" href="#">下载当前配准和目标数据 ↗</a></section>
<section id="tab-terrain" class="tab-content hidden"><div class="section-title"><h2>核对可通行区域</h2><span class="step-number">01</span></div><p class="muted">红色区域为识别出的障碍。重点核对围墙、门洞、水体、禁区和上下层通道。</p><label class="check-label"><input id="show-terrain" type="checkbox">在地图上显示障碍候选</label><button id="edit-terrain" class="primary-button wide section-gap">打开原图修正通行区域</button><div id="terrain-summary" class="quality-card"></div><button id="reset-static-data" class="text-button wide" disabled>恢复构建时数据</button><p class="muted">本机修正保存后需重新生成路线；修正只保存在本浏览器，不会随站点分发。二维相交不能证明桥上、桥下可以互通。</p><div class="section-title section-gap"><h2>随图图例</h2></div><div id="legend-links" class="legend-links"></div></section></aside>
<aside id="student-panel" class="sidebar student-panel hidden"><p class="eyebrow">TODAY'S ADVENTURE</p><h1 id="course-title">定向越野课</h1><p class="muted">选择队伍，打开定位，出发探索。</p><label class="field-label" for="student-team">我的队伍</label><select id="student-team"></select><div id="student-overview" class="quality-card"></div><div class="section-title"><h2>下一目标</h2><span id="student-progress" class="muted"></span></div><div id="next-control" class="next-control"></div><button id="check-in" class="primary-button wide">已完成这个目标</button><button id="undo-check-in" class="text-button wide">撤销上一个打点</button><div id="student-stops" class="student-stops"></div><p class="muted">打点由本人确认；当前位置精度不足时，请结合原图和现场地物判断。</p></aside>
<main class="map-panel"><div id="map" role="application" aria-label="校园标准地图与定向路线"></div><div class="map-topbar"><div class="map-title"><span class="map-kicker">CAMPUS EXPLORER</span><strong id="map-title">深圳大学 · 粤海校区</strong><span id="map-subtitle">标准地图 / WGS84</span></div><div class="layer-tools"><label class="check-label"><input id="standard-map" type="checkbox" checked>标准底图</label><label class="check-label teacher-only"><input id="stitch-maps" type="checkbox">叠合三图</label><label class="opacity-control">原图 <input id="overlay-opacity" type="range" min="0" max="100" value="70" aria-label="原图透明度"></label></div></div><div id="map-action-hint" class="map-action-hint hidden" role="status"></div><button id="north-button" class="compass-button" title="北向上" aria-label="切换北向上"><span>N</span><svg id="compass-arrow" viewBox="0 0 24 28" aria-hidden="true"><path d="M12 1 21 25 12 20 3 25Z" fill="#295c46"/><path d="M12 1v19l9 5Z" fill="#abbdb1"/></svg></button><div class="map-bottom"><div class="navigation-card"><button id="follow-button" class="follow-button"><span class="locate-icon">◎</span><span id="follow-label">开始定位与朝向跟随</span></button><div id="sensor-status" class="sensor-status" role="status">手机打开 HTTPS 链接即可启用</div></div><div class="map-key"><span><i class="key-control"></i>目标点</span><span><i class="key-route"></i>队伍路线</span><span id="location-status">位置尚未开启</span></div></div></main></div><div id="toast" class="toast hidden" role="status"></div>`;
document.querySelector('#app')!.innerHTML = APP_TEMPLATE;

const campusMap = new CampusMap($('map'));
const sensors = new NavigationSensors();
$('objective').insertAdjacentHTML('beforeend', '<option value="min_distance">全班总路程最短</option>');
$('course-mode').insertAdjacentHTML('beforeend', '<option value="mixed_coverage">全员必到 + 其余点全班分担</option>');
$('course-mode').insertAdjacentHTML('afterend', '<div id="mixed-rules" class="quality-card hidden"><strong>混合访问规则</strong><p id="mixed-rule-summary"></p><p class="muted compact">在下方勾选“每队必到”；其余选中点由队伍分担。整队同行，每名队员都会经过本队必到点。</p><label class="check-label"><input id="mandatory-order" type="checkbox">指定每队必到点的先后顺序</label><div id="mandatory-order-list"></div><p id="mandatory-rule-error" class="danger"></p></div>');
$<HTMLInputElement>('mandatory-order').onchange = () => { resetPlan(); persistRules(); renderState(); };
$('sensor-status').insertAdjacentHTML('afterend', '<button id="calibrate-compass" class="text-button compass-calibration">校准罗盘</button>');
$('objective').insertAdjacentHTML('afterend', '<label class="field-label" for="solver-budget">规划计算预算</label><select id="solver-budget"><option value="5">快速规划 · 5 秒</option><option value="30">继续优化 · 30 秒</option></select>');
$<HTMLSelectElement>('solver-budget').onchange = resetPlan;
if (import.meta.env.PROD) {
  void installOffline();
  $('course-title').insertAdjacentHTML('afterend', '<p id="offline-status" class="muted" role="status"></p><button id="save-offline" class="text-button">保存本课供离线查看</button>');
  $('save-offline').onclick = () => { void cacheStudentCourse(); };
}
// Expose deterministic hooks in development and for static end-to-end checks.
// Production users only get these when a `?e2e` query parameter is present.
if (import.meta.env.DEV || new URLSearchParams(location.search).has('e2e')) {
  (window as Window & { dirmap?: unknown }).dirmap = {
    campusMap, sensors, getPlanner,
    getState: () => state, getRelease: () => currentRelease, getPlan: () => plan,
  };
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
  const ids = ['plan-button', 'review-button', 'edit-terrain', 'edit-controls'];
  for (const id of ids) { const element = $(id) as HTMLButtonElement | null; if (element) element.disabled = true; }
  try { await action(); }
  catch (error) { notify(error instanceof Error ? error.message : String(error), true); }
  finally {
    busy = false;
    for (const id of ids) { const element = $(id) as HTMLButtonElement | null; if (element) element.disabled = false; }
    hideProgress();
  }
}
function randomId32(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
}
function updateStaticNotice() {
  const note = document.getElementById('static-source-note');
  if (note?.textContent) note.textContent = `静态版：识别、配准与通行图已在构建时完成（${catalog.generatedAt?.slice(0, 10) ?? '构建数据'}）；可直接校核目标并生成路线。`;
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
  $('map-cards').innerHTML = catalog.maps.map((map, index) => `<button class="map-card ${map.id === activeId ? 'active' : ''}" data-map="${e(map.id)}"><span class="map-index">0${index + 1}</span><span class="map-card-name">${e(map.name)}<small>${map.processed ? `${map.controls} 个候选目标` : '原图待处理'}</small></span><span class="map-card-state">${map.processed ? '✓' : '○'}</span></button>`).join('');
  $('legend-links').innerHTML = catalog.legends.map(legend => `<a href="${e(legendUrl(legend.url))}" target="_blank" rel="noreferrer">${e(legend.name.includes('aigc') ? '随图定向运动图例（含 AIGC 标记）' : legend.name)} <span>↗</span></a>`).join('');
}
async function loadState(id: string, fit = true) {
  activeId = id;
  const baked = await loadMapState(id);
  bakedState = baked;
  state = baked ? applyStateOverride(baked, loadStateOverride(id)) : null;
  resetPlan(); startPixel = finishPixel = null; mapAction = null;
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
      .map(map => loadMapState(map.id).then(value => value ? applyStateOverride(value, loadStateOverride(map.id)) : null)));
    states = [...others.filter((value): value is MapState => value !== null), ...states];
  }
  campusMap.setOverlays(states, number('overlay-opacity') / 100, activeId);
  campusMap.setTerrain(state, $<HTMLInputElement>('show-terrain').checked);
}
function renderState() {
  renderMixedRules();
  campusMap.setControls(state?.controls ?? []);
  $('release-badge').textContent = state?.reviewed ? '已核验' : (hasStateOverride(activeId) ? '本机草稿' : '备课草稿');
  const reset = document.getElementById('reset-static-data') as HTMLButtonElement | null;
  if (reset) { reset.disabled = !hasStateOverride(activeId); reset.title = reset.disabled ? '当前显示构建时数据' : '清除本机修改并恢复构建时数据'; }
  if (!state) {
    $('map-summary').textContent = '原图尚未处理';
    $('control-list').innerHTML = '<div class="empty-small">这张图没有构建数据。</div>';
    $('control-count').textContent = '尚未识别目标';
    $('registration-quality').textContent = '没有可用的配准数据。';
    return;
  }
  $('map-title').textContent = state.name;
  $('map-subtitle').textContent = state.registrationStale ? '基准图已更新，请重新配准' : state.registration ? '已叠合至标准地图 · 待现场核对' : '地理参照待补充';
  $('map-summary').innerHTML = `<span><b>${state.controls.length}</b> 候选目标</span><span><b>${state.quality?.fitCount ?? 0}</b> 配准点</span><span class="review-dot">${hasStateOverride(activeId) ? '本机修改' : '待复核'}</span>`;
  $('control-count').textContent = `${state.controls.filter(control => control.required).length} / ${state.controls.length} 个目标已选`;
  $('control-list').innerHTML = state.controls.map((control, index) => `<div class="control-row" data-id="${e(control.id)}"><label class="control-toggle"><input type="checkbox" data-required="${e(control.id)}" ${control.role && control.role !== 'control' ? 'disabled' : ''} ${control.required ? 'checked' : ''}><span class="control-number ${control.calibrated ? '' : 'uncalibrated'}">${e(control.label)}</span></label>${value('course-mode') === 'mixed_coverage' && (control.role ?? 'control') === 'control' ? `<label class="every-team-toggle"><input type="checkbox" data-mandatory="${e(control.id)}" aria-label="${e(control.label)} 每队必到" ${mandatory.has(control.id) ? 'checked' : ''}>每队必到</label>` : ''}<span class="control-state">${!control.calibrated ? '待补配准' : control.state === 'accepted' ? '已确认' : '待核对'}</span><button class="text-button" data-control-edit="${e(control.id)}" aria-label="编辑目标 ${e(control.label)}">↗</button>${value('course-mode') === 'fixed_order' ? `<button class="text-button" data-up="${index}" aria-label="上移 ${e(control.label)}">↑</button><button class="text-button" data-down="${index}" aria-label="下移 ${e(control.label)}">↓</button>` : ''}</div>`).join('');
  const quality = state.quality;
  $('registration-quality').innerHTML = quality
    ? `<span class="quality-label">${quality.baseAssetId ? '共同地物对齐' : '标准地图地物匹配'}</span><div class="quality-values"><div><strong>${quality.fitCount}</strong><span>拟合同名点</span></div><div><strong>${quality.p95ErrorM?.toFixed(1) ?? '—'}<small> m</small></strong><span>留出误差 P95</span></div></div><p>${e(quality.validationScope)}</p>${quality.inheritedGeographicP95M ? `<p>基准图地理匹配 P95：${quality.inheritedGeographicP95M.toFixed(1)} m</p>` : ''}`
    : '自动匹配证据不足；静态版不支持重新运行配准。';
  $('terrain-summary').innerHTML = `<strong>${Math.round(state.statistics.forbiddenPixelFraction * 100)}%</strong> 像素被识别为障碍候选<p>${state.edits.length} 处本机通行修正</p><p>路线会检查每条边经过的原图像素。未知层级及模糊的门洞仍需核验。</p>${hasStateOverride(activeId) ? '<p id="local-edit-note">正在使用本机修改；本机修改可规划路线，但不会随静态站点分发给其他设备。</p>' : ''}`;
  $('export-map').setAttribute('download', `${activeId}.json`);
}
async function saveEdits(controls: Control[], edits: TerrainEdit[]) {
  if (!state || !bakedState) return;
  // Static build: persist mutable fields in this browser. The baked state stays
  // the immutable base, so a later "restore build data" is always lossless.
  const saved = saveStateOverride(activeId, bakedState, controls, edits);
  state = applyStateOverride(bakedState, saved);
  // The edited costs need a rebuilt graph; release cached graphs eagerly so a
  // phone with several edit generations cannot exhaust memory.
  clearPlannerCache();
  resetPlan(); renderState(); await updateOverlays();
  notify('修改已保存到本机，请重新生成路线');
}
function openEditor(mode: 'inspect' | 'close' | 'open' = 'inspect', id: string | null = null) {
  if (!state) { notify('请先载入地图'); return; }
  new PixelEditor(state, saveEdits, null, plan?.approaches ?? []).open(mode, id);
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
  const element = target.closest<HTMLElement>('[data-control-edit],[data-up],[data-down],[data-team]');
  if (!element) return;
  if (element.dataset.controlEdit) openEditor('inspect', element.dataset.controlEdit);
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
$('export-map').onclick = event => { event.preventDefault(); if (state) download(`${activeId}.json`, state); };
$('reset-static-data').onclick = () => task(async () => {
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
  const planner = await getPlanner(state, progress);
  const result = await planner.plan(request, state.controls, progress);
  if (revision !== planRevision) throw new Error('计算期间规则已改变，请按新规则重新生成路线');
  showPlan(result);
});

$('follow-button').onclick = async () => { if (courseSuperseded) { notify('旧路线已失效，请使用教师提供的新课程链接', true); return; } northOnly = false; follow = true; await sensors.start(); if (sensors.active) $('follow-label').textContent = '跟随已开启'; };
$('north-button').onclick = () => { northOnly = true; campusMap.northUp(); $('compass-arrow').style.transform = 'rotate(0deg)'; $('sensor-status').textContent = '北向上 · 点击跟随按钮恢复朝向跟随'; };
sensors.onHeading = (sample: HeadingSample | null) => {
  if (!sample) { if (sensors.active && !northOnly) $('sensor-status').textContent = '等待可靠朝向 · 可缓慢走动或校准罗盘'; return; }
  if (follow && !northOnly) { campusMap.setHeading(sample.degrees); $('compass-arrow').style.transform = `rotate(${-sample.degrees}deg)`; }
  if (!northOnly) $('sensor-status').textContent = `${sample.source === 'compass' ? sample.reference === 'true' ? '罗盘方向' : '罗盘方向（北向待校准）' : '行进方向'} ${Math.round(sample.degrees)}°${follow ? ' · 跟随中' : ' · 已暂停跟随'}`;
};
sensors.onPosition = position => {
  lastPosition = position; campusMap.setLocation(position, follow);
  $('location-status').textContent = `定位精度 ±${Math.round(position.coords.accuracy)} m`;
  if (courseId) { renderStudent(); void updateNavigation(position); }
};
sensors.onStatus = message => { $('sensor-status').textContent = message; notify(message); };
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
$('calibrate-compass').onclick = () => {
  const dialog = simpleDialog('校准手机朝向', `<p>在开阔处平放手机，将屏幕上沿对准已知的真北方向。默认真北为 0°；也可以输入已知地物的方位角。</p><label class="field-label">已知方位角（顺时针，度）<input id="known-bearing" type="number" min="0" max="359" value="0"></label><p id="compass-calibration-error" class="danger" role="status"></p><button id="confirm-compass" class="primary-button wide">已对准，校准当前方向</button><button id="clear-compass-calibration" class="text-button wide">恢复设备原始方向</button><p class="muted">校准只影响罗盘；行进方向仍由定位数据计算。远离金属和磁铁后可重新校准。</p>`);
  dialog.querySelector<HTMLButtonElement>('#confirm-compass')!.onclick = () => {
    try { sensors.calibrate(number('known-bearing')); dialog.close(); dialog.remove(); notify('已校准罗盘方向'); }
    catch (error) { $('compass-calibration-error').textContent = error instanceof Error ? error.message : String(error); }
  };
  dialog.querySelector<HTMLButtonElement>('#clear-compass-calibration')!.onclick = () => { sensors.clearCalibration(); dialog.close(); dialog.remove(); notify('已恢复设备方向'); };
};
$('open-help').onclick = () => simpleDialog('使用 DirMap', `<ol class="help-list"><li>课程地图、目标点、配准和通行图在站点构建时预计算，网页本身不需要后端服务。</li><li>在“原图校核”中核对目标编号和到访位置；修改保存在本机浏览器，可随时恢复构建时数据。</li><li>核对围墙、门洞、禁区与桥隧；在“通行核验”中描画修正后需重新生成路线。</li><li>填写人数、队数与课程规则，选择起终点并生成路线；小规模课程使用精确动态规划，大规模使用时间预算内的可行解。</li><li>发布前确认课程；静态版会下载课程 JSON 并保存在本机。把 JSON 放入 <code>web/public/data/releases/</code> 并更新 <code>releases.json</code> 后重新部署，其他设备即可打开课程链接。</li></ol><p class="muted">静态部署不支持重新识别、OCR、自动配准和上传新图；标准地图及校园参照 © OpenStreetMap contributors。自动匹配误差不等价于现场精度。手机传感器需 HTTPS 和浏览器权限。</p>`);
$('review-button').onclick = () => {
  if (!state || !plan?.routes.length) return;
  const suggested = plan.approaches?.filter(approach => approach.automatic) ?? [];
  const dialog = simpleDialog('发布前核验与生成静态课程', `<p class="muted">静态版没有服务器：核验结果与课程保存在本机浏览器，同时下载课程 JSON 供你提交到站点数据目录。</p><label class="check-label"><input id="review-registration" type="checkbox">配准位置与实际地物一致</label><label class="check-label"><input id="review-controls" type="checkbox">本次目标编号、圈心与到访位置已核对</label><label class="check-label"><input id="review-passages" type="checkbox">路线不穿越禁区、围墙及不可通行的层级</label><label class="field-label">核验备注<textarea id="review-notes" rows="3" placeholder="现场核验日期、临时封闭路段等"></textarea></label><p id="review-error" class="danger" role="status"></p><button id="publish-confirm" class="primary-button wide">生成学生课程链接</button><p class="muted compact">让其他设备访问：把下载的 <code>release-&lt;id&gt;.json</code> 放入 <code>web/public/data/releases/</code>，在 <code>web/public/data/releases.json</code> 的 <code>releases</code> 数组加入 <code>{"id":"…","title":"…","assetId":"…","createdAt":…}</code>，重新构建部署。</p>`);
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
      const start = startPixel!, finish = finishPixel ?? start;
      await saveEdits(controls, state!.edits);
      const request: PlanRequest = {
        assetId: activeId, version: state!.version,
        controlIds: controls.filter(control => control.required).map(control => control.id),
        startPixel: start, finishPixel: $<HTMLInputElement>('return-start').checked ? start : finish,
        mode: value('course-mode'), objective: value('objective'), ...rulePayload(),
        teams: previousPlan.routes.map(route => ({ id: route.teamId, people: route.people, speedMps: number('speed'), departureSec: route.departureSec })),
        timeLimitSec: number('solver-budget'),
      };
      const planner = await getPlanner(state!, progress);
      const result = await planner.plan(request, state!.controls, progress);
      showPlan(result);
      if (!result.routes.length) throw new Error('核验后有点位不可达，请重新检查');
      const releaseId = randomId32();
      const release: StaticRelease = {
        id: releaseId, title: '深大定向越野课', assetId: activeId, createdAt: Date.now() / 1000,
        plan: result, controls: state!.controls, frame: state!.frame, extent: state!.extent ?? [],
        quality: state!.quality, review: { at: Date.now() / 1000, static: true, notes: value('review-notes') },
        version: state!.version, routingSchema: __DIRMAP_ROUTING_SCHEMA__, superseded: false,
        static: true, overlay: `data/maps/${activeId}/overlay.png`,
      };
      saveLocalRelease(release);
      downloadRelease(release);
      const link = new URL(location.href); link.search = `?course=${releaseId}`; link.hash = '';
      $('share-result').innerHTML = `<label class="field-label">学生课程链接<input readonly value="${e(link.href)}" aria-label="学生课程链接"></label><a href="${e(link.href)}" target="_blank" rel="noreferrer" class="text-link">本机预览学生页面 ↗</a><p class="muted compact">已下载 <code>release-${releaseId}.json</code>；按提示放入静态数据目录后重新部署，链接即可在其他设备打开。${state!.edits.length ? '注意：本课程的通行修正只在本机存在，其他设备需要重新构建站点数据。' : ''}</p>`;
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
  if (next?.lnglat && lastPosition && Date.now() - lastPosition.timestamp < 15000) { const lat = next.lnglat[1] * Math.PI / 180; metres = Math.hypot((next.lnglat[0] - lastPosition.coords.longitude) * 111320 * Math.cos(lat), (next.lnglat[1] - lastPosition.coords.latitude) * 110700); }
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
    // The offline shell intentionally caches only the course package (release
    // JSON + overlay), not the multi-hundred-KiB map state/graph bundles.
    // Rendering, check-in and progress must still work offline; only the
    // off-route re-planner needs `state` and therefore degrades to the cached
    // route when it is unavailable.
    const baked = await loadMapState(course.assetId);
    state = baked;
    plan = course.plan; studentControls = course.controls; $('course-title').textContent = course.title;
    $('release-badge').textContent = course.offline ? '离线 · 状态未更新' : '课堂导航';
    $('student-team').innerHTML = plan.routes.map(route => `<option value="${e(route.teamId)}">${e(route.teamId)} · ${route.people} 人</option>`).join('');
    campusMap.setReleaseOverlay(course, .65); campusMap.setControls(course.controls); campusMap.setApproaches(plan.approaches ?? []);
    $<HTMLSelectElement>('student-team').dispatchEvent(new Event('change'));
    if (import.meta.env.PROD) {
      if (course.offline) $('offline-status').textContent = '正在使用已保存课程；联网后检查教师更新。';
      else void cacheStudentCourse();
    }
  } else {
    catalog = await loadCatalog();
    activeId = catalog.maps[0]?.id ?? activeId;
    updateStaticNotice(); renderCatalog();
    await loadState(activeId);
  }
}
async function cacheStudentCourse() {
  if (!courseId || courseSuperseded) return;
  $('offline-status').textContent = '正在保存原图、路线与目标…';
  try { await saveOfflineCourse(courseId); $('offline-status').textContent = '本课已保存；断网后可查看原图、路线并记录打点。'; }
  catch (error) { $('offline-status').textContent = error instanceof Error ? error.message : String(error); }
}
boot().catch(error => notify(error instanceof Error ? error.message : String(error), true));
