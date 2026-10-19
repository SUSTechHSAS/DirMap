import { escapeHtml, type Approach, type Control, type MapState, type Pixel, type TerrainEdit } from './types';
import { assetUrl } from './data';

type Mode = 'inspect' | 'control' | 'approach' | 'open' | 'close' | 'anchor';
export class PixelEditor {
  private dialog = document.createElement('dialog');
  private canvas!: HTMLCanvasElement;
  private image = new Image();
  private terrain = new Image();
  private controls: Control[];
  private edits: TerrainEdit[];
  private mode: Mode = 'inspect';
  private selected: string | null = null;
  private stroke: Pixel[] | null = null;
  private zoom = 1;
  private baseWidth = 800;
  private showTerrain = false;
  private history: { controls: Control[]; edits: TerrainEdit[] }[] = [];

  constructor(private state: MapState, private onSave: (controls: Control[], edits: TerrainEdit[]) => Promise<void>, private onAnchor: ((pixel: Pixel) => void) | null, private suggestions: Approach[] = []) {
    this.controls = structuredClone(state.controls);
    this.edits = structuredClone(state.edits);
  }
  open(mode: Mode = 'inspect', selected: string | null = null) {
    this.mode = mode; this.selected = selected;
    this.dialog.className = 'editor-dialog';
    this.dialog.innerHTML = `<div class="dialog-heading"><div><span class="eyebrow">原图校核</span><h2>${escapeHtml(this.state.name)}</h2></div><button data-close class="icon-button" aria-label="关闭原图">×</button></div>
      <div class="editor-toolbar"><select id="edit-mode" aria-label="编辑操作"><option value="inspect">检查 / 选择目标</option><option value="control">新增目标</option><option value="approach">选择目标到访位置</option><option value="close">标记封闭路段</option><option value="open">确认开放通道</option><option value="anchor">添加配准同名点</option></select>
      <label class="brush-label">笔刷 <input id="brush-size" type="number" value="6" min="1" max="100" aria-label="笔刷半径"></label><button id="undo-edit" class="soft-button">撤销</button><button id="zoom-out" class="icon-button" aria-label="缩小原图">−</button><button id="zoom-in" class="icon-button" aria-label="放大原图">＋</button><label class="check-label"><input id="editor-terrain" type="checkbox">识别障碍</label></div>
      <p id="editor-hint" class="muted editor-hint" role="status"></p><div class="editor-viewport"><canvas aria-label="原图及目标点编辑画布"></canvas></div>
      <div class="editor-footer"><span id="edit-selection" class="muted">点击目标可修改编号、删除或确认</span><button id="save-edits" class="primary-button">保存修改</button></div>`;
    document.body.append(this.dialog); this.dialog.showModal();
    this.canvas = this.dialog.querySelector('canvas')!;
    this.canvas.width = this.state.width; this.canvas.height = this.state.height;
    this.baseWidth = Math.min(this.state.width, Math.max(300, this.dialog.clientWidth - 40)); this.resize();
    const select = this.dialog.querySelector<HTMLSelectElement>('#edit-mode')!;
    if (!this.onAnchor) select.querySelector('option[value="anchor"]')?.remove();
    select.value = mode;
    select.onchange = () => { this.mode = select.value as Mode; this.hint(); };
    this.dialog.querySelector<HTMLButtonElement>('[data-close]')!.onclick = () => this.close();
    this.dialog.addEventListener('cancel', () => this.close());
    this.dialog.querySelector<HTMLButtonElement>('#zoom-in')!.onclick = () => { this.zoom = Math.min(5, this.zoom * 1.4); this.resize(); };
    this.dialog.querySelector<HTMLButtonElement>('#zoom-out')!.onclick = () => { this.zoom = Math.max(.7, this.zoom / 1.4); this.resize(); };
    this.dialog.querySelector<HTMLButtonElement>('#undo-edit')!.onclick = () => { const previous = this.history.pop(); if (previous) { this.controls = previous.controls; this.edits = previous.edits; this.draw(); this.selection(); } };
    this.dialog.querySelector<HTMLInputElement>('#editor-terrain')!.onchange = event => { this.showTerrain = (event.target as HTMLInputElement).checked; this.draw(); };
    this.canvas.onpointerdown = event => this.pointerDown(event);
    this.canvas.onpointermove = event => { if (this.stroke) { this.stroke.push(this.pixel(event)); this.draw(); } };
    this.canvas.onpointerup = () => this.endStroke();
    this.canvas.onpointercancel = () => this.endStroke();
    this.image.onload = () => this.draw(); this.image.src = assetUrl(this.state.assetId, 'original.jpg');
    this.terrain.onload = () => this.draw(); this.terrain.src = assetUrl(this.state.assetId, 'terrain.png', this.state.version);
    this.dialog.querySelector<HTMLButtonElement>('#save-edits')!.onclick = async () => {
      const button = this.dialog.querySelector<HTMLButtonElement>('#save-edits')!; button.disabled = true;
      try { await this.onSave(this.controls, this.edits); this.close(); }
      catch (error) { this.dialog.querySelector('#editor-hint')!.textContent = String(error); button.disabled = false; }
    };
    this.hint(); this.selection();
  }
  private remember() { this.history.push({ controls: structuredClone(this.controls), edits: structuredClone(this.edits) }); }
  private resize() { this.canvas.style.width = `${this.baseWidth * this.zoom}px`; this.canvas.style.height = 'auto'; }
  private pixel(event: PointerEvent): Pixel {
    const rect = this.canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(this.state.width - 1, (event.clientX - rect.left) / rect.width * this.state.width)), Math.max(0, Math.min(this.state.height - 1, (event.clientY - rect.top) / rect.height * this.state.height))];
  }
  private pointerDown(event: PointerEvent) {
    const point = this.pixel(event);
    if (this.mode === 'anchor') { this.onAnchor?.(point); this.close(); return; }
    if (this.mode === 'open' || this.mode === 'close') { this.remember(); this.canvas.setPointerCapture(event.pointerId); this.stroke = [point]; this.canvas.style.touchAction = 'none'; return; }
    if (this.mode === 'control') {
      this.remember(); const id = `manual-${crypto.randomUUID().slice(0, 8)}`;
      this.controls.push({ id, label: `新点${this.controls.length + 1}`, pixel: point, required: true, radiusPx: this.state.width / 60, state: 'accepted', serviceBaseSec: 5, servicePerPersonSec: 1 });
      this.selected = id;
    } else if (this.mode === 'approach') {
      const control = this.controls.find(c => c.id === this.selected);
      if (control) { this.remember(); control.approachPixel = point; control.state = 'accepted'; }
      else this.dialog.querySelector('#editor-hint')!.textContent = '请先切到“检查”，点击需要设置到访位置的目标。';
    } else {
      let nearest = this.controls[0], distance = Infinity;
      for (const c of this.controls) { const d = Math.hypot(c.pixel[0] - point[0], c.pixel[1] - point[1]); if (d < distance) { nearest = c; distance = d; } }
      this.selected = nearest && distance < Math.max(nearest.radiusPx * 1.7, 25) ? nearest.id : null;
    }
    this.draw(); this.selection();
  }
  private endStroke() {
    if (!this.stroke) return;
    this.edits.push({ kind: this.mode as 'open' | 'close', pixels: this.stroke, radiusPx: Math.max(1, Math.min(100, Number(this.dialog.querySelector<HTMLInputElement>('#brush-size')!.value) || 6)), cost: 1 });
    this.stroke = null; this.draw();
  }
  private hint() {
    const messages: Record<Mode, string> = { inspect: '点击圈心选择目标。缩放后可核对细墙、门洞和圈内实际位置。', control: '点击原图新增一个目标；随后修改编号。', approach: '点击实际可以到访的位置，保留原圈心。请确认与目标在围墙同侧。', open: '沿已确认可通行的门洞或路径描画。仅修改教师明确核验的区域。', close: '沿围墙、施工区或课程禁行线描画。保存后旧路线立即失效。', anchor: '先点击原图中的建筑角或路口，再到标准地图点击同一个位置。' };
    this.dialog.querySelector('#editor-hint')!.textContent = messages[this.mode];
    this.canvas.style.touchAction = ['open', 'close'].includes(this.mode) ? 'none' : 'auto';
  }
  private selection() {
    const el = this.dialog.querySelector('#edit-selection')!;
    const control = this.controls.find(c => c.id === this.selected);
    if (!control) { el.textContent = '点击目标可修改编号、删除或确认'; return; }
    const suggestion = this.suggestions.find(a => a.controlId === control.id && a.automatic);
    el.innerHTML = `<label>编号 <input id="control-label" value="${escapeHtml(control.label)}" maxlength="40"></label><label>类型 <select id="control-role"><option value="control">目标</option><option value="start">起点</option><option value="finish">终点</option><option value="start_finish">起终点</option></select></label><button id="confirm-control" class="soft-button">确认目标</button>${suggestion ? `<button id="accept-approach" class="soft-button">采用建议到访位置（距圈心 ${suggestion.offsetM.toFixed(1)} m）</button>` : ''}<button id="delete-control" class="text-button danger">删除</button>`;
    const role = el.querySelector<HTMLSelectElement>('#control-role')!; role.value = control.role ?? 'control';
    role.onchange = () => { this.remember(); control.role = role.value as Control['role']; control.required = control.role === 'control'; this.draw(); };
    el.querySelector<HTMLButtonElement>('#accept-approach')?.addEventListener('click', () => {
      this.remember(); control.approachPixel = [...suggestion!.pixel]; control.state = 'accepted'; this.draw();
      this.dialog.querySelector('#editor-hint')!.textContent = '已采用到访位置，原圈心保持原位。保存后重新规划。';
    });
    el.querySelector<HTMLInputElement>('#control-label')!.onchange = event => { control.label = (event.target as HTMLInputElement).value.trim() || control.label; this.draw(); };
    el.querySelector<HTMLButtonElement>('#confirm-control')!.onclick = () => { this.remember(); control.state = 'accepted'; this.draw(); this.dialog.querySelector('#editor-hint')!.textContent = '目标已确认，点击“保存修改”提交。'; };
    el.querySelector<HTMLButtonElement>('#delete-control')!.onclick = () => { this.remember(); this.controls = this.controls.filter(c => c.id !== control.id); this.selected = null; this.draw(); this.selection(); };
  }
  private draw() {
    const ctx = this.canvas.getContext('2d')!; ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.image.complete && this.image.naturalWidth) ctx.drawImage(this.image, 0, 0, this.state.width, this.state.height);
    if (this.showTerrain && this.terrain.complete && this.terrain.naturalWidth) ctx.drawImage(this.terrain, 0, 0, this.state.width, this.state.height);
    const strokes = [...this.edits, ...(this.stroke ? [{ kind: this.mode, pixels: this.stroke, radiusPx: Number(this.dialog.querySelector<HTMLInputElement>('#brush-size')!.value) || 6 }] : [])];
    for (const edit of strokes) { ctx.strokeStyle = edit.kind === 'close' ? '#ef4444aa' : '#16a077aa'; ctx.fillStyle = ctx.strokeStyle; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = edit.radiusPx * 2;
      if (edit.pixels.length === 1) { ctx.beginPath(); ctx.arc(...edit.pixels[0], edit.radiusPx, 0, Math.PI * 2); ctx.fill(); }
      else { ctx.beginPath(); edit.pixels.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); } }
    for (const c of this.controls) {
      ctx.strokeStyle = c.id === this.selected ? '#ed6c35' : c.state === 'accepted' ? '#11815f' : '#a22bbe'; ctx.lineWidth = Math.max(2, this.state.width / 350); ctx.beginPath(); ctx.arc(...c.pixel, c.radiusPx, 0, Math.PI * 2); ctx.stroke();
      ctx.font = `bold ${Math.max(18, this.state.width / 60)}px sans-serif`; ctx.strokeStyle = '#fff'; ctx.lineWidth = 4; ctx.strokeText(c.label, c.pixel[0] + c.radiusPx, c.pixel[1]); ctx.fillStyle = '#7d238b'; ctx.fillText(c.label, c.pixel[0] + c.radiusPx, c.pixel[1]);
      if (c.approachPixel) { ctx.strokeStyle = '#0a8966'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(...c.pixel); ctx.lineTo(...c.approachPixel); ctx.stroke(); ctx.fillStyle = '#0a8966'; ctx.beginPath(); ctx.arc(...c.approachPixel, 6, 0, Math.PI * 2); ctx.fill(); }
    }
    for (const a of this.suggestions) {
      if (!a.automatic || this.controls.find(c => c.id === a.controlId)?.approachPixel) continue;
      ctx.strokeStyle = '#e58824'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(...a.pixel, 6, 0, Math.PI * 2); ctx.stroke();
    }
  }
  close() { this.dialog.close(); this.dialog.remove(); }
}
