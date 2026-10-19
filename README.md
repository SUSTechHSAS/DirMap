# DirMap · 深大定向（静态部署版）

深圳大学定向越野课的教师工作台与手机导航网页。本仓库已从“Python/FastAPI 后端 + 浏览器前端”迁移为
**可直接部署在 GitHub Pages 的纯静态站点**：

- 地图识别、配准、通行图构建在**构建时**由 Python baker 预计算，产物提交在 `web/public/data/`；
- 教师规划与手机偏航重规划在**浏览器内**完成（TypeScript 移植的最短路、精确动态规划与限时启发式）；
- 页面运行时**没有任何 `/api` 请求，也不需要后端进程**；教师修改与已发布课程保存在本机浏览器。

详细模块契约见 [静态架构说明](docs/static-architecture.md)，部署步骤见 [DEPLOY.md](DEPLOY.md)。

## 快速开始

```bash
npm ci
npm run dev          # http://127.0.0.1:5173（Vite 直接服务已提交的静态数据）
```

构建与本地验证：

```bash
npx tsc --noEmit
npm test             # Node 单元测试 + 静态数据 golden 测试（48 项）
npm run build        # tsc + vite，产物在 web/dist
python3 tools/static_server.py --port 4173 --base /
# 打开 http://127.0.0.1:4173/ ；`npm run preview` 也可预览构建产物
```

浏览器端到端测试（Playwright，会自行构建并用 `tools/static_server.py` 提供服务）：

```bash
npx playwright install chromium   # 仅首次，或设置 DIRMAP_BROWSER_PATH
npm run test:browser              # 7 项：根路径 6 项 + /DIRMAP/ 子路径 1 项
```

## 部署到 GitHub Pages

1. 把仓库推送到 GitHub 的 `main` 分支。
2. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 之后的每次推送由 [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) 自动运行
   `npm ci → npm test → npm run build` 并发布 `web/dist`；项目站点自动使用 `DIRMAP_BASE=/<仓库名>/`。
4. 用户站点或自定义域名：把工作流里的 `DIRMAP_BASE` 改为 `/` 即可。

完整说明（含本地模拟子路径、发布课程、常见问题）见 [DEPLOY.md](DEPLOY.md)。

## 静态版功能

- **地图与配准结果**：三张真实定向图与两张图例；叠合三图、显示障碍候选、查看配准质量（P95）与原始扫描图。
- **教师工作台**：勾选目标、设置人数/队数/速度/错峰与四种访问方式（全队全点、全班分担、固定顺序、混合覆盖）；
  修改目标编号/类型/到访位置、描画通行修正。修改保存在浏览器 `localStorage`，可一键“恢复构建时数据”。
- **路线规划（浏览器内）**：
  - 小规模课程（`all_visit` ≤13 点等）使用与 Python 相同的 Held-Karp 精确动态规划，状态显示“模型内已证明最优”；
  - 大规模课程使用时间预算内的贪心 + 2-opt + 迁移启发式，状态如实显示“当前最佳可行”；
  - 支持服务时间、速度、错峰、混合覆盖的每队必到点与必到顺序；可导出路线 GeoJSON。
- **静态课程发布**：核验后生成课程 JSON 并保存在本机、同时下载 `release-<id>.json`；学生链接 `?course=<id>`
  可直接在本机预览。把该 JSON 放入 `web/public/data/releases/` 并在 `web/public/data/releases.json` 加入索引后重新部署，
  其他设备即可打开同一课程。
- **课堂导航**：选队、打点/撤销、本机进度保存；偏离路线时在浏览器内按本队剩余点顺序与发布路口重新规划；
  支持 iOS 罗盘/绝对方向、北向上、GPS 航向回退与罗盘校准。
- **离线**：Service Worker 缓存应用外壳与最近三份已保存课程（课程 JSON + 叠层 + 地图状态），断网可重载、查看与打点；
  离线时偏航重规划降级为保留已下载路线。

## 与后端版的差异与边界

| 项目 | 静态部署版 |
| --- | --- |
| 识别 / OCR / 自动配准 / 通行图重算 | 仅在构建时运行 `tools/bake_static.py`；页面内不再提供 |
| 同名点重校准、上传新图 | 不支持；需在本地重新处理数据后重跑 baker |
| 大班最优性证明 | 不运行 CP-SAT；小规模与 Python 完全一致（精确 DP），大规模为限时可行解 |
| 教师修改 | 仅保存在本机浏览器；跨设备共享需提交课程 JSON 并重新部署 |
| 偏航重规划 | 浏览器内计算；离线时不可用（缺少通行图 bundle），保留已下载路线 |
| 服务端任务队列 / 教师口令 / HTTPS 反向代理 | 已移除（静态托管自带 HTTPS） |

自动配准精度沿用构建时结果（图 1/2/3 留出误差 P95 约 6.43 / 1.66 / 8.49 m）；图 2 的 35 号点仍按已确认规则
保留在禁入区、31 号点通道待核验，因此完整图 2 课程会如实返回“需要核对”的未解决位置。这些限制没有因静态迁移而改变。

## 仓库结构

```
web/                    浏览器应用（Vite + TypeScript + OpenLayers）
  src/                  main / map / editor / planner / raster-graph / optimizer / geometry …
  public/data/          ★ 构建时生成的静态数据（约 20 MB，随仓库提交）
  sw.js                 Service Worker（离线外壳与课程缓存）
dirmap/                 构建时 Python 库（识别、配准、通行图、优化器；不再包含 FastAPI 服务）
tools/
  bake_static.py        ★ 把 data/projects 的项目数据转换成 web/public/data
  verify_static_data.py 静态数据完整性与图结构校验（12277 项）
  prepare_maps.py       重新处理三张原图（可选，需 OCR 时另装 Tesseract）
  static_server.py      本地/Pages 模拟静态服务器（支持 --base 子路径）
tests/
  *.test.ts             Node 单元测试（朝向、导航工具）
  static/               浏览器规划核心 golden 测试（37 项）
  fixtures/static/      由 Python baker 生成的 golden 数据
  browser/              Playwright 端到端测试（7 项）
  test_*.py             Python 构建库单元测试
.github/workflows/       GitHub Pages 部署工作流
docs/static-architecture.md  迁移架构与模块契约
DEPLOY.md               部署与发布课程说明
design/ evidence/       历史设计与后端阶段的验证记录（保留为设计依据）
```

## 静态数据与重新构建

`web/public/data/` 已提交，CI 不需要 Python。只有在需要更新识别/配准/通行结果时才重新运行 baker：

```bash
npm run setup:bake    # python3 -m pip install --target .pydeps -r requirements.txt
npm run bake          # Python baker，约 20 秒
PYTHONPATH=.pydeps python3 tools/verify_static_data.py
```

baker 读取仓库根目录的三张原图、`data/reference/` 的校园参照与 `data/projects/` 的本地处理状态
（`data/projects/` 已被 `.gitignore` 忽略，是本地工作数据）。完整重建三张图可先运行
`python3 -B tools/prepare_maps.py`（需要 OpenCV 环境；编号 OCR 可选安装 Tesseract）。当前静态数据由同一份
源数据确定性生成：连续两次完整 bake 的 38 个文件字节一致。

## 验证记录（本次迁移，2026-10-19）

| 检查 | 结果 |
| --- | --- |
| 静态数据校验 `tools/verify_static_data.py` | 12277 项全部通过 |
| Python 构建库测试 | 62 passed / 1 skipped；2 项 `test_overprint` 失败在迁移前的原始检出中同样存在，与本次迁移无关 |
| Node 单元 + golden 测试 `npm test` | 48 passed / 0 failed（11 项工具测试 + 37 项规划 golden） |
| 浏览器 E2E `npm run test:browser` | 7 passed / 0 failed（根路径 6 + `/DIRMAP/` 子路径 1），离线、发布、重规划均覆盖 |
| 构建与子路径 | `npm run build`、`DIRMAP_BASE=/DIRMAP/ npm run build` 均通过；`dist/index.html`、`sw.js`、数据 URL 全部带子路径前缀 |
| 浏览器内性能（szu-3） | 通行图构建 1.39 s；19×19 通行时间矩阵 2.27 s；教师规划 szu-1 全队全点 1.16 s；学生偏航重规划 1.55 s |
| 规划一致性 | szu-1 三个队伍的点序与 Python golden 完全一致（状态 OPTIMAL）；szu-3 重规划路线距离与发布路线相差 0.008% |

## 来源与许可

- 校园参照 © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)，遵循 ODbL。
- 三张原图与两张图例保持原文件不变（仓库根目录）。
- 标准地图瓦片、在线底图与字体使用外部 CDN；离线时使用系统字体与已缓存数据。
- 原实施方案、接口草案与验收计划保留在 `design/`；后端阶段的实施状态见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)（历史文档）。
