# DirMap · 深大定向（静态部署版）

深圳大学定向越野课的教师工作台与手机导航网页。本仓库已从“Python/FastAPI 后端 + 浏览器前端”迁移为
**可直接部署在 GitHub Pages 的纯静态站点**：

- 地图识别、配准、通行图构建、同名点校准与**导入新图**全部在**浏览器内**运行（opencv.js WASM + tesseract.js OCR）；
- 教师规划、发布、学生导航与偏航重规划也在浏览器内完成（TypeScript 移植的最短路、精确动态规划与限时启发式）；
- 构建时 Python baker 仍会预生成三张内置图的结果作为**快速默认值**并作为浏览器管线的 golden 基准；
- 页面运行时**没有任何 `/api` 请求，也不需要后端进程**；教师修改、本机处理结果与已发布课程保存在本机浏览器。

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
npm run test:browser              # 12 项：根路径 11 项 + /DIRMAP/ 子路径 1 项
```

## 部署到 GitHub Pages

1. 把仓库推送到 GitHub 的 `main` 分支。
2. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 之后的每次推送由 [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) 自动运行
   `npm ci → npm test → npm run build` 并发布 `web/dist`；项目站点自动使用 `DIRMAP_BASE=/<仓库名>/`。
4. 用户站点或自定义域名：把工作流里的 `DIRMAP_BASE` 改为 `/` 即可。

完整说明（含本地模拟子路径、发布课程、常见问题）见 [DEPLOY.md](DEPLOY.md)。

## 静态版功能

- **浏览器内地图处理**：自动识别控制圈/编号/起终点、地物匹配与配准、通行区域识别、更新通行识别、同名点重校准，
  以及**上传任意 JPEG/PNG 新图**后自动处理；处理结果显示配准质量（P95）、障碍候选与原始扫描图。首次处理时懒加载
  `vendor/opencv.js`（约 11 MB）与 OCR 资源（tesseract.js CDN，离线时编号保留待核，不阻塞其余流程）。
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
| 识别 / OCR / 自动配准 / 通行图重算 / 同名点校准 / 上传新图 | 全部在浏览器内运行（WebAssembly）；无服务器进程 |
| 构建时数据 | 三张内置图的识别与配准结果预生成，作为快速默认值；也可以随时在本机重新处理 |
| 大班最优性证明 | 不运行 CP-SAT；小规模与 Python 完全一致（精确 DP），大规模为时间预算内的可行解 |
| 处理性能与内存 | 桌面浏览器处理 szu-2 约 13–16 秒、szu-1 约 38 秒（含 OCR）；6.6 MP 的 szu-3 峰值内存较高，建议桌面处理大图 |
| 教师修改 / 本机处理结果 | 保存在本机 IndexedDB/localStorage；可一键删除本机处理结果恢复构建时数据 |
| 跨设备分享 | 课程 JSON 可下载；分享**本机处理的地图**（上传图或重新处理结果）需把处理数据纳入构建数据后重新部署（静态托管的固有限制） |
| 偏航重规划 | 浏览器内计算；离线时缺少通行图 bundle 则保留已下载路线 |
| 服务端任务队列 / 教师口令 / HTTPS 反向代理 | 已移除（静态托管自带 HTTPS） |

自动配准精度沿用构建时结果（图 1/2/3 留出误差 P95 约 6.43 / 1.66 / 8.49 m）；图 2 的 35 号点仍按已确认规则
保留在禁入区、31 号点通道待核验，因此完整图 2 课程会如实返回“需要核对”的未解决位置。这些限制没有因静态迁移而改变。

## 仓库结构

```
web/                    浏览器应用（Vite + TypeScript + OpenLayers）
  src/                  main / map / editor / planner / raster-graph / optimizer / geometry
                        cv / vision / ocr / image-registration / access / processing / workspace-store
                        （浏览器内 CV 管线：opencv.js WASM + tesseract.js）
  public/data/          ★ 构建时生成的静态数据（约 32 MB，随仓库提交）
  public/vendor/        构建时从 node_modules 复制的 opencv.js（不提交，prepare 脚本生成）
  sw.js                 Service Worker（离线外壳与课程缓存）
dirmap/                 构建时 Python 库（识别、配准、通行图、优化器；不再包含 FastAPI 服务）
tools/
  bake_static.py        ★ 把 data/projects 的项目数据转换成 web/public/data
  verify_static_data.py 静态数据完整性与图结构校验（64678 项）
  vendor-assets.mjs     把 opencv.js 复制到 web/public/vendor（npm prepare/prebuild 自动运行）
  prepare_maps.py       用 Python 重新处理三张原图（可选；浏览器管线通常已足够）
  static_server.py      本地/Pages 模拟静态服务器（支持 --base 子路径）
tests/
  *.test.ts             Node 单元测试（朝向、导航工具）
  static/               浏览器规划核心 golden 测试（37 项）
  client/               浏览器 CV/配准/到访点 golden 测试（21+ 项，Node + opencv.js WASM）
  fixtures/static/      由 Python baker 生成的 golden 数据（含 client-processing.json）
  browser/              Playwright 端到端测试（12 项：浏览器内处理/上传/校准/发布/离线）
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
（`data/projects/` 已被 `.gitignore` 忽略，是本地工作数据），并生成浏览器管线的输入：
`original.png`（与 Python 识别输入逐像素一致的无损原图）、`semantics.bin.gz`（语义掩膜）、
`reference-features.json`（校园参照特征）与 `tests/fixtures/static/client-processing.json` golden。
完整重建三张图可先运行 `python3 -B tools/prepare_maps.py`（需要 OpenCV 环境；编号 OCR 可选安装 Tesseract）。
当前静态数据由同一份源数据确定性生成：连续两次完整 bake 的 46 个文件字节一致。
`npm run setup:bake` 安装构建时 Python 依赖；CI 不需要 Python（数据已提交）。

## 验证记录（本次迁移，2026-10-19）

| 检查 | 结果 |
| --- | --- |
| 静态数据校验 `tools/verify_static_data.py` | 64678 项全部通过（含 original.png 逐像素、semantics 全掩膜、client golden 复算） |
| Python 构建库测试 | 62 passed / 1 skipped；2 项 `test_overprint` 失败在迁移前的原始检出中同样存在，与本次迁移无关 |
| Node 单元 + golden 测试 `npm test` | 83 项中 80 passed / 0 failed / 3 skipped（3 项为 heavy/OCR 开关用例；覆盖工具、规划 golden、浏览器 CV/配准/到访点 golden） |
| 浏览器 E2E `npm run test:browser` | 12 passed / 0 failed（3.1–3.4 min；根路径 11 + `/DIRMAP/` 子路径 1），覆盖浏览器内识别/上传/通行刷新/同名点校准/本机发布/学生重规划/离线 |
| 构建与子路径 | `npm run build`、`DIRMAP_BASE=/DIRMAP/ npm run build` 均通过；`dist/index.html`、`sw.js`、数据 URL 全部带子路径前缀 |
| 浏览器管线 golden（Node WASM，对照当前 Python 参考实现） | 控制点像素误差 0.000 px；地物计数全部一致；blocked 一致率 100.000/99.967/100.000%；OCR 编号 11/11、7/7 与 Python 一致 |
| 浏览器内性能（桌面 Chrome） | szu-2 处理（含 OCR）13–16 s；szu-1 处理约 38 s；更新通行识别约 4 s；导入新图约 14 s；教师规划 szu-1 2.8 s；学生偏航重规划 1.55 s；szu-3 通行图构建 1.39 s / 19×19 矩阵 2.27 s |
| 规划一致性 | szu-1 三个队伍的点序与 Python golden 完全一致（状态 OPTIMAL）；szu-3 重规划路线距离与发布路线相差 0.008% |

## 来源与许可

- 校园参照 © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)，遵循 ODbL。
- 三张原图与两张图例保持原文件不变（仓库根目录）。
- 标准地图瓦片、在线底图与字体使用外部 CDN；离线时使用系统字体与已缓存数据。
- 原实施方案、接口草案与验收计划保留在 `design/`；后端阶段的实施状态见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)（历史文档）。
