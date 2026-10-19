# DirMap 静态版部署指南（GitHub Pages）

本文件说明如何把 **纯静态版** DirMap 发布到 GitHub Pages、如何在本地验证、如何更新烘焙数据、以及教师如何把课程发布到静态站点。
应用功能、开发方式和数据来源见 [README.md](README.md)，静态化架构契约见 [docs/static-architecture.md](docs/static-architecture.md)，本文件不重复它们。

静态版没有服务器：所有数据都是 `web/public/data/**` 里的普通文件，规划在浏览器内完成，教师编辑保存在浏览器 localStorage。

## 1. 部署结构

- 构建产物 **`web/dist/`** 就是整个网站：`index.html`、带哈希的 `assets/*`、`sw.js`，以及从 `web/public/**` 原样复制的文件（包括 `data/**` 烘焙数据和 `.nojekyll`）。
- [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) 在 **push 到 `main`** 或 **手动 `workflow_dispatch`** 时构建并部署，不需要 `gh-pages` 分支。
- 站点地址：
  - 项目站点 `https://<user>.github.io/<repo>/` → 构建 base 为 `/<repo>/`（workflow 默认）；
  - 用户站点 `https://<user>.github.io/` 或自定义域名 → base 为 `/`。
- 构建时 Vite 会把 base 前缀写进 `index.html` 的资源 URL；`web/sw.js` 在运行时从 `self.registration.scope` 推导自己的 base，因此同一份源码同时支持根路径和子路径。

## 2. 一次性仓库设置

1. 打开仓库 **Settings → Pages → Build and deployment**，把 **Source** 设为 **GitHub Actions**。
2. 仓库 **Settings → Actions → General** 保持默认权限即可；workflow 已声明最小权限（`contents: read`、`pages: write`、`id-token: write`）。
3. 如需绑定自定义域名：在 **Settings → Pages → Custom domain** 填写域名，并把 workflow 中 `Build static site` 步骤的 `DIRMAP_BASE` 改成 `/`（详见第 4 节）。
4. 若仓库来自 fork，需在 fork 中同样启用 Pages；首次部署成功后，Pages 页面会显示站点地址。

## 3. 自动部署流程

`deploy.yml` 分为两个 job：

- **build**：`actions/checkout@v4` → `actions/setup-node@v4`（Node 24，npm 缓存）→ `npm ci` → `npm test` → `DIRMAP_BASE=/<repo>/ npm run build` → `actions/configure-pages@v5` → `actions/upload-pages-artifact@v3`（`path: web/dist`）。
- **deploy**：`needs: build`，环境 `github-pages`，用 `actions/deploy-pages@v4` 发布。部署 URL 显示在这次运行的 environment 中。

并发组为 `pages`，同一时间只允许一个部署且不会中途取消。

- **base 前缀自动取仓库名**：构建步骤使用 `DIRMAP_BASE=/${{ github.event.repository.name }}/`，所以项目站点无需手工修改；只有用户站点或自定义域名才需要改成 `/`（见第 4 节）。
- CI **只运行 Node 构建和测试**（`npm ci` → `npm test` → `npm run build`），不会运行 Python baker，也不会执行 `tools/verify_static_data.py`；因此 `web/public/data/**` 必须已经提交并保持最新（见第 6、7 节）。

## 4. base path（子路径）处理

- Vite 配置 `web/vite.config.ts` 读取环境变量 **`DIRMAP_BASE`**，缺省为 `/`；输入会被规范化为以 `/` 开头和结尾（例如 `DirMap` → `/DirMap/`）。
- 该 base 同时作用于 `npm run dev`、`npm run build` 和 `npm run preview`。
- 应用代码一律通过 `import.meta.env.BASE_URL`（`web/src/data.ts` 的 `dataUrl()` 等）拼接数据 URL，禁止写死 `/data/...`。
- 服务 worker 注册为 `${BASE_URL}sw.js`，scope 为 `${BASE_URL}`，并从 `self.registration.scope` 推导缓存用的前缀。

常用构建：

```bash
npm run build                       # 根路径部署（用户站点 / 自定义域 / 本机）
DIRMAP_BASE=/DirMap/ npm run build  # 仓库子路径（把 DirMap 换成实际仓库名）
```

构建后检查 base 是否生效：

```bash
grep -o 'src="[^"]*"' web/dist/index.html
grep -o 'href="[^"]*\.css"' web/dist/index.html
# 子路径构建应看到 /DirMap/assets/...；根路径构建应看到 /assets/...
grep -o 'dirmap-shell-[0-9a-f]*' web/dist/sw.js   # 构建指纹已写入
```

## 5. 本地验证（不依赖 GitHub）

```bash
npm ci
npm test
npm run build                      # 或 DIRMAP_BASE=/DirMap/ npm run build
python3 tools/static_server.py --dir web/dist --base /DirMap/ --port 4173
```

然后访问 <http://127.0.0.1:4173/DirMap/>。`npm test` 只运行 Node 单元测试（包括 `tests/static/*.test.ts` 的通行图/规划 golden 测试），**不需要 Python**；只有重新生成或校验静态数据时才需要 Python baker（第 6 节）。

`tools/static_server.py` 是零依赖的验证服务器，模拟 Pages 的关键行为：剥离 base 前缀、只服务 `--dir` 下的文件、导航请求回退 `index.html`、正确 Content-Type、HTML/JSON/`sw.js` 用 `no-cache`、哈希资源用 immutable、支持 HEAD、无目录列表。

```bash
curl -sI http://127.0.0.1:4173/DirMap/                                  # 200 text/html, Cache-Control: no-cache
curl -s  http://127.0.0.1:4173/DirMap/ | head -n 5                     # 打印 index.html
curl -sI http://127.0.0.1:4173/DirMap/sw.js                            # 200 text/javascript
curl -s  http://127.0.0.1:4173/DirMap/data/catalog.json | head -c 200  # 200 application/json（需已烘焙并构建）
curl -sI http://127.0.0.1:4173/DirMap/no-such-page                     # 200 + index.html（导航回退）
curl -sI -H 'Accept: application/json' http://127.0.0.1:4173/DirMap/no-such-page  # 404
```

其它参数：`--port 0` 让系统分配端口（启动行会打印实际端口），`--quiet` 关闭逐请求日志，`--host` 修改绑定地址。
`npm run preview`（Vite 预览，默认 127.0.0.1:4173）用于快速查看构建产物，和 `static_server.py` 的默认端口相同，二者选其一即可。

浏览器端回归测试（可选，需已安装 Playwright Chromium）：`npm run test:browser`。

## 6. 更新烘焙数据

`data/projects/**`、`data/course-presets.json` 等是源数据；**`web/public/data/**` 是 baker 生成的静态数据，需要提交到仓库**。CI 不运行 Python，只做 Node 构建/测试，所以让 `web/public/data/**` 保持最新是提交者的责任。重新生成：

```bash
PYTHONPATH=.pydeps DIRMAP_CACHE_DIR=$PWD/.cache python3 tools/bake_static.py
PYTHONPATH=.pydeps python3 tools/verify_static_data.py
npm run build
```

- 只重烘一张图：追加 `--asset szu-3`；快速迭代可加 `--skip-fixtures`。
- baker 是幂等且确定性的，但 `generatedAt` 会变化，属正常现象。
- 生成的文件布局见 [docs/static-architecture.md](docs/static-architecture.md) §4：`catalog.json`、`health.json`、`reference.json`、`releases.json`、`releases/<id>.json`、`maps/<assetId>/*`、`legends/*`。
- 提交 `web/public/data/**` 后 push 到 `main`，Actions 会重新构建并部署。**不要手改生成文件**；磁偏角表过期时重新运行 baker 即可。

## 7. 教师发布课程后如何上线静态站

静态站没有“上传课程”接口，发布 = 提交两个文件并 push：

1. 教师在应用里核验课程并点击“发布”，浏览器会下载 `release-<id>.json`（`id` 为 32 位小写十六进制）。
2. 把文件放到 **`web/public/data/releases/<id>.json`**（文件名必须与 `id` 完全一致）。
3. 在 **`web/public/data/releases.json`** 的 `releases` 数组中增加一条索引（`createdAt` 为毫秒时间戳）：

   ```json
   { "id": "0123456789abcdef0123456789abcdef", "title": "深大定向越野课", "assetId": "szu-3", "createdAt": 1760000000000 }
   ```

4. 提交并 push 到 `main`，等待 Actions 部署完成。

说明：

- 学生刷新课程列表即可看到新课；`release.id`、`assetId` 必须与文件内容和地图一致，否则会报“无法载入课程”。
- 教师本地“发布”的课程只存在当前浏览器的 localStorage，并可随时下载 JSON；不会自动同步到其它设备。
- 若课程 JSON 含 `"superseded": true`，service worker 会拒绝离线保存（撤回的课程不会因断网重新可用）。
- 如需撤回课程：删除 `data/releases/<id>.json`、从 `releases.json` 移除索引，重新部署；已离线保存的学生设备会保留旧副本，联网后不再更新。

## 8. 静态版的功能边界

- **不支持**：OCR/图片识别、地形重新分析、人工重新配准、上传新的地图扫描件、服务端作业队列、跨设备状态同步。
- **路线规划**：在浏览器内完成，并需要首次在线下载 `graph.bin.gz` 等数据。CP-SAT 已移除：小规模精确 DP 保持最优证明，其余情况为限时启发式 `FEASIBLE`（`proof.globalOptimalityProven: false`），界面上不会再显示“已证明最优”。
- **教师编辑**：目标点、标签、地形开闭、课程规则等改动只保存在本机 localStorage；可一键“恢复构建时数据”。这些编辑不会写回静态文件。
- **手机导航**：定位/罗盘能力与浏览器和 HTTPS 有关；静态站部署在 GitHub Pages 上即 HTTPS，但离线时无法获取教师更新（会显示离线标记）。

## 9. 隐私与离线说明

- 静态站是**公开文件**：`web/public/data/**`（含烘焙的课程、地图、队伍/路线规划）对任何访问者可见。不要把学生名单、个人信息、未公开成绩等隐私数据放进仓库；本机 localStorage 和下载的 JSON 才适合保存私密内容。
- 没有后端、账号、Cookie 或统计脚本；除按需加载的 OpenStreetMap 瓦片（跨域，service worker 不拦截、不缓存）外，不向第三方发送请求。
- Service worker 仅在 HTTPS 或 localhost 下注册（`npm run dev` 不注册）。它会缓存**应用外壳**和最多 **3 门**“保存离线”课程（release JSON + 地图 overlay + `health.json`/`reference.json`），见 [`web/sw.js`](web/sw.js)；`/assets/*` 哈希资源 cache-first，导航 network-first 回退缓存首页，数据文件 network-first 并在离线时给 JSON 加 `offline: true` / 响应头 `X-DirMap-Offline: 1`。
- 原图扫描件、`graph.bin.gz`、`state.json` 等大文件**不在** SW 的保存课程清单内，依赖浏览器 HTTP 缓存；缓存被清理后需联网重新加载。
- 清除站点数据/浏览器存储会删除离线课程和本机教师编辑；浏览器可能因空间不足回收缓存，离线前请确认已显示“本课已保存”。

## 10. 故障排查

| 现象 | 处理 |
| --- | --- |
| 页面白屏、资源 404 | base 与部署路径不一致。确认 workflow 的 `DIRMAP_BASE` 与 Pages 站点类型，检查 `web/dist/index.html` 里的资源前缀（第 4 节）。 |
| Actions 失败：Pages 未启用 | 完成 Settings → Pages → Source: GitHub Actions（第 2 节）。 |
| 部署成功但页面仍是旧版 | 刷新页面（SW 更新需一次联网访问，已设置 `updateViaCache: 'none'`）；必要时在 DevTools → Application → Service Workers 选择 Update/Unregister。 |
| 课程列表为空 | 检查 `web/public/data/releases.json` 是否存在、索引是否合法，以及 `/DIRMAP/data/releases.json` 返回 200。 |
| 离线保存失败 | 需要 HTTPS/localhost、首次保存时联网，且课程未被标记 `superseded`；保存超时为 20 秒。 |
| 自定义域名下 404 | 把 `DIRMAP_BASE` 改为 `/` 后重新部署，并确认 DNS/域名设置。 |
| 地图/路线数据加载慢或失败 | `graph.bin.gz` 较大，确认文件已烘焙并部署（`/DIRMAP/data/maps/<assetId>/graph.bin.gz` 返回 200 且 `Content-Type: application/gzip`）。 |
