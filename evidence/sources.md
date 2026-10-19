# 方案依据与证据边界

核验日期：2026-10-14。仅检索网页文字与接口文档，没有读取在线地图图片或下载大数据集。以下“支持”指文档支持相应接口/规则，不代表本项目已经完成集成测试。

| 编号 | 来源与核验方式 | 支持的决策；不支持的推断 |
| --- | --- | --- |
| S0 | [IOF O-Map Wiki：ISSprOM](https://omapwiki.orienteering.sport/specifications/issprom)，直接读取官方页面文字；[411](https://omapwiki.orienteering.sport/symbols/411-uncrossable-vegetation)；[2019-2 修订说明](https://orienteering.sport/new-map-specifications-for-mtbo-and-orienteering-sprint-have-been-published) | 禁行植被、墙、围栏、临时禁区、多层结构必须按语义解释。当前在线目录将 512.3 描述为 “Area runnable at lower level”，历史材料有旧名称；因此按实际版次保存字典。不能推断本地 PDF 或 PNG 已符合某一版。 |
| S1 | [高德：其他坐标转高德坐标](https://lbs.amap.com/api/javascript-api-v2/guide/transform/convertfrom)，官方文档检索摘录；[JS 2.0 定位服务](https://lbs.amap.com/api/javascript-api-v2/guide/services/geolocation) | 高德使用 GCJ-02，`AMap.convertFrom(...,"gps",...)` 接收 GPS 原始坐标；定位插件融合多个来源。不能把浏览器原始 WGS84 与高德已偏移坐标混用，也不能把 IP 结果当精确 GPS。 |
| S2 | [高德 JS 2.0 示例目录](https://lbs.amap.com/demo/list/js-api-v2)，官方目录检索摘录，包含“XYZ 栅格图层”“函数加载栅格图”“图片图层” | 支持 `TileLayer.getTileUrl` 自定义自有栅格，以及 ImageLayer 等显示方案。文档不替应用进行局部形变；必须自行生成正确地理位置的瓦片。 |
| S3 | [高德 JS API 2.0 参考快照](https://a.amap.com/jsapi/static/doc/20230922/index.html?v=2)，官方文档索引摘录 | `rotation` 为“初始地图顺时针旋转的角度”；据此设计 `rotation=-heading`。本次抓取不带日期的动态 API 页面未得到方法正文，故未以它证明当前 SDK 运行行为；集成时须四方位真机验证符号、取值范围和动画。 |
| S4 | [W3C Device Orientation and Motion](https://www.w3.org/TR/orientation-event/)，直接读取；[MDN requestPermission](https://developer.mozilla.org/en-US/docs/Web/API/DeviceOrientationEvent/requestPermission_static) 和 [DeviceOrientationEvent](https://developer.mozilla.org/en-US/docs/Web/API/DeviceOrientationEvent)，文档检索 | 安全上下文、用户手势、absolute 与姿态坐标语义；W3C 平放示例指出朝向可为 `360-alpha`，不能推广到任意倾角与横屏。支持设计质量回退；不证明每款手机/微信内嵌浏览器均有可靠罗盘。 |
| S5 | [MDN GeolocationCoordinates.heading](https://developer.mozilla.org/en-US/docs/Web/API/GeolocationCoordinates/heading)，文档检索 | 航向是运动方向，北 0°、东 90°、静止/缺少信息时可为空。英文总览中 “facing” 表述不如本属性专页明确，方案采用专页的 traveling 语义。 |
| S6 | [OpenCV 官方仓库：特征匹配与单应矩阵教程源码](https://github.com/opencv/opencv/blob/4.x/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.markdown)，直接读取 raw 源文件 | 单应矩阵至少需要四个正确对应点，RANSAC 掩膜区分内/外点。官方生成文档站本次返回 HTTP 403，改读公开官方源码。教程是同一物体图像匹配例子，不能作为定向图对卫星图必然匹配成功的证据。 |
| S7 | [GDAL gdalwarp](https://gdal.org/en/stable/programs/gdalwarp.html)，直接读取 | `-tps` 是基于 GCP 的薄板样条；多于数千 GCP 时有二次内存代价；`-wm` 限制 warp 缓存并可分块。支持模型备选和资源规划，未在本机安装/运行 GDAL。 |
| S8 | [OR-Tools Routing 概览](https://developers.google.com/optimization/routing)、[Routing Options](https://developers.google.com/optimization/routing/routing_options)、[Penalties and Dropping Visits](https://developers.google.com/optimization/routing/penalties)；Options 直接读取，其余官方文档检索 | Routing 可能得到好但非最优结果；有 time_limit、局部搜索、成功/超时/不可行等状态；AddDisjunction 会允许丢点。支持区分精确解、限时可行解、无解和超时。本次未运行 OR-Tools，因此 CP-SAT 集成仍需实施验收。 |
| S9 | [NOAA/NCEI World Magnetic Model](https://www.ncei.noaa.gov/products/world-magnetic-model)，官方文档检索 | WMM2025 有效期 2025–2030，可计算磁偏角 D；支持校园级偏角缓存。未计算深圳大学当前具体偏角，不提供猜测常数。 |
| S10 | [OSM Wiki：Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)，文档检索 | 可以只读取得指定范围、对象类型和标签的矢量数据。未查询本地三图所在校区的实际覆盖，所以方案将覆盖审查作为 P0 条件。 |
| S11 | [高德：JS API 安全密钥使用](https://lbs.amap.com/api/jsapi-v2/guide/abc/prepare)，官方文档检索 | JS API Key 的安全密钥与代理方式；使用 `serviceHost` 由服务端保管安全密钥。未申请或修改用户账号/Key。 |
| S12 | [GitHub CLI：gh codespace ssh](https://cli.github.com/manual/gh_codespace_ssh)，直接读取；本机 `gh codespace ssh --help`、`gh codespace cp --help` | SSH 与 cp 的 `-c`、`remote:` 语法可用。只核实命令帮助，未查询账号、启动/创建 Codespace、上传原图或产生远端运行费用。 |

## 本地事实

初始目录只有用户提供的五份资料，没有代码、方案或 Git 仓库。根文件系统当时显示约 888 MiB 可用；`/tmp` 为 tmpfs，不能以其空余空间替代持久磁盘容量评估。没有发现可以核验的既有后台进程句柄，本次从文件与真实命令结果继续工作，不把 active goal 状态当作已完成进度。

本机有 Python 3.14.4、Pillow、pdfinfo、pdftotext、Node、gh；NumPy、SciPy、OpenCV、pyproj、Shapely 未安装。仅在最初用既有 Pillow 打开头部/getexif 核对元数据，没有调用像素解码；随后标准库 [inspect_inputs.py](../tools/inspect_inputs.py) 独立完成可复现盘点。原文件只做哈希、头部和 PDF 文字检查。

PDF 由 `pdfinfo` 返回一页、OCAD 创建、允许复制，`pdftotext -layout` 只有换页符。文字提取失败的合理解释可能有多种，不能据此断言 PDF 必然是扫描图，也没有绕过限制做 OCR/栅格化。

## 合成实验的实际结果

完整输出保存在 [design-probe.json](design-probe.json)，由 [design_probe.py](../tools/design_probe.py) 生成。当前 21 项检查全部通过。

- 9 个候选对应点中，穷举最小子集的鲁棒拟合保留 8 个正确内点；该做法是小规模 RANSAC 思路验证，不是生产 RANSAC 性能测试。
- 独立解析分片形变的 8 个核查点：全局仿射 RMSE 5.099 m，分片仿射约 6.28e-16 m；反变换最大误差约 1.42e-14 m。近零误差来自人造模型恰好可表达，不用于承诺真实配准精度。
- 359°→1° 半步滤波为 0°，四方位地图旋转代数一致；NaN 被拒绝。未读取手机传感器或实际高德地图相机。
- 合成网格中两点直线距离 6 m，合法经门路线 14 m；关门即不可达，斜向切角被禁止，同位置不同层不误连，显式 portal 可连通。
- 五个目标的单队精确巡回为 28 m；两队速度 1 与 1.5 m/s 时的分担模式，最小最大耗时为 18.667 s，和独立“全排列+切分”穷举结果相同。该样例没有服务时长或动态拥堵。
- all_visit 为每队独立访问全部目标，split_coverage 每点恰好访问一次；优化后的路线可展开回合法图边。

修订后通过 `/usr/bin/time -v` 运行一次完整探针：进程退出码 0，整条命令墙钟约 0.34 s、最大 RSS 17,640 KiB；脚本测得算法段约 0.014 s。此数据只描述小型标准库实验，不外推到完整识别/求解 worker。首次直接启动时高水位 RSS 受启动环境影响偏大，因此最终采用独立 time 进程记录。
