#!/usr/bin/env python3
"""Structural verifier for the baked DirMap static bundle (no node required).

Checks, from the committed files alone (plus their source `data/projects/`
counterparts), that:

* every static data file exists, parses as strict JSON (no NaN/Infinity), and
  the catalog/health/reference/release index shapes match the contract;
* `graph.bin.gz` decompresses to the exact §5.7 layout, arrays have the right
  lengths and value domains, and the node arrays rebuild the server owner grid;
* every `graph.json` counter (`nodeCount`, `directedEdgeCount`,
  `componentCount`, `expected.lengthSum/weightSum/edgeCount`) matches an
  independent reconstruction from the binary bundle;
* every graph hash is reproduced from the source state + binary costs, and
  szu-3's hash equals the hash recorded in `data/projects/szu-3/state.json`
  (2570e1372e5059e7a540);
* `approaches.json` covers every role=control control, nodes are in range,
  `component`/`startComponent` are consistent, and failed entries keep a
  Chinese message;
* `state.json`, images, releases and fixtures are internally consistent and
  releases resolve to existing files;
* the golden fixtures parse, stay below 2 MB, and the optimizer golden is
  reproduced by re-running the Python exact solver.

Run with: PYTHONPATH=.pydeps python3 tools/verify_static_data.py
Exit code 0 means the bundle is good; 1 prints every failure found.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import struct
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "public" / "data"
FIXTURES = ROOT / "tests" / "fixtures" / "static"
SOURCE = ROOT / "data" / "projects"
EXPECTED_SZU3_HASH = "2570e1372e5059e7a540"
MAP_IDS = ("szu-1", "szu-2", "szu-3")
RELEASE_PLAN_ID = "5684df3f33f04390"
MAX_FIXTURE_BYTES = 2 * 1024 * 1024
MAX_DATA_FILE_BYTES = 100 * 1024 * 1024
SEMANTICS_MAGIC = b"DMSEM1"
SEMANTICS_NAMES = ("building", "water", "point_ink", "vegetation", "restricted")
BLOCKED_SAMPLE_COUNT = 3000
COSTS_SAMPLE_COUNT = 2000
VISION_SOURCE_FILES = ("vision.py", "symbols.py", "overprint.py", "reference.py",
                       "ocr.py", "protocol.py", "geometry.py", "schemas.json")


def vision_source_hash():
    package = Path(service.__file__).resolve().parent
    digest = hashlib.sha256()
    for name in VISION_SOURCE_FILES:
        path = package / name
        if not path.is_file():
            raise ValueError(f"缺少参考实现源文件 dirmap/{name}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

from dirmap import service, store  # noqa: E402
from dirmap.geometry import Frame, Registration  # noqa: E402
from dirmap.optimizer import solve  # noqa: E402
from dirmap.protocol import ROUTING_SCHEMA, TERRAIN_SCHEMA  # noqa: E402
from dirmap.reference import load_reference  # noqa: E402
from dirmap.vision import analyze, read_image  # noqa: E402


class Report:
    def __init__(self):
        self.errors = []
        self.checks = 0

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def equal(self, actual, expected, message):
        return self.check(actual == expected, f"{message}: {actual!r} != {expected!r}")

    def close(self, actual, expected, tolerance, message):
        try:
            good = math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            good = False
        return self.check(good, f"{message}: {actual!r} vs {expected!r} (tol {tolerance})")


def reject_constant(value):
    raise ValueError(f"JSON 含非有限数值 {value}")


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(256 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_size(count):
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024


def is_hex32(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def finite_point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(finite_number(item) for item in value))


# --------------------------------------------------------------------------- #
# graph.bin.gz parsing and the browser-side reconstruction.
# --------------------------------------------------------------------------- #


def parse_bundle(path):
    compressed = Path(path).read_bytes()
    if compressed[:2] != b"\x1f\x8b":
        raise ValueError("不是 gzip 流")
    data = gzip.decompress(compressed)
    if len(data) < 24 or data[:8] != b"DMGRAPH1":
        raise ValueError("magic DMGRAPH1 不正确")
    version, = struct.unpack_from("<H", data, 8)
    if version != 1:
        raise ValueError(f"格式版本 {version} != 1")
    width, height, node_count, palette_count = struct.unpack_from("<IIIH", data, 10)
    offset = 24
    palette = np.frombuffer(data, dtype="<f4", count=palette_count, offset=offset).astype(np.float32)
    offset += 4 * palette_count
    costs = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=offset).copy()
    offset += width * height
    cx = np.frombuffer(data, dtype="<f4", count=node_count, offset=offset).astype(np.float64)
    offset += 4 * node_count
    cy = np.frombuffer(data, dtype="<f4", count=node_count, offset=offset).astype(np.float64)
    offset += 4 * node_count
    sizes = np.frombuffer(data, dtype=np.uint8, count=node_count, offset=offset).copy()
    offset += node_count
    terrain = np.frombuffer(data, dtype=np.uint8, count=node_count, offset=offset).copy()
    offset += node_count
    if offset != len(data):
        raise ValueError(f"解压后长度 {len(data)} != 表头+数组 {offset}")
    return {
        "width": width,
        "height": height,
        "nodeCount": node_count,
        "paletteCount": palette_count,
        "palette": palette,
        "costs": costs.reshape((height, width)),
        "cx": cx,
        "cy": cy,
        "pixels": np.column_stack((cx, cy)),
        "sizes": sizes,
        "terrain": terrain,
        "decompressed": len(data),
    }


def reconstruct(state, bundle):
    """Rebuild local positions, the owner grid, edges and components from the bundle."""
    width, height, count = bundle["width"], bundle["height"], bundle["nodeCount"]
    pixels = bundle["pixels"]
    sizes = bundle["sizes"].astype(np.int64)
    registration = Registration.from_dict(state["registration"])
    local, valid = registration.forward(pixels)
    if not bool(valid.all()):
        raise ValueError(f"{int((~valid).sum())} 个节点中心超出配准范围")
    owner = np.full((height, width), -1, np.int32)
    x0 = np.rint(pixels[:, 0] - (sizes.astype(np.float64) - 1) / 2).astype(np.int64)
    y0 = np.rint(pixels[:, 1] - (sizes.astype(np.float64) - 1) / 2).astype(np.int64)
    for size in sorted({int(value) for value in np.unique(sizes)}):
        ids = np.flatnonzero(sizes == size)
        for dy in range(size):
            for dx in range(size):
                owner[y0[ids] + dy, x0[ids] + dx] = ids
    if int(owner.max()) >= count:
        raise ValueError("owner 网格节点编号越界")
    keys = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        sy, ty = slice(0, height - dy), slice(dy, height)
        sx = slice(max(0, -dx), min(width, width - dx))
        tx = slice(max(0, dx), min(width, width + dx))
        a, b = owner[sy, sx], owner[ty, tx]
        keep = (a >= 0) & (b >= 0) & (a != b)
        if dx and dy:
            keep &= (owner[sy, tx] >= 0) & (owner[ty, sx] >= 0)
        a, b = a[keep].astype(np.int64), b[keep].astype(np.int64)
        keys.append(np.minimum(a, b) * count + np.maximum(a, b))
    pairs = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    a, b = (pairs // count).astype(np.int64), (pairs % count).astype(np.int64)
    lengths = np.hypot(local[b, 0] - local[a, 0], local[b, 1] - local[a, 1])
    terrain_values = bundle["palette"][bundle["terrain"]].astype(np.float32)
    sizes32 = sizes.astype(np.float32)
    ratio = (terrain_values[a] * sizes32[a] + terrain_values[b] * sizes32[b]) / (sizes32[a] + sizes32[b])
    weights = lengths * ratio
    if len(a):
        matrix = coo_matrix((np.ones(2 * len(a), np.float64),
                             (np.concatenate((a, b)), np.concatenate((b, a)))),
                            shape=(count, count)).tocsr()
        component_count, component_labels = connected_components(matrix, directed=False)
    else:
        component_count, component_labels = count, np.arange(count)
    return {
        "local": local,
        "owner": owner,
        "a": a,
        "b": b,
        "lengths": lengths,
        "weights": weights,
        "componentCount": int(component_count),
        "components": component_labels,
    }


def recompute_graph_hash(state, bundle):
    costs = bundle["palette"][bundle["costs"]].astype(np.float32)
    registration = Registration.from_dict(state["registration"])
    payload = json.dumps({
        "source": state["sourceHash"],
        "registration": registration.as_dict(),
        "edits": state.get("edits") or [],
        "step": 4,
        "schema": ROUTING_SCHEMA,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload + costs.tobytes()).hexdigest()[:20]


def verify_samples(report, asset_id, metadata, rebuilt, count):
    samples = metadata.get("samples")
    if not report.check(isinstance(samples, list) and len(samples) == 40,
                        f"{asset_id}: graph.json.samples 应为 40 条，实际 {len(samples) if isinstance(samples, list) else samples}"):
        return
    pairs = []
    for index, sample in enumerate(samples):
        label = f"{asset_id}: samples[{index}]"
        if not report.check(isinstance(sample, dict), f"{label} 不是对象"):
            continue
        source, target = sample.get("from"), sample.get("to")
        ok = (report.check(isinstance(source, int) and 0 <= source < count, f"{label}.from 越界: {source}")
              and report.check(isinstance(target, int) and 0 <= target < count, f"{label}.to 越界: {target}")
              and report.check(source != target, f"{label} from == to")
              and report.check(finite_number(sample.get("time")) and sample["time"] >= 0, f"{label}.time 无效")
              and report.check(finite_number(sample.get("distance")) and sample["distance"] >= 0, f"{label}.distance 无效"))
        if ok:
            pairs.append((source, target, float(sample["time"]), float(sample["distance"])))
    if not report.equal(len({(source, target) for source, target, _t, _d in pairs}), len(pairs),
                        f"{asset_id}: 最短路样本存在重复节点对"):
        return
    a, b = rebuilt["a"], rebuilt["b"]
    n = count
    time_graph = csr_matrix((np.concatenate((rebuilt["weights"], rebuilt["weights"])),
                             (np.concatenate((a, b)), np.concatenate((b, a)))), shape=(n, n))
    distance_graph = csr_matrix((np.concatenate((rebuilt["lengths"], rebuilt["lengths"])),
                                 (np.concatenate((a, b)), np.concatenate((b, a)))), shape=(n, n))
    sources = sorted({source for source, _t, _x, _y in pairs})
    time_rows = dijkstra(time_graph, indices=sources, directed=True)
    distance_rows = dijkstra(distance_graph, indices=sources, directed=True)
    row_of = {source: index for index, source in enumerate(sources)}
    for source, target, time_value, distance_value in pairs:
        row = row_of[source]
        report.close(time_rows[row, target], time_value, 1e-6, f"{asset_id}: 样本 ({source},{target}) time")
        report.close(distance_rows[row, target], distance_value, 1e-6, f"{asset_id}: 样本 ({source},{target}) distance")


# --------------------------------------------------------------------------- #
# Per-map checks.
# --------------------------------------------------------------------------- #


def verify_map(report, asset_id, catalog_entry, summaries):
    directory = OUT / "maps" / asset_id
    state_path = directory / "state.json"
    for name in ("state.json", "original.jpg", "original.png", "overlay.png", "terrain-warp.png",
                 "terrain.png", "semantics.bin.gz", "approaches.json", "graph.json", "graph.bin.gz"):
        if not report.check((directory / name).is_file(), f"{asset_id}: 缺少 {name}"):
            return
    source_state = load_json(SOURCE / asset_id / "state.json")
    state = load_json(state_path)
    graph_metadata = load_json(directory / "graph.json")
    approaches = load_json(directory / "approaches.json")
    bundle = parse_bundle(directory / "graph.bin.gz")
    rebuilt = reconstruct(state, bundle)
    count = bundle["nodeCount"]
    report.equal(graph_metadata.get("assetId"), asset_id, f"{asset_id}: graph.json.assetId")
    report.equal(graph_metadata.get("schema"), ROUTING_SCHEMA, f"{asset_id}: graph.json.schema")
    report.equal(graph_metadata.get("sourceHash"), state.get("sourceHash"), f"{asset_id}: graph.json.sourceHash")
    report.equal(graph_metadata.get("step"), 4, f"{asset_id}: graph.json.step")
    report.equal(graph_metadata.get("adaptive"), True, f"{asset_id}: graph.json.adaptive")
    report.equal(graph_metadata.get("width"), bundle["width"], f"{asset_id}: graph.json.width")
    report.equal(graph_metadata.get("height"), bundle["height"], f"{asset_id}: graph.json.height")
    report.equal(graph_metadata.get("nodeCount"), count, f"{asset_id}: graph.json.nodeCount")
    if source_state.get("graphHash"):
        report.equal(graph_metadata.get("graphHash"), source_state["graphHash"],
                     f"{asset_id}: graph.json.graphHash 未复现 state.json 记录")
    if asset_id == "szu-3":
        report.equal(graph_metadata.get("graphHash"), EXPECTED_SZU3_HASH,
                     f"{asset_id}: graph.json.graphHash 不等于记录的 2570e1372e5059e7a540")
    recomputed = recompute_graph_hash(state, bundle)
    report.equal(graph_metadata.get("graphHash"), recomputed, f"{asset_id}: 二进制 bundle 不能复现 graphHash")
    # Header/array domains.
    report.equal(bundle["width"], state.get("width"), f"{asset_id}: bundle.width")
    report.equal(bundle["height"], state.get("height"), f"{asset_id}: bundle.height")
    report.check(bundle["paletteCount"] == len(bundle["palette"]) >= 2,
                 f"{asset_id}: paletteCount 无效 {bundle['paletteCount']}")
    report.check(all(finite_number(value) for value in bundle["palette"].tolist()),
                 f"{asset_id}: costsPalette 含非有限值")
    report.check(bool(np.all(np.diff(bundle["palette"].astype(np.float64)) > 0)),
                 f"{asset_id}: costsPalette 未严格递增")
    report.check(bundle["palette"][0] == 0.0, f"{asset_id}: costsPalette 首个值不是 0")
    report.check(bool(np.all(bundle["costs"] < bundle["paletteCount"])),
                 f"{asset_id}: costs 调色板索引越界")
    report.check(bool(np.all(bundle["terrain"] < bundle["paletteCount"])),
                 f"{asset_id}: terrain 调色板索引越界")
    report.check(bool(np.all(np.isin(bundle["sizes"], (1, 2, 4, 8, 16)))),
                 f"{asset_id}: 单元格尺寸越界")
    report.check(bool(np.all(np.diff(bundle["sizes"].astype(np.int64)) <= 0)),
                 f"{asset_id}: 节点未按单元格尺寸降序排列")
    report.check(bool(np.all(np.rint(bundle["pixels"] * 2) == bundle["pixels"] * 2)),
                 f"{asset_id}: 节点中心不是精确半像素")
    inside = bool(np.all((bundle["cx"] >= 0) & (bundle["cx"] <= bundle["width"] - 1)
                         & (bundle["cy"] >= 0) & (bundle["cy"] <= bundle["height"] - 1)))
    report.check(inside, f"{asset_id}: 节点中心超出画幅")
    # Counters.
    edge_count = int(len(rebuilt["a"]))
    report.equal(graph_metadata.get("directedEdgeCount"), 2 * edge_count,
                 f"{asset_id}: directedEdgeCount 与重建无向边数不符")
    expected = graph_metadata.get("expected") or {}
    report.equal(expected.get("edgeCount"), edge_count, f"{asset_id}: expected.edgeCount")
    report.close(expected.get("lengthSum"), float(np.sum(rebuilt["lengths"])), 1e-6,
                 f"{asset_id}: expected.lengthSum")
    report.close(expected.get("weightSum"), float(np.sum(rebuilt["weights"])), 1e-6,
                 f"{asset_id}: expected.weightSum")
    report.equal(graph_metadata.get("componentCount"), rebuilt["componentCount"],
                 f"{asset_id}: componentCount")
    palette_list = graph_metadata.get("costsPalette")
    report.check(isinstance(palette_list, list) and len(palette_list) == bundle["paletteCount"]
                 and all(finite_number(value) for value in palette_list)
                 and np.allclose(np.asarray(palette_list, np.float32), bundle["palette"], rtol=0, atol=0),
                 f"{asset_id}: graph.json.costsPalette 与 graph.bin.gz 不一致")
    # Samples.
    verify_samples(report, asset_id, graph_metadata, rebuilt, count)
    # Baked state (§5.3).
    for forbidden in ("artifacts", "lastOperationId", "processingElapsedSec",
                      "terrainProcessingElapsedSec", "failures"):
        report.check(forbidden not in state, f"{asset_id}: state.json 不应包含 {forbidden}")
    for key in ("assetId", "name", "filename", "sourceHash", "width", "height", "version",
                "reviewed", "controls", "edits", "registration", "frame", "extent",
                "warnings", "statistics", "quality", "terrainSchema"):
        report.check(key in state, f"{asset_id}: state.json 缺少 {key}")
    report.equal(state.get("static"), True, f"{asset_id}: state.json.static")
    report.equal(state.get("graphHash"), graph_metadata.get("graphHash"), f"{asset_id}: state.json.graphHash")
    report.equal(state.get("version"), source_state.get("version"), f"{asset_id}: state.json.version")
    report.equal(state.get("reviewed"), source_state.get("reviewed"), f"{asset_id}: state.json.reviewed")
    report.equal(state.get("sourceHash"), source_state.get("sourceHash"), f"{asset_id}: state.json.sourceHash")
    report.equal(sha256_file(directory / "original.jpg"), catalog_entry.get("sha256"),
                 f"{asset_id}: original.jpg 哈希与 catalog 不符")
    report.equal(sha256_file(directory / "original.jpg"), source_state.get("sourceHash"),
                 f"{asset_id}: original.jpg 哈希与源 state 不符")
    processing_sizes = verify_processing_inputs(report, asset_id, state, source_state, directory)
    source_controls = source_state.get("controls") or []
    baked_controls = state.get("controls") or []
    report.equal([control.get("id") for control in baked_controls],
                 [control.get("id") for control in source_controls],
                 f"{asset_id}: state.json 控制点 id 顺序不一致")
    report.equal([control.get("label") for control in baked_controls],
                 [control.get("label") for control in source_controls],
                 f"{asset_id}: state.json 控制点编号不一致")
    for name in ("overlay.png", "terrain-warp.png", "terrain.png"):
        try:
            source_path = store.artifact_path(source_state, name)
            report.check(Path(source_path).read_bytes() == (directory / name).read_bytes(),
                         f"{asset_id}: {name} 与源派生文件不一致")
        except Exception as error:  # noqa: BLE001 - report, do not abort the run
            report.check(False, f"{asset_id}: 无法比较 {name}: {error}")
    # Approaches (§5.5 + approved canonical-start metadata).
    report.equal(approaches.get("assetId"), asset_id, f"{asset_id}: approaches.assetId")
    report.equal(approaches.get("routingSchema"), ROUTING_SCHEMA, f"{asset_id}: approaches.routingSchema")
    report.equal(approaches.get("graphHash"), graph_metadata.get("graphHash"), f"{asset_id}: approaches.graphHash")
    report.check(isinstance(approaches.get("note"), str) and approaches["note"],
                 f"{asset_id}: approaches.note 缺失")
    start_node = approaches.get("startNode")
    start_component = approaches.get("startComponent")
    report.check(isinstance(start_node, int) and 0 <= start_node < count,
                 f"{asset_id}: approaches.startNode 越界: {start_node}")
    report.check(isinstance(start_component, int) and 0 <= start_component < rebuilt["componentCount"],
                 f"{asset_id}: approaches.startComponent 越界: {start_component}")
    if isinstance(start_node, int) and 0 <= start_node < count and isinstance(start_component, int):
        report.equal(int(rebuilt["components"][start_node]), start_component,
                     f"{asset_id}: approaches.startNode 的连通块与 startComponent 不符")
    report.check(finite_point(approaches.get("startPixel")), f"{asset_id}: approaches.startPixel 无效")
    report.check(finite_point(approaches.get("finishPixel")), f"{asset_id}: approaches.finishPixel 无效")
    approach_controls = approaches.get("controls") or {}
    wanted = [control["id"] for control in source_controls
              if control.get("role", "control") == "control"]
    report.equal(sorted(approach_controls), sorted(wanted), f"{asset_id}: approaches.controls 覆盖不完整")
    ok_count = 0
    for control_id in wanted:
        entry = approach_controls.get(control_id)
        label = f"{asset_id}: approaches.{control_id}"
        if not report.check(isinstance(entry, dict), f"{label} 缺失"):
            continue
        if entry.get("ok") is True:
            ok_count += 1
            node = entry.get("node")
            report.check(isinstance(node, int) and 0 <= node < count, f"{label}.node 越界: {node}")
            report.check(isinstance(entry.get("component"), int), f"{label}.component 缺失")
            if isinstance(node, int) and 0 <= node < count and isinstance(entry.get("component"), int):
                report.equal(int(rebuilt["components"][node]), entry["component"],
                             f"{label}.component 与实际连通块不符")
                report.equal(entry["component"], start_component,
                             f"{label}.component 与 startComponent 不一致")
            report.check(finite_point(entry.get("pixel")), f"{label}.pixel 无效")
            report.check(finite_point(entry.get("local")), f"{label}.local 无效")
            report.check(finite_point(entry.get("lnglat")), f"{label}.lnglat 无效")
            report.check(entry.get("targetLnglat") is None or finite_point(entry.get("targetLnglat")),
                         f"{label}.targetLnglat 无效")
            report.check(isinstance(entry.get("method"), str) and entry["method"], f"{label}.method 无效")
            report.check(isinstance(entry.get("automatic"), bool), f"{label}.automatic 无效")
            report.check(finite_number(entry.get("offsetM")) and entry["offsetM"] >= 0,
                         f"{label}.offsetM 无效")
            report.check(finite_point(entry.get("targetPixel")), f"{label}.targetPixel 无效")
        elif entry.get("ok") is False:
            message = entry.get("message")
            report.check(isinstance(message, str) and message, f"{label}.message 缺失")
            report.check(bool(re.search(r"[\u4e00-\u9fff]", message or "")), f"{label}.message 不是中文原因")
            report.check("node" not in entry or entry.get("node") is None, f"{label}: 失败项不应带 node")
    summaries[asset_id] = {
        "width": bundle["width"],
        "height": bundle["height"],
        "nodes": count,
        "edges": int(graph_metadata.get("directedEdgeCount") or 0),
        "components": rebuilt["componentCount"],
        "approaches": f"{ok_count}/{len(wanted)}",
        "graph_size": (directory / "graph.bin.gz").stat().st_size,
        "png_size": processing_sizes.get("png", 0),
        "semantics_size": processing_sizes.get("semantics", 0),
        "graph_hash": graph_metadata.get("graphHash"),
    }
    # Matrix fixture.
    fixture = load_json(FIXTURES / f"{asset_id}.matrix.json")
    verify_matrix_fixture(report, asset_id, fixture, approaches, source_state)
    # Dijkstra fixture.
    dijkstra_fixture = load_json(FIXTURES / f"{asset_id}.dijkstra.json")
    verify_dijkstra_fixture(report, asset_id, dijkstra_fixture, graph_metadata, bundle["pixels"])


def verify_matrix_fixture(report, asset_id, fixture, approaches, source_state):
    label = f"{asset_id}: matrix fixture"
    report.equal(fixture.get("assetId"), asset_id, f"{label}.assetId")
    report.equal(fixture.get("metric"), "time", f"{label}.metric")
    scenario = fixture.get("scenario") or {}
    wanted = [control["id"] for control in source_state["controls"]
              if control.get("role", "control") == "control" and control.get("state") != "rejected"]
    report.equal(scenario.get("controlIds"), wanted, f"{label}.scenario.controlIds")
    report.equal(fixture.get("request", {}).get("controlIds"), wanted, f"{label}.request.controlIds")
    report.check(finite_point(scenario.get("startPixel")), f"{label}.scenario.startPixel")
    report.check(finite_point(scenario.get("finishPixel")), f"{label}.scenario.finishPixel")
    report.check(scenario.get("startPixel") == scenario.get("finishPixel"),
                 f"{label}: startPixel 与 finishPixel 应相同")
    report.equal(fixture.get("startPixel"), scenario.get("startPixel"), f"{label}.startPixel")
    report.equal(fixture.get("finishPixel"), scenario.get("finishPixel"), f"{label}.finishPixel")
    report.equal(fixture.get("graphHash"), approaches.get("graphHash"), f"{label}.graphHash")
    report.equal(fixture.get("startNode"), approaches.get("startNode"), f"{label}.startNode")
    unresolved = fixture.get("unresolved")
    report.check(isinstance(unresolved, list), f"{label}.unresolved 不是列表")
    matrix = fixture.get("matrix")
    nodes = fixture.get("nodes")
    if matrix is None:
        report.check(bool(unresolved), f"{label}: 未解决场景应记录 unresolved")
        report.equal(fixture.get("nodes"), [], f"{label}: 未解决场景 nodes 应为空")
        report.equal(fixture.get("endpoints"), [], f"{label}: 未解决场景 endpoints 应为空")
        for item in unresolved or []:
            if not report.check(isinstance(item, dict) and item.get("id") in wanted,
                                f"{label}: unresolved 项无效 {item!r}"):
                continue
            report.check(isinstance(item.get("message"), str) and item["message"],
                         f"{label}.unresolved[{item.get('id')}] 缺少消息")
            baked = approaches["controls"].get(item["id"], {})
            report.check(baked.get("ok") is not True,
                         f"{label}: {item['id']} 有 ok 到访点却被标记未解决")
        return
    if not report.check(isinstance(matrix, list) and isinstance(nodes, list),
                        f"{label}: matrix/nodes 类型无效"):
        return
    size = len(wanted) + 2
    report.equal(len(nodes), size, f"{label}.nodes 长度")
    report.equal(len(matrix), size, f"{label}.matrix 行数")
    report.equal(unresolved, [], f"{label}: 已解决场景 unresolved 应为空")
    connectors = fixture.get("connectors")
    weights = fixture.get("connectorWeights")
    report.check(isinstance(connectors, list) and len(connectors) == size, f"{label}.connectors 长度")
    report.check(isinstance(weights, list) and len(weights) == size, f"{label}.connectorWeights 长度")
    report.check(all(isinstance(node, int) for node in nodes), f"{label}.nodes 必须是整数")
    expected_ids = ["start", *wanted, "finish"]
    endpoints = fixture.get("endpoints") or []
    report.equal([item.get("id") for item in endpoints], expected_ids, f"{label}.endpoints id 顺序")
    report.equal([item.get("node") for item in endpoints], nodes, f"{label}.endpoints node 与 nodes 不符")
    for index, row in enumerate(matrix):
        report.check(isinstance(row, list) and len(row) == size, f"{label}.matrix[{index}] 列数")
        if not isinstance(row, list) or len(row) != size:
            continue
        report.check(all(finite_number(value) and value >= -1e-9 for value in row),
                     f"{label}.matrix[{index}] 含无效数值")
        report.check(abs(float(row[index])) <= 1e-6, f"{label}.matrix[{index}][{index}] 对角线不为 0")
    for index, control_id in enumerate(wanted):
        endpoint = endpoints[index + 1] if index + 1 < len(endpoints) else {}
        baked = approaches["controls"].get(control_id) or {}
        if baked.get("ok") is not True:
            report.check(False, f"{label}: 已解决矩阵包含未解决到访点 {control_id}")
            continue
        report.equal(endpoint.get("node"), baked.get("node"), f"{label}: {control_id} 端点节点与 approaches 不符")
        report.check(finite_point(endpoint.get("approachPixel")), f"{label}: {control_id} 缺少 approachPixel")
        report.check(np.allclose(endpoint.get("approachPixel"), baked.get("pixel"), rtol=0, atol=0)
                     if finite_point(endpoint.get("approachPixel")) else False,
                     f"{label}: {control_id} approachPixel 与 approaches.pixel 不符")
    report.check(len(fixture.get("approaches") or []) == len(wanted), f"{label}.approaches 数量")


def verify_dijkstra_fixture(report, asset_id, fixture, graph_metadata, pixels):
    label = f"{asset_id}: dijkstra fixture"
    report.equal(fixture.get("assetId"), asset_id, f"{label}.assetId")
    report.equal(fixture.get("graphHash"), graph_metadata.get("graphHash"), f"{label}.graphHash")
    samples = fixture.get("samples")
    graph_samples = graph_metadata.get("samples") or []
    if not report.check(isinstance(samples, list) and len(samples) == len(graph_samples) == 40,
                        f"{label}.samples 数量与 graph.json 不一致"):
        return
    for index, (sample, graph_sample) in enumerate(zip(samples, graph_samples)):
        report.equal(sample.get("from"), graph_sample.get("from"), f"{label}.samples[{index}].from")
        report.equal(sample.get("to"), graph_sample.get("to"), f"{label}.samples[{index}].to")
        report.close(sample.get("time"), graph_sample.get("time"), 1e-12, f"{label}.samples[{index}].time")
        report.close(sample.get("distance"), graph_sample.get("distance"), 1e-12,
                     f"{label}.samples[{index}].distance")
        source = sample.get("from")
        if isinstance(source, int) and 0 <= source < len(pixels):
            report.check(np.allclose(sample.get("fromPixel"), pixels[source], rtol=0, atol=0),
                         f"{label}.samples[{index}].fromPixel 与节点坐标不符")
        target = sample.get("to")
        if isinstance(target, int) and 0 <= target < len(pixels):
            report.check(np.allclose(sample.get("toPixel"), pixels[target], rtol=0, atol=0),
                         f"{label}.samples[{index}].toPixel 与节点坐标不符")


# --------------------------------------------------------------------------- #
# C.5 processing inputs: lossless original PNG, semantics bundle, client goldens.
# --------------------------------------------------------------------------- #


def parse_semantics(path):
    compressed = Path(path).read_bytes()
    if compressed[:2] != b"\x1f\x8b":
        raise ValueError("semantics.bin.gz 不是 gzip 流")
    data = gzip.decompress(compressed)
    if data[:6] != SEMANTICS_MAGIC:
        raise ValueError(f"semantics magic 不是 {SEMANTICS_MAGIC!r}")
    version, = struct.unpack_from("<H", data, 6)
    width, height, count = struct.unpack_from("<IIB", data, 8)
    offset = 17
    names = []
    for _ in range(count):
        if offset >= len(data):
            raise ValueError("语义文件名称表被截断")
        length = data[offset]
        offset += 1
        names.append(data[offset:offset + length].decode("utf-8"))
        offset += length
    mask_bytes = (width * height + 7) // 8
    expected = offset + count * mask_bytes
    if len(data) != expected:
        raise ValueError(f"语义文件长度 {len(data)} != 表头+掩码 {expected}")
    masks = {}
    for name in names:
        chunk = np.frombuffer(data, dtype=np.uint8, count=mask_bytes, offset=offset)
        offset += mask_bytes
        unpacked = np.unpackbits(chunk, bitorder="little")[:width * height]
        masks[name] = unpacked.reshape((height, width)).astype(bool)
    return {"version": version, "width": width, "height": height, "names": names, "masks": masks}


def verify_processing_inputs(report, asset_id, state, source_state, directory):
    label = f"{asset_id}: original.png"
    png_path = directory / "original.png"
    semantics_path = directory / "semantics.bin.gz"
    if not report.check(png_path.is_file(), f"{label} 不存在"):
        return {}
    report.check(png_path.stat().st_size < MAX_DATA_FILE_BYTES, f"{label} 超过 100 MB")
    image = read_image(store.get_asset(asset_id)["path"])
    png = cv2.imdecode(np.fromfile(str(png_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if not report.check(png is not None, f"{label} 无法解码"):
        return {}
    report.equal(png.shape, image.shape, f"{label} 尺寸")
    report.equal(png.shape[1], state.get("width"), f"{label} 宽度")
    report.equal(png.shape[0], state.get("height"), f"{label} 高度")
    report.equal(hashlib.sha256(png.tobytes()).hexdigest(),
                 hashlib.sha256(image.tobytes()).hexdigest(),
                 f"{label} 像素哈希与 vision.read_image 不一致")
    report.check(np.array_equal(png, image), f"{label} 像素与 cv2 JPEG 解码不一致")
    label_sem = f"{asset_id}: semantics.bin.gz"
    if not report.check(semantics_path.is_file(), f"{label_sem} 不存在"):
        return {"png": png_path.stat().st_size}
    report.check(semantics_path.stat().st_size < MAX_DATA_FILE_BYTES, f"{label_sem} 超过 100 MB")
    try:
        semantics = parse_semantics(semantics_path)
    except (OSError, ValueError) as error:
        report.check(False, f"{label_sem} 解析失败: {error}")
        return {"png": png_path.stat().st_size}
    report.equal(semantics["version"], 1, f"{label_sem}.version")
    report.equal((semantics["width"], semantics["height"]),
                 (state.get("width"), state.get("height")), f"{label_sem} 画幅")
    report.equal(semantics["names"], list(SEMANTICS_NAMES), f"{label_sem} 掩码名称/顺序")
    terrain_path = store.artifact_path(source_state, "terrain.npz")
    with np.load(terrain_path) as artifact:
        for name in SEMANTICS_NAMES:
            if not report.check(name in artifact, f"{label_sem}: terrain.npz 缺少 {name}"):
                continue
            source_mask = np.asarray(artifact[name], dtype=bool)
            report.check(np.array_equal(semantics["masks"].get(name), source_mask),
                         f"{label_sem}: 掩码 {name} 与 terrain.npz 不一致")
    return {"png": png_path.stat().st_size, "semantics": semantics_path.stat().st_size}


def verify_reference_features(report):
    label = "reference-features.json"
    path = OUT / label
    if not report.check(path.is_file(), f"{label} 不存在"):
        return 0
    report.check(path.stat().st_size < MAX_DATA_FILE_BYTES, f"{label} 超过 100 MB")
    payload = load_json(path)
    frame = payload.get("frame") or {}
    report.equal(frame.get("crs"), "WGS84", f"{label}.frame.crs")
    report.equal(frame.get("unit"), "metre", f"{label}.frame.unit")
    report.equal(frame.get("axes"), "east_north", f"{label}.frame.axes")
    origin = frame.get("origin")
    if report.check(finite_point(origin), f"{label}.frame.origin 无效"):
        report.check(abs(origin[0] - Frame().lng) < 1e-9 and abs(origin[1] - Frame().lat) < 1e-9,
                     f"{label}.frame.origin 与 Frame 原点不符")
    features = payload.get("features")
    if not report.check(isinstance(features, list) and len(features) > 0, f"{label}.features 为空"):
        return 0
    counts = {}
    for index, feature in enumerate(features):
        label_feature = f"{label}.features[{index}]"
        if not report.check(isinstance(feature, dict), f"{label_feature} 不是对象"):
            continue
        report.check(isinstance(feature.get("id"), str) and feature["id"], f"{label_feature}.id")
        kind = feature.get("kind")
        report.check(kind in ("building", "water", "pitch"), f"{label_feature}.kind 无效: {kind!r}")
        counts[kind] = counts.get(kind, 0) + 1
        report.check(finite_point(feature.get("center")), f"{label_feature}.center 无效")
        report.check(finite_number(feature.get("area")) and feature["area"] > 0, f"{label_feature}.area 无效")
        report.check(finite_number(feature.get("aspect")) and feature["aspect"] > 0,
                     f"{label_feature}.aspect 无效")
        points = feature.get("points")
        report.check(isinstance(points, list) and len(points) >= 3
                     and all(finite_point(point) for point in points), f"{label_feature}.points 无效")
    reference = load_reference(service.reference_path())
    expected_counts, expected_ids = {}, []
    for feature in reference:
        if feature.get("kind") not in ("building", "water", "pitch"):
            continue
        expected_counts[feature["kind"]] = expected_counts.get(feature["kind"], 0) + 1
        expected_ids.append(str(feature["id"]))
    report.equal([feature.get("id") for feature in features], expected_ids, f"{label}: 地物 id/顺序")
    report.equal(counts, expected_counts, f"{label}: 各地物类型数量")
    return len(features)


def _check_samples(report, label, samples, count, width, height, value_check):
    if not report.check(isinstance(samples, list) and len(samples) == count,
                        f"{label} 应为 {count} 条，实际 {len(samples) if isinstance(samples, list) else samples}"):
        return
    seen = set()
    for index, sample in enumerate(samples):
        if not report.check(isinstance(sample, dict), f"{label}[{index}] 不是对象"):
            continue
        x, y, value = sample.get("x"), sample.get("y"), sample.get("v")
        if not report.check(isinstance(x, int) and isinstance(y, int)
                            and 0 <= x < width and 0 <= y < height,
                            f"{label}[{index}] 坐标越界: {x},{y}"):
            continue
        seen.add((x, y))
        value_check(f"{label}[{index}]", x, y, value)
    report.equal(len(seen), count, f"{label} 坐标应互不重复")


def _verify_coordinates(report, label, coordinates, count, width, height):
    if not report.check(isinstance(coordinates, list) and len(coordinates) == count,
                        f"{label} 应为 {count} 个坐标，实际 "
                        f"{len(coordinates) if isinstance(coordinates, list) else coordinates}"):
        return None
    seen = set()
    for index, item in enumerate(coordinates):
        if not report.check(isinstance(item, dict) and isinstance(item.get("x"), int)
                            and isinstance(item.get("y"), int)
                            and 0 <= item["x"] < width and 0 <= item["y"] < height,
                            f"{label}[{index}] 坐标越界"):
            return None
        seen.add((item["x"], item["y"]))
    report.equal(len(seen), count, f"{label} 坐标应互不重复")
    return [(item["x"], item["y"]) for item in coordinates]


def verify_client_processing(report):
    label = "tests/fixtures/static/client-processing.json"
    path = FIXTURES / "client-processing.json"
    if not report.check(path.is_file(), f"{label} 不存在"):
        return None
    report.check(path.stat().st_size < MAX_FIXTURE_BYTES, f"{label} 超过 2 MB")
    fixture = load_json(path)
    report.check(isinstance(fixture.get("sampling"), dict), f"{label}.sampling")
    report.check(isinstance(fixture.get("note"), str) and fixture["note"], f"{label}.note")
    maps = fixture.get("maps")
    if not report.check(isinstance(maps, list), f"{label}.maps 不是列表"):
        return None
    report.equal([entry.get("assetId") for entry in maps], list(MAP_IDS), f"{label}.maps 顺序")
    current_code_hash = vision_source_hash()
    for entry in maps:
        asset_id = entry.get("assetId")
        source_state = load_json(SOURCE / asset_id / "state.json")
        label_map = f"{label}[{asset_id}]"
        width, height = source_state["width"], source_state["height"]
        report.equal(entry.get("width"), width, f"{label_map}.width")
        report.equal(entry.get("height"), height, f"{label_map}.height")
        controls = entry.get("controls")
        source_controls = source_state["controls"]
        if report.check(isinstance(controls, list) and len(controls) == len(source_controls),
                        f"{label_map}.controls 数量"):
            for index, (control, source) in enumerate(zip(controls, source_controls)):
                label_control = f"{label_map}.controls[{index}]"
                report.equal(control.get("id"), source.get("id"), f"{label_control}.id")
                report.equal(control.get("label"), source.get("label"), f"{label_control}.label")
                report.equal(control.get("role", "control"), source.get("role", "control"),
                             f"{label_control}.role")
                report.equal(control.get("state"), source.get("state"), f"{label_control}.state")
                report.equal(control.get("labelSource"), source.get("labelSource"),
                             f"{label_control}.labelSource")
                report.check(finite_point(control.get("pixel")), f"{label_control}.pixel 无效")
                if finite_point(control.get("pixel")):
                    report.check(np.allclose(control["pixel"], source["pixel"], rtol=0, atol=0),
                                 f"{label_control}.pixel 与 state 不一致")
                report.check(finite_number(control.get("radiusPx"))
                             and abs(float(control["radiusPx"]) - float(source.get("radiusPx") or 0)) < 1e-9,
                             f"{label_control}.radiusPx 与 state 不一致")
        samples = entry.get("samples") or {}
        blocked_coordinates = _verify_coordinates(
            report, f"{label_map}.samples.blocked", samples.get("blocked"),
            BLOCKED_SAMPLE_COUNT, width, height)
        costs_coordinates = _verify_coordinates(
            report, f"{label_map}.samples.costs", samples.get("costs"),
            COSTS_SAMPLE_COUNT, width, height)
        terrain_path = store.artifact_path(source_state, "terrain.npz")
        with np.load(terrain_path) as artifact:
            costs = np.asarray(artifact["costs"], dtype=np.float32)
        blocked_path = store.artifact_path(source_state, "blocked.png")
        blocked = cv2.imread(str(blocked_path), cv2.IMREAD_GRAYSCALE)
        if not report.check(blocked is not None and blocked.shape == costs.shape,
                            f"{label_map}: 源 blocked.png 缺失或尺寸不一致"):
            continue
        # --- secondary tolerance block: committed baked artifacts ---
        artifact_block = entry.get("artifact")
        if report.check(isinstance(artifact_block, dict), f"{label_map}.artifact 不是对象"):
            statistics = source_state.get("statistics") or {}
            artifact_counts = artifact_block.get("featureCounts") or {}
            report.equal(artifact_counts.get("building"), statistics.get("buildingCandidates"),
                         f"{label_map}.artifact.featureCounts.building 与 state.statistics 不一致")
            report.equal(artifact_counts.get("water"), statistics.get("waterCandidates"),
                         f"{label_map}.artifact.featureCounts.water 与 state.statistics 不一致")
            report.check("pitch" not in artifact_counts,
                         f"{label_map}.artifact.featureCounts 无 pitch 计数（旧工件未保存）")
            report.equal(artifact_block.get("statistics"), statistics,
                         f"{label_map}.artifact.statistics 与 state.statistics 不一致")
            report.equal(artifact_block.get("forbiddenPixelFraction"),
                         statistics.get("forbiddenPixelFraction"),
                         f"{label_map}.artifact.forbiddenPixelFraction")
            blocked_values = artifact_block.get("blockedSampleValues")
            if (blocked_coordinates is not None
                    and report.check(isinstance(blocked_values, list)
                                     and len(blocked_values) == BLOCKED_SAMPLE_COUNT,
                                     f"{label_map}.artifact.blockedSampleValues 数量")):
                for index, (x, y) in enumerate(blocked_coordinates):
                    report.equal(blocked_values[index], int(blocked[y, x]),
                                 f"{label_map}.artifact.blockedSampleValues[{index}]")
            cost_values = artifact_block.get("costsSampleValues")
            if (costs_coordinates is not None
                    and report.check(isinstance(cost_values, list)
                                     and len(cost_values) == COSTS_SAMPLE_COUNT,
                                     f"{label_map}.artifact.costsSampleValues 数量")):
                for index, (x, y) in enumerate(costs_coordinates):
                    report.check(np.float32(cost_values[index]) == costs[y, x],
                                 f"{label_map}.artifact.costsSampleValues[{index}]")
        # --- primary golden: current reference implementation ---
        reference = entry.get("referenceImplementation")
        if not report.check(isinstance(reference, dict), f"{label_map}.referenceImplementation 不是对象"):
            continue
        if report.check(isinstance(reference.get("controls"), list),
                        f"{label_map}.referenceImplementation.controls 不是列表"):
            report.equal(reference.get("controls"), controls,
                         f"{label_map}.referenceImplementation.controls 与共享 controls 不一致")
        report.equal(reference.get("generatedBy"), "dirmap.vision.analyze",
                     f"{label_map}.referenceImplementation.generatedBy")
        report.equal(reference.get("sourceCodeHash"), current_code_hash,
                     f"{label_map}.referenceImplementation.sourceCodeHash 与当前参考代码不符")
        analysis = analyze(read_image(store.get_asset(asset_id)["path"]), labeler=None)
        counts = {"building": 0, "water": 0, "pitch": 0}
        for feature in analysis["features"]:
            if feature["kind"] in counts:
                counts[feature["kind"]] += 1
        report.equal(reference.get("featureCounts"), counts,
                     f"{label_map}.referenceImplementation.featureCounts")
        report.equal(reference.get("statistics"), analysis["statistics"],
                     f"{label_map}.referenceImplementation.statistics")
        report.equal(reference.get("forbiddenPixelFraction"),
                     analysis["statistics"]["forbiddenPixelFraction"],
                     f"{label_map}.referenceImplementation.forbiddenPixelFraction")
        reference_blocked = np.asarray(analysis["blocked"])
        reference_costs = np.asarray(analysis["costs"], dtype=np.float32)
        blocked_values = reference.get("blockedSampleValues")
        if (blocked_coordinates is not None
                and report.check(isinstance(blocked_values, list)
                                 and len(blocked_values) == BLOCKED_SAMPLE_COUNT,
                                 f"{label_map}.referenceImplementation.blockedSampleValues 数量")):
            for index, (x, y) in enumerate(blocked_coordinates):
                report.equal(blocked_values[index], int(reference_blocked[y, x]),
                             f"{label_map}.referenceImplementation.blockedSampleValues[{index}]")
        cost_values = reference.get("costsSampleValues")
        if (costs_coordinates is not None
                and report.check(isinstance(cost_values, list)
                                 and len(cost_values) == COSTS_SAMPLE_COUNT,
                                 f"{label_map}.referenceImplementation.costsSampleValues 数量")):
            for index, (x, y) in enumerate(costs_coordinates):
                report.check(np.float32(cost_values[index]) == reference_costs[y, x],
                             f"{label_map}.referenceImplementation.costsSampleValues[{index}]")
    return fixture


# --------------------------------------------------------------------------- #
# Global checks: catalog, health, reference, releases, optimizer fixture.
# --------------------------------------------------------------------------- #


def verify_catalog(report, summaries):
    label = "catalog.json"
    catalog = load_json(OUT / label)
    report.equal(catalog.get("static"), True, f"{label}.static")
    report.equal(catalog.get("routingSchema"), ROUTING_SCHEMA, f"{label}.routingSchema")
    report.equal(catalog.get("terrainSchema"), TERRAIN_SCHEMA, f"{label}.terrainSchema")
    generated = catalog.get("generatedAt")
    try:
        datetime_value = __import__("datetime").datetime.fromisoformat(generated)
        report.check(datetime_value.tzinfo is not None, f"{label}.generatedAt 缺少时区")
    except (TypeError, ValueError):
        report.check(False, f"{label}.generatedAt 不是 ISO 时间: {generated!r}")
    reference = catalog.get("reference") or {}
    report.equal(reference.get("crs"), "WGS84", f"{label}.reference.crs")
    report.check(isinstance(reference.get("name"), str) and reference["name"], f"{label}.reference.name")
    report.equal(reference.get("attribution"), "© OpenStreetMap contributors", f"{label}.reference.attribution")
    report.equal(reference.get("licenseUrl"), "https://www.openstreetmap.org/copyright",
                 f"{label}.reference.licenseUrl")
    maps = catalog.get("maps") or []
    report.equal([entry.get("id") for entry in maps], list(MAP_IDS), f"{label}.maps id 顺序")
    for entry in maps:
        asset_id = entry.get("id")
        label_map = f"{label}.maps[{asset_id}]"
        report.equal(entry.get("processed"), True, f"{label_map}.processed")
        report.equal(entry.get("hasGraph"), True, f"{label_map}.hasGraph")
        report.equal(entry.get("hasApproaches"), True, f"{label_map}.hasApproaches")
        report.check(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
                     f"{label_map}.sha256 无效")
        report.check(isinstance(entry.get("version"), int), f"{label_map}.version 无效")
        report.check(isinstance(entry.get("controls"), int) and entry["controls"] > 0,
                     f"{label_map}.controls 无效")
        for name in ("state.json", "original.jpg", "original.png", "overlay.png", "terrain-warp.png",
                     "terrain.png", "semantics.bin.gz", "approaches.json", "graph.json", "graph.bin.gz"):
            report.check((OUT / "maps" / asset_id / name).is_file(), f"{label_map}: 缺少 {name}")
        report.check((OUT / "maps" / asset_id / "original.jpg").is_file()
                     and sha256_file(OUT / "maps" / asset_id / "original.jpg") == entry.get("sha256"),
                     f"{label_map}: sha256 与原图不符")
    legends = catalog.get("legends") or []
    report.equal(len(legends), 2, f"{label}.legends 数量")
    expected_names = ["ISSprOM2019-2图例.pdf", "____________aigc_label.png"]
    report.equal([item.get("name") for item in legends], expected_names, f"{label}.legends 名称")
    for index, item in enumerate(legends):
        report.check(item.get("url") == f"data/legends/legend-{index}{Path(expected_names[index]).suffix}",
                     f"{label}.legends[{index}].url 无效")
        relative = str(item.get("url", ""))[len("data/"):]
        report.check((OUT / relative).is_file(), f"{label}.legends[{index}] 文件不存在")
    return catalog


def verify_health(report):
    label = "health.json"
    health = load_json(OUT / label)
    report.equal(health.get("status"), "ok", f"{label}.status")
    report.equal(health.get("static"), True, f"{label}.static")
    report.equal(health.get("adminTokenRequired"), False, f"{label}.adminTokenRequired")
    report.equal(health.get("routingSchema"), ROUTING_SCHEMA, f"{label}.routingSchema")
    report.equal(health.get("terrainSchema"), TERRAIN_SCHEMA, f"{label}.terrainSchema")
    frame = health.get("frame") or {}
    report.equal(frame.get("crs"), "WGS84", f"{label}.frame.crs")
    origin = frame.get("origin")
    report.check(finite_point(origin), f"{label}.frame.origin 无效")
    if finite_point(origin):
        report.check(abs(origin[0] - Frame().lng) < 1e-9 and abs(origin[1] - Frame().lat) < 1e-9,
                     f"{label}.frame.origin 与 Frame 原点不符")
    report.check(finite_number(health.get("magneticDeclinationDeg")), f"{label}.magneticDeclinationDeg")
    try:
        model_date = date.fromisoformat(health.get("modelDate"))
        report.check(True, f"{label}.modelDate 无效")
    except (TypeError, ValueError):
        model_date = None
        report.check(False, f"{label}.modelDate 不是 ISO 日期: {health.get('modelDate')!r}")
    table = health.get("declinationTable")
    if not report.check(isinstance(table, list) and len(table) == 72,
                        f"{label}.declinationTable 应为 72 条，实际 "
                        f"{len(table) if isinstance(table, list) else table}"):
        return
    expected_months = []
    cursor = date(2025, 1, 1)
    while cursor <= date(2030, 12, 1):
        expected_months.append(cursor.isoformat())
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    report.equal([item.get("date") for item in table], expected_months,
                 f"{label}.declinationTable 月份序列")
    for index, item in enumerate(table):
        report.check(finite_number(item.get("deg")) and -30 <= item["deg"] <= 30,
                     f"{label}.declinationTable[{index}].deg 无效")
    if model_date is not None:
        report.check("2025-01-01" <= model_date.isoformat() <= "2030-12-31",
                     f"{label}.modelDate 超出表范围")


def verify_reference(report):
    label = "reference.json"
    collection = load_json(OUT / label)
    report.equal(collection.get("type"), "FeatureCollection", f"{label}.type")
    features = collection.get("features")
    if not report.check(isinstance(features, list) and len(features) > 100, f"{label}.features 数量不足"):
        return
    kinds = set()
    for index, feature in enumerate(features):
        if not report.check(isinstance(feature, dict) and isinstance(feature.get("geometry"), dict),
                            f"{label}.features[{index}] 无效"):
            return
        geometry = feature["geometry"]
        if not report.check(geometry.get("type") in ("LineString", "Polygon"),
                            f"{label}.features[{index}] 几何类型"):
            return
        coordinates = geometry.get("coordinates") or []
        points = coordinates if geometry["type"] == "LineString" else coordinates[0]
        if not report.check(isinstance(points, list) and len(points) >= 2,
                            f"{label}.features[{index}] 坐标不足"):
            return
        if not all(isinstance(point, list) and len(point) == 2
                   and all(finite_number(value) for value in point) for point in points):
            report.check(False, f"{label}.features[{index}] 坐标含非有限值")
            return
        kinds.add((feature.get("properties") or {}).get("kind"))
    report.check({"building", "water", "road"} & kinds, f"{label}: 缺少预期地物类型")


def verify_releases(report, catalog):
    label = "releases.json"
    index = load_json(OUT / label)
    releases = index.get("releases")
    if not report.check(isinstance(releases, list) and releases, f"{label}.releases 应为非空列表"):
        return
    graph_hashes = {}
    states = {}
    for asset_id in MAP_IDS:
        graph_metadata = load_json(OUT / "maps" / asset_id / "graph.json")
        graph_hashes[asset_id] = graph_metadata.get("graphHash")
        states[asset_id] = load_json(OUT / "maps" / asset_id / "state.json")
    found_expected = False
    for entry in releases:
        release_id = entry.get("id")
        label_entry = f"{label}.releases[{release_id}]"
        report.check(is_hex32(release_id), f"{label_entry}.id 不是 32 位十六进制")
        report.check(isinstance(entry.get("title"), str) and entry["title"], f"{label_entry}.title")
        report.check(entry.get("assetId") in MAP_IDS, f"{label_entry}.assetId 无效")
        report.check(finite_number(entry.get("createdAt")), f"{label_entry}.createdAt 无效")
        path = OUT / "releases" / f"{release_id}.json"
        if not report.check(path.is_file(), f"{label_entry}: 发布课程文件不存在"):
            continue
        release = load_json(path)
        report.equal(release.get("id"), release_id, f"{label_entry}: 文件 id 不一致")
        report.check(is_hex32(release.get("id")), f"{label_entry}: 文件 id 格式无效")
        report.equal(release.get("static"), True, f"{label_entry}.static")
        report.equal(release.get("superseded"), False, f"{label_entry}.superseded")
        report.equal(release.get("title"), entry.get("title"), f"{label_entry}.title 与索引不一致")
        report.equal(release.get("assetId"), entry.get("assetId"), f"{label_entry}.assetId 与索引不一致")
        report.equal(release.get("createdAt"), entry.get("createdAt"), f"{label_entry}.createdAt 与索引不一致")
        report.equal(release.get("routingSchema"), ROUTING_SCHEMA, f"{label_entry}.routingSchema")
        asset_id = release.get("assetId")
        if asset_id not in MAP_IDS:
            continue
        state = states[asset_id]
        graph_metadata = load_json(OUT / "maps" / asset_id / "graph.json")
        report.equal(release.get("version"), state.get("version"), f"{label_entry}.version")
        report.equal(release.get("frame"), state.get("frame"), f"{label_entry}.frame")
        report.equal(release.get("extent"), state.get("extent"), f"{label_entry}.extent")
        release_controls = release.get("controls") or []
        state_controls = state.get("controls") or []
        report.equal([control.get("id") for control in release_controls],
                     [control.get("id") for control in state_controls],
                     f"{label_entry}.controls id 与地图状态不一致")
        report.equal([control.get("label") for control in release_controls],
                     [control.get("label") for control in state_controls],
                     f"{label_entry}.controls 编号与地图状态不一致")
        plan = release.get("plan") or {}
        report.check(isinstance(plan.get("routes"), list) and plan["routes"], f"{label_entry}.plan.routes 无效")
        report.equal(plan.get("version"), release.get("version"), f"{label_entry}.plan.version")
        report.equal(plan.get("graphHash"), graph_metadata.get("graphHash"),
                     f"{label_entry}.plan.graphHash 与 graph.json 不一致")
        report.equal(plan.get("proof", {}).get("graphHash"), graph_metadata.get("graphHash"),
                     f"{label_entry}.plan.proof.graphHash 与 graph.json 不一致")
        plan_id = plan.get("id")
        if isinstance(plan_id, str) and asset_id:
            deterministic = hashlib.sha256(
                f"static-release:{asset_id}:{plan_id}".encode("utf-8")).hexdigest()[:32]
            report.equal(release_id, deterministic, f"{label_entry}: id 不是确定性发布 id")
        overlay = release.get("overlay")
        report.check(isinstance(overlay, str) and overlay.startswith(f"data/maps/{asset_id}/"),
                     f"{label_entry}.overlay 路径无效")
        if isinstance(overlay, str) and overlay.startswith("data/"):
            report.check((OUT / overlay[len("data/"):]).is_file(), f"{label_entry}.overlay 文件不存在")
        if asset_id == "szu-3" and plan_id == RELEASE_PLAN_ID:
            found_expected = True
            report.equal(plan.get("graphHash"), EXPECTED_SZU3_HASH,
                         f"{label_entry}: 发布课程未使用记录的通行图哈希")
    report.check(found_expected, f"{label}: 缺少 szu-3/{RELEASE_PLAN_ID} 的发布课程")
    for relative in (OUT / "releases").glob("*.json"):
        report.check(is_hex32(relative.stem), f"releases/{relative.name} 文件名不是 32 位十六进制")


def verify_optimizer_fixture(report):
    label = "tests/fixtures/static/optimizer.json"
    if not report.check((FIXTURES / "optimizer.json").is_file(), f"{label} 不存在"):
        return
    fixture = load_json(FIXTURES / "optimizer.json")
    report.check((FIXTURES / "optimizer.json").stat().st_size < MAX_FIXTURE_BYTES,
                 f"{label} 超过 2 MB")
    report.equal(fixture.get("assetId"), "szu-1", f"{label}.assetId")
    graph_metadata = load_json(OUT / "maps" / "szu-1" / "graph.json")
    report.equal(fixture.get("graphHash"), graph_metadata.get("graphHash"), f"{label}.graphHash")
    request = fixture.get("request") or {}
    report.equal(request.get("mode"), "all_visit", f"{label}.request.mode")
    report.equal(request.get("objective"), "min_max_finish_then_sum", f"{label}.request.objective")
    report.equal(len(request.get("teams") or []), 3, f"{label}.request.teams 数量")
    report.equal(request.get("timeLimitSec"), 5, f"{label}.request.timeLimitSec")
    control_ids = request.get("controlIds") or []
    matrix = fixture.get("matrix")
    size = len(control_ids) + 2
    if not report.check(isinstance(matrix, list) and len(matrix) == size,
                        f"{label}.matrix 尺寸无效"):
        return
    for index, row in enumerate(matrix):
        report.check(isinstance(row, list) and len(row) == size
                     and all(finite_number(value) for value in row),
                     f"{label}.matrix[{index}] 无效")
    expected = fixture.get("expected") or {}
    report.equal(expected.get("status"), "OPTIMAL", f"{label}.expected.status")
    report.equal(expected.get("proof", {}).get("algorithm"), "held_karp",
                 f"{label}.expected.proof.algorithm")
    expected_routes = expected.get("routes") or []
    report.equal(len(expected_routes), 3, f"{label}.expected.routes 数量")
    controls = [{"id": control_id} for control_id in control_ids]
    teams = request.get("teams") or []
    try:
        result = solve(np.asarray(matrix, dtype=np.float64), controls, teams,
                       request.get("mode"), request.get("objective"), request.get("timeLimitSec"))
    except Exception as error:  # noqa: BLE001 - report and continue
        report.check(False, f"{label}: 重新求解优化基准失败: {error}")
        return
    report.equal(result.get("status"), expected.get("status"), f"{label}: 重新求解状态不一致")
    report.equal(result.get("proof", {}).get("algorithm"), expected.get("proof", {}).get("algorithm"),
                 f"{label}: 重新求解算法分支不一致")
    for index, route in enumerate(result.get("routes") or []):
        expected_route = expected_routes[index] if index < len(expected_routes) else {}
        report.equal(route.get("teamId"), expected_route.get("teamId"), f"{label}.routes[{index}].teamId")
        report.equal(route.get("controlIds"), expected_route.get("controlIds"),
                     f"{label}.routes[{index}].controlIds")
        report.equal([int(value) for value in route.get("order") or []],
                     expected_route.get("order"), f"{label}.routes[{index}].order")
    report.check((FIXTURES / "optimizer.json").stat().st_size < MAX_FIXTURE_BYTES,
                 f"{label} 超过 2 MB")


def verify_fixture_sizes(report):
    if not report.check(FIXTURES.is_dir(), "tests/fixtures/static/ 不存在"):
        return
    for path in sorted(FIXTURES.glob("*.json")):
        report.check(path.stat().st_size < MAX_FIXTURE_BYTES, f"{path.name} 超过 2 MB")
        report.check(path.stat().st_size > 0, f"{path.name} 为空")
    for asset_id in MAP_IDS:
        for suffix in ("dijkstra", "matrix"):
            report.check((FIXTURES / f"{asset_id}.{suffix}.json").is_file(),
                         f"缺少 fixtures/{asset_id}.{suffix}.json")


def print_summary(report, summaries, catalog):
    print()
    print("静态数据摘要")
    print(f"{'地图':<8}{'画幅':<12}{'节点':>9}{'有向边':>10}{'连通块':>8}{'到访点':>8}"
          f"{'graph.bin.gz':>13}{'original.png':>13}{'semantics':>11}  图哈希")
    for asset_id in MAP_IDS:
        summary = summaries.get(asset_id)
        if not summary:
            print(f"{asset_id:<8}(未校验)")
            continue
        print(f"{asset_id:<8}{str(summary['width']) + 'x' + str(summary['height']):<12}"
              f"{summary['nodes']:>9}{summary['edges']:>10}{summary['components']:>8}"
              f"{summary['approaches']:>8}{human_size(summary['graph_size']):>13}"
              f"{human_size(summary.get('png_size', 0)):>13}"
              f"{human_size(summary.get('semantics_size', 0)):>11}  {summary['graph_hash']}")
    if catalog:
        print(f"目录地图 {len(catalog.get('maps') or [])} 张，图例 {len(catalog.get('legends') or [])} 个，"
              f"generatedAt {catalog.get('generatedAt')}")
    reference_path = OUT / "reference-features.json"
    if reference_path.is_file():
        try:
            count = len(load_json(reference_path).get("features") or [])
        except (OSError, ValueError):
            count = -1
        print(f"reference-features.json {human_size(reference_path.stat().st_size)}，"
              f"建筑/水体/球场地物 {count} 个")
    client_path = FIXTURES / "client-processing.json"
    if client_path.is_file():
        print(f"client-processing.json {human_size(client_path.stat().st_size)}")
    fixture_size = sum(path.stat().st_size for path in FIXTURES.glob("*.json")) if FIXTURES.is_dir() else 0
    data_size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) if OUT.is_dir() else 0
    print(f"数据 {human_size(data_size)}，fixtures {human_size(fixture_size)}，检查 {report.checks} 项")


def main():
    report = Report()
    summaries = {}
    catalog = None
    if not report.check(OUT.is_dir(), f"缺少 {OUT}"):
        print_summary(report, summaries, None)
        return 1
    try:
        catalog = verify_catalog(report, summaries)
        for asset_id in MAP_IDS:
            entry = next((item for item in (catalog.get("maps") or [])
                          if item.get("id") == asset_id), {})
            try:
                verify_map(report, asset_id, entry, summaries)
            except Exception as error:  # noqa: BLE001 - keep checking the rest
                report.check(False, f"{asset_id}: 校验中断: {type(error).__name__}: {error}")
        verify_health(report)
        verify_reference(report)
        reference_feature_total = verify_reference_features(report)
        verify_client_processing(report)
        verify_releases(report, catalog)
        verify_optimizer_fixture(report)
        verify_fixture_sizes(report)
    except Exception as error:  # noqa: BLE001 - malformed bundle must not crash with a traceback
        report.check(False, f"校验中断: {type(error).__name__}: {error}")
    print_summary(report, summaries, catalog)
    if report.errors:
        print()
        for index, message in enumerate(report.errors, 1):
            print(f"FAIL {index:>3}: {message}")
        print(f"\n{len(report.errors)} 项失败 / {report.checks} 项检查")
        return 1
    print(f"\n全部 {report.checks} 项检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
