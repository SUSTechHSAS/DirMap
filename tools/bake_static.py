#!/usr/bin/env python3
"""Bake the processed DirMap project data into the static GitHub Pages bundle.

Contract: docs/static-architecture.md §4-§6.  The browser never runs Python:
this script converts `data/projects/<asset>/` plus the referenced `derived/`
artifacts into `web/public/data/`, and writes the golden routing fixtures into
`tests/fixtures/static/`.

Run with:
    PYTHONPATH=.pydeps DIRMAP_CACHE_DIR=$PWD/.cache python3 tools/bake_static.py
Optional flags:
    --asset szu-1|szu-2|szu-3   bake only the given map(s) (repeatable)
    --skip-fixtures             do not rewrite tests/fixtures/static/

The output is deterministic: no wall-clock timestamps are written, the only
date-like fields (`catalog.generatedAt`, `health.modelDate`) are derived from
the newest source `state.json.updatedAt`, and every collection is written in a
stable order with sorted JSON keys.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import struct
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"
FIXTURES = ROOT / "tests" / "fixtures" / "static"
LEGEND_FILES = [ROOT / "ISSprOM2019-2图例.pdf", ROOT / "____________aigc_label.png"]
RELEASE_ASSET = "szu-3"
RELEASE_PLAN_ID = "5684df3f33f04390"
RELEASE_TITLE = "深大定向越野课"
EXPECTED_SZU3_GRAPH_HASH = "2570e1372e5059e7a540"
MAGIC = b"DMGRAPH1"
FORMAT_VERSION = 1
GRAPH_STEP = 4
APPROACH_NOTE = (
    "到访位置在构建时按“起终点符号所在通行区域”预解算：先取无约束的合法到访位置，"
    "若其通行区域与起点不同，再按起点区域重新解算；失败时保存中文原因。"
    "浏览器须校验每个 ok 节点的 component 是否等于本文件的 startComponent，"
    "不一致时按“合法到访侧与起点不连通，请核对通道”处理。"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dirmap import service, store  # noqa: E402  (ROOT is needed on sys.path first)
from dirmap.access import AccessResolver  # noqa: E402
from dirmap.geometry import Frame, Registration  # noqa: E402
from dirmap.graph import build_graph, plan_matrix  # noqa: E402
from dirmap.optimizer import balanced_people, solve  # noqa: E402
from dirmap.protocol import ROUTING_SCHEMA, TERRAIN_SCHEMA  # noqa: E402
from dirmap.reference import geojson  # noqa: E402

START_ROLES = ("start", "start_finish", "finish")
KEEP_STATE_KEYS = (
    "warnings", "statistics", "quality", "registration", "frame", "extent",
    "controls", "edits", "version", "reviewed", "terrainSchema", "sourceHash",
    "name", "filename", "width", "height", "assetId",
)


def log(message):
    print(f"[bake] {message}", flush=True)


def fail(message):
    print(f"[bake] ERROR: {message}", flush=True)
    raise SystemExit(1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(256 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False)
    write_bytes(path, (text + "\n").encode("utf-8"))


def write_gzip_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n"
    write_bytes(path, gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))


def human_size(count):
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}GiB"


def directory_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def newest_updated_at(states):
    stamps = [float(state["updatedAt"]) for state in states.values()
              if state and isinstance(state.get("updatedAt"), (int, float))]
    if not stamps:
        return datetime(2025, 1, 1, tzinfo=timezone.utc)
    return datetime.fromtimestamp(max(stamps), tz=timezone.utc)


def decimal_year(value):
    return value.year + (value - date(value.year, 1, 1)).days / (
        date(value.year + 1, 1, 1) - date(value.year, 1, 1)
    ).days


# --------------------------------------------------------------------------- #
# Graph data: build once per map and export every representation from it.
# --------------------------------------------------------------------------- #


def build_map(asset_id):
    """Build the routing graph exactly like `dirmap.service._get_graph`."""
    asset = store.get_asset(asset_id)
    state = store.get_state(asset_id)
    if not state:
        fail(f"{asset_id}: 缺少 data/projects/{asset_id}/state.json")
    if state.get("assetId") != asset_id:
        fail(f"{asset_id}: state.json 的 assetId 不一致")
    if state.get("sourceHash") != asset["sha256"]:
        fail(f"{asset_id}: 原图哈希与 state.json 不一致，请重新处理地图")
    if not state.get("registration"):
        fail(f"{asset_id}: 缺少配准数据，无法构建通行图")
    if state.get("terrainSchema") != TERRAIN_SCHEMA:
        fail(f"{asset_id}: 通行识别规则已更新（{state.get('terrainSchema')} != {TERRAIN_SCHEMA}）")
    terrain_path = store.artifact_path(state, "terrain.npz")
    with np.load(terrain_path) as artifact:
        costs = np.array(artifact["costs"], dtype=np.float32, copy=True)
        semantics = {kind: artifact[kind] for kind in
                     ("building", "water", "point_ink", "vegetation", "restricted")
                     if kind in artifact}
        if "costs" not in artifact:
            fail(f"{asset_id}: terrain.npz 缺少 costs")
    registration = Registration.from_dict(state["registration"])
    started = time.perf_counter()
    graph = build_graph(costs, registration, state["sourceHash"], state.get("edits") or [], step=GRAPH_STEP)
    graph.access_resolver = AccessResolver(graph, registration, semantics)
    elapsed = time.perf_counter() - started
    recorded = state.get("graphHash")
    if recorded and recorded != graph.content_hash:
        fail(f"{asset_id}: 通行图哈希 {graph.content_hash} 与 state.json 记录的 {recorded} 不一致")
    if asset_id == RELEASE_ASSET and graph.content_hash != EXPECTED_SZU3_GRAPH_HASH:
        fail(f"{asset_id}: 通行图哈希 {graph.content_hash} != 预期的 {EXPECTED_SZU3_GRAPH_HASH}")
    height, width = costs.shape
    if (state["width"], state["height"]) != (width, height):
        fail(f"{asset_id}: state.json 画幅与 terrain.npz 不一致")
    palette = np.unique(costs)
    if palette.size < 2 or palette.size > 256 or float(palette[0]) != 0.0:
        fail(f"{asset_id}: 通行代价调色板无效: {palette.tolist()}")
    cost_index = np.searchsorted(palette, costs)
    if not np.array_equal(palette[cost_index], costs):
        fail(f"{asset_id}: 通行代价不能无损映射到调色板")
    log(f"{asset_id}: 图 {width}x{height}, 节点 {len(graph.pixels)}, CSR 边 {graph.weights.nnz}, "
        f"连通块 {graph.component_count}, 哈希 {graph.content_hash} ({elapsed:.2f}s)")
    return {
        "assetId": asset_id,
        "asset": asset,
        "state": state,
        "graph": graph,
        "registration": registration,
        "costs": costs,
        "palette": palette.astype(np.float32),
        "costIndex": cost_index.astype(np.uint8),
        "width": width,
        "height": height,
    }


def export_bundle(info):
    """Serialize the GraphBundle (§5.7) and the node arrays used by fixtures."""
    graph = info["graph"]
    pixels = np.asarray(graph.pixels, dtype=np.float64)
    sizes = np.asarray(graph.cell_sizes, dtype=np.float32)
    count = len(pixels)
    if count != len(sizes) or count != len(graph.local):
        fail(f"{info['assetId']}: 节点数组长度不一致")
    if not np.array_equal(np.rint(pixels * 2.0), pixels * 2.0):
        fail(f"{info['assetId']}: 节点中心不是精确半像素")
    if not np.array_equal(pixels.astype(np.float32).astype(np.float64), pixels):
        fail(f"{info['assetId']}: 节点中心不能无损存为 float32")
    if not np.array_equal(sizes, np.rint(sizes)):
        fail(f"{info['assetId']}: 单元格尺寸不是整数")
    sizes_u8 = np.rint(sizes).astype(np.uint8)
    if not np.isin(sizes_u8, (1, 2, 4, 8, 16)).all():
        fail(f"{info['assetId']}: 单元格尺寸意外: {np.unique(sizes_u8).tolist()}")
    # Terrain of each leaf = the uniform cost value at its top-left native pixel.
    x0 = np.rint(pixels[:, 0] - (sizes - 1) / 2).astype(np.int64)
    y0 = np.rint(pixels[:, 1] - (sizes - 1) / 2).astype(np.int64)
    node_costs = info["costs"][y0, x0]
    terrain_index = np.searchsorted(info["palette"], node_costs)
    if not np.array_equal(info["palette"][terrain_index], node_costs):
        fail(f"{info['assetId']}: 节点地物代价不能映射到调色板")
    terrain_index = terrain_index.astype(np.uint8)
    float_palette = info["palette"].astype("<f4")
    header = MAGIC + struct.pack(
        "<HIIIH", FORMAT_VERSION, info["width"], info["height"], count, len(float_palette)
    )
    raw = b"".join((
        header,
        float_palette.tobytes(),
        info["costIndex"].astype(np.uint8).tobytes(order="C"),
        pixels[:, 0].astype("<f4").tobytes(),
        pixels[:, 1].astype("<f4").tobytes(),
        sizes_u8.tobytes(),
        terrain_index.tobytes(),
    ))
    return {
        "raw": raw,
        "count": count,
        "sizes": sizes_u8,
        "terrain": terrain_index,
        "nodeCosts": node_costs.astype(np.float32),
        "pixels": pixels,
    }


def rebuild_edges(info, exported):
    """Reproduce `_quadtree_graph` adjacency from the exported node arrays only.

    This is the same algorithm the browser runs, and gives the `expected`
    counters of graph.json plus the cross-check of the baked node order.
    """
    graph = info["graph"]
    width, height = info["width"], info["height"]
    count = exported["count"]
    pixels = exported["pixels"]
    sizes = exported["sizes"]
    local = np.asarray(graph.local, dtype=np.float64)
    owner = np.full((height, width), -1, np.int32)
    x0 = np.rint(pixels[:, 0] - (sizes.astype(np.float64) - 1) / 2).astype(np.int64)
    y0 = np.rint(pixels[:, 1] - (sizes.astype(np.float64) - 1) / 2).astype(np.int64)
    for size in sorted({int(value) for value in np.unique(sizes)}):
        ids = np.flatnonzero(sizes == size)
        for dy in range(size):
            for dx in range(size):
                owner[y0[ids] + dy, x0[ids] + dx] = ids
    if not np.array_equal(owner, np.asarray(graph.owner)):
        fail(f"{info['assetId']}: 导出的节点数组无法重建服务器 owner 网格")
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
    a = (pairs // count).astype(np.int64)
    b = (pairs % count).astype(np.int64)
    lengths = np.hypot(local[b, 0] - local[a, 0], local[b, 1] - local[a, 1])
    sizes32 = sizes.astype(np.float32)
    terrain = exported["nodeCosts"].astype(np.float32)
    ratio = (terrain[a] * sizes32[a] + terrain[b] * sizes32[b]) / (sizes32[a] + sizes32[b])
    weights = lengths * ratio
    return {"a": a, "b": b, "lengths": lengths.astype(np.float64),
            "weights": weights.astype(np.float64), "owner": owner}


def check_expected(info, exported, rebuilt):
    graph = info["graph"]
    count = exported["count"]
    if int(graph.weights.nnz) != 2 * len(rebuilt["a"]):
        fail(f"{info['assetId']}: 重建边数 {len(rebuilt['a'])} != CSR 边数/2 {graph.weights.nnz // 2}")
    length_sum = float(np.sum(rebuilt["lengths"]))
    weight_sum = float(np.sum(rebuilt["weights"]))
    server_length_sum = float(np.sum(graph.lengths.data)) / 2.0
    server_weight_sum = float(np.sum(graph.weights.data)) / 2.0
    for name, value, server in (("lengthSum", length_sum, server_length_sum),
                                ("weightSum", weight_sum, server_weight_sum)):
        if abs(value - server) > 1e-9 * max(1.0, abs(server)):
            fail(f"{info['assetId']}: 重建 {name} {value} 与服务器 {server} 不一致")
    expected = {"lengthSum": length_sum, "weightSum": weight_sum, "edgeCount": int(len(rebuilt["a"]))}
    components = _component_count(count, rebuilt)
    if components != int(graph.component_count):
        fail(f"{info['assetId']}: 重建连通块 {components} != 服务器 {graph.component_count}")
    return expected, components


def _component_count(count, rebuilt):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a, b = rebuilt["a"], rebuilt["b"]
    if not len(a):
        return count
    matrix = coo_matrix((np.ones(2 * len(a), np.float64),
                         (np.concatenate((a, b)), np.concatenate((b, a)))),
                        shape=(count, count)).tocsr()
    return int(connected_components(matrix, directed=False)[0])


def select_samples(info):
    """~40 reachable node pairs; deterministic and reproducible from the data."""
    graph = info["graph"]
    count = len(graph.pixels)
    components = np.asarray(graph.components)
    candidates = []

    def add(node):
        node = int(node)
        if 0 <= node < count and node not in candidates:
            candidates.append(node)

    add(0)
    for control in info["state"]["controls"]:
        if control.get("role", "control") != "control" or control.get("state") == "rejected":
            continue
        entry = info["approaches"].get(control["id"])
        if entry and entry["ok"]:
            add(entry["node"])
    add(info["startNode"])
    largest = int(np.bincount(components).argmax())
    owned = np.flatnonzero(components == largest)
    stride = max(1, len(owned) // 9)
    for index in range(9):
        add(int(owned[min(len(owned) - 1, index * stride)]))
    if len(candidates) < 4:
        fail(f"{info['assetId']}: 可用节点候选不足")
    groups = {}
    for source in candidates:
        targets = [target for target in candidates
                   if target != source and components[target] == components[source]]
        if len(targets) >= 2:
            groups[source] = targets
    sources = list(groups)[:6]
    if len(sources) < 2:
        fail(f"{info['assetId']}: 连通节点来源不足")
    chosen = []
    round_index = 0
    while len(chosen) < 40 and round_index < 40:
        progressed = False
        for source in sources:
            targets = groups[source]
            if round_index < len(targets):
                chosen.append((source, targets[round_index]))
                progressed = True
                if len(chosen) >= 40:
                    break
        if not progressed:
            break
        round_index += 1
    if len(chosen) < 40:
        fail(f"{info['assetId']}: 可用的连通节点对不足 40 个（{len(chosen)}）")
    samples = []
    for source in sorted({pair[0] for pair in chosen}):
        targets = [pair[1] for pair in chosen if pair[0] == source]
        time_row = graph.shortest_paths([source], "time", targets=targets)[0][0]
        distance_row = graph.shortest_paths([source], "distance", targets=targets)[0][0]
        for target in targets:
            time_value = float(time_row[target])
            distance_value = float(distance_row[target])
            if not (math.isfinite(time_value) and math.isfinite(distance_value)):
                continue
            samples.append({"from": int(source), "to": int(target),
                            "time": time_value, "distance": distance_value})
    if len(samples) < 40:
        fail(f"{info['assetId']}: 只得到 {len(samples)} 条有效最短路样本")
    return samples[:40]


# --------------------------------------------------------------------------- #
# Approaches (§5.5) and the plan matrix golden scenario (§7.4).
# --------------------------------------------------------------------------- #


def canonical_endpoints(state):
    controls = [control for control in state["controls"]
                if control.get("role", "control") == "control" and control.get("state") != "rejected"]
    symbols = [control for control in state["controls"]
               if control.get("role") in START_ROLES and control.get("state") != "rejected"]
    if not symbols:
        fail(f"{state['assetId']}: 缺少起终点符号，无法确定矩阵起点")
    symbol = symbols[0]
    pixel = symbol.get("approachPixel") or symbol["pixel"]
    return controls, symbol, [float(pixel[0]), float(pixel[1])]


def resolve_approach_entry(info, control, required_component):
    graph, registration = info["graph"], info["registration"]
    try:
        resolved = graph.access_resolver.resolve(control, required_component)
    except ValueError as error:
        return {"ok": False, "message": str(error)}
    node = int(resolved["node"])
    local = [float(value) for value in resolved["local"]]
    target_pixel = control["pixel"]
    target_local, valid = registration.forward([target_pixel])
    entry = {
        "ok": True,
        "node": node,
        "pixel": [float(value) for value in resolved["pixel"]],
        "local": local,
        "method": resolved["method"],
        "automatic": bool(resolved["automatic"]),
        "offsetM": float(resolved["offsetM"]),
        "lnglat": [float(value) for value in Frame().to_lnglat(np.asarray(local, dtype=np.float64))],
        "targetLnglat": [float(value) for value in Frame().to_lnglat(target_local[0])] if valid[0] else None,
        "targetPixel": [float(value) for value in target_pixel],
        "component": int(graph.components[node]),
    }
    if "featureKind" in resolved:
        entry["featureKind"] = resolved["featureKind"]
    if "reviewRequired" in resolved:
        entry["reviewRequired"] = bool(resolved["reviewRequired"])
    return entry


def bake_approaches(info):
    """Per-control legal approach, compatible with the canonical start component."""
    graph = info["graph"]
    controls, symbol, start_pixel = canonical_endpoints(info["state"])
    start_node = int(graph.snap(start_pixel))
    start_component = int(graph.components[start_node])
    entries = {}
    for control in controls:
        entry = resolve_approach_entry(info, control, None)
        if entry["ok"] and entry["component"] != start_component:
            entry = resolve_approach_entry(info, control, start_component)
        entries[control["id"]] = entry
    info["approaches"] = entries
    info["startNode"] = start_node
    info["startComponent"] = start_component
    info["startPixel"] = start_pixel
    info["finishPixel"] = start_pixel
    info["symbolControl"] = symbol["id"]
    return {
        "assetId": info["assetId"],
        "routingSchema": ROUTING_SCHEMA,
        "graphHash": graph.content_hash,
        "startPixel": start_pixel,
        "finishPixel": start_pixel,
        "startNode": start_node,
        "startComponent": start_component,
        "note": APPROACH_NOTE,
        "controls": entries,
    }


def bake_matrix(info):
    controls, symbol, start_pixel = canonical_endpoints(info["state"])
    graph, registration = info["graph"], info["registration"]
    bundle = plan_matrix(graph, controls, start_pixel, start_pixel, registration)
    result = {
        "assetId": info["assetId"],
        "graphHash": graph.content_hash,
        "metric": "time",
        "request": {
            "assetId": info["assetId"],
            "version": int(info["state"]["version"]),
            "controlIds": [control["id"] for control in controls],
            "startPixel": start_pixel,
            "finishPixel": start_pixel,
            "metric": "time",
        },
        "scenario": {
            "controlIds": [control["id"] for control in controls],
            "startPixel": start_pixel,
            "finishPixel": start_pixel,
            "metric": "time",
        },
        "start": {"controlId": symbol["id"], "pixel": start_pixel},
        "startPixel": start_pixel,
        "finishPixel": start_pixel,
        "startNode": int(graph.snap(start_pixel)),
        "nodes": [],
        "endpoints": [],
        "connectors": [],
        "connectorWeights": [],
        "matrix": None,
        "unresolved": [{"id": item["id"], "message": item["message"]} for item in bundle["unresolved"]],
        "approaches": [],
    }
    if "matrix" not in bundle:
        return result, None
    nodes = [int(node) for node in bundle["nodes"]]
    connectors = [float(value) for value in bundle["connectors"]]
    connector_weights = [float(value) for value in bundle["connectorWeights"]]
    endpoints = []
    for endpoint, node in zip(bundle["endpoints"], nodes):
        pixel = endpoint.get("approachPixel") or endpoint["pixel"]
        item = {"id": endpoint["id"], "pixel": [float(value) for value in pixel], "node": int(node)}
        if "approachPixel" in endpoint:
            item["approachPixel"] = [float(value) for value in endpoint["approachPixel"]]
            item["targetPixel"] = [float(value) for value in endpoint["pixel"]]
        endpoints.append(item)
    resolved_nodes = dict(zip([endpoint["id"] for endpoint in bundle["endpoints"]], nodes))
    approaches = []
    for entry in bundle["approaches"]:
        control_id = entry["controlId"]
        node = resolved_nodes.get(control_id)
        local = [float(value) for value in info["graph"].local[node]] if node is not None else None
        approaches.append({
            "controlId": control_id,
            "node": node,
            "targetPixel": [float(value) for value in entry.get("targetPixel") or []],
            "pixel": [float(value) for value in entry.get("pixel") or []],
            "local": local,
            "method": entry.get("method"),
            "automatic": bool(entry.get("automatic")),
            "offsetM": float(entry.get("offsetM") or 0.0),
            "lnglat": [float(value) for value in entry.get("lnglat") or []],
            "targetLnglat": [float(value) for value in entry["targetLnglat"]] if entry.get("targetLnglat") else None,
        })
    result.update(
        nodes=nodes,
        endpoints=endpoints,
        connectors=connectors,
        connectorWeights=connector_weights,
        matrix=[[round(float(value), 6) for value in row] for row in bundle["matrix"]],
        approaches=approaches,
    )
    return result, np.asarray(bundle["matrix"], dtype=np.float64)


def unresolved_js_simulation(info):
    """What the browser (which trusts baked approaches) must report for the matrix scenario."""
    graph = info["graph"]
    controls, _symbol, start_pixel = canonical_endpoints(info["state"])
    item = {"startNode": int(graph.snap(start_pixel))}
    start_component = int(graph.components[item["startNode"]])
    unresolved = []
    component_counts = np.bincount(graph.components)
    for control in controls:
        entry = info["approaches"][control["id"]]
        if not entry["ok"]:
            unresolved.append({"id": control["id"], "message": entry["message"]})
        elif int(entry["component"]) != start_component:
            count = int(component_counts[int(entry["component"])])
            unresolved.append({"id": control["id"],
                               "message": f"位于独立通行区域（{count} 个节点），检查门洞或目标的到访侧"})
    return unresolved


# --------------------------------------------------------------------------- #
# Output writers.
# --------------------------------------------------------------------------- #


def write_map_bundle(info, exported, expected, components, samples):
    asset_id = info["assetId"]
    graph = info["graph"]
    directory = OUT / "maps" / asset_id
    directory.mkdir(parents=True, exist_ok=True)
    state = info["state"]
    baked_state = {key: state[key] for key in KEEP_STATE_KEYS if key in state}
    baked_state["static"] = True
    baked_state["processed"] = True
    baked_state["graphHash"] = graph.content_hash
    write_json(directory / "state.json", baked_state)
    shutil.copyfile(info["asset"]["path"], directory / "original.jpg")
    for name in ("overlay.png", "terrain-warp.png", "terrain.png"):
        source = store.artifact_path(state, name)
        if not source.exists():
            fail(f"{asset_id}: 缺少派生文件 {name}")
        shutil.copyfile(source, directory / name)
    bundle_path = directory / "graph.bin.gz"
    write_bytes(bundle_path, gzip.compress(exported["raw"], compresslevel=9, mtime=0))
    metadata = {
        "assetId": asset_id,
        "schema": ROUTING_SCHEMA,
        "sourceHash": state["sourceHash"],
        "graphHash": graph.content_hash,
        "step": GRAPH_STEP,
        "adaptive": True,
        "width": info["width"],
        "height": info["height"],
        "nodeCount": int(exported["count"]),
        "directedEdgeCount": int(graph.weights.nnz),
        "componentCount": int(components),
        "costsPalette": [float(value) for value in info["palette"]],
        "expected": expected,
        "samples": samples,
    }
    write_json(directory / "graph.json", metadata)
    write_json(directory / "approaches.json", info["approaches_payload"] if "approaches_payload" in info
               else _approaches_payload(info))
    return metadata, bundle_path


def _approaches_payload(info):
    return {
        "assetId": info["assetId"],
        "routingSchema": ROUTING_SCHEMA,
        "graphHash": info["graph"].content_hash,
        "startPixel": info["startPixel"],
        "finishPixel": info["finishPixel"],
        "startNode": info["startNode"],
        "startComponent": info["startComponent"],
        "note": APPROACH_NOTE,
        "controls": info["approaches"],
    }


def write_map_fixtures(info, samples):
    asset_id = info["assetId"]
    graph = info["graph"]
    pixels = np.asarray(graph.pixels)
    dijkstra = {
        "assetId": asset_id,
        "graphHash": graph.content_hash,
        "schema": ROUTING_SCHEMA,
        "note": "图节点编号即 graph.bin.gz 的节点顺序；time 为权重米（metric=time），distance 为米（metric=distance）。",
        "samples": [
            {**sample,
             "fromPixel": [float(pixels[sample["from"], 0]), float(pixels[sample["from"], 1])],
             "toPixel": [float(pixels[sample["to"], 0]), float(pixels[sample["to"], 1])]}
            for sample in samples
        ],
    }
    write_json(FIXTURES / f"{asset_id}.dijkstra.json", dijkstra)


def write_optimizer_fixture(info, raw_matrix):
    asset_id = info["assetId"]
    controls, _symbol, start_pixel = canonical_endpoints(info["state"])
    people = balanced_people(15, 3)
    teams = [{"id": f"第 {index + 1} 队", "people": people[index], "speedMps": 1.4,
              "departureSec": [0.0, 20.0, 40.0][index]} for index in range(3)]
    request = {
        "assetId": asset_id,
        "version": int(info["state"]["version"]),
        "controlIds": [control["id"] for control in controls],
        "startPixel": start_pixel,
        "finishPixel": start_pixel,
        "mode": "all_visit",
        "objective": "min_max_finish_then_sum",
        "teams": teams,
        "timeLimitSec": 5,
    }
    started = time.perf_counter()
    result = solve(raw_matrix, controls, teams, "all_visit", "min_max_finish_then_sum", 5)
    elapsed = time.perf_counter() - started
    if result["status"] != "OPTIMAL" or result["proof"]["algorithm"] != "held_karp":
        fail(f"{asset_id}: 优化基准未走精确 DP（{result['status']} / {result['proof']['algorithm']}）")
    fixture = {
        "assetId": asset_id,
        "graphHash": info["graph"].content_hash,
        "controlIds": [control["id"] for control in controls],
        "startPixel": start_pixel,
        "finishPixel": start_pixel,
        "mode": "all_visit",
        "objective": "min_max_finish_then_sum",
        "request": request,
        "matrix": [[float(value) for value in row] for row in raw_matrix],
        "expected": {
            "status": result["status"],
            "routes": [{"teamId": route["teamId"],
                        "controlIds": route["controlIds"],
                        "order": [int(value) for value in route["order"]]}
                       for route in result["routes"]],
            "proof": {
                "algorithm": result["proof"]["algorithm"],
                "globalOptimalityProven": bool(result["proof"]["globalOptimalityProven"]),
                "objectiveValue": [float(value) for value in result["proof"]["objectiveValue"]],
            },
        },
    }
    # n=10 all_visit takes the exact Held-Karp branch, which the JS port also
    # implements: the full python solve is already the JS-reproducible answer.
    fixture["expectedSolveFromPython"] = json.loads(json.dumps(fixture["expected"]))
    write_json(FIXTURES / "optimizer.json", fixture)
    log(f"{asset_id}: 优化基准 {result['status']} / {result['proof']['algorithm']} "
        f"{len(result['routes'])} 队 ({elapsed:.2f}s)")


def write_catalog(states, built_metadata, generated_at):
    maps = []
    for asset in store.catalog():
        asset_id = asset["id"]
        state = states.get(asset_id)
        has_graph = (OUT / "maps" / asset_id / "graph.json").is_file()
        entry = {
            "id": asset_id,
            "name": asset["name"],
            "filename": asset["filename"],
            "sha256": asset["sha256"],
            "processed": bool(state),
            "version": int(state["version"]) if state else None,
            "controls": len(state["controls"]) if state else None,
            "courseDefaults": store.course_defaults(asset["sha256"]),
            "hasGraph": has_graph,
            "hasApproaches": (OUT / "maps" / asset_id / "approaches.json").is_file(),
        }
        maps.append(entry)
    legends = [{"name": path.name, "url": f"data/legends/legend-{index}{path.suffix}"}
               for index, path in enumerate(LEGEND_FILES)]
    catalog = {
        "static": True,
        "generatedAt": generated_at.isoformat(),
        "routingSchema": ROUTING_SCHEMA,
        "terrainSchema": TERRAIN_SCHEMA,
        "reference": {
            "crs": "WGS84",
            "name": "深圳大学粤海校区",
            "attribution": "© OpenStreetMap contributors",
            "licenseUrl": "https://www.openstreetmap.org/copyright",
        },
        "maps": maps,
        "legends": legends,
    }
    write_json(OUT / "catalog.json", catalog)
    return catalog


def write_health(generated_at):
    from pygeomag import GeoMag
    model = GeoMag()
    origin = Frame()
    table = []
    cursor = date(2025, 1, 1)
    while cursor <= date(2030, 12, 1):
        declination = model.calculate(glat=origin.lat, glon=origin.lng, alt=0,
                                      time=decimal_year(cursor),
                                      allow_date_outside_lifespan=True).d
        table.append({"date": cursor.isoformat(), "deg": round(float(declination), 6)})
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    current = model.calculate(glat=origin.lat, glon=origin.lng, alt=0,
                              time=decimal_year(generated_at.date()),
                              allow_date_outside_lifespan=True).d
    health = {
        "status": "ok",
        "version": "0.1.0-static",
        "frame": {**origin.as_dict(), "crs": "WGS84"},
        "static": True,
        "adminTokenRequired": False,
        "modelDate": generated_at.date().isoformat(),
        "magneticDeclinationDeg": round(float(current), 6),
        "magneticModel": "WMM-2025",
        "declinationTable": table,
        "routingSchema": ROUTING_SCHEMA,
        "terrainSchema": TERRAIN_SCHEMA,
    }
    write_json(OUT / "health.json", health)
    return health


def write_reference():
    features = service.reference()
    collection = geojson(features)
    write_json(OUT / "reference.json", collection)
    return len(collection["features"])


def write_legends():
    (OUT / "legends").mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(LEGEND_FILES):
        if not source.exists():
            fail(f"缺少图例文件 {source.name}")
        shutil.copyfile(source, OUT / "legends" / f"legend-{index}{source.suffix}")


def write_release(states, built_hashes):
    asset_id = RELEASE_ASSET
    state = states.get(asset_id)
    if not state:
        log("release: 缺少 szu-3 state.json，跳过发布课程")
        return None
    plan_path = store.asset_dir(asset_id) / f"plan-{RELEASE_PLAN_ID}.json"
    plan = store.read_json(plan_path)
    if not plan:
        log(f"release: 缺少 {plan_path.name}，跳过发布课程")
        return None
    graph_hash = built_hashes.get(asset_id) or state.get("graphHash")
    if not graph_hash:
        log("release: szu-3 没有已构建通行图哈希，跳过发布课程")
        return None
    reason = None
    if plan.get("version") != state.get("version"):
        reason = f"路线版本 {plan.get('version')} != 当前地图版本 {state.get('version')}"
    elif plan.get("graphHash") != graph_hash:
        reason = f"路线通行图哈希 {plan.get('graphHash')} != 当前 {graph_hash}"
    elif plan.get("routingSchema") != ROUTING_SCHEMA:
        reason = f"路线通行规则 {plan.get('routingSchema')} != {ROUTING_SCHEMA}"
    if reason:
        log(f"release: 跳过 {plan_path.name}（{reason}）；不生成会失效的学生课程")
        return None
    release_id = hashlib.sha256(
        f"static-release:{asset_id}:{plan['id']}".encode("utf-8")).hexdigest()[:32]
    frozen_plan = json.loads(json.dumps(plan))
    frozen_plan["graphHash"] = graph_hash
    if isinstance(frozen_plan.get("proof"), dict):
        frozen_plan["proof"]["graphHash"] = graph_hash
    release = {
        "id": release_id,
        "title": RELEASE_TITLE,
        "assetId": asset_id,
        "createdAt": float(plan["createdAt"]),
        "plan": frozen_plan,
        "controls": state["controls"],
        "frame": state["frame"],
        "extent": state["extent"],
        "quality": state.get("quality"),
        "review": state.get("review"),
        "version": int(state["version"]),
        "routingSchema": ROUTING_SCHEMA,
        "superseded": False,
        "static": True,
        "overlay": f"data/maps/{asset_id}/overlay.png",
    }
    write_json(OUT / "releases" / f"{release_id}.json", release)
    index = {"releases": [{"id": release_id, "title": RELEASE_TITLE,
                           "assetId": asset_id, "createdAt": float(plan["createdAt"])}]}
    write_json(OUT / "releases.json", index)
    log(f"release: {release_id}（{RELEASE_TITLE}，{asset_id} v{state['version']}，{len(plan['routes'])} 队）")
    return release


def clean_outputs(skip_fixtures):
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if not skip_fixtures and FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    if not skip_fixtures:
        FIXTURES.mkdir(parents=True, exist_ok=True)


def collect_states(asset_ids):
    states = {}
    for asset_id in asset_ids:
        state = store.get_state(asset_id)
        if state:
            states[asset_id] = state
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bake DirMap static data and golden fixtures")
    parser.add_argument("--asset", action="append", default=None,
                        help="asset id to bake (repeatable; default: all processed maps)")
    parser.add_argument("--skip-fixtures", action="store_true",
                        help="do not rewrite tests/fixtures/static/")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    catalog_entries = store.catalog()
    catalog_ids = [entry["id"] for entry in catalog_entries]
    requested = args.asset or catalog_ids
    unknown = [asset_id for asset_id in requested if asset_id not in catalog_ids]
    if unknown:
        fail(f"未知地图: {unknown}")
    partial = bool(args.asset)
    if not partial:
        clean_outputs(args.skip_fixtures)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        if not args.skip_fixtures:
            FIXTURES.mkdir(parents=True, exist_ok=True)

    states = collect_states(catalog_ids)
    generated_at = newest_updated_at(states)
    log(f"工件时间 {generated_at.isoformat()}（由最新 state.json.updatedAt 派生，保证确定性）")

    built = {}
    metadata = {}
    raw_matrices = {}
    samples_by_asset = {}
    fixture_count = 0
    for asset_id in requested:
        started_map = time.perf_counter()
        info = build_map(asset_id)
        exported = export_bundle(info)
        rebuilt = rebuild_edges(info, exported)
        expected, components = check_expected(info, exported, rebuilt)
        approaches_payload = bake_approaches(info)
        info["approaches_payload"] = approaches_payload
        samples = select_samples(info)
        samples_by_asset[asset_id] = samples
        matrix_fixture, raw_matrix = bake_matrix(info)
        graph_metadata, bundle_path = write_map_bundle(info, exported, expected, components, samples)
        metadata[asset_id] = graph_metadata
        built[asset_id] = info
        log(f"{asset_id}: graph.bin.gz {human_size(bundle_path.stat().st_size)}, "
            f"{len(samples)} 条最短路样本, 到访点 "
            f"{sum(1 for entry in info['approaches'].values() if entry['ok'])}/{(len(info['approaches']))}")
        if not args.skip_fixtures:
            write_map_fixtures(info, samples)
            write_json(FIXTURES / f"{asset_id}.matrix.json", matrix_fixture)
            fixture_count += 2
        js_unresolved = unresolved_js_simulation(info)
        if js_unresolved != matrix_fixture["unresolved"]:
            fail(f"{asset_id}: 浏览器预解算结果与 plan_matrix 不一致: "
                 f"{js_unresolved} != {matrix_fixture['unresolved']}")
        raw_matrices[asset_id] = raw_matrix
        log(f"{asset_id}: 矩阵 {'已解算 (' + str(len(matrix_fixture['nodes'])) + ' 端点)' if matrix_fixture['matrix'] else '未解决 ' + str(matrix_fixture['unresolved'])}"
            f"，耗时 {time.perf_counter() - started_map:.2f}s")

    write_legends()
    legend_count = len(LEGEND_FILES)
    reference_count = write_reference()
    health = write_health(generated_at)
    catalog = write_catalog(states, metadata, generated_at)

    if not args.skip_fixtures and not partial:
        if "szu-1" in raw_matrices:
            write_optimizer_fixture(built["szu-1"], raw_matrices["szu-1"])
            fixture_count += 1
    elif not args.skip_fixtures and partial and "szu-1" in raw_matrices:
        write_optimizer_fixture(built["szu-1"], raw_matrices["szu-1"])
        fixture_count += 1

    built_hashes = {asset_id: info["graph"].content_hash for asset_id, info in built.items()}
    release = write_release(states, built_hashes)

    log("")
    log("汇总 | 地图 | 尺寸 | 节点 | 有向边 | 连通块 | 到访点 ok | 图大小")
    for asset_id in requested:
        info, graph_metadata = built[asset_id], metadata[asset_id]
        ok = sum(1 for entry in info["approaches"].values() if entry["ok"])
        total = len(info["approaches"])
        size = (OUT / "maps" / asset_id / "graph.bin.gz").stat().st_size
        log(f"     | {asset_id} | {info['width']}x{info['height']} | {graph_metadata['nodeCount']} | "
            f"{graph_metadata['directedEdgeCount']} | {graph_metadata['componentCount']} | "
            f"{ok}/{total} | {human_size(size)}")
    log(f"数据目录 {OUT.relative_to(ROOT)}: {human_size(directory_size(OUT))}")
    if FIXTURES.exists():
        log(f"fixtures {FIXTURES.relative_to(ROOT)}: {human_size(directory_size(FIXTURES))}")
    log(f"图例 {legend_count} 个, 校园参照要素 {reference_count} 个, 目录地图 {len(catalog['maps'])} 张, "
        f"发布课程 {release['id'] if release else '无'}")
    log(f"总耗时 {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
