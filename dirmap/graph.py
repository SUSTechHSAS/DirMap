"""Native-pixel obstacle checks with a sparse, metre-weighted navigation grid."""

from dataclasses import dataclass
import hashlib
import json

import cv2
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .geometry import Frame
from .native import shortest_paths as native_shortest_paths
from .protocol import ROUTING_SCHEMA


def segment_pixels(a, b):
    """All native pixel squares touched by a segment, including corners.

    Sampling at a fixed spacing can miss a thin wall or the two orthogonal
    neighbours of a diagonal crossing. Split at the exact half-pixel grid
    boundaries, and inspect both sides when the segment touches a boundary.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError('线段坐标无效')
    delta = b-a
    times = [0., 1.]
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-12:
            continue
        low, high = sorted((a[axis], b[axis]))
        boundaries = np.arange(np.ceil(low-.5), np.floor(high-.5)+1)+.5
        times.extend(((boundaries-a[axis])/delta[axis]).tolist())
    times = np.unique(np.clip(times, 0, 1))
    sample_times = np.concatenate((times, (times[:-1]+times[1:])/2))
    points = a+sample_times[:, None]*delta
    cell = np.floor(points+.5+1e-9).astype(int)
    on_boundary = np.isclose(points+.5, np.rint(points+.5), rtol=0, atol=1e-8)
    touched = [cell]
    for dx, dy in ((1, 0), (0, 1), (1, 1)):
        keep = (on_boundary[:, 0] if dx else np.ones(len(cell), bool)) & (on_boundary[:, 1] if dy else True)
        touched.append(cell[keep]-[dx, dy])
    return np.unique(np.concatenate(touched), axis=0)


@dataclass
class RasterGraph:
    weights: csr_matrix
    lengths: csr_matrix
    pixels: np.ndarray
    local: np.ndarray
    costs: np.ndarray
    content_hash: str
    step: int
    adaptive: bool = False

    def __post_init__(self):
        self.tree = cKDTree(self.pixels)
        self.component_count, self.components = connected_components(self.weights, directed=False)

    def clear_segment(self, a, b):
        px = segment_pixels(a, b)
        if (px < 0).any() or (px[:, 0] >= self.costs.shape[1]).any() or (px[:, 1] >= self.costs.shape[0]).any():
            return False
        return bool(np.all(self.costs[px[:, 1], px[:, 0]] > 0))

    def node_at(self, pixel, max_distance_px=3, near=False):
        """Nearest node owning a free native pixel, keeping the exact position.

        A quadtree leaf centre can sit several metres from a point that hugs a
        wall, so searching only node centres rejects legal map positions. The
        ownership grid answers the same question at native resolution.

        `near=True` returns the closest free-owned pixel even when the straight
        line from `pixel` clips a blocked corner (a circle centre drawn inside
        a hatched corridor). The caller must keep such a move review-required.
        """
        point = np.asarray(pixel, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("到访坐标无效")
        owner = getattr(self, 'owner', None)
        if owner is None:
            return None
        height, width = owner.shape
        x, y = np.rint(point).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            return None
        limit = int(max(1, max_distance_px))
        closest = closest_distance = None
        for radius in range(limit+1):
            x0, x1 = max(0, x-radius), min(width, x+radius+1)
            y0, y1 = max(0, y-radius), min(height, y+radius+1)
            window = owner[y0:y1, x0:x1]
            hits = np.argwhere(window >= 0)
            if not len(hits):
                continue
            points = np.column_stack((hits[:, 1]+x0, hits[:, 0]+y0)).astype(float)
            distance = np.linalg.norm(points-point, axis=1)
            for index in np.argsort(distance):
                node = int(window[hits[index, 0], hits[index, 1]])
                if node < 0:
                    continue
                if closest is None or distance[index] < closest_distance:
                    closest, closest_distance = node, float(distance[index])
                if self.clear_segment(point, points[index]) and self.clear_segment(point, self.pixels[node]):
                    return node
        return closest if near else None

    def snap(self, pixel, max_distance_px=24):
        point = np.asarray(pixel, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("到访坐标无效")
        exact = self.node_at(point, max_distance_px=1)
        if exact is not None:
            return exact
        candidates = self.tree.query_ball_point(point, max_distance_px)
        candidates.sort(key=lambda i: np.linalg.norm(self.pixels[i]-point))
        for node in candidates:
            if self.clear_segment(point, self.pixels[node]):
                return node
        raise ValueError("点位没有无障碍连接，请在同一侧选择实际可到达的位置")

    def shortest_paths(self, node_ids, metric='time', targets=None):
        matrix = self.lengths if metric == 'distance' else self.weights
        native = native_shortest_paths(matrix, node_ids, targets)
        if native is not None:
            return native
        return dijkstra(matrix,
                        indices=node_ids, directed=True, return_predecessors=True)

    def expand(self, parents, start, finish):
        path = [int(finish)]
        while path[-1] != start:
            previous = int(parents[path[-1]])
            if previous < 0 or len(path) > len(self.pixels):
                raise ValueError("必到点不连通")
            path.append(previous)
        path.reverse()
        return path


def build_graph(costs, registration, source_hash, edits=None, step=4, adaptive=True):
    costs = costs.copy()
    closed_edits = np.zeros(costs.shape, np.uint8)
    height, width = costs.shape
    edits = edits or []
    for edit in edits:
        points = np.rint(edit['pixels']).astype(np.int32)
        if len(points) < 1 or not np.isfinite(points).all() or (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
            raise ValueError("修正区域超出地图")
        value = 0. if edit['kind'] == 'close' else float(edit.get('cost', 1.))
        radius = int(edit.get('radiusPx', 6))
        if not 1 <= radius <= 100 or value < 0 or not np.isfinite(value):
            raise ValueError("通行修正无效")
        if len(points) == 1:
            cv2.circle(costs, tuple(points[0]), radius, value, cv2.FILLED)
            cv2.circle(closed_edits, tuple(points[0]), radius, int(edit['kind'] == 'close'), cv2.FILLED)
        else:
            cv2.polylines(costs, [points], False, value, radius*2)
            cv2.polylines(closed_edits, [points], False, int(edit['kind'] == 'close'), radius*2)
            for p in (points[0], points[-1]):
                cv2.circle(costs, tuple(p), radius, value, cv2.FILLED)
                cv2.circle(closed_edits, tuple(p), radius, int(edit['kind'] == 'close'), cv2.FILLED)
    if not isinstance(step, int) or step < 1 or step > 16:
        raise ValueError('通行网格步长无效')
    if adaptive:
        graph = _quadtree_graph(costs, registration, source_hash, edits, step)
        graph.closed_edits = closed_edits
        return graph
    xx, yy = np.meshgrid(np.arange(step//2, width, step), np.arange(step//2, height, step))
    coarse_pixels = np.column_stack((xx.ravel(), yy.ravel()))
    selected = np.zeros((height, width), dtype=bool)
    selected[yy, xx] = True
    selected &= costs > 0
    cy, cx = np.nonzero(selected)
    candidates = np.column_stack((cx, cy))
    local, valid = registration.forward(candidates)
    pixels = candidates[valid]
    positions = local[valid]
    id_grid = np.full((height, width), -1, dtype=np.int32)
    id_grid[pixels[:, 1], pixels[:, 0]] = np.arange(len(pixels))
    if len(pixels) < 2:
        raise ValueError("配准范围内没有足够的可通行区域")
    rows, cols, weights, lengths = [], [], [], []
    offsets = [(dx, dy, pixels) for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1))] if adaptive or step == 1 else []
    if step > 1:
        offsets += [(dx*step, dy*step, coarse_pixels) for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1))]
    for dx, dy, starts in offsets:
        ends = starts+[dx, dy]
        inside = (ends[:, 0] >= 0) & (ends[:, 0] < width) & (ends[:, 1] >= 0) & (ends[:, 1] < height)
        p, q = starts[inside], ends[inside]
        a, b = id_grid[p[:, 1], p[:, 0]], id_grid[q[:, 1], q[:, 0]]
        keep = (a >= 0) & (b >= 0)
        if dx and dy:
            keep &= (id_grid[p[:, 1], q[:, 0]] >= 0) & (id_grid[q[:, 1], p[:, 0]] >= 0)
        a, b = a[keep], b[keep]
        clear = np.ones(len(a), dtype=bool)
        if not len(a):
            continue
        # Supersample every edge at native image resolution. A coarse graph
        # must not skip a one-pixel wall lying between two free cell centres.
        edge_cost = np.zeros(len(a))
        samples = 2*max(abs(dx), abs(dy))+1
        for ratio in np.linspace(0, 1, samples):
            p = np.rint(pixels[a]+(pixels[b]-pixels[a])*ratio).astype(int)
            sample = costs[p[:, 1], p[:, 0]]
            clear &= sample > 0
            edge_cost += sample/samples
        if dx and dy:
            # A long diagonal also has native-pixel corners BETWEEN its end
            # nodes. Checking only the two coarse axial neighbours misses
            # a thin L-shaped barrier in the middle of the edge.
            sx, sy = int(np.sign(dx)), int(np.sign(dy))
            for offset in range(abs(dx)):
                p = pixels[a]+[sx*offset, sy*offset]
                clear &= costs[p[:, 1], p[:, 0]+sx] > 0
                clear &= costs[p[:, 1]+sy, p[:, 0]] > 0
        a, b, edge_cost = a[clear], b[clear], edge_cost[clear]
        length = np.linalg.norm(positions[a]-positions[b], axis=1)
        rows.extend((a, b)); cols.extend((b, a))
        weights.extend((length*edge_cost, length*edge_cost)); lengths.extend((length, length))
    row = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
    col = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
    shape = (len(pixels), len(pixels))
    graph_weights = coo_matrix((np.concatenate(weights) if weights else [], (row, col)), shape=shape).tocsr()
    graph_lengths = coo_matrix((np.concatenate(lengths) if lengths else [], (row, col)), shape=shape).tocsr()
    content_hash = hashlib.sha256(json.dumps({'source': source_hash, 'registration': registration.as_dict(),
                                              'edits': edits, 'step': step, 'adaptive': adaptive, 'version': 2}, sort_keys=True).encode()
                                  +costs.tobytes()).hexdigest()[:20]
    graph = RasterGraph(graph_weights, graph_lengths, pixels, positions, costs, content_hash, step, adaptive)
    graph.closed_edits = closed_edits
    graph.owner = id_grid
    return graph


def _quadtree_graph(costs, registration, source_hash, edits, step):
    """Merge fully free homogeneous squares; split mixed cells to one pixel.

    Adjacency comes from actual native-pixel shared faces/corners. Thus a
    one-pixel gate and a short alternative passage survive even if the rest
    of the map uses larger cells. No blocked pixel is opened by coarsening.
    """
    if step & (step-1):
        raise ValueError('自适应网格步长必须是 1、2、4、8 或 16')
    height, width = costs.shape
    if registration.tri is None:
        raise ValueError('缺少有效配准范围')
    domain = np.zeros((height, width), np.uint8)
    hull = cv2.convexHull(registration.source.astype(np.float32))
    cv2.fillConvexPoly(domain, np.rint(hull).astype(np.int32), 1)
    eroded = cv2.erode(domain, np.ones((3, 3), np.uint8))
    y, x = np.nonzero(domain-eroded)
    if len(x):
        domain[y, x] = registration.tri.find_simplex(np.column_stack((x, y))) >= 0
    padded_h = (height+step-1)//step*step
    padded_w = (width+step-1)//step*step
    padded = np.zeros((padded_h, padded_w), np.float32)
    padded[:height, :width] = np.where(domain, costs, 0)
    owners = np.full(padded.shape, -1, np.int32)
    centres, sizes, terrain_values = [], [], []
    count, size = 0, step
    while size:
        shape = (padded_h//size, size, padded_w//size, size)
        fresh = ((padded > 0) & (owners < 0)).reshape(shape).all(axis=(1, 3))
        tiles = padded.reshape(shape)
        fresh &= (tiles.max(axis=(1, 3))-tiles.min(axis=(1, 3))) < 1e-6
        yy, xx = np.nonzero(fresh)
        ids = np.full(fresh.shape, -1, np.int32)
        ids[yy, xx] = np.arange(count, count+len(xx))
        expanded = np.repeat(np.repeat(ids, size, axis=0), size, axis=1)
        assigned = expanded >= 0
        owners[assigned] = expanded[assigned]
        centres.append(np.column_stack((xx*size+(size-1)/2, yy*size+(size-1)/2)))
        sizes.append(np.full(len(xx), size, np.float32))
        terrain_values.append(padded[yy*size, xx*size])
        count += len(xx)
        size //= 2
    if count < 2:
        raise ValueError('配准范围内没有足够可通行区域')
    pixels = np.concatenate(centres)
    cell_sizes = np.concatenate(sizes)
    terrain = np.concatenate(terrain_values)
    positions, valid = registration.forward(pixels)
    if not valid.all():
        mapping = np.full(count, -1, np.int32)
        mapping[valid] = np.arange(int(valid.sum()))
        occupied = owners >= 0
        owners[occupied] = mapping[owners[occupied]]
        pixels, positions = pixels[valid], positions[valid]
        cell_sizes, terrain = cell_sizes[valid], terrain[valid]
        count = len(pixels)
    owners = owners[:height, :width]
    keys = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        sy, ty = slice(0, height-dy), slice(dy, height)
        sx = slice(max(0, -dx), min(width, width-dx))
        tx = slice(max(0, dx), min(width, width+dx))
        a, b = owners[sy, sx], owners[ty, tx]
        keep = (a >= 0) & (b >= 0) & (a != b)
        if dx and dy:
            keep &= (owners[sy, tx] >= 0) & (owners[ty, sx] >= 0)
        a, b = a[keep].astype(np.int64), b[keep].astype(np.int64)
        keys.append(np.minimum(a, b)*count+np.maximum(a, b))
    pairs = np.unique(np.concatenate(keys))
    a, b = (pairs//count).astype(np.int32), (pairs % count).astype(np.int32)
    lengths = np.linalg.norm(positions[a]-positions[b], axis=1)
    # The line between centres of adjacent square cells crosses their shared
    # face at this size ratio. Each leaf has a uniform terrain cost.
    weights = lengths*(terrain[a]*cell_sizes[a]+terrain[b]*cell_sizes[b])/(cell_sizes[a]+cell_sizes[b])
    row, col = np.concatenate((a, b)), np.concatenate((b, a))
    shape = (count, count)
    weighted = coo_matrix((np.concatenate((weights, weights)), (row, col)), shape=shape).tocsr()
    distances = coo_matrix((np.concatenate((lengths, lengths)), (row, col)), shape=shape).tocsr()
    digest = hashlib.sha256(json.dumps({'source': source_hash, 'registration': registration.as_dict(),
                                        'edits': edits, 'step': step, 'schema': ROUTING_SCHEMA}, sort_keys=True).encode()+costs.tobytes()).hexdigest()[:20]
    result = RasterGraph(weighted, distances, pixels, positions, costs, digest, step, True)
    result.cell_sizes = cell_sizes
    result.owner = owners
    return result


def plan_matrix(graph, controls, start_pixel, finish_pixel, registration, metric='time'):
    endpoints = [{'id': 'start', 'pixel': start_pixel}, *controls, {'id': 'finish', 'pixel': finish_pixel}]
    nodes, points, unresolved, approaches = [], [], [], []
    start_component = None
    resolver = getattr(graph, 'access_resolver', None)
    for endpoint in endpoints:
        pixel = endpoint.get('approachPixel') or endpoint['pixel']
        try:
            if resolver is not None and endpoint['id'] not in ('start', 'finish'):
                resolved = resolver.resolve(endpoint, start_component)
                node, pixel = resolved['node'], resolved['pixel']
                world = np.array([resolved['local']])
                exact, valid = registration.forward([endpoint['pixel']])
                approaches.append({'controlId': endpoint['id'], 'targetPixel': list(endpoint['pixel']),
                                   'pixel': pixel, 'method': resolved['method'], 'automatic': resolved['automatic'],
                                   'offsetM': float(np.linalg.norm(world[0]-exact[0])) if valid[0] else None,
                                   'lnglat': Frame().to_lnglat(world[0]).tolist(),
                                   'targetLnglat': Frame().to_lnglat(exact[0]).tolist() if valid[0] else None})
                endpoint = {**endpoint, 'approachPixel': pixel}
            else:
                world, inside = registration.forward([pixel])
                if not inside[0]:
                    raise ValueError("点位超出配准有效范围")
                node = graph.snap(pixel)
            if endpoint['id'] == 'start':
                start_component = graph.components[node]
            endpoints[len(nodes)] = endpoint
            nodes.append(node)
            points.append(world[0])
        except ValueError as exc:
            unresolved.append({'id': endpoint['id'], 'message': str(exc)})
    if unresolved:
        return {'unresolved': unresolved, 'approaches': approaches}
    components = [int(graph.components[n]) for n in nodes]
    if len(set(components)) != 1:
        counts = np.bincount(graph.components)
        return {'unresolved': [{'id': e['id'], 'message': f'位于独立通行区域（{int(counts[c])} 个节点），检查门洞或目标的到访侧'}
                               for e, c in zip(endpoints, components) if c != components[0]], 'approaches': approaches}
    distances, parents = graph.shortest_paths(nodes, metric, targets=nodes)
    connectors = np.linalg.norm(np.asarray(points)-graph.local[nodes], axis=1)
    factors = []
    for endpoint in endpoints:
        x, y = np.rint(endpoint.get('approachPixel') or endpoint['pixel']).astype(int)
        factors.append(float(graph.costs[y, x]))
    connector_costs = connectors if metric == 'distance' else connectors*np.array(factors)
    matrix = distances[:, nodes] + connector_costs[:, None] + connector_costs[None, :]
    np.fill_diagonal(matrix, 0.)
    return {'matrix': matrix, 'parents': parents, 'nodes': nodes, 'endpoints': endpoints,
            'world': np.asarray(points), 'connectors': connectors,
            'connectorWeights': connectors*np.asarray(factors), 'unresolved': [], 'approaches': approaches}


def expand_route(graph, bundle, order, frame=None):
    frame = frame or Frame()
    positions, node_ids, edge_ids, legs = [], [], [], []
    distance = 0.
    travel_weight = 0.
    for a, b in zip(order, order[1:]):
        start_index, before_distance, before_weight = len(positions), distance, travel_weight
        start, finish = bundle['nodes'][a], bundle['nodes'][b]
        path = graph.expand(bundle['parents'][a], start, finish)
        positions.append(bundle['world'][a].tolist())
        positions.extend(graph.local[path].tolist())
        positions.append(bundle['world'][b].tolist())
        node_ids.extend(path)
        for u, v in zip(path, path[1:]):
            edge_ids.append([u, v])
            distance += float(graph.lengths[u, v])
            travel_weight += float(graph.weights[u, v])
        distance += float(bundle['connectors'][a]+bundle['connectors'][b])
        travel_weight += float(bundle['connectorWeights'][a]+bundle['connectorWeights'][b])
        legs.append({'fromId': bundle['endpoints'][a]['id'], 'toId': bundle['endpoints'][b]['id'],
                     'startIndex': start_index, 'endIndex': len(positions)-1,
                     'distanceM': distance-before_distance, 'travelWeightM': travel_weight-before_weight})
    return {'geometry': frame.to_lnglat(positions).tolist(), 'localGeometry': positions,
            'distanceM': distance, 'travelWeightM': travel_weight, 'nodeIds': node_ids, 'edgeIds': edge_ids,
            'legs': legs}
