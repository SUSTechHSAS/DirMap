import cv2
import numpy as np
import pytest

from dirmap.access import AccessResolver
from dirmap.geometry import Registration, affine_points
from dirmap.graph import build_graph, plan_matrix, expand_route


def scene():
    shape = (90, 100)
    semantics = {k: np.zeros(shape, np.uint8) for k in ('building', 'water', 'point_ink', 'vegetation', 'restricted')}
    pixels = [[0, 0], [99, 0], [99, 89], [0, 89]]
    affine = [[.5, 0, 0], [0, -.5, 44.5]]
    reg = Registration(affine, pixels, affine_points(affine, pixels), regularization=0)
    return semantics, reg


def graph_for(semantics, reg):
    cost = np.ones((90, 100), np.float32)
    for mask in semantics.values():
        cost[mask > 0] = 0
    graph = build_graph(cost, reg, 'access-fixture', step=4)
    graph.access_resolver = AccessResolver(graph, reg, semantics)
    return graph


def test_point_symbol_gets_legal_nearby_approach_without_moving_target_or_opening_terrain():
    masks, reg = scene()
    cv2.line(masks['point_ink'], (45, 40), (49, 44), 1, 1)
    cv2.line(masks['point_ink'], (45, 44), (49, 40), 1, 1)
    graph = graph_for(masks, reg)
    control = {'id': '31', 'pixel': [47, 42], 'radiusPx': 15}
    before = graph.costs.copy()
    bundle = plan_matrix(graph, [control], [20, 42], [20, 42], reg)
    assert not bundle['unresolved']
    approach = bundle['approaches'][0]
    assert approach['automatic'] and approach['method'] == 'point_symbol_edge'
    assert approach['offsetM'] <= 3
    assert control['pixel'] == [47, 42]
    np.testing.assert_array_equal(graph.costs, before)
    route = expand_route(graph, bundle, [0, 1, 2])
    assert all(graph.weights[a, b] > 0 for a, b in route['edgeIds'])
    assert reg.forward([[47, 42]])[0][0].tolist() not in route['localGeometry']


def test_building_edge_allows_nearby_facade_not_a_path_inside_roof():
    masks, reg = scene()
    masks['building'][20:65, 40:80] = 1
    graph = graph_for(masks, reg)
    c = {'id': 'b', 'pixel': [41, 42], 'radiusPx': 15}
    result = graph.access_resolver.resolve(c)
    assert result['method'] == 'building_edge'
    assert result['pixel'][0] < 40 and result['offsetM'] <= 3


def test_long_wall_never_becomes_a_point_symbol_even_when_sides_connect_far_away():
    masks, reg = scene()
    masks['point_ink'][8:80, 48:50] = 1
    graph = graph_for(masks, reg)
    with pytest.raises(ValueError):
        graph.access_resolver.resolve({'id': 'wall', 'pixel': [48, 42], 'radiusPx': 16})


def test_tree_cannot_be_approached_through_another_wall():
    masks, reg = scene()
    cv2.circle(masks['vegetation'], (40, 44), 2, 1, -1)
    masks['point_ink'][:, 43] = 1
    graph = graph_for(masks, reg)
    right = graph.snap([60, 44])
    with pytest.raises(ValueError):
        graph.access_resolver.resolve({'id': 'tree', 'pixel': [40, 44], 'radiusPx': 15}, graph.components[right])


def test_forbidden_area_is_not_replaced_by_a_nearby_road():
    masks, reg = scene()
    masks['restricted'][35:50, 35:50] = 1
    masks['point_ink'][41:44, 41:44] = 1
    graph = graph_for(masks, reg)
    with pytest.raises(ValueError, match='禁入'):
        graph.access_resolver.resolve({'id': 'closed', 'pixel': [42, 42], 'radiusPx': 15})


def test_small_round_water_feature_has_perimeter_approach_but_large_lake_does_not():
    masks, reg = scene()
    cv2.circle(masks['water'], (50, 45), 10, 1, -1)
    graph = graph_for(masks, reg)
    resolved = graph.access_resolver.resolve({'id': 'fountain', 'pixel': [50, 45], 'radiusPx': 15})
    assert resolved['method'] == 'small_water_feature_edge'
    assert 5 < resolved['offsetM'] < 7
    cv2.circle(masks['water'], (50, 45), 32, 1, -1)
    large = graph_for(masks, reg)
    with pytest.raises(ValueError):
        large.access_resolver.resolve({'id': 'lake', 'pixel': [50, 45], 'radiusPx': 15})


def test_near_bank_target_and_compound_round_symbol_have_perimeter_approaches():
    masks, reg = scene()
    cv2.ellipse(masks['water'], (65, 40), (20, 11), 25, 0, 360, 1, -1)
    graph = graph_for(masks, reg)
    result = graph.access_resolver.resolve({'id': 'bank', 'pixel': [48, 36], 'radiusPx': 15})
    assert result['method'] == 'water_edge' and result['offsetM'] <= 3
    masks, reg = scene()
    for center in ((35, 35), (45, 45), (55, 55)):
        cv2.circle(masks['building'], center, 7, 1, 2)
    graph = graph_for(masks, reg)
    point = {'id': 'sculpture', 'pixel': [45, 45], 'radiusPx': 30}
    # The symbol is hollow on paper, but its centre has no route to outside.
    main = graph.components[graph.snap([15, 45])]
    result = graph.access_resolver.resolve(point, main)
    assert result['method'] == 'compound_point_symbol_edge'
    assert result['offsetM'] <= 8
    assert graph.costs[45, 45] > 0 and not graph.clear_segment(point['pixel'], result['pixel'])


def test_teacher_closure_of_a_point_feature_is_not_bypassed_by_an_approach():
    masks, reg = scene()
    masks['point_ink'][40:43, 45:48] = 1
    costs = np.where(masks['point_ink'], 0, 1).astype(np.float32)
    graph = build_graph(costs, reg, 'closed-sculpture', [{'kind': 'close', 'pixels': [[46, 41]], 'radiusPx': 1}])
    resolver = AccessResolver(graph, reg, masks)
    with pytest.raises(ValueError, match='封闭'):
        resolver.resolve({'id': 'closed-point', 'pixel': [46, 41], 'radiusPx': 15})
