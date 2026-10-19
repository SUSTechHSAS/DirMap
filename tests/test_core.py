import itertools
import math
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from dirmap.geometry import Frame, Registration, affine_points, fit_affine, mercator, unmercator
from dirmap.graph import build_graph, expand_route, plan_matrix
from dirmap.optimizer import balanced_people, solve, subset_tours
from dirmap.vision import analyze, detect_controls, read_image


def registration(width, height, scale=1):
    pixels = [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]
    affine = [[scale, 0, 0], [0, -scale, (height-1)*scale]]
    return Registration(affine, pixels, affine_points(affine, pixels), regularization=0)


def test_frame_and_mesh_roundtrip_with_nonuniform_distortion():
    frame = Frame()
    points = np.array([[113.929, 22.535], [113.934, 22.529]])
    np.testing.assert_allclose(frame.to_lnglat(frame.to_local(points)), points)
    np.testing.assert_allclose(unmercator(mercator(points)), points)
    source = [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]]
    target = [[0, 100], [100, 100], [100, 0], [0, 0], [61, 43]]
    reg = Registration([[1, 0, 0], [0, -1, 100]], source, target, regularization=0)
    withheld = np.array([[45, 32], [12, 77], [80, 55], [51, 53]])
    projected, valid = reg.forward(withheld)
    assert valid.all()
    np.testing.assert_allclose(reg.inverse(projected), withheld, atol=1e-8)
    assert np.isnan(reg.forward([[101, 33]])[0]).all()
    assert np.isnan(reg.inverse([[150, 150]])).all()


def test_affine_outlier_and_fold_rejection():
    source = np.array(list(itertools.product(range(0, 100, 25), repeat=2)), dtype=float)
    truth = np.array([[1.3, .1, 9], [-.1, -1.5, 7]])
    target = affine_points(truth, source)
    target[3] += [500, -500]
    fitted, inside = fit_affine(source, target, threshold=2)
    assert inside.sum() == len(source)-1
    np.testing.assert_allclose(fitted, truth, atol=1e-6)
    with pytest.raises(ValueError, match='共线'):
        fit_affine([[0, 0], [1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError, match='折叠|重叠'):
        Registration([[1, 0, 0], [0, -1, 100]], [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
                     [[0, 100], [100, 100], [100, 0], [0, 0], [200, 50]], regularization=0)


def test_native_wall_not_lost_between_grid_centres_and_gate_closure_invalidates():
    costs = np.ones((25, 25), np.float32)
    costs[:, 11] = 0  # Grid centres are 2,6,10,14: wall is never a centre.
    costs[2:7, 11] = 1
    reg = registration(25, 25)
    graph = build_graph(costs, reg, 'fixture', step=4)
    start, finish = graph.snap([6, 18]), graph.snap([18, 18])
    distance, parents = graph.shortest_paths([start])
    path = graph.expand(parents[0], start, finish)
    assert distance[0, finish] > 25
    assert min(graph.pixels[path, 1]) <= 6
    assert all(graph.clear_segment(graph.pixels[a], graph.pixels[b]) for a, b in zip(path, path[1:]))
    closed = build_graph(costs, reg, 'fixture', [{'kind': 'close', 'pixels': [[11, 4]], 'radiusPx': 4}], step=4)
    assert graph.content_hash != closed.content_hash
    a, b = closed.snap([6, 18]), closed.snap([18, 18])
    assert closed.components[a] != closed.components[b]


def test_no_diagonal_corner_cut_or_snap_through_wall():
    costs = np.ones((5, 5), np.float32)
    costs[0, 1] = 0
    costs[1, 0] = 0
    graph = build_graph(costs, registration(5, 5), 'corner', step=1)
    a, b = graph.snap([0, 0]), graph.snap([1, 1])
    assert graph.components[a] != graph.components[b]
    with pytest.raises(ValueError):
        graph.snap([1, 0])


def test_route_geometry_expands_to_edges_and_separate_control_position():
    costs = np.ones((33, 33), np.float32)
    costs[7:25, 16] = 0
    reg = registration(33, 33)
    graph = build_graph(costs, reg, 'geometry', step=2)
    controls = [{'id': 'a', 'pixel': [25, 19], 'approachPixel': [25, 17]}]
    bundle = plan_matrix(graph, controls, [5, 17], [5, 17], reg)
    assert not bundle['unresolved']
    expanded = expand_route(graph, bundle, [0, 1, 2])
    assert expanded['edgeIds']
    assert all(graph.weights[a, b] > 0 for a, b in expanded['edgeIds'])
    assert controls[0]['pixel'] == [25, 19]
    assert expanded['distanceM'] > 40


def euclidean_matrix(points):
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(points[:, None]-points[None], axis=2)


def brute_split(matrix, controls, teams):
    n = len(controls)
    best = (math.inf, math.inf)
    for permutation in itertools.permutations(range(1, n+1)):
        for cut in range(1, n):
            routes = [[0, *permutation[:cut], n+1], [0, *permutation[cut:], n+1]]
            finishes = []
            for team, route in zip(teams, routes):
                travel = sum(matrix[a, b] for a, b in zip(route, route[1:]))/team['speedMps']
                service = sum(controls[i-1]['serviceBaseSec']+controls[i-1]['servicePerPersonSec']*team['people'] for i in route[1:-1])
                finishes.append(travel+service+team['departureSec'])
            best = min(best, (max(finishes), sum(finishes)))
    return best


def test_exact_split_matches_independent_permutation_and_cut_oracle():
    matrix = euclidean_matrix([[0, 0], [3, 6], [9, 1], [4, 4], [-3, 2], [9, 8], [1, 0]])
    controls = [{'id': str(i), 'serviceBaseSec': i % 3, 'servicePerPersonSec': .7+i*.1} for i in range(5)]
    teams = [{'id': 'a', 'people': 2, 'speedMps': 1.1, 'departureSec': 5}, {'id': 'b', 'people': 5, 'speedMps': 1.8, 'departureSec': 2}]
    result = solve(matrix, controls, teams, mode='split_coverage')
    assert result['status'] == 'OPTIMAL'
    np.testing.assert_allclose(result['proof']['objectiveValue'], brute_split(matrix, controls, teams), atol=1e-8)
    covered = [c for route in result['routes'] for c in route['controlIds']]
    assert sorted(covered) == [str(i) for i in range(5)]
    assert all(route['controlIds'] for route in result['routes'])


def test_all_visit_fixed_order_people_and_open_finish():
    matrix = euclidean_matrix([[0, 0], [6, 4], [3, 0], [0, 8], [10, 0]])
    controls = [{'id': 'A'}, {'id': 'B'}, {'id': 'C'}]
    teams = [{'id': 'a', 'people': 2, 'speedMps': 1.4}, {'id': 'b', 'people': 7, 'speedMps': 1.4}]
    result = solve(matrix, controls, teams, mode='all_visit')
    for route in result['routes']:
        assert set(route['controlIds']) == {'A', 'B', 'C'}
        assert route['order'][0] == 0 and route['order'][-1] == 4
    assert result['routes'][1]['durationSec']-result['routes'][0]['durationSec'] == pytest.approx(15)
    oracle = min(sum(matrix[a, b] for a, b in zip([0, *p], [*p, 4])) for p in itertools.permutations([1, 2, 3]))
    assert result['routes'][0]['travelSec']*1.4 == pytest.approx(oracle)
    ordered = solve(matrix, controls, teams, mode='fixed_order')
    assert all(route['controlIds'] == ['A', 'B', 'C'] for route in ordered['routes'])
    assert balanced_people(11, 3) == [4, 4, 3]
    with pytest.raises(ValueError):
        balanced_people(2, 3)


def test_bounded_large_solve_preserves_every_target_and_reports_proof_scope():
    rng = np.random.default_rng(6)
    matrix = euclidean_matrix(rng.random((18, 2))*200)
    controls = [{'id': str(i)} for i in range(16)]
    teams = [{'id': str(i), 'people': 4, 'speedMps': 1.3} for i in range(5)]
    result = solve(matrix, controls, teams, mode='split_coverage', time_limit=.3)
    proof = result['proof']
    assert proof['elapsedSec'] < 2, 'The 0.3-second budget must not be consumed by an uninterruptible presolve pass'
    assert result['status'] in ('FEASIBLE', 'OPTIMAL')
    if result['status'] == 'OPTIMAL':
        assert proof['globalOptimalityProven'] and all(stage['status'] == 'OPTIMAL' for stage in proof['stages'])
    else:
        assert not proof['globalOptimalityProven'] and proof['relativeGap'] is None
    if proof['bestBound'] is not None:
        assert proof['bestBound'][0] <= proof['modelObjectiveValue'][0]+1e-6
        assert proof['quantization']['timeSec'] == .001
    assert sorted(c for r in result['routes'] for c in r['controlIds']) == sorted(c['id'] for c in controls)


def test_reject_invalid_optimizer_input_and_unreachable_nodes():
    controls = [{'id': '1'}]
    teams = [{'id': 'a', 'people': 1, 'speedMps': 1}]
    matrix = np.array([[0, np.inf, 2], [np.inf, 0, 1], [2, 1, 0]])
    with pytest.raises(ValueError):
        solve(matrix, controls, teams)
    with pytest.raises(ValueError):
        solve(np.zeros((3, 3)), controls, teams*2, mode='split_coverage')


def test_circle_detection_excludes_hatched_area_and_off_image_text():
    purple = np.zeros((900, 1000), np.uint8)
    cv2.circle(purple, (220, 220), 22, 255, 3)
    cv2.circle(purple, (640, 650), 22, 255, 3)
    for x in range(380, 550, 7):
        cv2.line(purple, (x, 120), (x, 360), 255, 2)
    cv2.putText(purple, '31', (248, 200), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2)
    controls = detect_controls(purple)
    assert len(controls) == 2
    assert all(min(np.linalg.norm(np.asarray(c['pixel'])-p) for p in ([220, 220], [640, 650])) < 3 for c in controls)


def test_cyan_north_guides_do_not_become_water_barriers():
    image = np.full((1000, 1000, 3), 250, np.uint8)
    cv2.line(image, (300, 0), (300, 999), (230, 190, 90), 2)
    cv2.rectangle(image, (500, 300), (700, 480), (230, 190, 90), -1)
    result = analyze(image)
    assert result['blocked'][600, 300] == 0
    assert result['blocked'][350, 600] > 0


def test_oversized_image_rejected_before_decompression(tmp_path, monkeypatch):
    path = tmp_path/'huge.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+struct.pack('>I4s', 13, b'IHDR')
                     +struct.pack('>IIBBBBB', 40000, 40000, 8, 2, 0, 0, 0))
    monkeypatch.setattr(cv2, 'imdecode', lambda *args: pytest.fail('oversized pixels must not be decoded'))
    with pytest.raises(ValueError, match='三千万'):
        read_image(path)
