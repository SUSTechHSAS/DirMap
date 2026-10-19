"""Mixed compulsory/shared visits, checked against independent enumeration."""
from collections import Counter
import itertools
import time

import numpy as np
import pytest

from dirmap.exact import optimize
from dirmap.optimizer import solve


def score(matrix, routes, controls, teams, objective):
    distance = [sum(matrix[a, b] for a, b in zip(route, route[1:])) for route in routes]
    finish = [d/team['speedMps']+team.get('departureSec', 0)
              +sum(controls[n-1]['serviceBaseSec']+team['people']*controls[n-1]['servicePerPersonSec'] for n in route[1:-1])
              for d, route, team in zip(distance, routes, teams)]
    return (sum(distance),) if objective == 'min_distance' else (sum(finish),) if objective == 'min_total_time' else (max(finish), sum(finish))


def oracle(matrix, controls, teams, ordered):
    # Every team visits 1 and 3; points 2 and 4 each go to exactly one team.
    candidates = []
    for assignment in itertools.product(range(len(teams)), repeat=2):
        options = []
        for team in range(len(teams)):
            selected = [1, 3]+[n for n, owner in zip((2, 4), assignment) if owner == team]
            routes = [[0, *p, 5] for p in itertools.permutations(selected) if not ordered or p.index(1) < p.index(3)]
            options.append(routes)
        candidates += list(itertools.product(*options))
    return candidates


@pytest.mark.parametrize('objective', ['min_max_finish_then_sum', 'min_total_time', 'min_distance'])
@pytest.mark.parametrize('ordered', [False, True])
def test_mixed_dp_and_circuit_match_independent_partitions_and_permutations(objective, ordered):
    rng = np.random.default_rng(427)
    matrix = rng.integers(2, 100, (6, 6)).astype(float); np.fill_diagonal(matrix, 0)
    controls = [{'id': str(i), 'serviceBaseSec': i+1, 'servicePerPersonSec': i % 2} for i in range(1, 5)]
    teams = [{'id': 'a', 'people': 2, 'speedMps': 1., 'departureSec': 0},
             {'id': 'b', 'people': 5, 'speedMps': 2., 'departureSec': 7}]
    routes = oracle(matrix, controls, teams, ordered)
    best = min(score(matrix, r, controls, teams, objective) for r in routes)
    result = solve(matrix, controls, teams, 'mixed_coverage', objective, 5, ['1', '3'], ordered)
    assert result['status'] == 'OPTIMAL'
    assert score(matrix, [r['order'] for r in result['routes']], controls, teams, objective) == best
    exact = optimize(matrix, controls, teams, 'mixed_coverage', objective, routes[0], time.perf_counter()+5,
                     mandatory_nodes=(1, 3), ordered_nodes=(1, 3) if ordered else ())
    assert exact['proven']
    assert score(matrix, exact['sequences'], controls, teams, objective) == best


def test_identical_teams_may_have_identical_compulsory_tours_and_no_shared_points():
    matrix = np.array([[0, 5, 9, 0], [5, 0, 2, 4], [9, 2, 0, 8], [0, 4, 8, 0]], float)
    controls = [{'id': '31'}, {'id': '32'}]
    teams = [{'id': str(i), 'people': 3, 'speedMps': 1} for i in range(5)]
    result = solve(matrix, controls, teams, 'mixed_coverage', time_limit=2,
                   mandatory_ids=['31', '32'], mandatory_order=True)
    assert result['status'] == 'OPTIMAL'
    assert all(r['controlIds'] == ['31', '32'] for r in result['routes'])
    exact = optimize(matrix, controls, teams, 'mixed_coverage', 'min_total_time',
                     [[0, 1, 2, 3] for _ in teams], time.perf_counter()+3,
                     mandatory_nodes=(1, 2), ordered_nodes=(1, 2))
    assert exact['proven'] and all(route == [0, 1, 2, 3] for route in exact['sequences'])


def test_large_timed_out_mixed_plan_keeps_all_compulsory_and_shared_visits():
    rng = np.random.default_rng(387)
    points = rng.random((18, 2))*100
    matrix = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
    controls = [{'id': str(i)} for i in range(16)]
    teams = [{'id': str(i), 'people': 4, 'speedMps': 1.4} for i in range(10)]
    compulsory = ['0', '7', '14']
    result = solve(matrix, controls, teams, 'mixed_coverage', time_limit=.00001,
                   mandatory_ids=compulsory, mandatory_order=True)
    assert result['status'] == 'FEASIBLE'
    for route in result['routes']:
        assert [c for c in route['controlIds'] if c in compulsory] == compulsory
        assert len(set(route['controlIds'])) == len(route['controlIds'])
    counts = Counter(c for r in result['routes'] for c in r['controlIds'])
    assert counts == Counter({c['id']: 10 if c['id'] in compulsory else 1 for c in controls})


def test_mixed_requires_selected_compulsory_ids():
    controls = [{'id': '31'}]; matrix = np.ones((3, 3)); np.fill_diagonal(matrix, 0)
    teams = [{'id': 't', 'people': 1, 'speedMps': 1}]
    for ids in ([], ['32'], ['31', '31']):
        with pytest.raises(ValueError):
            solve(matrix, controls, teams, 'mixed_coverage', mandatory_ids=ids)
