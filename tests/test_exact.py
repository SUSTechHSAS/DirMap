"""Independent permutation/partition oracles for the bounded circuit solver."""

import itertools
import math
import time

import numpy as np
import pytest

from dirmap.exact import optimize
from dirmap.optimizer import solve, subset_tours


def model_score(matrix, controls, teams, routes, objective):
    times, distances = [], []
    for team, route in zip(teams, routes):
        edges = list(zip(route, route[1:]))
        distances.append(sum(round(matrix[a, b]*1000) for a, b in edges))
        times.append(sum(math.ceil(matrix[a, b]/team['speedMps']*1000) for a, b in edges)
                     +sum(math.ceil((controls[i-1].get('serviceBaseSec', 5)
                                     +team['people']*controls[i-1].get('servicePerPersonSec', 1))*1000)
                          for i in route[1:-1])
                     +math.ceil(team.get('departureSec', 0)*1000))
    return (sum(distances),) if objective == 'min_distance' else (sum(times),) if objective == 'min_total_time' else (max(times), sum(times))


@pytest.mark.parametrize('objective', ['min_max_finish_then_sum', 'min_total_time', 'min_distance'])
def test_circuit_partition_matches_complete_independent_enumeration(objective):
    rng = np.random.default_rng(67)
    matrix = rng.integers(1, 80, (7, 7)).astype(float)  # deliberately directed/non-Euclidean
    np.fill_diagonal(matrix, 0)
    controls = [{'id': str(i), 'serviceBaseSec': i+1, 'servicePerPersonSec': i % 3} for i in range(5)]
    teams = [{'id': 'slow', 'people': 5, 'speedMps': 1., 'departureSec': 0},
             {'id': 'fast', 'people': 2, 'speedMps': 2., 'departureSec': 8}]
    best = None
    for sequence in itertools.permutations(range(1, 6)):
        for split in range(1, 5):
            routes = [[0, *sequence[:split], 6], [0, *sequence[split:], 6]]
            score = model_score(matrix, controls, teams, routes, objective)
            best = score if best is None or score < best else best
    result = optimize(matrix, controls, teams, 'split_coverage', objective,
                      [[0, 1, 2, 6], [0, 3, 4, 5, 6]], time.perf_counter()+5)
    assert result['proven']
    assert model_score(matrix, controls, teams, result['sequences'], objective) == best
    np.testing.assert_allclose(result['proof']['modelObjectiveValue'], np.asarray(best)/1000)
    assert result['proof']['bestBound'] == result['proof']['modelObjectiveValue']
    assert result['proof']['relativeGap'] == 0
    assert len(result['proof']['stages']) == (2 if objective == 'min_max_finish_then_sum' else 1)


def test_all_visit_circuit_matches_directed_tour_oracle_and_distinct_endpoints():
    rng = np.random.default_rng(513)
    matrix = rng.integers(1, 100, (8, 8)).astype(float)
    np.fill_diagonal(matrix, 0)
    controls = [{'id': str(i)} for i in range(6)]
    teams = [{'id': 'a', 'people': 6, 'speedMps': 1.3, 'departureSec': 0},
             {'id': 'b', 'people': 3, 'speedMps': 2.1, 'departureSec': 25}]
    best = min(sum(matrix[a, b] for a, b in zip([0, *p], [*p, 7]))
               for p in itertools.permutations(range(1, 7)))
    result = optimize(matrix, controls, teams, 'all_visit', 'min_distance',
                      [list(range(8))]*2, time.perf_counter()+5)
    assert result['proven']
    for route in result['sequences']:
        assert route[0] == 0 and route[-1] == 7
        assert sorted(route[1:-1]) == list(range(1, 7))
        assert sum(matrix[a, b] for a, b in zip(route, route[1:])) == best
    assert result['proof']['modelObjectiveValue'] == [best*2]


def test_expired_exact_budget_does_not_claim_infeasibility_or_discard_incumbent():
    matrix = np.ones((18, 18)); np.fill_diagonal(matrix, 0)
    controls = [{'id': str(i)} for i in range(16)]
    teams = [{'id': 't', 'people': 1, 'speedMps': 1}]
    assert optimize(matrix, controls, teams, 'all_visit', 'min_distance', [list(range(18))], time.perf_counter()-1) is None
    with pytest.raises(TimeoutError):
        subset_tours(matrix, deadline=time.perf_counter()-1)
    # Even a tiny budget returns all mandatory controls, without a spurious
    # optimality proof or an INFEASIBLE label for an interrupted search.
    result = solve(matrix, controls, teams, time_limit=.00001)
    assert result['status'] == 'FEASIBLE'
    assert not result['proof']['globalOptimalityProven']
    assert sorted(result['routes'][0]['controlIds']) == sorted(c['id'] for c in controls)


def test_solver_never_reports_raw_float_optimality_for_quantized_proof():
    matrix = np.array([[0, 1.00051, 1.00049, 3], [1, 0, 2.00002, 1],
                       [1, 2, 0, 1], [3, 1, 1, 0]])
    controls = [{'id': 'a'}, {'id': 'b'}]
    teams = [{'id': 't', 'people': 1, 'speedMps': 1}]
    result = optimize(matrix, controls, teams, 'all_visit', 'min_distance', [[0, 1, 2, 3]], time.perf_counter()+2)
    assert result['proof']['quantization']['distanceM'] == .001
    assert '量化模型' in result['proof']['scope']
