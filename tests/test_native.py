import shutil

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from dirmap.native import library, shortest_paths


@pytest.mark.skipif(not shutil.which('c++'), reason='Optional native accelerator needs a C++ compiler')
def test_native_directed_disconnected_zero_cost_graph_matches_scipy():
    assert library() is not None
    rng = np.random.default_rng(5317)
    for size in (5, 35, 200):
        row, col = rng.integers(0, size-2, (2, size*4))
        weights = rng.integers(0, 40, len(row)).astype(float)/7
        matrix = coo_matrix((weights, (row, col)), shape=(size, size)).tocsr()
        sources = [0, size//2, size-1, 0]
        expected, _ = dijkstra(matrix, indices=sources, directed=True, return_predecessors=True)
        actual, parents = shortest_paths(matrix, sources)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        for source_row, source in enumerate(sources):
            for finish in np.flatnonzero(np.isfinite(expected[source_row])):
                node, total, seen = int(finish), 0., set()
                while node != source:
                    assert node not in seen
                    seen.add(node)
                    previous = int(parents[source_row, node])
                    assert previous >= 0
                    begin, end = matrix.indptr[previous:previous+2]
                    edges = dict(zip(matrix.indices[begin:end], matrix.data[begin:end]))
                    assert node in edges
                    total += edges[node]; node = previous
                assert total == pytest.approx(actual[source_row, finish])
        targets = [1, size//2, 1]
        limited, _ = shortest_paths(matrix, sources, targets)
        np.testing.assert_allclose(limited[:, targets], expected[:, targets], rtol=1e-12, atol=1e-12)
