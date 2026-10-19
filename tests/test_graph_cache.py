import numpy as np

from dirmap import graph_cache
from dirmap.geometry import Registration, affine_points
from dirmap.graph import build_graph


def test_scratch_cache_preserves_walls_weights_and_closures_and_detects_new_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('DIRMAP_CACHE_DIR', str(tmp_path/'cache'))
    source = [[0, 0], [39, 0], [39, 39], [0, 39]]
    affine = [[1, 0, 0], [0, -1, 39]]
    registration = Registration(affine, source, affine_points(affine, source))
    costs = np.ones((40, 40), np.float32); costs[:30, 20] = 0
    terrain = tmp_path/'terrain.npz'; np.savez_compressed(terrain, costs=costs)
    state = {'registration': registration.as_dict(), 'sourceHash': 'test',
             'edits': [{'kind': 'close', 'pixels': [[10, 10]], 'radiusPx': 2}]}
    key = graph_cache.key(state, terrain)
    graph = build_graph(costs, registration, state['sourceHash'], state['edits'])
    graph_cache.write(key, graph)
    restored = graph_cache.read(key)
    assert restored.content_hash == graph.content_hash
    np.testing.assert_array_equal(restored.costs, graph.costs)
    np.testing.assert_array_equal(restored.closed_edits, graph.closed_edits)
    np.testing.assert_array_equal(restored.owner, graph.owner)
    np.testing.assert_array_equal(restored.weights.toarray(), graph.weights.toarray())
    np.testing.assert_array_equal(restored.lengths.toarray(), graph.lengths.toarray())
    assert not restored.clear_segment([10, 20], [30, 20])
    state['edits'] = []
    assert graph_cache.key(state, terrain) != key
    costs[15, 20] = 1; np.savez_compressed(terrain, costs=costs)
    changed = graph_cache.key(state, terrain)
    assert changed != key and graph_cache.read(changed) is None
    (graph_cache.directory()/(key+'.npz')).write_bytes(b'incomplete zip cache')
    assert graph_cache.read(key) is None
