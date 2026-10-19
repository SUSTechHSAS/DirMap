"""Bounded disposable graph artifacts; no pickle or workspace-sized cache."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from scipy.sparse import csr_matrix

from .graph import RasterGraph, ROUTING_SCHEMA


def directory():
    path = Path(os.environ.get('DIRMAP_CACHE_DIR', str(Path(tempfile.gettempdir())/'dirmap-cache')))/'graphs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def key(state, terrain_path, step=4):
    digest = hashlib.sha256(json.dumps({'schema': ROUTING_SCHEMA, 'source': state['sourceHash'],
        'registration': state['registration'], 'edits': state['edits'], 'step': step}, sort_keys=True).encode())
    with Path(terrain_path).open('rb') as stream:
        for block in iter(lambda: stream.read(256*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(cache_key):
    path = directory()/(cache_key+'.npz')
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['key'] != cache_key or metadata['schema'] != ROUTING_SCHEMA:
                return None
            pixels = data['pixels']
            count = len(pixels)
            indexes, pointers = data['indices'], data['indptr']
            weighted = csr_matrix((data['weights'], indexes, pointers), shape=(count, count))
            lengths = csr_matrix((data['lengths'], indexes.copy(), pointers.copy()), shape=(count, count))
            graph = RasterGraph(weighted, lengths, pixels, data['local'], data['costs'],
                                metadata['hash'], metadata['step'], True)
            graph.owner = data['owner']
            graph.cell_sizes = data['cell_sizes']
            graph.closed_edits = data['closed_edits']
        try:
            path.touch()
        except OSError:
            pass
        return graph
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # A lost/incomplete scratch cache is reconstructed from source data.
        return None


def write(cache_key, graph):
    root = directory()
    if not np.array_equal(graph.weights.indices, graph.lengths.indices) or not np.array_equal(graph.weights.indptr, graph.lengths.indptr):
        raise ValueError('通行图边与长度索引不一致')
    metadata = {'key': cache_key, 'schema': ROUTING_SCHEMA, 'hash': graph.content_hash, 'step': graph.step}
    temporary = tempfile.NamedTemporaryFile(dir=root, suffix='.tmp', delete=False)
    try:
        with temporary:
            np.savez_compressed(temporary, metadata=json.dumps(metadata), pixels=graph.pixels, local=graph.local,
                costs=graph.costs, owner=graph.owner, cell_sizes=graph.cell_sizes, closed_edits=graph.closed_edits,
                indices=graph.weights.indices, indptr=graph.weights.indptr,
                weights=graph.weights.data, lengths=graph.lengths.data)
        Path(temporary.name).replace(root/(cache_key+'.npz'))
    finally:
        Path(temporary.name).unlink(missing_ok=True)
    entries = sorted(root.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    used = 0
    for index, path in enumerate(entries):
        used += path.stat().st_size
        if index >= 6 or used > 256*1024*1024:
            path.unlink(missing_ok=True)
