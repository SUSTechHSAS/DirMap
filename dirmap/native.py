"""Small optional C++ CSR shortest-path kernel, compiled only into /tmp.

SciPy remains the portable fallback. The kernel preserves every graph edge
and weight; it changes neither terrain resolution nor checkpoint semantics.
"""
from functools import lru_cache
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_name('shortest_paths.cpp')
    compiler = shutil.which('c++')
    if not compiler or os.environ.get('DIRMAP_DISABLE_NATIVE') == '1':
        return None
    directory = Path(os.environ.get('DIRMAP_CACHE_DIR', str(Path(tempfile.gettempdir())/'dirmap-cache')))/'native'
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    destination = directory/(digest+'.so')
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix='compile-', dir=directory) as temporary:
            output = Path(temporary)/'paths.so'
            try:
                subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC', '-pthread', str(source), '-o', str(output)],
                               check=True, capture_output=True, timeout=45)
                output.replace(destination)
            except (OSError, subprocess.SubprocessError):
                return None
    try:
        kernel = ctypes.CDLL(str(destination))
    except OSError:
        return None
    vector = np.ctypeslib.ndpointer
    kernel.dirmap_dijkstra.argtypes = [ctypes.c_int, vector(np.int64, flags='C_CONTIGUOUS'),
        vector(np.int32, flags='C_CONTIGUOUS'), vector(np.float64, flags='C_CONTIGUOUS'),
        ctypes.c_int, vector(np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, vector(np.int32, flags='C_CONTIGUOUS'),
        vector(np.float64, flags='C_CONTIGUOUS'), vector(np.int32, flags='C_CONTIGUOUS'), ctypes.c_int]
    kernel.dirmap_dijkstra.restype = ctypes.c_int
    return kernel


def shortest_paths(matrix, sources, targets=None):
    kernel = library()
    if kernel is None:
        return None
    n = matrix.shape[0]
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    targets = np.ascontiguousarray([] if targets is None else targets, dtype=np.int32)
    if matrix.shape != (n, n) or sources.ndim != 1 or targets.ndim != 1:
        raise ValueError('最短路输入形状无效')
    if (sources < 0).any() or (sources >= n).any() or (targets < 0).any() or (targets >= n).any():
        raise ValueError('最短路节点越界')
    weights = np.ascontiguousarray(matrix.data, dtype=np.float64)
    offsets = np.ascontiguousarray(matrix.indptr, dtype=np.int64)
    neighbours = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    if not np.isfinite(weights).all() or (weights < 0).any() or (neighbours < 0).any() or (neighbours >= n).any():
        raise ValueError('最短路边无效')
    distance = np.empty((len(sources), n), dtype=np.float64)
    parent = np.empty((len(sources), n), dtype=np.int32)
    status = kernel.dirmap_dijkstra(n, offsets, neighbours, weights, len(sources), sources,
                                   len(targets), targets, distance, parent, 2)
    if status:
        raise MemoryError('最短路计算资源不足')
    return distance, parent
