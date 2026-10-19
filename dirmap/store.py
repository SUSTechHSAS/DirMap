"""Atomic manifests, immutable derived files, and cross-process edit locking."""

import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DIRMAP_DATA_DIR', str(ROOT/'data')))

class DataLock:
    """Reentrant in a thread; serializes writers in other server processes.

    flock is held by a live file descriptor, not inferred from a stale PID or
    a timestamp. A crashed process releases it through the operating system.
    """
    def __init__(self):
        self.mutex = threading.RLock()
        self.local = threading.local()

    def __enter__(self):
        self.mutex.acquire()
        try:
            depth = getattr(self.local, 'depth', 0)
            if depth == 0:
                DATA.mkdir(parents=True, exist_ok=True)
                self.local.stream = (DATA/'.write.lock').open('a+b')
                try:
                    fcntl.flock(self.local.stream, fcntl.LOCK_EX)
                except BaseException:
                    self.local.stream.close()
                    raise
            self.local.depth = depth+1
            return self
        except BaseException:
            self.mutex.release()
            raise

    def __exit__(self, *exc):
        self.local.depth -= 1
        if self.local.depth == 0:
            self.local.stream.close()
        self.mutex.release()


LOCK = DataLock()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(256*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def asset_dir(asset_id):
    if not asset_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in asset_id):
        raise ValueError('地图标识无效')
    path = DATA/'projects'/asset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def catalog():
    originals = []
    for index, path in enumerate(sorted(ROOT.glob('*.jpg'))):
        originals.append({'id': f'szu-{index+1}', 'path': str(path), 'name': f'深大定向图 {index+1}',
                          'filename': path.name, 'role': 'map', 'sha256': file_hash(path)})
    originals.extend(read_json(DATA/'uploads.json', []))
    return originals


def get_asset(asset_id):
    for asset in catalog():
        if asset['id'] == asset_id:
            return asset
    raise ValueError('地图不存在')


def course_defaults(source_hash):
    # Defaults describe this source sheet and explicit teacher rules. Other
    # uploaded maps remain freely configurable and inherit no SZU point IDs.
    return read_json(ROOT/'data/course-presets.json', {}).get(source_hash)


def get_state(asset_id):
    return read_json(asset_dir(asset_id)/'state.json')


def save_state(asset_id, value, operation_id=None, result=None):
    """Commit the state and a replay receipt with one atomic manifest switch.

    A receipt alone is only staged; lastOperationId proves the state was
    committed. Ordinary edits retain that marker. The single live queue
    worker always acknowledges/reconciles this operation before the next.
    """
    with LOCK:
        current = get_state(asset_id)
        previous_operation = (current or {}).get('lastOperationId')
        if operation_id:
            value['lastOperationId'] = operation_id
            atomic_json(DATA/'jobs'/'results'/f'{operation_id}.json', result if result is not None else value)
        elif previous_operation:
            value['lastOperationId'] = previous_operation
        atomic_json(asset_dir(asset_id)/'state.json', value)


def operation_result(asset_id, operation_id):
    with LOCK:
        state = get_state(asset_id)
        if (state or {}).get('lastOperationId') == operation_id:
            return read_json(DATA/'jobs'/'results'/f'{operation_id}.json')
    return None


ARTIFACT_NAMES = {'terrain.npz', 'terrain.png', 'terrain-warp.png', 'blocked.png', 'overlay.png'}


def artifact_path(state, name):
    if name not in ARTIFACT_NAMES:
        raise ValueError('地图派生文件无效')
    if 'artifacts' in state and name not in state['artifacts']:
        raise FileNotFoundError(name)
    relative = Path(state.get('artifacts', {}).get(name, name))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('地图派生文件路径无效')
    return asset_dir(state['assetId'])/relative


def stage_artifacts(state, directory, replace_all=False):
    """Stage a complete immutable generation; save_state publishes its paths.

    Existing manifests keep referring to their old files throughout the
    operation, including if the process dies between any two file writes.
    """
    root = asset_dir(state['assetId'])
    relative = Path('derived')/uuid.uuid4().hex
    destination = root/relative
    destination.mkdir(parents=True)
    manifest = {} if replace_all else dict(state.get('artifacts', {name: name for name in ARTIFACT_NAMES if (root/name).exists()}))
    for source in Path(directory).iterdir():
        if source.name not in ARTIFACT_NAMES:
            raise ValueError('处理结果包含未知文件')
        source.replace(destination/source.name)
        with (destination/source.name).open('rb') as stream:
            os.fsync(stream.fileno())
        manifest[source.name] = str(relative/source.name)
    sync_directory(destination)
    sync_directory(destination.parent)
    sync_directory(root)
    state['artifacts'] = manifest
    return state


def prune_artifacts(asset_id, minimum_age_sec=86400):
    """Bound old derived generations without touching sources or releases.

    Keep all currently referenced directories plus the two newest ones, and
    never delete a recent generation that an in-flight response could use.
    The caller is the exclusive queue worker, between operations.
    """
    with LOCK:
        state = get_state(asset_id)
        if not state:
            return
        root = asset_dir(asset_id)
        names = state.get('artifacts', {name: name for name in ARTIFACT_NAMES})
        live = {artifact_path(state, name).parent for name in names}
        generations = sorted((root/'derived').glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
        live.update(generations[:2])
        for path in generations:
            if (path.is_dir() and path not in live and len(path.name) == 32
                    and all(c in '0123456789abcdef' for c in path.name)
                    and time.time()-path.stat().st_mtime > minimum_age_sec):
                shutil.rmtree(path)
