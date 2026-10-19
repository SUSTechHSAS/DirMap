"""Unit/API tests must not evict graphs prepared for the user's courses."""
import pytest


@pytest.fixture(autouse=True)
def isolated_scratch_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('DIRMAP_CACHE_DIR', str(tmp_path/'scratch'))
