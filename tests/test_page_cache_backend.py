# coding=utf-8
"""Page-cache-aware automatic backend selection."""

import os
import sys
import time
import types
import warnings

import pytest

# `siphon._impl` imports the compiled `siphon._C` at module scope.  The logic under
# test here never touches it, so allow the suite to run against a stub when the
# extension is not built in this tree.
if "siphon._C" not in sys.modules:
    try:
        import siphon._C  # noqa: F401
    except Exception:
        _stub = types.ModuleType("siphon._C")
        _stub.backend_values = lambda: {"AIO": 0, "AIO_BUFFERED": 1, "URING": 2,
                                        "URING_BUFFERED": 3, "CUFILE": 4, "MMAP": 5}
        _stub.MAX_IO_DEPTH = 512
        _stub.backend_status = lambda value: (True, None, None)
        _stub.file_in_memory = lambda path: False
        _stub.required_buffer_size_for_io = lambda chunk, depth, world: chunk * depth * world
        _stub.cleanup = lambda: None
        sys.modules["siphon._C"] = _stub

from siphon import _impl                                    # noqa: E402
from siphon._impl import Backend                            # noqa: E402


# --------------------------------------------------------------------------- 阈值
def test_threshold_default(monkeypatch):
    monkeypatch.delenv("SIPHON_CACHE_RESIDENT_THRESHOLD", raising=False)
    assert _impl.env_cache_resident_threshold() == _impl.DEFAULT_CACHE_RESIDENT_THRESHOLD


def test_threshold_override(monkeypatch):
    monkeypatch.setenv("SIPHON_CACHE_RESIDENT_THRESHOLD", "0.5")
    assert _impl.env_cache_resident_threshold() == 0.5


def test_threshold_invalid_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("SIPHON_CACHE_RESIDENT_THRESHOLD", "not-a-number")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = _impl.env_cache_resident_threshold()
    assert got == _impl.DEFAULT_CACHE_RESIDENT_THRESHOLD
    assert any("SIPHON_CACHE_RESIDENT_THRESHOLD" in str(w.message) for w in caught)


def test_threshold_above_one_disables_probe(monkeypatch):
    """A threshold > 1.0 can never be met, so direct I/O must be kept and no probe run."""
    monkeypatch.setenv("SIPHON_CACHE_RESIDENT_THRESHOLD", "2.0")

    def boom(_):
        raise AssertionError("probe must not run when the threshold disables it")

    monkeypatch.setattr(_impl, "page_cache_resident_ratio", boom)
    assert _impl.choose_disk_backend_candidates(["/nonexistent"]) == list(_impl.default_backend)


# --------------------------------------------------------------------------- 决策
def test_choose_warm_returns_buffered(monkeypatch):
    monkeypatch.delenv("SIPHON_CACHE_RESIDENT_THRESHOLD", raising=False)
    monkeypatch.setattr(_impl, "page_cache_resident_ratio", lambda f: 1.0)
    assert _impl.choose_disk_backend_candidates(["a", "b"]) == list(_impl.default_buffered_io_backend)


def test_choose_cold_returns_direct(monkeypatch):
    monkeypatch.delenv("SIPHON_CACHE_RESIDENT_THRESHOLD", raising=False)
    monkeypatch.setattr(_impl, "page_cache_resident_ratio", lambda f: 0.0)
    assert _impl.choose_disk_backend_candidates(["a"]) == list(_impl.default_backend)


def test_choose_unknown_probe_returns_direct(monkeypatch):
    monkeypatch.delenv("SIPHON_CACHE_RESIDENT_THRESHOLD", raising=False)
    monkeypatch.setattr(_impl, "page_cache_resident_ratio", lambda f: -1.0)
    assert _impl.choose_disk_backend_candidates(["a"]) == list(_impl.default_backend)


def test_choose_mixed_files_returns_direct(monkeypatch):
    """One cold file is enough to keep the direct-I/O strategy for the whole set."""
    monkeypatch.delenv("SIPHON_CACHE_RESIDENT_THRESHOLD", raising=False)
    ratios = {"warm": 1.0, "cold": 0.1}
    monkeypatch.setattr(_impl, "page_cache_resident_ratio", lambda f: ratios[f])
    assert _impl.choose_disk_backend_candidates(["warm", "cold"]) == list(_impl.default_backend)


def test_choose_uses_configured_threshold(monkeypatch):
    monkeypatch.setenv("SIPHON_CACHE_RESIDENT_THRESHOLD", "0.9")
    monkeypatch.setattr(_impl, "page_cache_resident_ratio", lambda f: 0.85)
    assert _impl.choose_disk_backend_candidates(["a"]) == list(_impl.default_backend)
    monkeypatch.setattr(_impl, "page_cache_resident_ratio", lambda f: 0.95)
    assert _impl.choose_disk_backend_candidates(["a"]) == list(_impl.default_buffered_io_backend)


# --------------------------------------------------------------------------- mincore 探针
def test_probe_missing_file_returns_unknown(tmp_path):
    assert _impl.page_cache_resident_ratio(str(tmp_path / "nope.safetensors")) == -1.0


def _write(path, size=4 << 20):
    with open(path, "wb") as f:
        f.write(os.urandom(size))
        f.flush()
        os.fsync(f.fileno())   # dirty pages cannot be evicted by fadvise
    return str(path)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mincore is Linux-only")
def test_probe_resident_file_is_high(tmp_path):
    path = _write(tmp_path / "hot.bin")
    with open(path, "rb") as f:
        while f.read(1 << 20):
            pass
    assert _impl.page_cache_resident_ratio(path) >= 0.9


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mincore is Linux-only")
def test_probe_evicted_file_is_low(tmp_path):
    """POSIX_FADV_DONTNEED lets us build a genuinely cold file without root."""
    path = _write(tmp_path / "cold.bin")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    deadline = time.time() + 5.0
    ratio = _impl.page_cache_resident_ratio(path)
    while ratio > 0.1 and time.time() < deadline:
        time.sleep(0.2)
        ratio = _impl.page_cache_resident_ratio(path)
    assert ratio <= 0.1


# --------------------------------------------------------------------------- 调用点接线
class _FakeLoader:
    """Minimal stand-in: _determine_io_params only reads a handful of attributes."""

    filename = ["/tmp/whatever.safetensors"]
    world_size = 1
    rank = 0
    process_group = None
    device = 0
    device_idx = 0
    framework = "pt"
    copy = True

    def __getattr__(self, name):
        return None


def _config(backend):
    return _impl._OpenConfig(buffer_size=None, chunk_size=None, concurrency=None,
                             io_depth=None, max_free_mem_usage=None, backend=backend)


def test_disk_branch_consults_probe_when_backend_auto(monkeypatch):
    seen = {}

    def fake_choose(filenames):
        seen["filenames"] = list(filenames)
        return [Backend.MMAP]

    monkeypatch.setattr(_impl, "choose_disk_backend_candidates", fake_choose)
    monkeypatch.setattr(_impl, "select_backend", lambda cands, *a, **k: cands[0])
    _impl.safe_open._determine_io_params(_FakeLoader(), _config(None))
    assert seen["filenames"] == _FakeLoader.filename


def test_explicit_backend_skips_probe(monkeypatch):
    calls = {}

    def boom(_):
        raise AssertionError("probe must not run when the backend is pinned")

    def fake_select(cands, *a, **k):
        calls["cands"] = list(cands)
        return cands[0]

    monkeypatch.setattr(_impl, "choose_disk_backend_candidates", boom)
    monkeypatch.setattr(_impl, "select_backend", fake_select)
    _impl.safe_open._determine_io_params(_FakeLoader(), _config([Backend.AIO]))
    assert calls["cands"] == [Backend.AIO]
