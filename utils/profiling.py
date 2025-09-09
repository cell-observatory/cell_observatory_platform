import os
import time
import warnings
from pathlib import Path

import errno
import atexit
import ctypes
import functools
import threading

from omegaconf import DictConfig


# ---------- enable profiling flags ----------


def enable_profiling(cfg: DictConfig):
    os.makedirs(cfg.profiling.outdir, exist_ok=True)
    os.makedirs(cfg.profiling.lockfile_dir, exist_ok=True)
    # out directory of profiling results
    os.environ["PPROF_OUTDIR"] = str(cfg.profiling.outdir)
    # enable/disable profiling
    os.environ["PPROF_ENABLE"] = "1" if bool(cfg.profiling.enable) else "0"
    # optionally set time limit of profiling
    os.environ["PPROF_DURATION_SECS"] = str(cfg.profiling.duration)
    # optionally set profiling frequency
    os.environ["PPROF_FREQ"] = str(cfg.profiling.freq)
    # sample based on wall-clock time (vs. CPU time)
    os.environ["PPROF_REALTIME"] = "1" if bool(cfg.profiling.realtime) else "0"
    # lockfile directory for profiling
    os.environ["PPROF_LOCKFILE_DIR"] = str(cfg.profiling.lockfile_dir)


# ---------- sentinels, locks, etc. ----------


_prof = None          # will be set on first use
STATE = None          # will be set on first use
_init_lock = threading.Lock()
_atexit_armed = False


# ---------- pprof init / guard functionality ----------


def _env_bool(name: str, default=False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "t", "yes", "y")


def _start_pprof():
    global _prof, STATE

    def _load_libprofiler():
        for soname in ("libprofiler.so", "libprofiler.so.0"):
            try:
                # try loading shared library into process
                return ctypes.CDLL(soname)
            except OSError:
                pass
        return None

    _prof = _load_libprofiler()
    if _prof is not None:
        # ProfilerStart takes a const char* (path to the profile output file)
        # and ProfilerStop takes no args. Thus we set argtypes/restype, so
        # ctypes can do the right conversions.
        _prof.ProfilerStart.argtypes = [ctypes.c_char_p]
        _prof.ProfilerStart.restype = ctypes.c_int
        _prof.ProfilerStop.argtypes = []
        _prof.ProfilerStop.restype = ctypes.c_int

    class _State:
        def __init__(self):
            self.lock = threading.Lock()
            self.global_started = False      # process-wide profiler state
            self.active_keys = set()         # functions that "own" the run
            self.outfiles = {}               # key -> outfile path
            self.lockfile_paths = {}         # key -> lockfile path
            self.stop_threads = {}           # key -> Thread

    STATE = _State()


def _ensure_initialized() -> bool:
    """Lazy init on first decorated call, guarded from driver."""
    global _atexit_armed
    if _prof is not None and STATE is not None:
        return True
    with _init_lock:
        # ensure prof was not initialized while 
        # waiting for lock
        if _prof is not None and STATE is not None:
            return True
        _start_pprof()
        if _prof is None or STATE is None:
            # libprofiler missing—silently no-op thereafter
            warnings.warn("libprofiler not found; pprof decorators will be no-ops.")
            return False
        if not _atexit_armed:
            atexit.register(_stop_all_at_exit)
            _atexit_armed = True
        return True


# ---------- helpers ----------


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _claim_global_lockfile(lockfile_path: Path) -> bool:
    try:
        # atomic create; fail if exists, open for write, 644 perms (octal)
        fd = os.open(str(lockfile_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        # stale PID recovery
        try:
            import psutil
            pid = int(Path(lockfile_path).read_text().strip())
            # suboptimal since PID may have been recycled
            if not psutil.pid_exists(pid):
                Path(lockfile_path).unlink(missing_ok=True)
                return _claim_global_lockfile(lockfile_path)
        except Exception:
            pass
        return False


def _release_global_lockfile(key: str):
    p = STATE.lockfile_paths.pop(key, None)
    if p:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def _stop_profiler(key: str):
    if _prof is None:
        return
    with STATE.lock:
        _release_global_lockfile(key)
        STATE.active_keys.discard(key)
        if STATE.global_started and not STATE.active_keys:
            try:
                _prof.ProfilerStop()
            except Exception:
                pass
            STATE.global_started = False


def _stopper_wait(duration: float, stop_file: str | None, key: str):
    # duration <= 0 means "run until stop_file appears" or "forever"
    if duration > 0:
        time.sleep(duration)
    else:
        if stop_file:
            stop_path = Path(stop_file)
            while not stop_path.exists():
                time.sleep(0.5)
        else:
            return
    _stop_profiler(key)


def _stop_all_at_exit():
    if _prof is None or STATE is None:
        return
    with STATE.lock:
        keys = list(STATE.active_keys)
    for k in keys:
        _release_global_lockfile(k)
    with STATE.lock:
        if STATE.global_started:
            try:
                _prof.ProfilerStop()
            except Exception:
                pass
            STATE.global_started = False
            STATE.active_keys.clear()


# TODO: could probably be rewritten to be more robust
def _maybe_start_profiler(label_hint: str | None, func_key: str):
    if not _env_bool("PPROF_ENABLE", False):
        return
    if not _ensure_initialized():
        return

    outdir_str = os.environ.get("PPROF_OUTDIR")
    if not outdir_str:
        raise ValueError("PPROF_OUTDIR must be set to a valid directory path.")
    outdir = Path(outdir_str)
    outdir.mkdir(parents=True, exist_ok=True)

    lockfile_dir = Path(os.environ.get("PPROF_LOCKFILE_DIR", str(outdir)))
    lockfile_dir.mkdir(parents=True, exist_ok=True)
    lockfile_path = lockfile_dir / f".pprof.{_sanitize(func_key)}.lock"

    duration_env = os.environ.get("PPROF_DURATION_SECS", "0").strip().lower()
    assert duration_env.isdigit() or duration_env in ("forever", "", "none"), \
        "PPROF_DURATION_SECS must be a non-negative number of seconds, 'forever', or 'none'"
    duration = 0.0 if duration_env in ("forever", "", "none") else float(duration_env)

    if "PPROF_FREQ" in os.environ:
        os.environ["CPUPROFILE_FREQUENCY"] = os.environ["PPROF_FREQ"]
    os.environ.setdefault("CPUPROFILE_FREQUENCY", "1000")
    os.environ.setdefault("CPUPROFILE_REALTIME", os.environ.get("PPROF_REALTIME", "0"))

    with STATE.lock:
        if func_key in STATE.active_keys:
            return  # already registered in this proc

        # TODO: consider allowing multiple concurrent 
        # profilers of same key
        if not _claim_global_lockfile(lockfile_path):
            return

        label = os.environ.get("PPROF_LABEL") or label_hint or "pprof"
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        outfile = outdir / f"{_sanitize(label)}.pid{os.getpid()}.{ts}.prof"

        if not STATE.global_started:
            # treat 0 and 1 as non-fatal return codes and do best-effort cleanup
            rc = int(_prof.ProfilerStart(str(outfile).encode("utf-8")))
            if rc not in (0, 1):
                try:
                    Path(lockfile_path).unlink(missing_ok=True)
                except Exception:
                    pass
                return
            STATE.global_started = True

            # bookkeeping
            STATE.active_keys.add(func_key)
            STATE.outfiles[func_key] = str(outfile)
            STATE.lockfile_paths[func_key] = str(lockfile_path)

            # daemon thread waits until time to end profiling window
            stop_file = os.environ.get("PPROF_STOP_FILE")
            th = threading.Thread(
                target=_stopper_wait,
                args=(duration, stop_file, func_key),
                daemon=True,
            )
            th.start()
            STATE.stop_threads[func_key] = th


# ---------- public decorators / context managers ----------


def pprof_func(func=None, label: str | None = None):
    def _decorate(f):
        key = f"{f.__module__}.{f.__qualname__}"
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            _maybe_start_profiler(label_hint=label or f.__qualname__, func_key=key)
            try:
                return f(*args, **kwargs)
            finally:
                _stop_profiler(key)
        return _wrapper
    return _decorate if func is None else _decorate(func)


def pprof_class(cls, label: str | None = None):
    if not hasattr(cls, "__call__"):
        return cls
    orig_call = cls.__call__
    key = f"{cls.__module__}.{cls.__name__}.__call__"

    @functools.wraps(orig_call)
    def _wrapped(self, *args, **kwargs):
        _maybe_start_profiler(label_hint=label or f"{cls.__name__}", func_key=key)
        return orig_call(self, *args, **kwargs)

    cls.__call__ = _wrapped
    return cls


class pprof_context:
    def __init__(self, label: str | None = None):
        self.label = label or "pprof"
        self.key = f"context.{os.getpid()}.{threading.get_ident()}." + _sanitize(self.label)
    def __enter__(self):
        _maybe_start_profiler(label_hint=self.label, func_key=self.key)
        return self
    def __exit__(self, exc_type, exc, tb):
        if STATE is not None and self.key in STATE.active_keys:
            _stop_profiler(self.key)
        return False