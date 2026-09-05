from __future__ import annotations

import os
import subprocess
import threading

import psutil

SAMPLE_INTERVAL_S = 0.05


def gpu_clock_temp() -> dict | None:
    """SM clock (MHz) and temperature (C) via nvidia-smi -- CLAUDE.md: the
    1650 Mobile shares a 50W thermal envelope with the CPU, so a run that
    throttles must be visible in the metrics, not just "felt slow". Returns
    None (not zeros) when nvidia-smi isn't available or the call fails --
    "not measured" must never look like "measured and idle"."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        clock_mhz, temp_c = (x.strip() for x in out.stdout.strip().split(","))
        return {"clock_sm_mhz": int(clock_mhz), "temperature_c": int(temp_c)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def apply_env(threads: int) -> None:
    """Must run before torch is imported anywhere in this process -- OpenMP
    reads OMP_NUM_THREADS at import time, not at call time, so setting it
    after `import torch` is silently ignored."""
    if threads and threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OMP_PROC_BIND"] = "CLOSE"
        os.environ["OMP_SCHEDULE"] = "STATIC"


def configure(threads: int) -> str:
    """Call once per process (main process and each worker process), before
    any model gets loaded. torch.set_num_threads() always takes precedence
    over the env vars, so both are set to remove any ambiguity about what
    actually applied. Returns torch.__config__.parallel_info() as the run
    header's proof of what was configured."""
    apply_env(threads)
    import torch

    if threads and threads > 0:
        torch.set_num_threads(threads)
    return torch.__config__.parallel_info()


def is_cuda_oom(exc: BaseException) -> bool:
    """The OOM boundary is a headline GPU result (CLAUDE.md), not an error
    to bury in a generic traceback -- callers use this to route it to a
    distinct status instead of the catch-all error path."""
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except ImportError:
        return False


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def sync(device: str) -> None:
    """Call immediately before stopping any timer that may have touched the
    GPU. CUDA kernel launches are async -- without this, a timer measures
    how long it took to queue work, not to do it, and the resulting GPU
    numbers look impossibly fast and are wrong. No-op on CPU."""
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


class RSSSampler:
    """Samples this process's RSS on a background thread every 50ms and
    tracks the peak -- a single before/after reading misses the peak during
    a model's forward pass, which is what actually matters for a memory
    budget. Also tracks peak VRAM when CUDA is actually usable in this
    process; stays None (not 0) when it isn't, so "not measured" is never
    confused with "measured and empty"."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self.peak_rss = 0
        self.peak_vram_mb: float | None = None
        self._cuda = False
        try:
            import torch

            self._cuda = torch.cuda.is_available()
        except ImportError:
            self._cuda = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "RSSSampler":
        self.peak_rss = self._total_rss()
        if self._cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self._total_rss()
            self.peak_rss = max(self.peak_rss, rss)
            self._stop.wait(SAMPLE_INTERVAL_S)

    def _total_rss(self) -> int:
        # A worker process's memory belongs to this measurement too --
        # sampling only self.memory_info() would report multi-worker cells
        # as cheaper than the single-process case, which is backwards.
        total = 0
        try:
            total += self._process.memory_info().rss
            for child in self._process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        return total

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        if self._cuda:
            import torch

            self.peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
