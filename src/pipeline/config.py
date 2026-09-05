from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

ENV_PREFIX = "NLP_"


def _physical_cores() -> int:
    import psutil

    # cpu_count(logical=False) returns None on some platforms (e.g. NetBSD).
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def _env_bool(raw: str, default: bool) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on") if raw else default


@dataclass(frozen=True)
class Config:
    """Resolved run configuration. Every field here is one NLP_<FIELD> env var."""

    max_pages: int = 0
    injection_guard: bool = False  # CLAUDE.md: code stays, cost off benchmark runs by default
    backend: str = "torch"
    device: str = "cpu"
    dtype: str = "fp32"
    onnx_provider: str = "CPUExecutionProvider"
    attn_impl: str = "sdpa"
    cpu_threads: int = 1
    workers: int = 1
    batch_size: int = 1  # recorded in the run header; no stage batches across documents yet
    summary_max_new_tokens: int = 64
    reduced_context_chars: int = 2000
    gemma_gguf: str = "models_gguf/gemma-4-E2B_q4_0-it.gguf"
    zeroshot_max_chunks: int = 3
    min_words: int = 120  # validate stage: "too_short" floor
    min_chars: int = 0  # validate stage: 0 disables the check; --match-denzel sets 100
    protocol: str = "A"  # A: single pass, no warm-up (matches Denzel). B: 1 warm-up + 3 timed, median.
    taxonomy: str = "both"  # assessment | denzel | both
    # Matches transformers' own ZeroShotClassificationPipeline default (what
    # the torch path already used implicitly). onnx_zero_shot previously
    # hardcoded "This example is about {}." instead -- a latent mismatch
    # between backends that this field removes. Wording measurably changes
    # accuracy (CLAUDE.md), so it is config, not a constant.
    hypothesis_template: str = "This example is {}."


def load_config(env: dict[str, str] | None = None) -> Config:
    """Defaults plus NLP_* overrides. `env` defaults to os.environ; pass an
    explicit dict in tests instead of mutating the real environment."""
    env = os.environ if env is None else env
    defaults = Config(cpu_threads=_physical_cores())
    overrides: dict[str, object] = {}
    for f in fields(Config):
        env_name = ENV_PREFIX + f.name.upper()
        if env_name not in env:
            continue
        raw = env[env_name]
        current = getattr(defaults, f.name)
        if isinstance(current, bool):
            overrides[f.name] = _env_bool(raw, current)
        elif isinstance(current, int):
            overrides[f.name] = int(raw)
        else:
            overrides[f.name] = raw
    return replace(defaults, **overrides)


def config_fingerprint(config: Config, keys: tuple[str, ...]) -> str:
    """Fingerprint of a declared subset of config fields, keyed by the
    stage's own config_keys (e.g. "MAX_PAGES" -> config.max_pages)."""
    values = {k: getattr(config, k.lower()) for k in keys}
    blob = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def write_config(config: Config, run_id: str, runs_dir: Path = Path("runs")) -> Path:
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
    return path
