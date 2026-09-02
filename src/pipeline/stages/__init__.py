from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pipeline.config import Config


@dataclass
class DocContext:
    doc_id: str
    pdf_bytes: bytes
    config: Config
    # upstream payloads computed so far for this doc, keyed by stage name
    payloads: dict[str, dict] = field(default_factory=dict)


class Stage(Protocol):
    name: str
    version: str
    depends_on: tuple[str, ...]
    config_keys: tuple[str, ...]

    def run(self, doc: DocContext) -> dict: ...


STAGES: dict[str, Stage] = {}


def register(stage: Stage) -> Stage:
    STAGES[stage.name] = stage
    return stage


def ordered_stages() -> list[Stage]:
    """Dependency order over STAGES (Kahn's algorithm), so the runner never
    hardcodes the stage sequence."""
    remaining = dict(STAGES)
    ordered: list[Stage] = []
    while remaining:
        ready = [s for s in remaining.values() if all(d not in remaining for d in s.depends_on)]
        if not ready:
            raise RuntimeError(f"circular stage dependency among {sorted(remaining)}")
        ready.sort(key=lambda s: s.name)
        for s in ready:
            ordered.append(s)
            del remaining[s.name]
    return ordered


def downstream_closure(stage_names: set[str]) -> set[str]:
    """`stage_names` plus every stage that transitively depends on one of them."""
    closure = set(stage_names)
    changed = True
    while changed:
        changed = False
        for s in STAGES.values():
            if s.name not in closure and any(dep in closure for dep in s.depends_on):
                closure.add(s.name)
                changed = True
    return closure


# Imported for registration side effects only -- each module calls register()
# at import time. Must come after Stage/STAGES/register are defined above.
from pipeline.stages import (  # noqa: E402,F401
    classify,
    clean,
    context,
    extract,
    keywords,
    summarize,
    validate,
)
