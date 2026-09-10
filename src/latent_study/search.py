from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schema import CorpusUnit, ToolAction, ToolHit


@dataclass(frozen=True)
class SearchLimits:
    max_results: int = 5
    max_bytes_per_hit: int = 1600
    max_total_bytes: int = 6000


class CorpusSearch:
    """One deterministic search implementation shared by study and evaluation."""

    def __init__(self, units: Iterable[CorpusUnit], limits: SearchLimits = SearchLimits()):
        self.units = tuple(units)
        self.limits = limits

    def execute(self, action: ToolAction) -> tuple[bool, tuple[ToolHit, ...], str]:
        query = action.query.strip()
        if not query or len(query) > 256 or action.max_results < 1:
            return False, (), "ERROR: invalid search action"
        terms = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", query)]
        if not terms:
            return False, (), "ERROR: search query has no searchable terms"
        scored = []
        for unit in self.units:
            hay = f"{unit.source_path}\n{unit.name}\n{unit.text}".lower()
            score = sum(hay.count(term) for term in terms)
            if score:
                scored.append((-score, unit.source_path, unit.start_line, unit))
        cap = min(action.max_results, self.limits.max_results)
        used, hits, rendered = 0, [], []
        for rank, (_, _, _, unit) in enumerate(sorted(scored)[:cap], 1):
            header = f"[{unit.unit_id}] {unit.source_path}:{unit.start_line}-{unit.end_line}\n"
            available = min(self.limits.max_bytes_per_hit, self.limits.max_total_bytes - used)
            if available <= len(header.encode()):
                break
            body_bytes = unit.text.encode("utf-8")[: available - len(header.encode())]
            body = body_bytes.decode("utf-8", errors="ignore")
            visible = header + body
            size = len(visible.encode("utf-8"))
            used += size
            hits.append(ToolHit(unit.unit_id, unit.source_path, unit.start_line, unit.end_line,
                                body, rank, size))
            rendered.append(visible)
        return True, tuple(hits), "\n---\n".join(rendered) if rendered else "NO RESULTS"

