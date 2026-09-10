from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .io import canonical_hash
from .schema import CorpusUnit, ToolAction, ToolHit


TOOL_SCHEMAS = {
    "grep": {"type": "object", "additionalProperties": False, "required": ["tool", "query"],
             "properties": {"tool": {"const": "grep"}, "query": {"type": "string", "minLength": 1},
                            "path": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1}}},
    "glob": {"type": "object", "additionalProperties": False, "required": ["tool", "pattern"],
             "properties": {"tool": {"const": "glob"}, "pattern": {"type": "string", "minLength": 1},
                            "max_results": {"type": "integer", "minimum": 1}}},
    "read_file": {"type": "object", "additionalProperties": False,
                  "required": ["tool", "path", "start_line", "end_line"],
                  "properties": {"tool": {"const": "read_file"}, "path": {"type": "string", "minLength": 1},
                                 "start_line": {"type": "integer", "minimum": 1},
                                 "end_line": {"type": "integer", "minimum": 1}}},
}
TOOL_SCHEMA_HASH = canonical_hash(TOOL_SCHEMAS)


@dataclass(frozen=True)
class SearchLimits:
    max_results: int = 5
    max_bytes_per_hit: int = 1600
    max_total_bytes: int = 6000
    grep_context_lines: int = 4
    max_query_chars: int = 256


class CodingTools:
    """Pinned local reproduction of grep/glob/read_file over the raw corpus."""

    def __init__(self, root: str | Path, units: Iterable[CorpusUnit], limits: SearchLimits = SearchLimits()):
        self.root = Path(root).resolve()
        self.units = tuple(units)
        self.limits = limits
        self._by_path: dict[str, list[CorpusUnit]] = {}
        for unit in self.units:
            self._by_path.setdefault(unit.source_path, []).append(unit)

    def _path(self, relative: str) -> Path | None:
        try:
            candidate = (self.root / relative).resolve()
            candidate.relative_to(self.root)
        except (ValueError, OSError):
            return None
        return candidate

    def _paths(self, selector: str = ".") -> list[str]:
        if selector in ("", "."):
            return sorted(self._by_path)
        if any(char in selector for char in "*?["):
            return sorted(path for path in self._by_path if fnmatch.fnmatch(path, selector))
        prefix = selector.rstrip("/")
        return sorted(path for path in self._by_path if path == prefix or path.startswith(prefix + "/"))

    def _container(self, path: str, start: int, end: int) -> str:
        overlaps = [u for u in self._by_path.get(path, ()) if u.start_line <= end and u.end_line >= start]
        return min(overlaps, key=lambda u: (u.end_line - u.start_line, u.unit_id)).unit_id if overlaps else ""

    @staticmethod
    def _clip_utf8(text: str, limit: int) -> str:
        return text.encode("utf-8")[:max(0, limit)].decode("utf-8", errors="ignore")

    def _render(self, hits: list[ToolHit]) -> tuple[tuple[ToolHit, ...], str]:
        rendered, final_hits, used = [], [], 0
        for hit in hits:
            header = f"{hit.source_path}:{hit.start_line}-{hit.end_line}\n"
            available = min(self.limits.max_bytes_per_hit, self.limits.max_total_bytes - used)
            if available <= len(header.encode("utf-8")):
                break
            body = self._clip_utf8(hit.text, available - len(header.encode("utf-8")))
            visible = header + body
            size = len(visible.encode("utf-8")); used += size
            final_hits.append(ToolHit(hit.chunk_id, hit.source_path, hit.start_line, hit.end_line,
                                      body, len(final_hits) + 1, size, hit.tool))
            rendered.append(visible)
        return tuple(final_hits), "\n---\n".join(rendered) if rendered else "NO RESULTS"

    def execute(self, action: ToolAction) -> tuple[bool, tuple[ToolHit, ...], str]:
        if action.tool == "grep": return self._grep(action)
        if action.tool == "glob": return self._glob(action)
        if action.tool == "read_file": return self._read_file(action)
        return False, (), "ERROR: unknown tool"

    def _grep(self, action: ToolAction):
        query = action.query.strip()
        if not query or len(query) > self.limits.max_query_chars or action.max_results < 1:
            return False, (), "ERROR: invalid grep action"
        try: regex = re.compile(query)
        except re.error as error: return False, (), f"ERROR: invalid grep regex: {error}"
        cap, hits = min(action.max_results, self.limits.max_results), []
        for relative in self._paths(action.path):
            candidate = self._path(relative)
            if candidate is None or not candidate.is_file(): continue
            try: lines = candidate.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError: continue
            for index, line in enumerate(lines, 1):
                if regex.search(line) is None: continue
                start = max(1, index - self.limits.grep_context_lines)
                end = min(len(lines), index + self.limits.grep_context_lines)
                body = "\n".join(lines[start - 1:end])
                hits.append(ToolHit(self._container(relative, start, end), relative, start, end,
                                    body, len(hits) + 1, 0, "grep"))
                if len(hits) >= cap: return (True, *self._render(hits))
        return (True, *self._render(hits))

    def _glob(self, action: ToolAction):
        if not action.pattern or len(action.pattern) > self.limits.max_query_chars:
            return False, (), "ERROR: invalid glob action"
        cap = min(action.max_results, self.limits.max_results)
        paths = sorted(path for path in self._by_path if fnmatch.fnmatch(path, action.pattern))[:cap]
        hits = [ToolHit("", path, 0, 0, path, i + 1, 0, "glob") for i, path in enumerate(paths)]
        return (True, *self._render(hits))

    def _read_file(self, action: ToolAction):
        if not action.path or action.start_line is None or action.end_line is None:
            return False, (), "ERROR: read_file requires path/start_line/end_line"
        if action.start_line < 1 or action.end_line < action.start_line:
            return False, (), "ERROR: invalid read_file line range"
        candidate = self._path(action.path)
        if candidate is None or not candidate.is_file() or action.path not in self._by_path:
            return False, (), "ERROR: path is outside or absent from pinned corpus"
        try: lines = candidate.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError: return False, (), "ERROR: file is not UTF-8"
        if action.start_line > len(lines): return False, (), "ERROR: start_line exceeds file length"
        end = min(action.end_line, len(lines)); body = "\n".join(lines[action.start_line - 1:end])
        hit = ToolHit(self._container(action.path, action.start_line, end), action.path,
                      action.start_line, end, body, 1, 0, "read_file")
        return (True, *self._render([hit]))


class CorpusSearch(CodingTools):
    """Compatibility adapter for isolated tests; production uses CodingTools(raw_root)."""
    def __init__(self, units: Iterable[CorpusUnit], limits: SearchLimits = SearchLimits()):
        import tempfile
        self._temporary = tempfile.TemporaryDirectory(prefix="latent-study-test-corpus-")
        root = Path(self._temporary.name); grouped: dict[str, list[CorpusUnit]] = {}
        for unit in units: grouped.setdefault(unit.source_path, []).append(unit)
        for path, file_units in grouped.items():
            destination = root / path; destination.parent.mkdir(parents=True, exist_ok=True)
            lines = [""] * max(u.end_line for u in file_units)
            for unit in sorted(file_units, key=lambda u: u.start_line):
                body = unit.text.splitlines(); lines[unit.start_line - 1:unit.start_line - 1 + len(body)] = body
            destination.write_text("\n".join(lines), encoding="utf-8")
        super().__init__(root, tuple(u for values in grouped.values() for u in values), limits)


def serialize_tool_observation(action: ToolAction, observation: str) -> str:
    return json.dumps({"action": action.to_payload(), "observation": observation},
                      ensure_ascii=False, sort_keys=True)
