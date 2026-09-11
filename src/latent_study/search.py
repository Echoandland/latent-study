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


class ToolActionValidationError(ValueError):
    """A model/tool action does not satisfy the public JSON schema."""


def _is_integer(value) -> bool:
    # ``bool`` is an ``int`` subclass but is never a valid JSON-schema integer
    # for a line number or result count.
    return isinstance(value, int) and not isinstance(value, bool)


def parse_and_validate_tool_action(value) -> ToolAction:
    """Parse one action using the exact schema exposed in ``TOOL_SCHEMAS``.

    The validator intentionally derives allowed keys, required keys, types and
    minimums from ``TOOL_SCHEMAS`` rather than maintaining a second list in the
    agent loop.  It accepts a JSON string or an already-decoded object.
    """
    if isinstance(value, str):
        try:
            def reject_duplicate_keys(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        raise ToolActionValidationError(f"duplicate property: {key}")
                    result[key] = item
                return result
            value = json.loads(value, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ToolActionValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ToolActionValidationError("action must be a JSON object")
    tool = value.get("tool")
    if not isinstance(tool, str) or tool not in TOOL_SCHEMAS:
        raise ToolActionValidationError("tool must be one of grep, glob, read_file")
    schema = TOOL_SCHEMAS[tool]
    required = set(schema.get("required", ()))
    properties = schema.get("properties", {})
    unknown = set(value) - set(properties)
    missing = required - set(value)
    if unknown and schema.get("additionalProperties", True) is False:
        raise ToolActionValidationError("unknown properties: " + ", ".join(sorted(map(str, unknown))))
    if missing:
        raise ToolActionValidationError("missing properties: " + ", ".join(sorted(missing)))
    for name, spec in properties.items():
        if name not in value:
            continue
        item = value[name]
        if "const" in spec and item != spec["const"]:
            raise ToolActionValidationError(f"{name} must equal {spec['const']!r}")
        if "enum" in spec and item not in spec["enum"]:
            raise ToolActionValidationError(f"{name} must be one of {spec['enum']!r}")
        expected = spec.get("type")
        if expected == "string":
            if (not isinstance(item, str) or len(item) < int(spec.get("minLength", 0))
                    or ("maxLength" in spec and len(item) > int(spec["maxLength"]))
                    or ("pattern" in spec and re.search(spec["pattern"], item) is None)):
                raise ToolActionValidationError(f"{name} must satisfy the string schema")
        elif expected == "integer":
            if (not _is_integer(item) or item < int(spec.get("minimum", 0))
                    or ("maximum" in spec and item > int(spec["maximum"]))):
                raise ToolActionValidationError(f"{name} must satisfy the integer schema")
        elif expected == "number":
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or ("minimum" in spec and item < float(spec["minimum"]))
                    or ("maximum" in spec and item > float(spec["maximum"]))):
                raise ToolActionValidationError(f"{name} must satisfy the number schema")
        elif expected == "boolean" and not isinstance(item, bool):
            raise ToolActionValidationError(f"{name} must be a boolean")
        elif isinstance(expected, list):
            valid_types = {"integer": _is_integer, "string": lambda x: isinstance(x, str),
                           "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
                           "boolean": lambda x: isinstance(x, bool)}
            if not any(valid_types.get(kind, lambda _x: True)(item) for kind in expected):
                raise ToolActionValidationError(f"{name} has an invalid type")
    if tool == "grep":
        payload = {"tool": tool, "query": value["query"],
                   "path": value.get("path", "."), "max_results": value.get("max_results", 5)}
        if not isinstance(payload["path"], str):
            raise ToolActionValidationError("path must be a string")
        return ToolAction(**payload)
    if tool == "glob":
        return ToolAction(tool=tool, pattern=value["pattern"],
                          max_results=value.get("max_results", 5))
    start, end = value["start_line"], value["end_line"]
    if start > end:
        raise ToolActionValidationError("end_line must be >= start_line")
    return ToolAction(tool=tool, path=value["path"], start_line=start, end_line=end)


@dataclass(frozen=True)
class SearchLimits:
    max_results: int = 5
    max_bytes_per_hit: int = 1600
    max_total_bytes: int = 6000
    grep_context_lines: int = 4
    max_query_chars: int = 256


class CodingTools:
    """Pinned local reproduction of grep/glob/read_file over the raw corpus."""

    def __init__(self, root: str | Path, units: Iterable[CorpusUnit], limits: SearchLimits = SearchLimits(),
                 *, manifest: dict | None = None):
        self.root = Path(root).resolve()
        if manifest is not None:
            from .corpus import verify_manifest_against_live_corpus
            verify_manifest_against_live_corpus(manifest, self.root)
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
        try:
            raw = action.to_payload() if isinstance(action, ToolAction) else action
            action = parse_and_validate_tool_action(raw)
        except (AttributeError, TypeError, ToolActionValidationError) as exc:
            return False, (), f"ERROR: invalid tool action: {exc}"
        try:
            if action.tool == "grep": return self._grep(action)
            if action.tool == "glob": return self._glob(action)
            if action.tool == "read_file": return self._read_file(action)
            return False, (), "ERROR: unknown tool"
        except (AttributeError, TypeError, ValueError, re.error) as exc:
            # A malformed model action must be observable as a tool error, not
            # as an exception that aborts a study/root-agent episode.
            return False, (), f"ERROR: tool execution failed: {exc}"

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
