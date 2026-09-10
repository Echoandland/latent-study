from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from .io import canonical_hash, sha256_bytes, sha256_text
from .schema import CorpusUnit

_TEXT_EXTENSIONS = {".py", ".md", ".rst", ".txt"}
_EXCLUDED_DIRS = {".git", ".tox", ".venv", "node_modules", "dist", "build", "__pycache__"}


def _token_estimate(text: str) -> int:
    # Audit-only, tokenizer-independent estimate. Training reports actual tokenizer counts.
    return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


def _document_id(relative: str, digest: str) -> str:
    return f"doc_{sha256_text(relative + ':' + digest)[:16]}"


def _unit_id(relative: str, kind: str, name: str, start: int, end: int, digest: str) -> str:
    return f"unit_{sha256_text('|'.join(map(str, (relative, kind, name, start, end, digest))))[:20]}"


def _python_units(path: Path, relative: str, text: str, document_id: str, source_hash: str) -> list[CorpusUnit]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out: list[CorpusUnit] = []
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not isinstance(parents.get(node), ast.Module):
            continue  # nested definitions remain symbols inside their top-level semantic unit
        if not hasattr(node, "end_lineno") or node.end_lineno is None:
            continue
        pieces, cur = [node.name], parents.get(node)
        while isinstance(cur, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            pieces.append(cur.name)
            cur = parents.get(cur)
        name = ".".join(reversed(pieces))
        start, end = node.lineno, node.end_lineno
        body = "\n".join(lines[start - 1 : end])
        refs = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
        calls = sorted({
            n.func.id if isinstance(n.func, ast.Name) else n.func.attr
            for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Name, ast.Attribute))
        })
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        nested = sorted({f"{name}.{n.name}" for n in ast.walk(node)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node})
        defined_symbols = (name, *nested)
        digest = sha256_text(body)
        out.append(CorpusUnit(
            unit_id=_unit_id(relative, kind, name, start, end, digest),
            document_id=document_id, source_path=relative, kind=kind, name=name,
            start_line=start, end_line=end, content_hash=digest, source_hash=source_hash,
            eligible_tokens=_token_estimate(body), symbols=defined_symbols,
            definitions=defined_symbols, references=tuple(refs),
            relations=tuple((name, "calls", called) for called in calls), text=body,
        ))
    # Add the non-definition gaps as non-overlapping module sections. Together
    # with top-level definitions these partition the file; eligible tokens are
    # therefore never counted twice.
    spans = sorted((u.start_line, u.end_line) for u in out)
    cursor = 1
    gap_spans = []
    for start, end in spans:
        if cursor < start:
            gap_spans.append((cursor, start - 1))
        cursor = end + 1
    if cursor <= len(lines):
        gap_spans.append((cursor, len(lines)))
    for start, end in gap_spans:
        body = "\n".join(lines[start - 1:end])
        if not body.strip():
            continue
        gap_nodes = [n for n in tree.body if getattr(n, "lineno", 0) >= start
                     and getattr(n, "end_lineno", 0) <= end]
        definitions, imports = set(), set()
        for node in gap_nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    definitions.add(alias.asname or alias.name.split(".")[0])
                    imports.add(alias.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                definitions.update(t.id for target in targets for t in ast.walk(target)
                                   if isinstance(t, ast.Name))
        name = f"module@{start}"
        digest = sha256_text(body)
        out.append(CorpusUnit(
            unit_id=_unit_id(relative, "module_section", name, start, end, digest),
            document_id=document_id, source_path=relative, kind="module_section", name=name,
            start_line=start, end_line=end, content_hash=digest, source_hash=source_hash,
            eligible_tokens=_token_estimate(body), symbols=tuple(sorted(definitions)),
            definitions=tuple(sorted(definitions)), references=tuple(sorted(imports)),
            relations=tuple((relative, "imports", target) for target in sorted(imports)), text=body,
        ))
    return sorted(out, key=lambda u: (u.start_line, u.end_line, u.name))


def _prose_units(relative: str, text: str, document_id: str, source_hash: str) -> list[CorpusUnit]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines, 1) if re.match(r"^#{1,6}\s+\S", line)]
    bounds = starts or [1]
    out: list[CorpusUnit] = []
    for index, start in enumerate(bounds):
        end = (bounds[index + 1] - 1) if index + 1 < len(bounds) else len(lines)
        body = "\n".join(lines[start - 1 : end]).strip()
        if not body:
            continue
        heading = re.sub(r"^#{1,6}\s+", "", lines[start - 1]).strip() if starts else f"paragraphs@{start}"
        digest = sha256_text(body)
        out.append(CorpusUnit(
            unit_id=_unit_id(relative, "section", heading, start, end, digest),
            document_id=document_id, source_path=relative, kind="section", name=heading,
            start_line=start, end_line=end, content_hash=digest, source_hash=source_hash,
            eligible_tokens=_token_estimate(body), definitions=(heading,), text=body,
        ))
    return out


def build_manifest(root: str | Path) -> dict:
    root = Path(root).resolve()
    documents, units, excluded = [], [], []
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if any(part in _EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            excluded.append({"path": relative, "reason": "unsupported_extension"})
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            excluded.append({"path": relative, "reason": "non_utf8"})
            continue
        digest = sha256_bytes(raw)
        document_id = _document_id(relative, digest)
        file_units = (_python_units(path, relative, text, document_id, digest) if path.suffix == ".py"
                      else _prose_units(relative, text, document_id, digest))
        if not file_units and text.strip():
            body_hash = sha256_text(text)
            file_units = [CorpusUnit(
                unit_id=_unit_id(relative, "file", relative, 1, max(1, len(text.splitlines())), body_hash),
                document_id=document_id, source_path=relative, kind="file", name=relative,
                start_line=1, end_line=max(1, len(text.splitlines())), content_hash=body_hash,
                source_hash=digest,
                eligible_tokens=_token_estimate(text), text=text,
            )]
        documents.append({"document_id": document_id, "path": relative, "content_hash": digest,
                          "eligible_tokens": sum(u.eligible_tokens for u in file_units), "unit_count": len(file_units)})
        units.extend(file_units)
    corpus_hash = canonical_hash([{"path": d["path"], "content_hash": d["content_hash"]} for d in documents])
    return {"schema_version": 1, "root": str(root), "corpus_hash": corpus_hash,
            "documents": documents, "units": units, "excluded": excluded,
            "counts": {"documents": len(documents), "units": len(units),
                       "eligible_tokens": sum(u.eligible_tokens for u in units),
                       "symbols": sum(len(u.symbols) for u in units),
                       "verified_relations": sum(len(u.relations) for u in units)}}
