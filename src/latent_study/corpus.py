from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

from .io import canonical_hash, sha256_bytes, sha256_text
from .schema import CorpusUnit

_TEXT_EXTENSIONS = {".py", ".md", ".rst", ".txt"}
_EXCLUDED_DIRS = {".git", ".tox", ".venv", "node_modules", "dist", "build", "__pycache__"}


class CorpusIntegrityError(RuntimeError):
    """Raised when a live corpus no longer matches a frozen manifest."""


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


def _module_name(relative: str) -> str:
    value = relative[:-3] if relative.endswith(".py") else relative
    if value.endswith("/__init__"):
        value = value[:-9]
    return value.replace("/", ".").strip(".")


def _node_line_span(node: ast.AST, lines: list[str]) -> dict:
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    text = "\n".join(lines[start - 1:end])
    return {"start_line": start, "end_line": end, "text": text,
            "text_hash": sha256_text(text)}


def _declaration_line_span(node: ast.AST, lines: list[str]) -> dict:
    start = int(getattr(node, "lineno", 1))
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start = min(start, *(int(getattr(item, "lineno", start)) for item in decorators))
    body = getattr(node, "body", ())
    end = max(start, int(getattr(body[0], "lineno", start + 1)) - 1) if body else start
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        end = min(int(getattr(body[0], "end_lineno", end)), start + 12)
    text = "\n".join(lines[start - 1:end])
    return {"start_line": start, "end_line": end, "text": text, "text_hash": sha256_text(text)}


def _target_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)
            and isinstance(item.ctx, (ast.Store, ast.Del))}


def _scope_bindings(scope: ast.AST) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Return local, global and nonlocal names for one lexical function scope.

    Nested bodies are not traversed; their declaration names still bind in the
    enclosing scope.  Comprehension targets are conservatively treated as
    shadowing for calls inside the containing function.
    """
    local: set[str] = set()
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()
    assigned: set[str] = set()
    args = getattr(scope, "args", None)
    if args is not None:
        local.update(a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))
        if args.vararg: local.add(args.vararg.arg)
        if args.kwarg: local.add(args.kwarg.arg)

    def visit(node: ast.AST) -> None:
        if node is not scope and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local.add(node.name); assigned.add(node.name)
            return
        if isinstance(node, ast.Lambda) and node is not scope:
            return
        if isinstance(node, ast.Global): global_names.update(node.names)
        elif isinstance(node, ast.Nonlocal): nonlocal_names.update(node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if hasattr(node, "target") else [])
            for target in targets:
                names = _target_names(target); local.update(names); assigned.update(names)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names = _target_names(node.target); local.update(names); assigned.update(names)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    names = _target_names(item.optional_vars); local.update(names); assigned.update(names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            name = node.name if isinstance(node.name, str) else node.name.id
            local.add(name); assigned.add(name)
        elif isinstance(node, (ast.comprehension,)):
            names = _target_names(node.target); local.update(names); assigned.update(names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                local.add(name); assigned.add(name)
        for child in ast.iter_child_nodes(node): visit(child)

    for child in ast.iter_child_nodes(scope): visit(child)
    rebound_globals = local & global_names
    local.difference_update(global_names)
    return local, global_names, nonlocal_names, rebound_globals, assigned


def _resolve_python_relations(root: Path, units: list[CorpusUnit]) -> tuple[list[dict], dict]:
    """Resolve only call edges justified by explicit lexical/import bindings."""
    python_units = [u for u in units if u.source_path.endswith(".py")]
    units_by_path: dict[str, list[CorpusUnit]] = {}
    for unit in python_units:
        units_by_path.setdefault(unit.source_path, []).append(unit)
    parsed: dict[str, tuple[ast.Module, list[str]]] = {}
    definitions: dict[tuple[str, str], list[tuple[ast.AST, CorpusUnit]]] = {}
    module_paths: dict[str, list[str]] = {}
    for path, file_units in units_by_path.items():
        text = (root / path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        parsed[path] = (tree, lines)
        module_paths.setdefault(_module_name(path), []).append(path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                container = next((u for u in file_units if u.start_line == node.lineno), None)
                if container is None:
                    continue
                definitions.setdefault((path, node.name), []).append((node, container))
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            definitions.setdefault((path, f"{node.name}.{child.name}"), []).append((child, container))

    accepted: list[dict] = []
    counts = {"syntactic": 0, "uniquely_verified": 0, "ambiguous": 0,
              "shadowed": 0, "excluded": 0}
    excluded_by_reason: dict[str, int] = {}

    def exclude(reason: str, ambiguous: bool = False, shadowed: bool = False) -> None:
        counts["shadowed" if shadowed else "ambiguous" if ambiguous else "excluded"] += 1
        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1

    for path, (tree, lines) in parsed.items():
        import_symbols: dict[str, list[tuple[str, str]]] = {}
        import_modules: dict[str, list[str]] = {}
        module_binding_counts: dict[str, int] = {}
        def module_bind(name: str) -> None:
            module_binding_counts[name] = module_binding_counts.get(name, 0) + 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_bind(node.name)
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    if alias.name != "*":
                        local_name = alias.asname or alias.name
                        import_symbols.setdefault(local_name, []).append((node.module, alias.name))
                        module_bind(local_name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name.split(".")[0]
                    import_modules.setdefault(local_name, []).append(alias.name)
                    module_bind(local_name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target_node in targets:
                    for name in _target_names(target_node): module_bind(name)

        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            counts["syntactic"] += 1
            owner = call
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                owner = parents[owner]
            if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exclude("call_outside_semantic_definition")
                continue
            outer = owner
            while isinstance(parents.get(outer), (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                outer = parents[outer]
            caller = next((u for u in units_by_path[path] if u.start_line == outer.lineno), None)
            if caller is None:
                exclude("caller_not_indexed")
                continue
            target: tuple[ast.AST, CorpusUnit] | None = None
            rule = ""
            binding: dict = {}
            func = call.func
            lexical_scope = owner if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
            local_names, global_names, nonlocal_names, rebound_globals, assigned_names = (
                _scope_bindings(lexical_scope) if lexical_scope is not None
                else (set(), set(), set(), set(), set()))
            if isinstance(func, ast.Name):
                if func.id in nonlocal_names:
                    exclude("nonlocal_callable_binding", shadowed=True)
                    continue
                if func.id in local_names and func.id not in global_names:
                    exclude("lexically_shadowed_name_call", shadowed=True)
                    continue
                if func.id in rebound_globals:
                    exclude("rebound_global_callable", shadowed=True)
                    continue
                local = definitions.get((path, func.id), [])
                if len(local) == 1:
                    if module_binding_counts.get(func.id, 0) != 1:
                        exclude("ambiguous_module_binding", ambiguous=True)
                        continue
                    target, rule = local[0], "direct_local_function"
                elif len(local) > 1:
                    exclude("duplicate_local_definition", ambiguous=True)
                    continue
                elif func.id in import_symbols:
                    bindings = import_symbols[func.id]
                    if len(bindings) != 1:
                        exclude("duplicate_import_binding", ambiguous=True)
                        continue
                    module, symbol = bindings[0]
                    paths = module_paths.get(module, [])
                    candidates = [item for p in paths for item in definitions.get((p, symbol), [])]
                    if len(candidates) == 1:
                        target, rule = candidates[0], "from_import_symbol"
                        binding = {"local_name": func.id, "module": module, "symbol": symbol}
                    else:
                        exclude("ambiguous_or_missing_from_import_target", ambiguous=len(candidates) > 1)
                        continue
                else:
                    exclude("unresolved_name_call")
                    continue
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                base = func.value.id
                if base in {"self", "cls"} and base in assigned_names:
                    exclude("reassigned_self_or_cls", shadowed=True)
                    continue
                if base in rebound_globals:
                    exclude("rebound_global_receiver", shadowed=True)
                    continue
                if base in nonlocal_names or (base in local_names and base not in {"self", "cls"}
                                              and base not in global_names):
                    exclude("lexically_shadowed_receiver", shadowed=True)
                    continue
                if base in {"self", "cls"}:
                    cls = owner
                    while cls in parents and not isinstance(cls, ast.ClassDef):
                        cls = parents[cls]
                    candidates = definitions.get((path, f"{cls.name}.{func.attr}"), []) if isinstance(cls, ast.ClassDef) else []
                    if len(candidates) == 1:
                        target, rule = candidates[0], "class_local_method"
                    else:
                        exclude("ambiguous_or_missing_class_method", ambiguous=len(candidates) > 1)
                        continue
                elif base in import_modules:
                    modules = import_modules[base]
                    if len(modules) != 1:
                        exclude("duplicate_module_import_binding", ambiguous=True)
                        continue
                    module = modules[0]
                    paths = module_paths.get(module, [])
                    candidates = [item for p in paths for item in definitions.get((p, func.attr), [])]
                    if len(candidates) == 1:
                        target, rule = candidates[0], "import_alias_attribute"
                        binding = {"alias": base, "module": module, "symbol": func.attr}
                    else:
                        exclude("ambiguous_or_missing_module_attribute", ambiguous=len(candidates) > 1)
                        continue
                else:
                    exclude("unresolved_receiver_call")
                    continue
            else:
                exclude("dynamic_or_nested_callable")
                continue
            assert target is not None
            target_node, target_unit = target
            relation = {
                "relation_id": f"rel_{sha256_text(f'{path}:{call.lineno}:{rule}:{target_unit.unit_id}:{target_node.lineno}')[:20]}",
                "relation": "calls", "resolution_rule": rule,
                "caller_unit_id": caller.unit_id, "callee_unit_id": target_unit.unit_id,
                "caller_symbol": caller.name,
                "callee_symbol": getattr(target_node, "name", ""),
                "caller_path": path, "callee_path": target_unit.source_path,
                "call_span": _node_line_span(call, lines),
                "callee_declaration": _declaration_line_span(
                    target_node, (root / target_unit.source_path).read_text(encoding="utf-8").splitlines()),
                "binding": binding,
            }
            accepted.append(relation)
            counts["uniquely_verified"] += 1
    counts["excluded_by_reason"] = excluded_by_reason
    return sorted(accepted, key=lambda x: x["relation_id"]), counts


def _scan_corpus_files(root: Path):
    """Apply the one authoritative corpus eligibility/normalization policy."""
    eligible, excluded = [], []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
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
        eligible.append({"path": path, "relative": relative, "raw": raw, "text": text,
                         "content_hash": sha256_bytes(raw)})
    return eligible, excluded


def canonical_corpus_snapshot(root: str | Path) -> list[dict[str, str]]:
    """Return the semantic closed-snapshot identity used everywhere.

    Only eligible UTF-8 protocol files contribute.  VCS metadata, virtual
    environments, caches, unsupported extensions, and non-UTF-8 files are
    intentionally outside the corpus identity.
    """
    root = Path(root).resolve()
    eligible, _ = _scan_corpus_files(root)
    return [{"path": item["relative"], "content_hash": item["content_hash"]}
            for item in eligible]


def corpus_snapshot_hash(root: str | Path) -> str:
    return canonical_hash(canonical_corpus_snapshot(root))


def build_manifest(root: str | Path) -> dict:
    root = Path(root).resolve()
    documents, units = [], []
    eligible, excluded = _scan_corpus_files(root)
    for item in eligible:
        path, relative, text, digest = (item["path"], item["relative"],
                                        item["text"], item["content_hash"])
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
    verified_relations, relation_audit = _resolve_python_relations(root, units)
    corpus_hash = canonical_hash(
        [{"path": d["path"], "content_hash": d["content_hash"]} for d in documents])
    return {"schema_version": 2, "root": str(root), "corpus_hash": corpus_hash,
            "documents": documents, "units": units, "excluded": excluded,
            "verified_relations": verified_relations, "relation_audit": relation_audit,
            "counts": {"documents": len(documents), "units": len(units),
                       "eligible_tokens": sum(u.eligible_tokens for u in units),
                       "symbols": sum(len(u.symbols) for u in units),
                       "syntactic_relations": relation_audit["syntactic"],
                       "verified_relations": relation_audit["uniquely_verified"]}}


def verify_manifest_against_live_corpus(manifest: dict, root: str | Path | None = None) -> dict:
    """Verify every manifest file and hash against the current filesystem.

    The live index is built with the exact same inclusion/exclusion rules as
    :func:`build_manifest`, but it is never written back.  A mismatch is a hard
    error before a model or coding tool is initialized.
    """
    expected_root = Path(manifest.get("root", "")).resolve()
    live_root = Path(root or expected_root).resolve()
    if live_root != expected_root:
        raise CorpusIntegrityError(
            f"live corpus root differs from frozen manifest: {live_root} != {expected_root}")
    if not live_root.is_dir():
        raise CorpusIntegrityError(f"frozen corpus root is missing: {live_root}")
    live = build_manifest(live_root)
    expected_files = {item["path"]: item["content_hash"] for item in manifest.get("documents", ())}
    live_files = {item["path"]: item["content_hash"] for item in live.get("documents", ())}
    modified = sorted(path for path in expected_files.keys() & live_files.keys()
                      if expected_files[path] != live_files[path])
    missing = sorted(set(expected_files) - set(live_files))
    unexpected = sorted(set(live_files) - set(expected_files))
    live_excluded = {(item.get("path"), item.get("reason"))
                     for item in live.get("excluded", ())}
    if modified or missing or unexpected or live.get("corpus_hash") != manifest.get("corpus_hash"):
        raise CorpusIntegrityError(json.dumps({
            "root": str(live_root), "modified_files": modified,
            "missing_files": missing, "unexpected_eligible_files": unexpected,
            "manifest_corpus_hash": manifest.get("corpus_hash"),
            "live_corpus_hash": live.get("corpus_hash")}, sort_keys=True))
    return {"verified": True, "root": str(live_root), "corpus_hash": live["corpus_hash"],
            "documents": len(live_files), "excluded": len(live_excluded)}
