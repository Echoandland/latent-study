from __future__ import annotations

"""Deterministic identity for the actual model/tokenizer files being loaded."""

import hashlib
from functools import lru_cache
from pathlib import Path

from .io import canonical_hash


_MODEL_NAMES = {"config.json", "generation_config.json"}
_TOKENIZER_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "merges.txt", "vocab.json", "spiece.model",
    "sentencepiece.bpe.model", "chat_template.jinja",
}


def _is_model_file(path: Path) -> bool:
    name = path.name
    return (name in _MODEL_NAMES or name.startswith(("modeling_", "configuration_"))
            or name.endswith((".safetensors", ".bin", ".pt", ".pth"))
            or name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"})


def _is_tokenizer_file(path: Path) -> bool:
    name = path.name
    return (name in _TOKENIZER_NAMES or name.startswith(("vocab.", "tokenizer."))
            or name.endswith((".tiktoken", ".model"))
            or (name.startswith("chat_template") and name.endswith(".jinja")))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_files(root: Path, kind: str) -> list[Path]:
    predicate = _is_model_file if kind == "model" else _is_tokenizer_file
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and ".git" not in path.relative_to(root).parts
                  and predicate(path))


@lru_cache(maxsize=16)
def _fingerprint_cached(root_text: str, kind: str, stat_signature: tuple) -> tuple[str, int]:
    root = Path(root_text)
    entries = [{"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
               for path in _selected_files(root, kind)]
    if not entries:
        raise ValueError(f"local snapshot contains no recognized {kind} files: {root}")
    return canonical_hash({"schema_version": 1, "kind": kind, "files": entries}), len(entries)


def snapshot_fingerprint(root: str | Path, kind: str) -> dict:
    """Hash relevant snapshot contents; memoize only while file metadata is unchanged."""
    if kind not in {"model", "tokenizer"}:
        raise ValueError(f"unknown snapshot kind: {kind}")
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise ValueError(f"snapshot path is not a directory: {resolved}")
    files = _selected_files(resolved, kind)
    signature = tuple((path.relative_to(resolved).as_posix(), path.stat().st_size,
                       path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                      for path in files)
    digest, count = _fingerprint_cached(str(resolved), kind, signature)
    return {"sha256": digest, "file_count": count, "method": "selected_file_content_sha256_v1"}


def resolve_model_snapshot(reference: str | Path, revision: str, *,
                           local_files_only: bool = False) -> dict:
    """Resolve a local/Hugging Face reference and bind the files actually loaded."""
    candidate = Path(reference)
    if candidate.is_dir():
        root = candidate.resolve()
        resolution = "explicit_local_directory"
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to resolve a model snapshot") from exc
        root = Path(snapshot_download(repo_id=str(reference), revision=revision,
                                      local_files_only=local_files_only)).resolve()
        resolution = "huggingface_snapshot"
    model = snapshot_fingerprint(root, "model")
    tokenizer = snapshot_fingerprint(root, "tokenizer")
    return {
        "resolved_path": str(root), "resolution": resolution,
        "model_snapshot_sha256": model["sha256"],
        "tokenizer_snapshot_sha256": tokenizer["sha256"],
        "model_snapshot_files": model["file_count"],
        "tokenizer_snapshot_files": tokenizer["file_count"],
        "snapshot_method": model["method"],
    }
