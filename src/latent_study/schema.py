from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class EvidenceSpan:
    chunk_id: str
    start_line: int
    end_line: int
    text_hash: str


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    alternatives: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class CorpusUnit:
    unit_id: str
    document_id: str
    source_path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    content_hash: str
    source_hash: str
    eligible_tokens: int
    symbols: tuple[str, ...] = ()
    definitions: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    text: str = ""


@dataclass(frozen=True)
class ToolAction:
    query: str
    max_results: int = 5


@dataclass(frozen=True)
class ToolHit:
    chunk_id: str
    source_path: str
    start_line: int
    end_line: int
    text: str
    rank: int
    returned_bytes: int


@dataclass(frozen=True)
class ActionOutcome:
    action: ToolAction
    valid: bool
    hits: tuple[ToolHit, ...]
    visible_group_ids: tuple[str, ...]
    reward: float
    reward_components: dict[str, float]
    observation: str
    returned_bytes: int


@dataclass(frozen=True)
class StudyRecord:
    record_id: str
    family: Literal[
        "comprehension", "relation_induction", "navigation",
        "evidence_interpretation", "misconception_correction"
    ]
    source_id: str
    prompt: str
    observation: str
    evidence_groups: tuple[EvidenceGroup, ...]
    positive_chunk_ids: tuple[str, ...]
    verified_negative_chunk_ids: tuple[str, ...]
    candidate_actions: tuple[ToolAction, ...]
    outcomes: tuple[ActionOutcome, ...]
    validation: dict[str, Any]
    corpus_hash: str
    source_hash: str
    template_id: str
    random_seed: int
    weak_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExposureLedger:
    by_record: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)

    def add(self, record: StudyRecord) -> None:
        self.by_record[record.record_id] = self.by_record.get(record.record_id, 0) + 1
        self.by_source[record.source_id] = self.by_source.get(record.source_id, 0) + 1
