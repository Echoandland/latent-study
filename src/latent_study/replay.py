from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import ExposureLedger, StudyRecord


@dataclass(frozen=True)
class ReplayBatch:
    records: tuple[StudyRecord, ...]
    current_count: int
    previous_count: int


class SourceReplay:
    def __init__(self, seed: int, replay_fraction: float = 0.5):
        if replay_fraction not in (0.0, 0.5):
            raise ValueError("MVP replay_fraction must be 0.0 or 0.5")
        self.seed = seed
        self.replay_fraction = replay_fraction
        self.bank: dict[str, tuple[StudyRecord, ...]] = {}
        self.ledger = ExposureLedger()

    def add_source(self, source: str, records: list[StudyRecord]) -> None:
        if source in self.bank:
            raise ValueError(f"source already studied: {source}")
        self.bank[source] = tuple(sorted(records, key=lambda r: r.record_id))

    def batch(self, current_source: str, batch_size: int, step: int) -> ReplayBatch:
        current = self.bank[current_source]
        previous_sources = sorted(s for s in self.bank if s != current_source)
        rng = random.Random(f"{self.seed}:{current_source}:{step}")
        previous_n = batch_size // 2 if previous_sources and self.replay_fraction == 0.5 else 0
        current_n = batch_size - previous_n  # replay=0 compute-match: fill with current records
        order = list(range(len(current)))
        random.Random(f"{self.seed}:{current_source}:current-order").shuffle(order)
        cur = tuple(current[order[(step * current_n + i) % len(order)]] for i in range(current_n))
        prev = []
        for i in range(previous_n):
            source = previous_sources[i % len(previous_sources)]
            prev.append(self.bank[source][rng.randrange(len(self.bank[source]))])
        records = cur + tuple(prev)
        for record in records:
            self.ledger.add(record)
        return ReplayBatch(records, current_n, previous_n)
