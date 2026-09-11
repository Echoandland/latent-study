from __future__ import annotations

import random
import statistics
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
        self.current_ledger = ExposureLedger()
        self.previous_ledger = ExposureLedger()
        # Global cursors are intentionally independent of batch boundaries.  With a
        # two-record batch there is only one replay slot, so an index local to the
        # batch would select previous_sources[0] forever.
        self._replay_source_cursor = 0
        self._record_cursors: dict[str, int] = {}
        self._eligible_opportunities: dict[str, int] = {}

    def add_source(self, source: str, records: list[StudyRecord]) -> None:
        if source in self.bank:
            raise ValueError(f"source already studied: {source}")
        self.bank[source] = tuple(sorted(records, key=lambda r: r.record_id))

    def batch(self, current_source: str, batch_size: int, step: int) -> ReplayBatch:
        current = self.bank[current_source]
        previous_sources = sorted(s for s in self.bank if s != current_source)
        previous_n = batch_size // 2 if previous_sources and self.replay_fraction == 0.5 else 0
        if previous_n:
            for source in previous_sources:
                self._eligible_opportunities[source] = self._eligible_opportunities.get(source, 0) + previous_n
        current_n = batch_size - previous_n  # replay=0 compute-match: fill with current records
        order = list(range(len(current)))
        random.Random(f"{self.seed}:{current_source}:current-order").shuffle(order)
        cur = tuple(current[order[(step * current_n + i) % len(order)]] for i in range(current_n))
        prev = []
        for _ in range(previous_n):
            # Least-exposed scheduling is deterministic and balances the bank
            # that is actually eligible at this point in sequential training.
            minimum = min(self.previous_ledger.by_source.get(s, 0) for s in previous_sources)
            eligible = [s for s in previous_sources
                        if self.previous_ledger.by_source.get(s, 0) == minimum]
            source = eligible[self._replay_source_cursor % len(eligible)]
            self._replay_source_cursor += 1
            # Each source has a stable seeded permutation, then cycles through it.
            order = list(range(len(self.bank[source])))
            random.Random(f"{self.seed}:{source}:previous-order").shuffle(order)
            cursor = self._record_cursors.get(source, 0)
            prev.append(self.bank[source][order[cursor % len(order)]])
            self._record_cursors[source] = cursor + 1
        records = cur + tuple(prev)
        for record in cur:
            self.ledger.add(record); self.current_ledger.add(record)
        for record in prev:
            self.ledger.add(record); self.previous_ledger.add(record)
        return ReplayBatch(records, current_n, previous_n)

    def source_imbalance(self, sources: list[str] | None = None, *, kind: str = "all") -> dict[str, float | int]:
        """Report source exposure imbalance, including zero-exposure sources."""
        selected = sorted(sources if sources is not None else self.bank)
        ledgers = {"all": self.ledger, "current": self.current_ledger, "previous": self.previous_ledger}
        if kind not in ledgers: raise ValueError("kind must be all, current, or previous")
        counts = [ledgers[kind].by_source.get(source, 0) for source in selected]
        if not counts:
            return {"sources": 0, "min": 0, "max": 0, "range": 0, "mean": 0.0,
                    "population_stdev": 0.0, "max_to_min_ratio": 0.0}
        minimum, maximum = min(counts), max(counts)
        return {
            "sources": len(counts), "min": minimum, "max": maximum,
            "range": maximum - minimum, "mean": statistics.fmean(counts),
            "population_stdev": statistics.pstdev(counts),
            "max_to_min_ratio": (maximum / minimum) if minimum else (None if maximum else 0.0),
        }

    def replay_audit(self) -> dict:
        sources = sorted(self.bank)
        zero = [source for source in sources
                if self._eligible_opportunities.get(source, 0) > 0
                and self.previous_ledger.by_source.get(source, 0) == 0]
        no_opportunity = [source for source in sources if self._eligible_opportunities.get(source, 0) == 0]
        return {
            "scheduler": "deterministic_least_exposed_among_currently_eligible_sources",
            "exposures_per_source": {s: self.previous_ledger.by_source.get(s, 0) for s in sources},
            "eligible_replay_opportunities": {s: self._eligible_opportunities.get(s, 0) for s in sources},
            "zero_exposure_eligible_sources": zero,
            "no_replay_opportunity_sources": no_opportunity,
            "no_replay_opportunity_reasons": {
                source: "source was never a previous source before the sequential study ended"
                for source in no_opportunity
            },
            "no_opportunity_reason": "source was never a previous source before the sequential study ended",
            "imbalance": self.source_imbalance(sources, kind="previous"),
            "current_exposures": {s: self.current_ledger.by_source.get(s, 0) for s in sources},
            "previous_exposures": {s: self.previous_ledger.by_source.get(s, 0) for s in sources},
        }
