from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol


class Fuser[V](Protocol):
    def fuse(self, *values: Sequence[V]) -> list[V]: ...


@dataclass(slots=True)
class ReciprocalRankFuser[V: Hashable]:
    k: int = 60

    def fuse(self, *values: Sequence[V]) -> list[V]:
        scores: dict[V, float] = {}
        for ranking in values:
            for rank, value in enumerate(ranking, start=1):
                weight = 1 / (self.k + rank)
                scores[value] = scores.get(value, 0.0) + weight
        return sorted(scores, key=lambda value: scores[value], reverse=True)


@dataclass(slots=True)
class BordaCountFuser[V: Hashable]:
    def fuse(self, *values: Sequence[V]) -> list[V]:
        scores: dict[V, float] = {}
        for ranking in values:
            for rank, value in enumerate(ranking, start=1):
                weight = len(ranking) - rank + 1
                scores[value] = scores.get(value, 0.0) + weight
        return sorted(scores, key=lambda value: scores[value], reverse=True)
