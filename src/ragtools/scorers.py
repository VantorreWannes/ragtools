from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from sentence_transformers import CrossEncoder


class Scorer(Protocol):
    def score(self, query: str, chunk: str) -> float: ...


@dataclass(slots=True)
class CrossEncoderScorer:
    model_name: str

    @cached_property
    def model(self) -> CrossEncoder:
        return CrossEncoder(self.model_name)

    def score(self, query: str, chunk: str) -> float:
        return self.model.predict((query, chunk)).tolist()
