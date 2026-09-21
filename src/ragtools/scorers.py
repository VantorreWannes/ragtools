from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from sentence_transformers import CrossEncoder


class Scorer(Protocol):
    def score(self, query: str, chunk: str) -> float: ...


@dataclass
class CrossEncoderScorer:
    model_name: str

    @cached_property
    def model(self) -> CrossEncoder:
        return CrossEncoder(self.model_name)

    def score(self, query: str, chunk: str) -> float:
        return float(self.model.predict((query, chunk)).tolist())
