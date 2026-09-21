from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from transformers import TextGenerationPipeline, pipeline


class Generator(Protocol):
    def prompt(self, query: str) -> str: ...


@dataclass
class TransformersGenerator:
    model_name: str
    max_new_tokens: int

    @cached_property
    def model(self) -> TextGenerationPipeline:
        return pipeline("text-generation", model=self.model_name)

    def prompt(self, query: str) -> str:
        chat = [{"role": "user", "content": query}]
        outputs = self.model(
            chat,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
        )
        return outputs[0]["generated_text"][-1]["content"]