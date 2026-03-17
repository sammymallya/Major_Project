"""
Typed inputs/outputs for the Test LLM component.
"""

from typing import Optional


class LLMResponse:
    """
    Response from the Test LLM.

    Attributes:
        answer: Generated text.
        prompt_length: Length of input prompt.
    """

    def __init__(self, answer: str, prompt_length: int):
        self.answer = answer
        self.prompt_length = prompt_length