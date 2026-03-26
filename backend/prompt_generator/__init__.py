"""Prompt generator component package.

Exports `build_prompt()` for building a final prompt string from query + retrieval context.
"""

from .main import build_prompt, get_prompt_context_budget, set_prompt_context_budget  # noqa: F401
