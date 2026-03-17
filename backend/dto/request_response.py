"""
API request and response DTOs for the query/evaluate pipeline.

Defines the validated shape of the POST body and the JSON response,
plus optional context used for tracing (vector snippet, kg triples count).
"""

from typing import Literal

from pydantic import BaseModel, Field

TestMode = Literal["vectordb", "kg", "hybrid", "none"]


class QueryRequest(BaseModel):
    """
    Request body for the query/evaluate endpoint.

    Attributes:
        query: Natural language user query (e.g. Karnataka tourism question).
        test_mode: Routing mode: vectordb, kg, hybrid, or none.
    """

    query: str = Field(..., min_length=1, description="User query")
    test_mode: TestMode = Field(
        ...,
        description="Routing mode: vectordb | kg | hybrid | none",
    )


class ContextUsed(BaseModel):
    """
    Optional context passed to the model for tracing and debugging.

    Attributes:
        vector_snippet: Truncated text from the top vector result, if any.
        kg_triples_count: Number of KG triples included, if any.
        kg_snippet: Text representation of the top KG triple, if any.
    """

    vector_snippet: str | None = Field(default=None, description="Top vector result text snippet")
    kg_triples_count: int | None = Field(default=None, description="Number of KG triples used")
    kg_snippet: str | None = Field(default=None, description="Top KG triple text representation")


class QueryResponse(BaseModel):
    """
    Response body for the query/evaluate endpoint.

    Attributes:
        answer: Model-generated answer.
        test_mode: Echo of the test_mode used for this request.
        final_prompt: The final prompt that was sent to the LLM.
        context_used: Optional details about retrieval context (for tracing).
    """

    answer: str = Field(..., description="Model answer")
    test_mode: str = Field(..., description="Echo of test_mode")
    final_prompt: str = Field(..., description="Final prompt sent to the LLM")
    context_used: ContextUsed | None = Field(default=None, description="Retrieval context summary")
