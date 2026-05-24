"""Pydantic v2 schemas for the OpenAI-compatible HTTP surface.

We implement the subset of the OpenAI Chat Completions API that
unambiguously maps onto greedy speculative decoding:

  * ``role`` ∈ {system, user, assistant}
  * ``stream``: bool — drives SSE vs JSON response shape
  * ``max_tokens``: int — capped against ``ServerConfig.default_max_new_tokens``
  * ``model``: str — accepted for API parity but ignored at runtime
                    (the engine is bound to a specific verifier at process start)

Sampling controls (``temperature``, ``top_p``, ``presence_penalty``,
``frequency_penalty``, etc.) are accepted to keep OpenAI clients happy,
but are **not** honored by the current decoder, which is greedy
temperature-0 by design (speculative decoding's correctness proof
relies on greedy or aligned-distribution sampling). We document this
in the module docstring rather than rejecting the parameters at
runtime, to keep happy-path compatibility with off-the-shelf clients.

Streaming chunks follow the OpenAI ``ChatCompletionChunk`` format:

    data: {"id":..., "object":"chat.completion.chunk",
           "created":..., "model":..., "choices":[
               {"index":0, "delta":{"content":"text"}, "finish_reason":null}]}

terminated by:

    data: [DONE]

The terminal ``[DONE]`` line is emitted by the route handler, not by
the schema layer; this module only defines the shape of the JSON
events leading up to it.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Roles match OpenAI's documented values. ``tool`` and ``function``
    roles are not yet supported by this server (the underlying decoder
    has no tool-call mechanism); they are explicitly rejected at
    validation time so a misuse fails fast at the request boundary
    rather than producing garbled output.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("content must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """Request body for ``POST /v1/chat/completions``.

    ``model_config`` here is pydantic's, *not* the dataclass field —
    we set ``extra='ignore'`` so unknown fields from forward-compatible
    OpenAI clients (newer SDKs adding parameters we don't support yet)
    don't blow up the request.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: List[ChatMessage] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None

    @field_validator("messages")
    @classmethod
    def _last_message_is_user(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        # OpenAI itself accepts assistant-final histories (for
        # multi-turn replays), so we do too — we only reject empty
        # message lists, which the min_length=1 already covers. This
        # validator exists as a hook for future tightening if needed.
        return v


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = Field(ge=0)
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice] = Field(min_length=1)
    usage: ChatCompletionUsage


# ---------------------------------------------------------------------------
# Streaming response chunk
# ---------------------------------------------------------------------------


class ChatCompletionChunkDelta(BaseModel):
    """Per-chunk delta. ``role`` only set in the first chunk; ``content``
    only set when there is new text to emit (the final ``finish_reason``
    chunk has neither)."""

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = Field(ge=0)
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "kakeya"


class ListModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
