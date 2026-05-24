"""Unit tests for :mod:`inference_engine.server.schemas`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_engine.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
    ChatMessage,
    HealthResponse,
    ListModelsResponse,
    ModelInfo,
)


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_chat_message_accepts_valid_roles(role):
    m = ChatMessage(role=role, content="hi")
    assert m.role == role


@pytest.mark.parametrize("role", ["tool", "function", "agent", ""])
def test_chat_message_rejects_invalid_roles(role):
    with pytest.raises(ValidationError):
        ChatMessage(role=role, content="hi")


def test_chat_message_rejects_empty_content():
    with pytest.raises(ValidationError, match="non-empty string"):
        ChatMessage(role="user", content="")


def test_chat_message_accepts_whitespace_content():
    # We explicitly only reject the empty string (length 0); a single
    # space is content. Document that contract.
    m = ChatMessage(role="user", content=" ")
    assert m.content == " "


# ---------------------------------------------------------------------------
# ChatCompletionRequest
# ---------------------------------------------------------------------------


def test_request_minimal_construction():
    req = ChatCompletionRequest(
        model="kakeya-v1",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.stream is False
    assert req.max_tokens is None
    assert req.temperature is None


def test_request_with_max_tokens_and_stream():
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=42,
        stream=True,
    )
    assert req.max_tokens == 42
    assert req.stream is True


def test_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])


def test_request_rejects_negative_max_tokens():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=0,
        )


@pytest.mark.parametrize("temp", [-0.1, 2.1])
def test_request_rejects_out_of_range_temperature(temp):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=temp,
        )


@pytest.mark.parametrize("top_p", [-0.1, 1.1])
def test_request_rejects_out_of_range_top_p(top_p):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            top_p=top_p,
        )


def test_request_ignores_unknown_fields():
    req = ChatCompletionRequest.model_validate({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "presence_penalty": 0.5,
        "logit_bias": {"123": 1.0},
        "future_field_42": "anything",
    })
    assert req.model == "m"


def test_request_accepts_assistant_final_history():
    """Multi-turn replays often end with assistant — should be allowed."""
    ChatCompletionRequest(
        model="m",
        messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ],
    )


def test_request_validator_returns_messages_unchanged():
    """The _last_message_is_user validator passes through messages.

    Listed explicitly so the validator's body is exercised.
    """
    msgs = [ChatMessage(role="user", content="x")]
    req = ChatCompletionRequest(model="m", messages=msgs)
    assert len(req.messages) == 1


# ---------------------------------------------------------------------------
# Non-streaming response shapes
# ---------------------------------------------------------------------------


def test_response_construction():
    resp = ChatCompletionResponse(
        id="x", created=1, model="m",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].finish_reason == "stop"


def test_response_rejects_empty_choices():
    with pytest.raises(ValidationError):
        ChatCompletionResponse(
            id="x", created=1, model="m",
            choices=[],
            usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


def test_choice_rejects_invalid_finish_reason():
    with pytest.raises(ValidationError):
        ChatCompletionChoice(
            index=0,
            message=ChatCompletionResponseMessage(role="assistant", content="hi"),
            finish_reason="weird",
        )


def test_response_message_must_be_assistant_role():
    with pytest.raises(ValidationError):
        ChatCompletionResponseMessage(role="user", content="hi")


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_usage_rejects_negative(field):
    kwargs = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    kwargs[field] = -1
    with pytest.raises(ValidationError):
        ChatCompletionUsage(**kwargs)


# ---------------------------------------------------------------------------
# Streaming chunk shapes
# ---------------------------------------------------------------------------


def test_chunk_construction():
    chunk = ChatCompletionChunk(
        id="x", created=1, model="m",
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    assert chunk.object == "chat.completion.chunk"


def test_chunk_delta_can_have_content_only():
    d = ChatCompletionChunkDelta(content="hi")
    assert d.role is None
    assert d.content == "hi"


def test_chunk_delta_can_be_empty():
    """Final chunk's delta carries neither role nor content."""
    d = ChatCompletionChunkDelta()
    assert d.role is None
    assert d.content is None


def test_chunk_choice_finish_reason_optional():
    c = ChatCompletionChunkChoice(
        index=0, delta=ChatCompletionChunkDelta(), finish_reason="length"
    )
    assert c.finish_reason == "length"


# ---------------------------------------------------------------------------
# Models / health
# ---------------------------------------------------------------------------


def test_list_models_response():
    r = ListModelsResponse(data=[ModelInfo(id="m", created=1)])
    assert r.object == "list"
    assert r.data[0].owned_by == "kakeya"


def test_model_info_default_owned_by():
    m = ModelInfo(id="m", created=1)
    assert m.owned_by == "kakeya"
    assert m.object == "model"


def test_health_response():
    h = HealthResponse(status="ok", model="kakeya-v1")
    assert h.status == "ok"


def test_health_response_rejects_other_status():
    with pytest.raises(ValidationError):
        HealthResponse(status="degraded", model="m")
