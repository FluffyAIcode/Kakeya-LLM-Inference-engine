"""Tokenizer protocol — the subset of HF AutoTokenizer we depend on.

This module defines a structural interface (``Protocol``) that any
tokenizer plugged into the server must satisfy. Production code
constructs a real HuggingFace ``AutoTokenizer`` (already in scope via
the verifier loader); test code constructs a deterministic test class.
Either satisfies the protocol structurally, so the server code does
not branch on which one it has.

Why a protocol rather than the HF type directly? Two reasons:

  1. The server module is platform-neutral; it does not import
     ``transformers`` at top level. Depending on the protocol means
     the import-graph stays clean even if a future deployment uses a
     non-HF tokenizer (e.g. tiktoken for GPT-style models, or sentencepiece
     directly).
  2. The protocol documents *exactly* which methods we use. Any HF
     API surface beyond these four methods is irrelevant to the
     server, and changes to it cannot break the server contract.

The four methods:
    apply_chat_template — encode a list of OpenAI-style chat messages
                          into a flat list of token ids, applying the
                          tokenizer's chat template (system prompt,
                          role markers, generation prompt sentinel).
    decode               — turn a list of token ids back into a string.
    convert_tokens_to_ids — used to resolve ``<|im_end|>`` for EOS
                          detection in Qwen3 family models.
    eos_token_id          — read-only attribute; integer or None.

We rely on the HF semantics: ``apply_chat_template(..., tokenize=True,
return_dict=False)`` returns ``list[int]`` directly (in transformers
4.x). The ``return_dict=False`` argument is set explicitly because
transformers 5.x changes that default; pinning it here keeps the
protocol stable across version drift.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Subset of HF AutoTokenizer that the server consumes."""

    eos_token_id: Optional[int]
    unk_token_id: Optional[int]

    def apply_chat_template(
        self,
        messages: List[dict],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
        enable_thinking: bool = False,
    ) -> Any:
        """Encode chat messages to token ids.

        With ``tokenize=True, return_dict=False`` this must return a
        plain ``list[int]``. The ``enable_thinking`` flag is
        Qwen3-specific and toggles the chain-of-thought template
        prefix; we default it to ``False`` because thinking-mode
        output bloats the response and hurts perceived latency.
        """
        ...  # pragma: no cover - Protocol body

    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = False) -> str:
        ...  # pragma: no cover - Protocol body

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        ...  # pragma: no cover - Protocol body


def resolve_eos_ids(tokenizer: Tokenizer) -> List[int]:
    """Return the set of token ids that should terminate generation.

    Combines the tokenizer's canonical ``eos_token_id`` with Qwen3's
    ``<|im_end|>`` sentinel (which is the actual end-of-turn marker
    in chat-template output, distinct from the model's vocabulary
    EOS). De-duplicated; result is a list (not set) so the order is
    deterministic for downstream loggers.

    Returns an empty list only if the tokenizer reports no EOS at all,
    which is a real misconfiguration we want to surface — the engine
    refuses to start in that state, see :class:`SpeculativeEngine`.
    """
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    out: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
