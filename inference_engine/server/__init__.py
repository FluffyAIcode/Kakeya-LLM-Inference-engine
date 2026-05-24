"""HTTP serving stack for the Kakeya inference engine (E2).

Wraps the speculative decoding core in an OpenAI-compatible REST API
with Server-Sent-Events streaming. The serving layer is deliberately
thin: FastAPI handles routing, sse-starlette handles SSE framing and
disconnect detection, and an :class:`Engine` protocol cleanly separates
"how to generate" (real speculative decoder vs deterministic test
double) from "how to serve" (HTTP routes, status codes, content
negotiation).

Submodules:
    config      ServerConfig dataclass + env-var loading.
    schemas     Pydantic v2 request/response models matching OpenAI's
                /v1/chat/completions and /v1/models surfaces.
    tokenizer   Tokenizer protocol — the subset of the HF
                AutoTokenizer interface we actually rely on.
    engine      Engine protocol + SpeculativeEngine concrete impl.
    streaming   Sync-to-async bridge that converts the speculative
                decoder's blocking on_token callback into an async
                stream of text deltas suitable for SSE.
    app         FastAPI app factory and route handlers.

This package is platform-neutral: it imports neither MLX nor any
backend-specific library. Real backends are plugged in by the caller
(scripts/serve.py) which constructs the underlying speculative
decoder from the user's chosen verifier/proposer pair.
"""

from .config import ServerConfig
from .engine import Engine, EngineResult, SpeculativeEngine
from .tokenizer import Tokenizer

__all__ = [
    "ServerConfig",
    "Engine",
    "EngineResult",
    "SpeculativeEngine",
    "Tokenizer",
]
