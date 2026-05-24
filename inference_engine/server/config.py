"""Server configuration.

A single immutable :class:`ServerConfig` instance is constructed at
process start (typically by ``scripts/serve.py``) and threaded through
the FastAPI app via the application state. We do not re-read env vars
at request time; configuration is fixed for the process lifetime, which
matches uvicorn's worker model and avoids inconsistent state across
parallel requests.

Defaults are chosen so a bare ``./scripts/serve.py`` invocation works
on a developer's laptop without any flags or env vars:

    host                ``127.0.0.1``  (loopback only — flip to ``0.0.0.0``
                                       explicitly when exposing the engine)
    port                ``8000``       (no privileged port assumption)
    default_max_new_tokens   ``1024``  (matches scripts/chat.py default)
    request_timeout_s   ``120.0``      (chat completions cap; SSE keeps
                                       running until EOS / max_tokens
                                       even if the wall clock exceeds
                                       this — the limit is per-route
                                       request handler timeout, not
                                       per-token)
    model_id_label      ``"kakeya-v1"`` — string returned by /v1/models;
                                          surfaces as the ``model``
                                          field in completion responses.
                                          Production deployments
                                          override this to expose
                                          something meaningful to
                                          OpenAI-compatible clients.
    max_concurrent      ``1``          — number of concurrent inference
                                          sessions the scheduler admits.
                                          1 = single-user (default,
                                          matches the v0.1.0 behavior
                                          before the integration).
                                          Bump on multi-user deployments.
    admission_policy    ``"reject"``   — REJECT returns HTTP 429 when
                                          the pool is full. QUEUE makes
                                          the request wait up to
                                          ``queue_max_wait_s``.
    queue_max_wait_s    ``0.0``        — only honored under QUEUE policy;
                                          0 means wait forever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from inference_engine.scheduler.config import AdmissionPolicy


@dataclass(frozen=True)
class ServerConfig:
    """Process-wide HTTP server configuration.

    Frozen so it can be safely shared across worker threads / async
    tasks without ownership confusion. All fields are public.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    default_max_new_tokens: int = 1024
    request_timeout_s: float = 120.0
    model_id_label: str = "kakeya-v1"
    log_level: str = "info"
    # Scheduler tuning (see inference_engine.scheduler.config).
    max_concurrent: int = 1
    admission_policy: AdmissionPolicy = AdmissionPolicy.REJECT
    queue_max_wait_s: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.default_max_new_tokens <= 0:
            raise ValueError(
                "default_max_new_tokens must be positive, got "
                f"{self.default_max_new_tokens}"
            )
        if self.request_timeout_s <= 0:
            raise ValueError(
                "request_timeout_s must be positive, got "
                f"{self.request_timeout_s}"
            )
        if not self.model_id_label.strip():
            raise ValueError("model_id_label must be non-empty")
        if self.log_level not in {"trace", "debug", "info", "warning", "error", "critical"}:
            raise ValueError(
                f"log_level must be one of trace/debug/info/warning/error/critical, "
                f"got {self.log_level!r}"
            )
        if self.max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be positive, got {self.max_concurrent}"
            )
        if self.queue_max_wait_s < 0:
            raise ValueError(
                f"queue_max_wait_s must be >= 0, got {self.queue_max_wait_s}"
            )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ServerConfig":
        """Build a config from environment variables.

        Variable names are prefixed with ``KAKEYA_`` to avoid colliding
        with anything else the host might set. Unrecognized variables
        are ignored — we do not enforce an env-var allowlist because
        the user's shell is full of unrelated noise.

        Recognized variables:
            KAKEYA_HOST                  -> host
            KAKEYA_PORT                  -> port (int)
            KAKEYA_MAX_NEW_TOKENS        -> default_max_new_tokens (int)
            KAKEYA_REQUEST_TIMEOUT_S     -> request_timeout_s (float)
            KAKEYA_MODEL_ID_LABEL        -> model_id_label
            KAKEYA_LOG_LEVEL             -> log_level
            KAKEYA_MAX_CONCURRENT        -> max_concurrent (int)
            KAKEYA_ADMISSION_POLICY      -> admission_policy ("reject" or "queue")
            KAKEYA_QUEUE_MAX_WAIT_S      -> queue_max_wait_s (float)

        Defaults match the dataclass field defaults above.

        Parameters
        ----------
        env:
            Optional dict to read from. Defaults to ``os.environ``,
            which is what production callers want; tests pass an
            explicit dict so they don't pollute the process env.
        """
        source = os.environ if env is None else env
        kwargs: dict[str, object] = {}
        if "KAKEYA_HOST" in source:
            kwargs["host"] = source["KAKEYA_HOST"]
        if "KAKEYA_PORT" in source:
            kwargs["port"] = _parse_int(source["KAKEYA_PORT"], "KAKEYA_PORT")
        if "KAKEYA_MAX_NEW_TOKENS" in source:
            kwargs["default_max_new_tokens"] = _parse_int(
                source["KAKEYA_MAX_NEW_TOKENS"], "KAKEYA_MAX_NEW_TOKENS"
            )
        if "KAKEYA_REQUEST_TIMEOUT_S" in source:
            kwargs["request_timeout_s"] = _parse_float(
                source["KAKEYA_REQUEST_TIMEOUT_S"], "KAKEYA_REQUEST_TIMEOUT_S"
            )
        if "KAKEYA_MODEL_ID_LABEL" in source:
            kwargs["model_id_label"] = source["KAKEYA_MODEL_ID_LABEL"]
        if "KAKEYA_LOG_LEVEL" in source:
            kwargs["log_level"] = source["KAKEYA_LOG_LEVEL"]
        if "KAKEYA_MAX_CONCURRENT" in source:
            kwargs["max_concurrent"] = _parse_int(
                source["KAKEYA_MAX_CONCURRENT"], "KAKEYA_MAX_CONCURRENT"
            )
        if "KAKEYA_ADMISSION_POLICY" in source:
            raw = source["KAKEYA_ADMISSION_POLICY"].strip().lower()
            try:
                kwargs["admission_policy"] = AdmissionPolicy(raw)
            except ValueError as exc:
                raise ValueError(
                    f"environment variable KAKEYA_ADMISSION_POLICY={raw!r} "
                    "is not a valid AdmissionPolicy (expected 'reject' or 'queue')"
                ) from exc
        if "KAKEYA_QUEUE_MAX_WAIT_S" in source:
            kwargs["queue_max_wait_s"] = _parse_float(
                source["KAKEYA_QUEUE_MAX_WAIT_S"], "KAKEYA_QUEUE_MAX_WAIT_S"
            )
        return cls(**kwargs)


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"environment variable {name}={raw!r} is not an integer"
        ) from exc


def _parse_float(raw: str, name: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"environment variable {name}={raw!r} is not a float"
        ) from exc
