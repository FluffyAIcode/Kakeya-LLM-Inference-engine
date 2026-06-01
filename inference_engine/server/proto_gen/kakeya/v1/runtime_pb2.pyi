from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateSessionRequest(_message.Message):
    __slots__ = ("client_label", "eos_token_ids")
    CLIENT_LABEL_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    client_label: str
    eos_token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, client_label: _Optional[str] = ..., eos_token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CreateSessionResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class AppendTokensRequest(_message.Message):
    __slots__ = ("session_id", "token_ids")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, session_id: _Optional[str] = ..., token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class AppendTokensResponse(_message.Message):
    __slots__ = ("history_length",)
    HISTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    history_length: int
    def __init__(self, history_length: _Optional[int] = ...) -> None: ...

class CloseSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseSessionResponse(_message.Message):
    __slots__ = ("final_history_length",)
    FINAL_HISTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    final_history_length: int
    def __init__(self, final_history_length: _Optional[int] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("session_id", "max_tokens", "seed", "temperature", "top_p", "top_k")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    max_tokens: int
    seed: int
    temperature: float
    top_p: float
    top_k: int
    def __init__(self, session_id: _Optional[str] = ..., max_tokens: _Optional[int] = ..., seed: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("token_id", "done", "truncated")
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    done: GenerateDone
    truncated: HistoryTruncated
    def __init__(self, token_id: _Optional[int] = ..., done: _Optional[_Union[GenerateDone, _Mapping]] = ..., truncated: _Optional[_Union[HistoryTruncated, _Mapping]] = ...) -> None: ...

class GenerateDone(_message.Message):
    __slots__ = ("stop_reason", "generated_token_count", "prefill_duration_seconds", "total_duration_seconds")
    class StopReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STOP_REASON_UNSPECIFIED: _ClassVar[GenerateDone.StopReason]
        STOP_REASON_MAX_TOKENS: _ClassVar[GenerateDone.StopReason]
        STOP_REASON_EOS: _ClassVar[GenerateDone.StopReason]
        STOP_REASON_CANCELLED: _ClassVar[GenerateDone.StopReason]
        STOP_REASON_TRUNCATED: _ClassVar[GenerateDone.StopReason]
    STOP_REASON_UNSPECIFIED: GenerateDone.StopReason
    STOP_REASON_MAX_TOKENS: GenerateDone.StopReason
    STOP_REASON_EOS: GenerateDone.StopReason
    STOP_REASON_CANCELLED: GenerateDone.StopReason
    STOP_REASON_TRUNCATED: GenerateDone.StopReason
    STOP_REASON_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    PREFILL_DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    stop_reason: GenerateDone.StopReason
    generated_token_count: int
    prefill_duration_seconds: float
    total_duration_seconds: float
    def __init__(self, stop_reason: _Optional[_Union[GenerateDone.StopReason, str]] = ..., generated_token_count: _Optional[int] = ..., prefill_duration_seconds: _Optional[float] = ..., total_duration_seconds: _Optional[float] = ...) -> None: ...

class HistoryTruncated(_message.Message):
    __slots__ = ("dropped_token_count",)
    DROPPED_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    dropped_token_count: int
    def __init__(self, dropped_token_count: _Optional[int] = ...) -> None: ...

class GetSessionInfoRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetSessionInfoResponse(_message.Message):
    __slots__ = ("history_length", "kv_live_bytes", "cache_invariant_inv1_violations", "cache_invariant_inv2_violations", "idle_seconds")
    HISTORY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    KV_LIVE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CACHE_INVARIANT_INV1_VIOLATIONS_FIELD_NUMBER: _ClassVar[int]
    CACHE_INVARIANT_INV2_VIOLATIONS_FIELD_NUMBER: _ClassVar[int]
    IDLE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    history_length: int
    kv_live_bytes: int
    cache_invariant_inv1_violations: int
    cache_invariant_inv2_violations: int
    idle_seconds: float
    def __init__(self, history_length: _Optional[int] = ..., kv_live_bytes: _Optional[int] = ..., cache_invariant_inv1_violations: _Optional[int] = ..., cache_invariant_inv2_violations: _Optional[int] = ..., idle_seconds: _Optional[float] = ...) -> None: ...
