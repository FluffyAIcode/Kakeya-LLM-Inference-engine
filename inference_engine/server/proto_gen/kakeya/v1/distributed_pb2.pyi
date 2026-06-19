from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CapabilityRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAPABILITY_ROLE_UNSPECIFIED: _ClassVar[CapabilityRole]
    CAPABILITY_ROLE_VERIFIER: _ClassVar[CapabilityRole]
    CAPABILITY_ROLE_PROPOSER: _ClassVar[CapabilityRole]
    CAPABILITY_ROLE_EMBEDDER: _ClassVar[CapabilityRole]
    CAPABILITY_ROLE_TOOL: _ClassVar[CapabilityRole]
CAPABILITY_ROLE_UNSPECIFIED: CapabilityRole
CAPABILITY_ROLE_VERIFIER: CapabilityRole
CAPABILITY_ROLE_PROPOSER: CapabilityRole
CAPABILITY_ROLE_EMBEDDER: CapabilityRole
CAPABILITY_ROLE_TOOL: CapabilityRole

class ModelCapability(_message.Message):
    __slots__ = ("model_id", "role", "quantization", "tokens_per_second")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    TOKENS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    role: CapabilityRole
    quantization: str
    tokens_per_second: float
    def __init__(self, model_id: _Optional[str] = ..., role: _Optional[_Union[CapabilityRole, str]] = ..., quantization: _Optional[str] = ..., tokens_per_second: _Optional[float] = ...) -> None: ...

class NodeCapability(_message.Message):
    __slots__ = ("node_id", "grpc_address", "platform", "unified_memory_bytes", "mlx_version", "models", "announced_at_unix", "ttl_seconds", "ring_address")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    GRPC_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    UNIFIED_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MLX_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    ANNOUNCED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    RING_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    grpc_address: str
    platform: str
    unified_memory_bytes: int
    mlx_version: str
    models: _containers.RepeatedCompositeFieldContainer[ModelCapability]
    announced_at_unix: float
    ttl_seconds: float
    ring_address: str
    def __init__(self, node_id: _Optional[str] = ..., grpc_address: _Optional[str] = ..., platform: _Optional[str] = ..., unified_memory_bytes: _Optional[int] = ..., mlx_version: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelCapability, _Mapping]]] = ..., announced_at_unix: _Optional[float] = ..., ttl_seconds: _Optional[float] = ..., ring_address: _Optional[str] = ...) -> None: ...

class ExchangeCapabilitiesRequest(_message.Message):
    __slots__ = ("known_nodes",)
    KNOWN_NODES_FIELD_NUMBER: _ClassVar[int]
    known_nodes: _containers.RepeatedCompositeFieldContainer[NodeCapability]
    def __init__(self, known_nodes: _Optional[_Iterable[_Union[NodeCapability, _Mapping]]] = ...) -> None: ...

class ExchangeCapabilitiesResponse(_message.Message):
    __slots__ = ("known_nodes",)
    KNOWN_NODES_FIELD_NUMBER: _ClassVar[int]
    known_nodes: _containers.RepeatedCompositeFieldContainer[NodeCapability]
    def __init__(self, known_nodes: _Optional[_Iterable[_Union[NodeCapability, _Mapping]]] = ...) -> None: ...

class GetNodeCapabilityRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetNodeCapabilityResponse(_message.Message):
    __slots__ = ("node",)
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: NodeCapability
    def __init__(self, node: _Optional[_Union[NodeCapability, _Mapping]] = ...) -> None: ...

class ProposeBlockRequest(_message.Message):
    __slots__ = ("committed_token_ids", "block_size", "num_steps", "model_id")
    COMMITTED_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    committed_token_ids: _containers.RepeatedScalarFieldContainer[int]
    block_size: int
    num_steps: int
    model_id: str
    def __init__(self, committed_token_ids: _Optional[_Iterable[int]] = ..., block_size: _Optional[int] = ..., num_steps: _Optional[int] = ..., model_id: _Optional[str] = ...) -> None: ...

class ProposeBlockResponse(_message.Message):
    __slots__ = ("token_ids", "diffusion_steps", "forward_passes", "peak_activation_bytes")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    DIFFUSION_STEPS_FIELD_NUMBER: _ClassVar[int]
    FORWARD_PASSES_FIELD_NUMBER: _ClassVar[int]
    PEAK_ACTIVATION_BYTES_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    diffusion_steps: int
    forward_passes: int
    peak_activation_bytes: int
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., diffusion_steps: _Optional[int] = ..., forward_passes: _Optional[int] = ..., peak_activation_bytes: _Optional[int] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("dtype", "shape", "data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...

class LayerKV(_message.Message):
    __slots__ = ("layer", "k", "v")
    LAYER_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    V_FIELD_NUMBER: _ClassVar[int]
    layer: int
    k: Tensor
    v: Tensor
    def __init__(self, layer: _Optional[int] = ..., k: _Optional[_Union[Tensor, _Mapping]] = ..., v: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class RestoreRequest(_message.Message):
    __slots__ = ("session_id", "prompt_ids", "sink", "window", "s5_exact_full_attn", "model_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_IDS_FIELD_NUMBER: _ClassVar[int]
    SINK_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    S5_EXACT_FULL_ATTN_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    prompt_ids: _containers.RepeatedScalarFieldContainer[int]
    sink: int
    window: int
    s5_exact_full_attn: bool
    model_id: str
    def __init__(self, session_id: _Optional[str] = ..., prompt_ids: _Optional[_Iterable[int]] = ..., sink: _Optional[int] = ..., window: _Optional[int] = ..., s5_exact_full_attn: _Optional[bool] = ..., model_id: _Optional[str] = ...) -> None: ...

class RestoreResponse(_message.Message):
    __slots__ = ("restored", "evicted_positions", "prompt_len")
    RESTORED_FIELD_NUMBER: _ClassVar[int]
    EVICTED_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_LEN_FIELD_NUMBER: _ClassVar[int]
    restored: _containers.RepeatedCompositeFieldContainer[LayerKV]
    evicted_positions: _containers.RepeatedScalarFieldContainer[int]
    prompt_len: int
    def __init__(self, restored: _Optional[_Iterable[_Union[LayerKV, _Mapping]]] = ..., evicted_positions: _Optional[_Iterable[int]] = ..., prompt_len: _Optional[int] = ...) -> None: ...

class SeedContextRequest(_message.Message):
    __slots__ = ("session_id", "aux", "positions")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUX_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    aux: _containers.RepeatedCompositeFieldContainer[Tensor]
    positions: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, session_id: _Optional[str] = ..., aux: _Optional[_Iterable[_Union[Tensor, _Mapping]]] = ..., positions: _Optional[_Iterable[int]] = ...) -> None: ...

class SeedContextResponse(_message.Message):
    __slots__ = ("context_len",)
    CONTEXT_LEN_FIELD_NUMBER: _ClassVar[int]
    context_len: int
    def __init__(self, context_len: _Optional[int] = ...) -> None: ...

class DraftBlockRequest(_message.Message):
    __slots__ = ("session_id", "bonus_token_id", "context_len", "block_size")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    BONUS_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_LEN_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    bonus_token_id: int
    context_len: int
    block_size: int
    def __init__(self, session_id: _Optional[str] = ..., bonus_token_id: _Optional[int] = ..., context_len: _Optional[int] = ..., block_size: _Optional[int] = ...) -> None: ...

class DraftBlockResponse(_message.Message):
    __slots__ = ("draft_token_ids", "forward_passes", "peak_activation_bytes")
    DRAFT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    FORWARD_PASSES_FIELD_NUMBER: _ClassVar[int]
    PEAK_ACTIVATION_BYTES_FIELD_NUMBER: _ClassVar[int]
    draft_token_ids: _containers.RepeatedScalarFieldContainer[int]
    forward_passes: int
    peak_activation_bytes: int
    def __init__(self, draft_token_ids: _Optional[_Iterable[int]] = ..., forward_passes: _Optional[int] = ..., peak_activation_bytes: _Optional[int] = ...) -> None: ...

class ExtendContextRequest(_message.Message):
    __slots__ = ("session_id", "aux", "positions")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUX_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    aux: _containers.RepeatedCompositeFieldContainer[Tensor]
    positions: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, session_id: _Optional[str] = ..., aux: _Optional[_Iterable[_Union[Tensor, _Mapping]]] = ..., positions: _Optional[_Iterable[int]] = ...) -> None: ...

class ExtendContextResponse(_message.Message):
    __slots__ = ("context_len",)
    CONTEXT_LEN_FIELD_NUMBER: _ClassVar[int]
    context_len: int
    def __init__(self, context_len: _Optional[int] = ...) -> None: ...

class CloseDFlashSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseDFlashSessionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
