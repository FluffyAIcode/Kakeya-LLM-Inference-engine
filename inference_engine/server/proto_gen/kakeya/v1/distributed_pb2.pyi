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
    CAPABILITY_ROLE_PREFILL_CACHE: _ClassVar[CapabilityRole]
CAPABILITY_ROLE_UNSPECIFIED: CapabilityRole
CAPABILITY_ROLE_VERIFIER: CapabilityRole
CAPABILITY_ROLE_PROPOSER: CapabilityRole
CAPABILITY_ROLE_EMBEDDER: CapabilityRole
CAPABILITY_ROLE_TOOL: CapabilityRole
CAPABILITY_ROLE_PREFILL_CACHE: CapabilityRole

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
    __slots__ = ("node_id", "grpc_address", "platform", "unified_memory_bytes", "mlx_version", "models", "announced_at_unix", "ttl_seconds", "ring_address", "caches", "endpoints")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    GRPC_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    UNIFIED_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MLX_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    ANNOUNCED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    RING_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    CACHES_FIELD_NUMBER: _ClassVar[int]
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    grpc_address: str
    platform: str
    unified_memory_bytes: int
    mlx_version: str
    models: _containers.RepeatedCompositeFieldContainer[ModelCapability]
    announced_at_unix: float
    ttl_seconds: float
    ring_address: str
    caches: _containers.RepeatedCompositeFieldContainer[CacheCapability]
    endpoints: _containers.RepeatedCompositeFieldContainer[NodeEndpoint]
    def __init__(self, node_id: _Optional[str] = ..., grpc_address: _Optional[str] = ..., platform: _Optional[str] = ..., unified_memory_bytes: _Optional[int] = ..., mlx_version: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelCapability, _Mapping]]] = ..., announced_at_unix: _Optional[float] = ..., ttl_seconds: _Optional[float] = ..., ring_address: _Optional[str] = ..., caches: _Optional[_Iterable[_Union[CacheCapability, _Mapping]]] = ..., endpoints: _Optional[_Iterable[_Union[NodeEndpoint, _Mapping]]] = ...) -> None: ...

class NodeEndpoint(_message.Message):
    __slots__ = ("address", "network", "priority", "measured_rtt_ms")
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    MEASURED_RTT_MS_FIELD_NUMBER: _ClassVar[int]
    address: str
    network: str
    priority: int
    measured_rtt_ms: float
    def __init__(self, address: _Optional[str] = ..., network: _Optional[str] = ..., priority: _Optional[int] = ..., measured_rtt_ms: _Optional[float] = ...) -> None: ...

class CacheCompatibility(_message.Message):
    __slots__ = ("model_id", "model_revision", "tokenizer_revision", "cache_format_version", "quantization", "rope_hash", "layer_geometry_hash", "kv_dtype", "block_size_tokens")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_REVISION_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_REVISION_FIELD_NUMBER: _ClassVar[int]
    CACHE_FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    ROPE_HASH_FIELD_NUMBER: _ClassVar[int]
    LAYER_GEOMETRY_HASH_FIELD_NUMBER: _ClassVar[int]
    KV_DTYPE_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    model_revision: str
    tokenizer_revision: str
    cache_format_version: str
    quantization: str
    rope_hash: str
    layer_geometry_hash: str
    kv_dtype: str
    block_size_tokens: int
    def __init__(self, model_id: _Optional[str] = ..., model_revision: _Optional[str] = ..., tokenizer_revision: _Optional[str] = ..., cache_format_version: _Optional[str] = ..., quantization: _Optional[str] = ..., rope_hash: _Optional[str] = ..., layer_geometry_hash: _Optional[str] = ..., kv_dtype: _Optional[str] = ..., block_size_tokens: _Optional[int] = ...) -> None: ...

class CacheCapability(_message.Message):
    __slots__ = ("compatibility", "cache_address", "cache_bytes_used", "cache_bytes_free", "entry_count", "cache_epoch", "load", "tokens_served", "bloom_filter")
    COMPATIBILITY_FIELD_NUMBER: _ClassVar[int]
    CACHE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    CACHE_BYTES_USED_FIELD_NUMBER: _ClassVar[int]
    CACHE_BYTES_FREE_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    CACHE_EPOCH_FIELD_NUMBER: _ClassVar[int]
    LOAD_FIELD_NUMBER: _ClassVar[int]
    TOKENS_SERVED_FIELD_NUMBER: _ClassVar[int]
    BLOOM_FILTER_FIELD_NUMBER: _ClassVar[int]
    compatibility: CacheCompatibility
    cache_address: str
    cache_bytes_used: int
    cache_bytes_free: int
    entry_count: int
    cache_epoch: int
    load: float
    tokens_served: int
    bloom_filter: bytes
    def __init__(self, compatibility: _Optional[_Union[CacheCompatibility, _Mapping]] = ..., cache_address: _Optional[str] = ..., cache_bytes_used: _Optional[int] = ..., cache_bytes_free: _Optional[int] = ..., entry_count: _Optional[int] = ..., cache_epoch: _Optional[int] = ..., load: _Optional[float] = ..., tokens_served: _Optional[int] = ..., bloom_filter: _Optional[bytes] = ...) -> None: ...

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

class GetCacheSummaryRequest(_message.Message):
    __slots__ = ("compatibility",)
    COMPATIBILITY_FIELD_NUMBER: _ClassVar[int]
    compatibility: CacheCompatibility
    def __init__(self, compatibility: _Optional[_Union[CacheCompatibility, _Mapping]] = ...) -> None: ...

class GetCacheSummaryResponse(_message.Message):
    __slots__ = ("node_id", "caches")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CACHES_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    caches: _containers.RepeatedCompositeFieldContainer[CacheCapability]
    def __init__(self, node_id: _Optional[str] = ..., caches: _Optional[_Iterable[_Union[CacheCapability, _Mapping]]] = ...) -> None: ...

class LookupPrefixRequest(_message.Message):
    __slots__ = ("compatibility", "block_hashes")
    COMPATIBILITY_FIELD_NUMBER: _ClassVar[int]
    BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    compatibility: CacheCompatibility
    block_hashes: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, compatibility: _Optional[_Union[CacheCompatibility, _Mapping]] = ..., block_hashes: _Optional[_Iterable[bytes]] = ...) -> None: ...

class LookupPrefixResponse(_message.Message):
    __slots__ = ("node_id", "hit_block_count", "hit_token_count", "transfer_bytes", "cache_epoch", "lease_id", "lease_expires_at_unix", "payload_sha256")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HIT_BLOCK_COUNT_FIELD_NUMBER: _ClassVar[int]
    HIT_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_BYTES_FIELD_NUMBER: _ClassVar[int]
    CACHE_EPOCH_FIELD_NUMBER: _ClassVar[int]
    LEASE_ID_FIELD_NUMBER: _ClassVar[int]
    LEASE_EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_SHA256_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    hit_block_count: int
    hit_token_count: int
    transfer_bytes: int
    cache_epoch: int
    lease_id: str
    lease_expires_at_unix: float
    payload_sha256: bytes
    def __init__(self, node_id: _Optional[str] = ..., hit_block_count: _Optional[int] = ..., hit_token_count: _Optional[int] = ..., transfer_bytes: _Optional[int] = ..., cache_epoch: _Optional[int] = ..., lease_id: _Optional[str] = ..., lease_expires_at_unix: _Optional[float] = ..., payload_sha256: _Optional[bytes] = ...) -> None: ...

class FetchBlocksRequest(_message.Message):
    __slots__ = ("lease_id",)
    LEASE_ID_FIELD_NUMBER: _ClassVar[int]
    lease_id: str
    def __init__(self, lease_id: _Optional[str] = ...) -> None: ...

class KVBlockChunk(_message.Message):
    __slots__ = ("block_hash", "block_index", "token_count", "chunk_index", "total_chunks", "data", "block_sha256", "cache_epoch", "compatibility")
    BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SHA256_FIELD_NUMBER: _ClassVar[int]
    CACHE_EPOCH_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_FIELD_NUMBER: _ClassVar[int]
    block_hash: bytes
    block_index: int
    token_count: int
    chunk_index: int
    total_chunks: int
    data: bytes
    block_sha256: bytes
    cache_epoch: int
    compatibility: CacheCompatibility
    def __init__(self, block_hash: _Optional[bytes] = ..., block_index: _Optional[int] = ..., token_count: _Optional[int] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., data: _Optional[bytes] = ..., block_sha256: _Optional[bytes] = ..., cache_epoch: _Optional[int] = ..., compatibility: _Optional[_Union[CacheCompatibility, _Mapping]] = ...) -> None: ...

class PublishBlockResponse(_message.Message):
    __slots__ = ("stored", "cache_epoch")
    STORED_FIELD_NUMBER: _ClassVar[int]
    CACHE_EPOCH_FIELD_NUMBER: _ClassVar[int]
    stored: bool
    cache_epoch: int
    def __init__(self, stored: _Optional[bool] = ..., cache_epoch: _Optional[int] = ...) -> None: ...

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

class DFlashProposerServiceCloseSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class DFlashProposerServiceCloseSessionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
