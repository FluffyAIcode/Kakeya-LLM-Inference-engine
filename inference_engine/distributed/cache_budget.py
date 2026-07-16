"""Platform-neutral cache-budget policy for model-loaded offload workers."""


def adaptive_cache_budget(
    *,
    total_bytes: int,
    active_model_bytes: int,
    ceiling_bytes: int,
    minimum_bytes: int,
    reserve_bytes: int,
) -> int:
    if min(total_bytes, ceiling_bytes, minimum_bytes) <= 0 or reserve_bytes < 0:
        raise ValueError("adaptive cache budget inputs are invalid")
    available = max(0, total_bytes - active_model_bytes - reserve_bytes)
    return max(minimum_bytes, min(ceiling_bytes, available))
