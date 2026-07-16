"""FastAPI management/telemetry surface for the Kakeya inference network."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional, TYPE_CHECKING

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from inference_engine.network.dashboard import dashboard_html
from inference_engine.network.state import NetworkState

if TYPE_CHECKING:
    from inference_engine.distributed.cache_fill import CacheFillCapture


class RegisterNodeRequest(BaseModel):
    alias: str = Field(min_length=1, max_length=100)
    address: str = Field(min_length=3, max_length=255)
    region: str = Field(default="Private", max_length=100)
    role: str = Field(default="hybrid", pattern="^(head|cache|hybrid|inference)$")


class CreateGroupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    node_ids: list[str] = Field(min_length=1)


class TokenTelemetryRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=100)
    completed: int = Field(ge=0)
    kv_assisted: int = Field(default=0, ge=0)


class DrainCaptureRequest(BaseModel):
    max_items: int = Field(default=8, ge=1, le=64)


class BenchmarkCreateRequest(BaseModel):
    kind: str = Field(default="distributed_prefill_fleet_benchmark", max_length=100)
    config: dict = Field(default_factory=dict)
    started_at: Optional[float] = None


class BenchmarkUpdateRequest(BaseModel):
    stages: list[dict] = Field(default_factory=list)
    status: Optional[str] = None
    finished_at: Optional[float] = None


def create_network_app(
    state: NetworkState,
    *,
    api_key: Optional[str] = None,
    cache_fill_capture: Optional["CacheFillCapture"] = None,
) -> FastAPI:
    key = (api_key if api_key is not None else os.environ.get(
        "KAKEYA_NETWORK_API_KEY", "",
    )).strip()
    app = FastAPI(title="Kakeya Inference Network", version="0.1.0")

    def require_key(
        x_api_key: Optional[str] = Header(default=None),
    ) -> None:
        if key and x_api_key != key:
            raise HTTPException(status_code=401, detail="invalid X-API-Key")

    def require_maintenance_key(
        x_api_key: Optional[str] = Header(default=None),
    ) -> None:
        if not key or x_api_key != key:
            raise HTTPException(status_code=401, detail="maintenance API key required")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/network", response_class=HTMLResponse)
    def dashboard() -> str:
        return dashboard_html()

    @app.get("/healthz")
    def healthz():
        summary = state.summary()
        return {
            "status": "ok",
            "online_nodes": summary["online_nodes"],
            "cache_bytes_used": summary["cache_bytes_used"],
        }

    @app.get("/v1/network/summary")
    def summary():
        return state.summary()

    @app.get("/v1/network/nodes")
    def nodes():
        return state.nodes()

    @app.post("/v1/network/nodes/register", dependencies=[Depends(require_key)])
    def register(request: RegisterNodeRequest):
        try:
            return state.register_node(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/network/groups")
    def groups():
        return state.groups()

    @app.post("/v1/network/groups", dependencies=[Depends(require_key)])
    def create_group(request: CreateGroupRequest):
        try:
            return state.create_group(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/network/topology")
    def topology():
        return state.topology()

    @app.get("/v1/network/kvfs")
    def virtual_kv_file():
        return state.virtual_kv_file()

    @app.get("/v1/network/tokens")
    def tokens():
        summary = state.summary()
        return {
            "completed": summary["completed_tokens"],
            "kv_assisted": summary["kv_assisted_tokens"],
            "hit_rate": summary["kv_hit_rate"],
        }

    @app.get("/v1/network/prefill")
    def prefill():
        return state.prefill_stats()

    @app.get("/v1/network/benchmarks")
    def benchmarks(limit: int = 20, status: Optional[str] = None):
        try:
            return state.list_benchmarks(limit=limit, status=status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/network/benchmarks/live")
    def benchmark_live():
        return state.live_benchmark()

    @app.get("/v1/network/benchmarks/{run_id}")
    def benchmark_detail(run_id: str):
        try:
            return state.get_benchmark(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="benchmark not found") from exc

    @app.get("/v1/network/benchmarks/{run_id}/stages")
    def benchmark_stages(run_id: str, offset: int = 0, limit: int = 50):
        if offset < 0 or not 1 <= limit <= 200:
            raise HTTPException(status_code=400, detail="invalid stage pagination")
        try:
            stages = state.get_benchmark(run_id)["stages"]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="benchmark not found") from exc
        return {"items": stages[offset:offset + limit], "total": len(stages)}

    @app.post(
        "/v1/network/benchmarks",
        dependencies=[Depends(require_key)],
    )
    def create_benchmark(request: BenchmarkCreateRequest):
        try:
            return state.create_benchmark(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch(
        "/v1/network/benchmarks/{run_id}",
        dependencies=[Depends(require_key)],
    )
    def update_benchmark(run_id: str, request: BenchmarkUpdateRequest):
        try:
            return state.update_benchmark(run_id, **request.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="benchmark not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/v1/network/maintenance/capture",
        dependencies=[Depends(require_maintenance_key)],
    )
    def capture_status():
        if cache_fill_capture is None:
            raise HTTPException(status_code=404, detail="cache-fill capture disabled")
        return cache_fill_capture.stats()

    @app.post(
        "/v1/network/maintenance/capture/drain",
        dependencies=[Depends(require_maintenance_key)],
    )
    def drain_capture(request: DrainCaptureRequest):
        if cache_fill_capture is None:
            raise HTTPException(status_code=404, detail="cache-fill capture disabled")
        return {
            "items": [
                {
                    "capture_id": item.capture_id,
                    "token_count": item.token_count,
                    "token_ids": list(item.token_ids),
                }
                for item in cache_fill_capture.drain(request.max_items)
            ],
        }

    @app.post(
        "/v1/network/telemetry/tokens",
        dependencies=[Depends(require_key)],
    )
    def record_tokens(request: TokenTelemetryRequest):
        try:
            state.record_tokens(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "accepted"}

    @app.get("/v1/network/events")
    async def events(once: bool = False):
        async def stream():
            while True:
                payload = json.dumps({
                    "type": "summary",
                    "data": state.summary(),
                }, separators=(",", ":"))
                yield f"event: summary\ndata: {payload}\n\n"
                if once:
                    return
                await asyncio.sleep(5)  # pragma: no cover - persistent SSE loop

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
