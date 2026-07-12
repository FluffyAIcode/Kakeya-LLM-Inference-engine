"""FastAPI management/telemetry surface for the Kakeya inference network."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from inference_engine.network.dashboard import dashboard_html
from inference_engine.network.state import NetworkState


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


def create_network_app(
    state: NetworkState,
    *,
    api_key: Optional[str] = None,
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
