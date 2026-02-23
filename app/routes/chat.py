import os

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field




# ---- Config ----
MODEL_BASE = os.environ.get("MODEL_BASE", "http://127.0.0.1:9000").rstrip("/")
DEFAULT_MODEL_ROUTE = "chat/internvl2_5_2b"  # matches your Gradio config




# -------- Models --------
class ChatMessage(BaseModel):
    role: str
    content: str

# ---- Pydantic models (loose on purpose; proxy shouldn't be brittle) ----
class ChatProxyRequest(BaseModel):
    """
    Intentionally permissive. Whatever your frontend sends will be forwarded.
    Put your strict schema here later once your payload is stable.
    """
    model_config = ConfigDict(extra="allow")

    # Optional convenience fields if you want to standardize later:
    stream: Optional[bool] = Field(default=None, description="If True, request streaming from model service.")



router = APIRouter(tags=["chat"])




@router.post("/chat")
async def chat_proxy(
    body: ChatProxyRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> Any:
    """
    Proxies chat requests to the model service.

    - If upstream returns streaming (chunked / event-stream), we stream bytes through.
    - Otherwise we return JSON.

    The frontend can call:
      POST /api/chat

    Environment variables:
      MODEL_BASE  (default http://127.0.0.1:9000)
    """
    upstream_url = f"{MODEL_BASE}/{DEFAULT_MODEL_ROUTE}"

    # Forward headers (keep it minimal)
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization

    # Forward query params too (if you ever add ?model=... etc.)
    params = dict(request.query_params)

    payload = body.model_dump(exclude_none=True)

    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Use streaming request to support both streaming + non-streaming responses
            async with client.stream(
                "POST",
                upstream_url,
                headers=headers,
                params=params,
                json=payload,
            ) as resp:
                # Map upstream errors to a useful response
                if resp.status_code >= 400:
                    # Try to read upstream body safely (may be JSON or text)
                    try:
                        err_json = await resp.json()
                        detail = {"upstream_status": resp.status_code, "upstream": err_json}
                    except Exception:
                        err_text = (await resp.aread()).decode("utf-8", errors="replace")
                        detail = {"upstream_status": resp.status_code, "upstream": err_text}
                    
                    # TEMP debug
                    print("UPSTREAM STATUS:", resp.status_code)
                    print("UPSTREAM BODY:", detail)
                    raise HTTPException(status_code=502, detail=detail)

                content_type = resp.headers.get("content-type", "")

                # If upstream is SSE or otherwise streaming, pass-through bytes
                is_streaming = (
                    "text/event-stream" in content_type
                    or "application/x-ndjson" in content_type
                    or "chunked" in resp.headers.get("transfer-encoding", "").lower()
                )

                if is_streaming:
                    async def iter_bytes():
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk

                    passthrough_headers = {}
                    # Preserve content-type so the client knows how to parse it
                    if content_type:
                        passthrough_headers["Content-Type"] = content_type

                    return StreamingResponse(
                        iter_bytes(),
                        status_code=200,
                        headers=passthrough_headers,
                    )

                # Non-streaming: parse JSON and return
                #data = await resp.json()
                await resp.aread()
                data = resp.json()
                return JSONResponse(content=data, status_code=200)

        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Could not connect to model service at {upstream_url}: {e}",
            )
        except httpx.ReadError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Read error from model service at {upstream_url}: {e}",
            )
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout calling model service at {upstream_url}: {e}",
            )