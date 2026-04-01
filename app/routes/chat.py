import os

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Depends
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.models import QueryLog, User
from app.db import get_db
from app.deps import get_current_user
import time




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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None),
) -> Any:
    """
    Proxies chat requests to the model service.
    Logs all queries and responses to query_logs table.
    """
    upstream_url = f"{MODEL_BASE}/{DEFAULT_MODEL_ROUTE}"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    
    params = dict(request.query_params)
    payload = body.model_dump(exclude_none=True)
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST",
                upstream_url,
                headers=headers,
                params=params,
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    try:
                        err_json = await resp.json()
                        detail = {"upstream_status": resp.status_code, "upstream": err_json}
                    except Exception:
                        err_text = (await resp.aread()).decode("utf-8", errors="replace")
                        detail = {"upstream_status": resp.status_code, "upstream": err_text}
                    raise HTTPException(status_code=502, detail=detail)
                
                content_type = resp.headers.get("content-type", "")
                is_streaming = (
                    "text/event-stream" in content_type
                    or "application/x-ndjson" in content_type
                    or "chunked" in resp.headers.get("transfer-encoding", "").lower()
                )
                
                if is_streaming:
                    collected_response = []
                    
                    async def iter_bytes_and_log():
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                collected_response.append(chunk.decode("utf-8", errors="replace"))
                                yield chunk
                        
                        # Log after streaming completes
                        latency_ms = int((time.time() - start_time) * 1000)
                        full_response = "".join(collected_response)
                        log_query(
                            db=db,
                            user_id=current_user.id,
                            payload=payload,
                            response_text=full_response,
                            latency_ms=latency_ms,
                        )
                    
                    passthrough_headers = {}
                    if content_type:
                        passthrough_headers["Content-Type"] = content_type
                    return StreamingResponse(
                        iter_bytes_and_log(),
                        status_code=200,
                        headers=passthrough_headers,
                    )
                
                # Non-streaming
                await resp.aread()
                data = resp.json()
                
                latency_ms = int((time.time() - start_time) * 1000)
                response_text = data.get("response", "") if isinstance(data, dict) else str(data)
                
                log_query(
                    db=db,
                    user_id=current_user.id,
                    payload=payload,
                    response_text=response_text,
                    latency_ms=latency_ms,
                )
                
                return JSONResponse(content=data, status_code=200)
                
        except httpx.ConnectError as e:
            raise HTTPException(status_code=502, detail=f"Could not connect to model service: {e}")
        except httpx.ReadError as e:
            raise HTTPException(status_code=502, detail=f"Read error from model service: {e}")
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail=f"Timeout calling model service: {e}")


def log_query(
    db: Session,
    user_id: int,
    payload: dict,
    response_text: str,
    latency_ms: int,
):
    """Log query to database."""
    try:
        query_log = QueryLog(
            user_id=user_id,
            image_id=payload.get("image_id"),
            prompt=payload.get("prompt", ""),
            response=response_text,
            model_name=payload.get("model", "internvl2.5_8B"),
            latency_ms=latency_ms,
            extra_data={
                "history_length": len(payload.get("history", [])) if payload.get("history") else 0,
            }
        )
        db.add(query_log)
        db.commit()
    except Exception as e:
        print(f"Failed to log query: {e}")
        db.rollback()