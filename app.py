import asyncio
import uuid
import httpx
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from config import load_config, AppConfig
from critic import CriticService
from utils import write_debug_trace, format_response, request_id_ctx
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from reasoning_filter import ReasoningFilter

REASONING_FILTER_ENABLED = False

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

app = FastAPI(title="Mixture-of-Models Gateway")
DEBUG_REQUESTS_DIR: Optional[Path] = None

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared HTTP client"""
    config = load_config()
    app.state.config = config
    transport = httpx.AsyncHTTPTransport(retries=2)
    app.state.http_client = httpx.AsyncClient(
        timeout=config.timeout,
        transport=transport
    )
    app.state.critic = CriticService(config, app.state.http_client)
    yield
    await app.state.http_client.aclose()

app.router.lifespan_context = lifespan

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    request.state.id = rid
    token = request_id_ctx.set(rid)
    try:
        logger.info("Request started")
        response = await call_next(request)
        return response
    finally:
        request_id_ctx.reset(token)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_endpoint(client: httpx.AsyncClient, endpoint, payload: dict):
    """Unified endpoint caller with retries"""
    url = f"{endpoint.base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {endpoint.api_key}"}
    return await client.post(url, json=payload, headers=headers)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    logger.info("Starting request processing")
    rid = request.state.id
    config: AppConfig = request.app.state.config
    client = request.app.state.http_client

    logger.info(f"Preparing {len(config.models)} model tasks")
    # Parallelize context composition and model queries
    context_task = app.state.critic.compose_context_question(req.messages)
    model_tasks = []
    tasks_info = []

    for model in config.models:
        endpoint = next((e for e in config.endpoints if e.name == model.endpoint), None)
        if not endpoint:
            continue

        # Merge payloads (model params first, client overrides)
        payload = {**model.params, **req.dict(exclude_unset=True)}
        payload["model"] = model.model
        payload["stream"] = False  # Force non-streaming for base models

        task = call_endpoint(client, endpoint, payload)
        model_tasks.append(task)
        tasks_info.append({
            "endpoint": model.endpoint,
            "model": model.model,
            "payload": payload
        })

    logger.debug("Starting parallel execution")
    results = await asyncio.gather(*model_tasks, return_exceptions=True)
    logger.info(f"Collected {len(results)} responses")
    context = await context_task

    # Process responses
    successful = []
    for task_info, res in zip(tasks_info, results):
        if isinstance(res, httpx.Response) and res.status_code == 200:
            task_info["body"] = res.json()
            successful.append(task_info["body"])

    # Streaming path
    if req.stream:
        logger.info("Starting streaming critic execution")

        async def stream_generator():
            full_content = ""
            full_reasoning = ""
            # Instantiate the filter per request if enabled
            reasoning_filter = ReasoningFilter() if REASONING_FILTER_ENABLED else None
            try:
                async for chunk in app.state.critic.run_critic_stream(successful, context):
                    # Accumulate raw content/reasoning for debug trace
                    if chunk.get("choices") and chunk["choices"][0].get("delta"):
                        delta = chunk["choices"][0]["delta"]
                        if "reasoning" in delta:
                            # Mirror reasoning for better compatibility
                            delta["reasoning_content"] = delta["reasoning"]
                        full_content += delta.get("content", "") or ""
                        full_reasoning += delta.get("reasoning", "") or ""
                    # Apply reasoning filter if enabled
                    filtered_chunk = reasoning_filter.stream(chunk) if reasoning_filter else chunk
                    yield f"data: {json.dumps(filtered_chunk)}\n\n"
            finally:
                # Write debug trace after stream completes
                if DEBUG_REQUESTS_DIR:
                    final_resp = {
                        "choices": [{"message": {"content": full_content}}],
                        "usage": {"total_tokens": len(full_content.split())}
                    }
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        write_debug_trace,
                        req.messages,
                        tasks_info,
                        context,
                        final_resp,
                        DEBUG_REQUESTS_DIR,
                        rid,
                        full_reasoning
                    )

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    # Non-streaming path
    logger.info("Starting non-streaming critic execution")
    final_resp = await app.state.critic.run_critic(successful, context)

    # Extract reasoning if present
    reasoning = ""
    if final_resp.get("choices") and final_resp["choices"][0].get("message"):
        reasoning = final_resp["choices"][0]["message"].get("reasoning", "")

    if DEBUG_REQUESTS_DIR:
        await asyncio.get_running_loop().run_in_executor(
            None,
            write_debug_trace,
            req.messages,
            tasks_info,
            context,
            final_resp,
            DEBUG_REQUESTS_DIR,
            rid,
            reasoning
        )

    logger.info("Request processing finished")
    return final_resp
