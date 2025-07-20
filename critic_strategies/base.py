import httpx
import json
import uuid
import time
from abc import ABC, abstractmethod
from config import CriticConfig, EndpointConfig
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class BaseCriticStrategy(ABC):
    def __init__(self, cfg: CriticConfig, endpoint: EndpointConfig, http_client: httpx.AsyncClient):
        self.cfg = cfg
        self.endpoint = endpoint
        self.http = http_client

    @abstractmethod
    async def compose_context(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate context from conversation history"""
        pass

    @abstractmethod
    async def run_critic(self, candidates: List[dict], context: Optional[str]) -> dict:
        """Process candidates into final response"""
        pass

    @abstractmethod
    async def run_critic_stream(
        self,
        candidates: List[dict],
        context: Optional[str],
    ) -> AsyncGenerator[dict, None]:
        """Yield streaming response chunks from critic"""
        pass

    async def _call_endpoint(self, payload: dict) -> Optional[dict]:
        """Centralized HTTP call logic"""
        try:
            url = f"{self.endpoint.base_url.rstrip('/')}/v1/chat/completions"
            resp = await self.http.post(
                url,
                headers={"Authorization": f"Bearer {self.endpoint.api_key}"},
                json=payload
            )
            return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            logger.error(f"Endpoint error: {str(e)}")
            return None

    async def _call_endpoint_stream(self, payload: dict) -> AsyncGenerator[dict, None]:
        """Robust streaming HTTP caller with error handling"""
        url = f"{self.endpoint.base_url.rstrip('/')}/v1/chat/completions"
        try:
            async with self.http.stream(
                "POST",
                url,
                headers={"Authorization": f"Bearer {self.endpoint.api_key}"},
                json={**payload, "stream": True},
                timeout=None
            ) as resp:
                if resp.status_code != 200:
                    logger.error("Critic stream HTTP %s", resp.status_code)
                    yield {
                        "id": f"critic-error-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": payload.get("model", "unknown"),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: HTTP {resp.status_code}"},
                            "finish_reason": "stop"
                        }]
                    }
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning("Bad JSON chunk: %s", data)

        except Exception as exc:
            logger.exception("Streaming critic call failed: %s", exc)
            yield {
                "id": f"critic-error-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": payload.get("model", "unknown"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Error: {str(exc)}"},
                    "finish_reason": "stop"
                }]
            }

    def _fallback_response(self, candidate: dict) -> dict:
        """Standard fallback response"""
        return {
            "choices": [{"message": candidate["choices"][0]["message"]}],
            "usage": candidate.get("usage")
        }
