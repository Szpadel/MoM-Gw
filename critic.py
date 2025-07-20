from critic_strategies import build_strategy
import httpx
from config import AppConfig
from typing import List, Dict, Optional, Any, AsyncGenerator
import logging
import time
import uuid

logger = logging.getLogger(__name__)

class CriticService:
    def __init__(self, app_cfg: AppConfig, http_client: httpx.AsyncClient):
        self.strategy = None
        if app_cfg.critic:
            endpoint = self._get_endpoint(app_cfg, app_cfg.critic.endpoint)
            if endpoint:
                self.strategy = build_strategy(
                    app_cfg.critic,
                    endpoint,
                    http_client
                )

    def _get_endpoint(self, app_cfg: AppConfig, name: str):
        return next((e for e in app_cfg.endpoints if e.name == name), None)

    async def compose_context_question(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        return await self.strategy.compose_context(messages) if self.strategy else None

    async def run_critic(self, candidates: List[dict], context: Optional[str] = None) -> dict:
        if not self.strategy or not candidates:
            return self._fallback_response(candidates[0])
        return await self.strategy.run_critic(candidates, context)

    async def run_critic_stream(
        self,
        candidates: List[dict],
        context: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        if not self.strategy or not candidates:
            fallback_content = "Critic processing failed: No valid candidates"
            if candidates:
                fallback_content = candidates[0]["choices"][0]["message"]["content"]

            yield {
                "id": f"critic-fallback-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mixture-critic",
                "choices": [{
                    "index": 0,
                    "delta": {"content": fallback_content},
                    "finish_reason": "stop"
                }]
            }
            return

        async for chunk in self.strategy.run_critic_stream(candidates, context):
            yield chunk

    def _fallback_response(self, candidate: dict) -> dict:
        return {
            "choices": [{"message": candidate["choices"][0]["message"]}],
            "usage": candidate.get("usage")
        }
