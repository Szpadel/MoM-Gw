from .base import BaseCriticStrategy
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
from .merge_prompts import (
    DEFAULT_CONTEXT_SYSTEM_PROMPT,
    DEFAULT_CONTEXT_USER_PROMPT,
    DEFAULT_MERGE_SYSTEM_PROMPT,
    DEFAULT_MERGE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

class MergeStrategy(BaseCriticStrategy):
    async def compose_context(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        logger.debug("Composing context")
        if not self.cfg.context_system_prompt and not self.cfg.context_user_prompt:
            return None

        sys_prompt = self.cfg.context_system_prompt or DEFAULT_CONTEXT_SYSTEM_PROMPT
        user_prompt = (
            self.cfg.context_user_prompt or DEFAULT_CONTEXT_USER_PROMPT
        ).format(history=messages)

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.cfg.temperature
        }

        resp = await self._call_endpoint(payload)
        return resp["choices"][0]["message"]["content"] if resp else None

    async def run_critic(self, candidates: List[dict], context: Optional[str] = None) -> dict:
        logger.info(f"Running critic with {len(candidates)} candidates")
        answers = "\n\n".join(
            f"=!=!= Answer #{i+1}:\n{c['choices'][0]['message']['content']} =!=!="
            for i, c in enumerate(candidates)
        )
        user_prompt = (
            self.cfg.user_prompt or DEFAULT_MERGE_USER_PROMPT
        ).format(context=context or "", answers=answers)

        payload = {
            "model": self.cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.cfg.system_prompt or DEFAULT_MERGE_SYSTEM_PROMPT,
                },
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.cfg.temperature
        }

        resp = await self._call_endpoint(payload)
        return resp if resp else self._fallback_response(candidates[0])

    async def run_critic_stream(
        self,
        candidates: List[dict],
        context: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        logger.info("Streaming merge critic with %s candidates", len(candidates))

        answers = "\n\n".join(
            f"=!=!= Answer #{i+1}:\n{c['choices'][0]['message']['content']} =!=!="
            for i, c in enumerate(candidates)
        )

        user_prompt = (
            self.cfg.user_prompt or DEFAULT_MERGE_USER_PROMPT
        ).format(context=context or "", answers=answers)

        payload = {
            "model": self.cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.cfg.system_prompt
                    or DEFAULT_MERGE_SYSTEM_PROMPT,
                },
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.cfg.temperature
        }

        async for chunk in self._call_endpoint_stream(payload):
            yield chunk
