from .base import BaseCriticStrategy
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class MergeStrategy(BaseCriticStrategy):
    async def compose_context(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        logger.debug("Composing context")
        if not self.cfg.context_system_prompt and not self.cfg.context_user_prompt:
            return None

        sys_prompt = self.cfg.context_system_prompt or (
            "You are an assistant that extracts the latest user question plus any "
            "necessary context so another model can judge candidate answers."
        )
        user_prompt = (self.cfg.context_user_prompt or (
            "Given the following chat history JSON, output the final user question "
            "rephrased together with any essential context:\n\n{history}"
        )).format(history=messages)

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
        user_prompt = (self.cfg.user_prompt or (
            "Context:\n{context}\n\n"
            "Candidate answers (delimited by =!=!= ... =!=!=):\n{answers}\n\n"
            "Compose the SINGLE, self-contained, high-quality answer described "
            "above. Keep all useful details and reasoning. Do not mention that the "
            "answer was merged or reference the existence of other answers."
        )).format(context=context or "", answers=answers)

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt or (
                    "You are an expert answer composer. Analyse every candidate answer, "
                    "identify all valuable ideas, explanations, arguments, examples and "
                    "code snippets, then write ONE comprehensive, well-structured answer "
                    "that merges those good parts, removes contradictions and fills any "
                    "gaps. The final result must be richer and clearer than any single "
                    "candidate answer."
                )},
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

        user_prompt = (self.cfg.user_prompt or (
            "Context:\n{context}\n\n"
            "Candidate answers (delimited by =!=!= ... =!=!=):\n{answers}\n\n"
            "Compose the SINGLE, self-contained, high-quality answer that merges "
            "the best parts. Preserve all valuable content without contradictions."
        )).format(context=context or "", answers=answers)

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt or (
                    "You are an expert answer composer. Analyse every candidate answer, "
                    "identify valuable ideas, explanations, arguments, examples and "
                    "code snippets, then write ONE comprehensive, well-structured answer "
                    "that merges the best parts and fills gaps."
                )},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.cfg.temperature
        }

        async for chunk in self._call_endpoint_stream(payload):
            yield chunk
