import logging
import httpx
from typing import List, Dict, Optional, Any
from config import AppConfig, CriticConfig
from utils import format_response

logger = logging.getLogger(__name__)

class CriticService:
    def __init__(self, config: AppConfig, http_client: httpx.AsyncClient):
        self.config = config
        self.http_client = http_client
        self.critic_config = config.critic
        
    async def compose_context_question(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        logger.debug("Composing context")
        if not self.critic_config:
            return None
            
        endpoint = self._get_endpoint(self.critic_config.endpoint)
        if not endpoint:
            return None

        sys_prompt = self.critic_config.context_system_prompt or (
            "You are an assistant that extracts the latest user question plus any "
            "necessary context so another model can judge candidate answers."
        )
        user_prompt = (self.critic_config.context_user_prompt or (
            "Given the following chat history JSON, output the final user question "
            "rephrased together with any essential context:\n\n{history}"
        )).format(history=messages)

        payload = {
            "model": self.critic_config.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.critic_config.temperature
        }

        resp = await self._call_endpoint(endpoint, payload)
        if resp:
            logger.debug("Context composition succeeded")
            return resp["choices"][0]["message"]["content"]
        else:
            logger.warning("Context composition failed")
            return None

    async def run_critic(self, candidates: List[dict], context: Optional[str] = None) -> dict:
        logger.info(f"Running critic with {len(candidates)} candidates")
        if not self.critic_config or not candidates:
            return self._fallback_response(candidates[0])

        endpoint = self._get_endpoint(self.critic_config.endpoint)
        if not endpoint:
            return self._fallback_response(candidates[0])

        # Prepare prompt
        answers = "\n\n".join(
            f"=!=!= Answer #{i+1}:\n{c['choices'][0]['message']['content']} =!=!="
            for i, c in enumerate(candidates)
        )
        user_prompt = (self.critic_config.user_prompt or (
            "Context:\n{context}\n\n"
            "Candidate answers (delimited by =!=!= ... =!=!=):\n{answers}\n\n"
            "Compose the SINGLE, self-contained, high-quality answer described "
            "above. Keep all useful details and reasoning. Do not mention that the "
            "answer was merged or reference the existence of other answers."
        )).format(
            context=context or "", 
            answers=answers
        )

        payload = {
            "model": self.critic_config.model,
            "messages": [
                {"role": "system", "content": self.critic_config.system_prompt or (
                    "You are an expert answer composer. Analyse every candidate answer, "
                    "identify all valuable ideas, explanations, arguments, examples and "
                    "code snippets, then write ONE comprehensive, well-structured answer "
                    "that merges those good parts, removes contradictions and fills any "
                    "gaps. The final result must be richer and clearer than any single "
                    "candidate answer."
                )},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.critic_config.temperature
        }

        resp = await self._call_endpoint(endpoint, payload)
        if resp:
            logger.info("Critic succeeded")
            return format_response(resp["choices"][0]["message"], resp.get("usage"))
        else:
            logger.warning("Critic failed - using fallback")
            return self._fallback_response(candidates[0])
    
    def _fallback_response(self, candidate: dict) -> dict:
        """OpenAI-compatible fallback"""
        return format_response(
            candidate["choices"][0]["message"],
            candidate.get("usage")
        )
    
    def _get_endpoint(self, name: str):
        return next((e for e in self.config.endpoints if e.name == name), None)
    
    async def _call_endpoint(self, endpoint, payload: dict) -> Optional[dict]:
        logger.debug(f"Calling endpoint: {endpoint.name}")
        try:
            url = f"{endpoint.base_url.rstrip('/')}/v1/chat/completions"
            resp = await self.http_client.post(
                url,
                headers={"Authorization": f"Bearer {endpoint.api_key}"},
                json=payload
            )
            if resp.status_code != 200:
                logger.warning(f"Endpoint error: {resp.status_code}")
            return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            logger.error(f"Endpoint exception: {str(e)}")
            return None
