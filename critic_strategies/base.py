import httpx
from abc import ABC, abstractmethod
from config import CriticConfig, EndpointConfig
from typing import List, Dict, Any, Optional
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
            
    def _fallback_response(self, candidate: dict) -> dict:
        """Standard fallback response"""
        return {
            "choices": [{"message": candidate["choices"][0]["message"]}],
            "usage": candidate.get("usage")
        }
