import os
import yaml
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class EndpointConfig(BaseModel):
    name: str
    base_url: str
    api_key: str

class ModelConfig(BaseModel):
    endpoint: str
    model: str
    params: Dict[str, Any] = {}

class CriticConfig(BaseModel):
    strategy: str = "merge"
    endpoint: str
    model: str
    temperature: float = 0.6
    strategy_params: Dict[str, Any] = {}
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    context_system_prompt: Optional[str] = None
    context_user_prompt: Optional[str] = None

class AppConfig(BaseModel):
    endpoints: List[EndpointConfig]
    models: List[ModelConfig]
    critic: Optional[CriticConfig] = None
    timeout: float = 180.0
    api_key: Optional[str] = None

def _resolve_env(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders"""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.getenv(obj[2:-1], obj)  # Fallback to original
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(item) for item in obj]
    return obj

def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and validate configuration"""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    resolved = _resolve_env(raw)
    return AppConfig(**resolved)
