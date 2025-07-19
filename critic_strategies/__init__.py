from .merge import MergeStrategy
from .base import BaseCriticStrategy
from config import CriticConfig, EndpointConfig
from typing import Dict, Type
import httpx

_registry: Dict[str, Type[BaseCriticStrategy]] = {
    "merge": MergeStrategy
}

def build_strategy(
    cfg: CriticConfig,
    endpoint: EndpointConfig,
    http_client: httpx.AsyncClient
) -> BaseCriticStrategy:
    strategy_class = _registry.get(cfg.strategy.lower())
    if not strategy_class:
        raise ValueError(f"Invalid critic strategy: {cfg.strategy}")
    return strategy_class(cfg, endpoint, http_client)
