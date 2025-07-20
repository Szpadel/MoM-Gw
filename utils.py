import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import contextvars

logger = logging.getLogger(__name__)

# Context variable for request ID propagation
request_id_ctx = contextvars.ContextVar("request_id", default="-")

def format_response(message: dict, usage: Optional[dict] = None) -> dict:
    """Standardized OpenAI response format"""
    return {
        "id": f"mix-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mixture-v1",
        "choices": [{
            "index": 0,
            "message": {k: v for k, v in message.items() if v is not None},
            "finish_reason": "stop"
        }],
        "usage": {k: v for k, v in (usage or {}).items() if isinstance(v, int)} if usage else None
    }

def write_debug_trace(
    messages: List[Dict[str, Any]],
    tasks_info: List[Dict[str, Any]],
    context: Optional[str],
    final_resp: dict,
    debug_dir: Path,
    request_id: str
):
    """
    Write a human-readable text file containing:
    - initial chat history
    - every upstream model response
    - critic context
    - final critic answer
    """
    try:
        lines: list[str] = []
        lines.append("=== CHAT HISTORY ===")
        for m in messages:
            lines.append(f"{m.get('role', '').upper()}: {m.get('content', '')}")
        lines.append("")

        lines.append("=== CANDIDATE RESPONSES ===")
        for info in tasks_info:
            lines.append(
                f"--- from model {info['model']} (endpoint {info['endpoint']}) "
                f"status={info.get('status', 'n/a')} ---"
            )
            body = info.get("body")
            if isinstance(body, dict):
                # try to show only main content
                try:
                    content = body["choices"][0]["message"]["content"]
                except Exception:
                    content = str(body)[:1000]
            else:
                content = str(body)[:1000]
            lines.append(content)
            lines.append("")

        lines.append("=== CRITIC CONTEXT ===")
        lines.append(context or "(none)")
        lines.append("")

        lines.append("=== FINAL ANSWER ===")
        final_content = (
            final_resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        lines.append(final_content)

        fname = debug_dir / f"{int(time.time())}-{request_id}.txt"
        fname.write_text("\n".join(lines), encoding="utf-8")
    except Exception as exc:      # never break main flow
        logger.exception("Failed to write debug trace: %s", exc)
