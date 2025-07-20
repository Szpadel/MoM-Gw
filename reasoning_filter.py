from typing import Dict, Any

class ReasoningFilter:
    """
    Converts streaming 'reasoning' deltas into content wrapped in <think>...</think> tags.
    Maintains per-request state to ensure correct tag placement and buffering.
    """

    def __init__(self):
        # Track if we're inside a reasoning block for each event_id
        self.in_reasoning_block: Dict[str, bool] = {}
        # Buffer for content that needs to be prepended after closing a reasoning block
        self.buffer: Dict[str, str] = {}

    def stream(self, event: dict) -> dict:
        event_id = event.get("id")
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})

            # Prepend any buffered content if present
            buf = self.buffer.pop(event_id, None)
            if buf is not None:
                curr = delta.get("content", "")
                delta["content"] = f"{buf}{curr}"

            reasoning = delta.get("reasoning")
            if self.in_reasoning_block.get(event_id, False):
                if reasoning is None:
                    # End of reasoning block: close tag, buffer any content
                    self.in_reasoning_block.pop(event_id, None)
                    self.buffer[event_id] = "</think>\n\n" + delta.get("content", "")
                    delta["content"] = "\n\n"
                else:
                    # Continue reasoning block: append reasoning to content
                    delta["content"] = delta.get("content", "") + reasoning
            elif reasoning is not None:
                # Start of reasoning block: set flag and wrap reasoning in opening tag
                self.in_reasoning_block[event_id] = True
                delta["content"] = "<think>\n" + reasoning

            # Remove the original reasoning field to avoid leaking it
            if "reasoning" in delta:
                del delta["reasoning"]

        return event
