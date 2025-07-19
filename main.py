import argparse
import logging
from pathlib import Path
import uvicorn

from app import app, DEBUG_REQUESTS_DIR
from utils import request_id_ctx

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixture-of-Models gateway")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--debug-requests", action="store_true",
        help="Save human-readable trace of each request to ./debug-requests/"
    )
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [req:%(request_id)s] %(name)s: %(message)s"
    )

    # Override global LogRecord factory to inject request_id
    _old_factory = logging.getLogRecordFactory()

    def _record_factory(*args, **kwargs):
        record = _old_factory(*args, **kwargs)
        # Default to "-" if contextvar is unset (safe for startup/background)
        record.request_id = request_id_ctx.get("-")
        return record

    logging.setLogRecordFactory(_record_factory)

    if args.debug:
        logger.debug("Debug logging enabled")

    if args.debug_requests:
        DEBUG_REQUESTS_DIR = Path("debug-requests")
        DEBUG_REQUESTS_DIR.mkdir(exist_ok=True, parents=True)
        logger.info("Per-request debug traces will be written to %s", DEBUG_REQUESTS_DIR)

    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
