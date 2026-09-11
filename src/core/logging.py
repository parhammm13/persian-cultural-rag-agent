from __future__ import annotations

import logging


LOGGER_NAME = "persian_cultural_rag.api"


def configure_api_logging(level: int = logging.INFO) -> None:
    """Configure API logging without logging request/response bodies."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.propagate = False
