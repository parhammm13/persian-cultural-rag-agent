"""Phoenix/OpenTelemetry setup for RAG monitoring."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any


class _NoOpSpan:
    def set_input(self, value: Any) -> None:
        pass

    def set_output(self, value: Any) -> None:
        pass

    def set_attribute(self, name: str, value: Any) -> None:
        pass

    def record_exception(self, error: Exception) -> None:
        pass


class NoOpTracer:
    """Tracer used when Phoenix monitoring is disabled."""

    def start_as_current_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> Any:
        return nullcontext(_NoOpSpan())


_tracer: Any | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        f"{name} must be true/false, yes/no, on/off, or 1/0."
    )


def capture_content_enabled() -> bool:
    """Whether queries, retrieved text, and answers may enter traces."""
    return _env_bool("PHOENIX_CAPTURE_CONTENT", True)


def setup_phoenix_tracer() -> Any:
    """Initialize Phoenix once and return an OpenInference tracer.

    ``PHOENIX_COLLECTOR_ENDPOINT`` and ``PHOENIX_API_KEY`` are read by
    Phoenix itself. A local unauthenticated server needs no API key.
    """
    global _tracer

    if _tracer is not None:
        return _tracer

    enabled_by_default = bool(
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    )
    enabled = _env_bool("PHOENIX_TRACING_ENABLED", enabled_by_default)

    if not enabled:
        _tracer = NoOpTracer()
        return _tracer

    try:
        from phoenix.otel import register
    except ImportError as error:
        raise RuntimeError(
            "Phoenix tracing is enabled but arize-phoenix-otel is missing. "
            "Install it with: pip install \"arize-phoenix-otel>=0.16.0\""
        ) from error

    project_name = os.getenv(
        "PHOENIX_PROJECT_NAME",
        "persian-cultural-rag-agent",
    ).strip()

    if not project_name:
        raise RuntimeError("PHOENIX_PROJECT_NAME must not be empty.")

    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT",
        "http://localhost:6006",
    ).strip().rstrip("/")

    # OTLP over HTTP must POST to the traces ingestion route. Supplying the
    # full path also works across Phoenix OTEL SDK versions that do not append
    # it automatically, avoiding HTTP 405 from the Phoenix web root.
    if not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint}/v1/traces"

    tracer_provider = register(
        endpoint=endpoint,
        project_name=project_name,
        protocol="http/protobuf",
        # Immediate export is convenient for short CLI/test processes.
        batch=_env_bool("PHOENIX_BATCH", False),
        auto_instrument=False,
    )

    _tracer = tracer_provider.get_tracer(__name__)
    return _tracer


__all__ = [
    "NoOpTracer",
    "capture_content_enabled",
    "setup_phoenix_tracer",
]