"""HeadroomEngine — request/response hook facade (Chunk 2).

Composes the existing compression subsystems behind a clean hook interface.
Does NOT reimplement compression; delegates to injected ``CompressionPipeline``
instances via the ``ports.CompressionPipeline`` Protocol.

Design notes
------------
- **Dependency injection**: pipelines, config, usage_reporter are injected;
  no global state is read or written inside this module.
- **No silent fallbacks**: unregistered (provider, flavor) pairs raise loudly.
- **Passthrough fidelity**: when ``CompressionDecision.should_compress`` is
  False, ``on_request`` returns ``ctx.raw_body`` byte-identical (same object,
  no re-serialization).
- **CCR injection is out of scope**: CCR requires server-level session state
  (sticky tool tracking, frozen_message_count, workspace keys) that the
  facade does not own.  Chunk 4 wires that.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from headroom.engine.contract import (
    Flavor,
    Provider,
    RequestContext,
    RequestDecision,
    ResponseTelemetry,
    StreamContext,
)
from headroom.engine.ports import CompressionPipeline
from headroom.proxy.auth_mode import classify_auth_mode
from headroom.proxy.compression_decision import CompressionDecision
from headroom.transforms.compression_policy import resolve_policy


class HeadroomEngine:
    """Facade that composes Headroom compression behind hook-shaped entry points.

    ``on_request`` is the load-bearing method for Chunk 2.  The response hooks
    are stubs that the later chunks (telemetry, CCR expansion) will grow into.

    Parameters
    ----------
    pipelines:
        Mapping from ``(Provider, Flavor)`` to a ``CompressionPipeline``
        implementor.  Fakes satisfy this in tests; the real server passes
        ``TransformPipeline`` instances in Chunk 4.
    config:
        Config object forwarded verbatim to ``CompressionDecision.decide``.
        Only ``config.optimize: bool`` is read there.
    usage_reporter:
        Commercial gate forwarded to ``CompressionDecision.decide``.
        ``None`` means no licensing → always allow compression.
    salt:
        Salt bytes for session key derivation (kept for future CCR
        proactive-expansion wiring; not consumed in Chunk 2).
    """

    def __init__(
        self,
        *,
        pipelines: Mapping[tuple[Provider, Flavor], CompressionPipeline],
        config: Any,
        usage_reporter: Any | None,
        salt: bytes,
    ) -> None:
        self._pipelines = dict(pipelines)
        self._config = config
        self._usage_reporter = usage_reporter
        self._salt = salt

    # ── Request hook ──────────────────────────────────────────────────────────

    def on_request(self, ctx: RequestContext) -> RequestDecision:
        """Process an inbound request.

        For registered ``(provider, flavor)`` combos: classify auth mode,
        decide whether to compress, and either return the raw body unchanged
        (passthrough) or run the pipeline and return the mutated body.

        Raises
        ------
        KeyError
            If ``(ctx.provider, ctx.flavor)`` has no registered pipeline.
            Raised loudly; no silent fallback.
        ValueError
            If the raw body cannot be parsed as JSON (malformed request).
        """
        key = (ctx.provider, ctx.flavor)
        if key not in self._pipelines:
            raise KeyError(
                f"No pipeline registered for provider={ctx.provider!r}, "
                f"flavor={ctx.flavor!r}. Register it in the pipelines mapping."
            )

        pipeline = self._pipelines[key]

        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(ctx.raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"on_request: unparseable JSON body for "
                f"provider={ctx.provider!r}, flavor={ctx.flavor!r}: {exc}"
            ) from exc

        messages: list[dict[str, Any]] = body.get("messages") or []
        model: str = body.get("model", "")

        # Classify auth mode (pure, <10us, never raises)
        auth_mode = classify_auth_mode(ctx.headers_view)

        # Decision: should we compress?
        decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=self._config,
            usage_reporter=self._usage_reporter,
            messages=messages,
        )

        if not decision.should_compress:
            # Return raw body BYTE-IDENTICAL — same object, no re-serialization.
            # This is load-bearing for prefix-cache safety.
            return RequestDecision(
                body=ctx.raw_body,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # Resolve per-auth-mode compression policy
        policy = resolve_policy(auth_mode)

        # Delegate to the injected pipeline
        result = pipeline.apply(
            messages,
            model,
            compression_policy=policy,
        )

        # Reconstruct body with compressed messages
        body["messages"] = result.messages
        compressed_bytes = json.dumps(body).encode()

        bytes_saved = max(0, len(ctx.raw_body) - len(compressed_bytes))
        tokens_in = getattr(result, "tokens_before", 0)
        tokens_out = getattr(result, "tokens_after", 0)

        return RequestDecision(
            body=compressed_bytes,
            telemetry=ResponseTelemetry(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                bytes_saved=bytes_saved,
                compressed=True,
                ccr_fired=False,
            ),
        )

    # ── Response hooks (Chunk 2 stubs — Chunk 3+ extends these) ─────────────

    def on_response(self, ctx: RequestContext, raw_response: bytes) -> bytes:
        """Forward the upstream response unchanged.

        Chunk 3 will extend this with CCR proactive-expansion injection and
        token telemetry parsing.
        """
        return raw_response

    def on_response_chunk(self, sc: StreamContext, chunk: bytes) -> bytes:
        """Forward a streaming chunk unchanged.

        Chunk 3 will add SSE parsing for streaming token telemetry.
        """
        return chunk

    def on_response_end(self, sc: StreamContext, outcome: Any) -> ResponseTelemetry:
        """Finalize a streaming session and return its telemetry.

        Safe to call on normal completion OR abort (``outcome`` may be an
        Exception or ``None``).  Chunk 3 will accumulate streaming token
        counts here.
        """
        return ResponseTelemetry()
