"""Regression test: unresolvable model pricing warnings are logged once, not per-call.

See https://github.com/headroomlabs-ai/headroom/issues/2504 — a custom/unknown
model routed through litellm floods proxy.log with an identical WARNING on
every request.
"""

from __future__ import annotations

import logging

from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")


def test_pricing_warning_logged_once_per_model(caplog):
    from headroom.proxy import cost as cost_module
    from headroom.proxy.server import CostTracker

    cost_module._pricing_warned_models.clear()

    ct = CostTracker()
    unknown_model = "totally-unknown-model-xyz"

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for _ in range(5):
            ct.estimate_cost(unknown_model, input_tokens=100, output_tokens=50)

    warnings = [
        r
        for r in caplog.records
        if "Failed to get pricing" in r.message and unknown_model in r.message
    ]
    assert len(warnings) == 1
