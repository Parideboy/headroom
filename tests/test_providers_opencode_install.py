"""Tests for OpenCode install-time helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.install.models import ConfigScope, DeploymentManifest
from headroom.providers.opencode.config import HEADROOM_OPENCODE_MODELS
from headroom.providers.opencode.install import (
    apply_provider_scope,
    build_install_env,
    revert_provider_scope,
)


def _manifest(port: int = 8787) -> DeploymentManifest:
    return DeploymentManifest(
        profile="test",
        preset="persistent-task",
        runtime_kind="python",
        supervisor_kind="none",
        scope=ConfigScope.PROVIDER.value,
        provider_mode="auto",
        targets=[],
        port=port,
        host="127.0.0.1",
        backend="anthropic",
        proxy_args=[],
        base_env={},
        tool_envs={},
    )


def test_build_install_env() -> None:
    """build_install_env leaves OpenCode provider env vars untouched."""
    env = build_install_env(port=8787, backend="anthropic")
    assert env == {}


def test_apply_provider_scope_creates_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_provider_scope creates the opencode config with headroom provider."""
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

    manifest = _manifest(port=8787)
    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    assert mutation.target == "opencode"
    assert mutation.kind == "json-block"

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    assert config_file.exists()
    import json

    config = json.loads(config_file.read_text())
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
    assert "mcp" not in config


def test_apply_provider_scope_skips_when_scope_is_not_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_provider_scope returns None when scope is not PROVIDER."""
    manifest = _manifest()
    manifest.scope = ConfigScope.USER.value
    result = apply_provider_scope(manifest)
    assert result is None


def test_revert_provider_scope_restores_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """revert_provider_scope strips the Headroom block from the config."""
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('{"model": "openai/gpt-4o"}')

    from headroom.install.models import ManagedMutation

    mutation = ManagedMutation(
        target="opencode",
        kind="json-block",
        path=str(config_file),
    )
    manifest = _manifest()
    revert_provider_scope(mutation, manifest)
    assert config_file.exists()
    assert config_file.read_text().strip() == '{"model": "openai/gpt-4o"}'


def test_revert_provider_scope_noop_when_file_missing(
    tmp_path: Path,
) -> None:
    """revert_provider_scope is a safe no-op when the config file is gone."""
    from headroom.install.models import ManagedMutation

    mutation = ManagedMutation(
        target="opencode",
        kind="json-block",
        path=str(tmp_path / "nonexistent.json"),
    )
    manifest = _manifest()
    revert_provider_scope(mutation, manifest)
    # Should not raise


def _opencode_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    return tmp_path / ".config" / "opencode" / "opencode.json"


def _headroom_models(config_file: Path) -> dict:
    return json.loads(config_file.read_text())["provider"]["headroom"]["models"]


def test_apply_provider_scope_lists_default_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenCode only resolves headroom/<id> for listed ids, so install lists them.

    Extra models come from the manifest only: a scheduled restart does not run
    in the shell that ran ``install apply``.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    monkeypatch.setenv("HEADROOM_OPENCODE_EXTRA_MODELS", "from-shell")

    apply_provider_scope(_manifest())

    assert _headroom_models(config_file) == HEADROOM_OPENCODE_MODELS


def test_apply_provider_scope_adds_manifest_extra_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HEADROOM_OPENCODE_EXTRA_MODELS from ``install apply --env`` is written."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    manifest = _manifest()
    manifest.base_env = {
        "HEADROOM_OPENCODE_EXTRA_MODELS": "deepseek-chat=DeepSeek Chat=65536,qwen2.5-coder:7b"
    }

    apply_provider_scope(manifest)

    models = _headroom_models(config_file)
    assert models["deepseek-chat"] == {
        "name": "DeepSeek Chat",
        "limit": {"context": 65536, "output": 16384},
    }
    assert models["qwen2.5-coder:7b"]["name"] == "qwen2.5-coder:7b"
    assert "gpt-4o" in models


@pytest.mark.parametrize("prior_config", [True, False], ids=["with-backup", "no-prior-config"])
def test_reapply_keeps_manifest_extra_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_config: bool
) -> None:
    """Re-running install apply reverts the old deployment, then applies (#3970).

    The revert restores the pre-install backup when there is one, and the apply
    replaces the whole ``headroom`` provider, so the models must come back from
    the manifest.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    if prior_config:
        config_file.parent.mkdir(parents=True)
        config_file.write_text('{"theme": "system"}')
    manifest = _manifest()
    manifest.base_env = {"HEADROOM_OPENCODE_EXTRA_MODELS": "deepseek-chat"}

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest)
    apply_provider_scope(manifest)

    models = _headroom_models(config_file)
    assert "deepseek-chat" in models
    assert "gpt-4o" in models
