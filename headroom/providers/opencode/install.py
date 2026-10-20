"""OpenCode install-time helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from headroom import fsutil
from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from headroom.install.paths import opencode_config_path

from .config import (
    _inject_key_into_json,
    _parse_json_loose,
    extra_models_from_env,
    headroom_provider_entry,
    migrate_legacy_opencode_jsonc_backup,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)

# Options Headroom owns: the proxy URL always points at the local proxy, so a
# kept user option must never repoint it.
_PROXY_OWNED_OPTIONS = frozenset({"baseURL"})


def _named_upstream(manifest: DeploymentManifest) -> bool:
    """True when the deployment forwards OpenAI-compatible traffic to a named upstream.

    Only then may a third-party credential stay in the OpenCode config: with
    the default upstream the proxy forwards to OpenAI, and a kept DeepSeek key
    would be sent there. Mirrors the wrap path, which keeps the user's entries
    only for an explicit upstream.
    """
    base_env = manifest.base_env or {}
    if (base_env.get("OPENAI_TARGET_API_URL") or "").strip():
        return True
    return "--openai-api-url" in (manifest.proxy_args or [])


def _user_options(entry: Any) -> dict[str, Any]:
    """Return the user-owned options of a ``headroom`` provider entry."""
    if not isinstance(entry, dict):
        return {}
    options = entry.get("options")
    if not isinstance(options, dict):
        return {}
    return {k: v for k, v in options.items() if k not in _PROXY_OWNED_OPTIONS}


def _headroom_entry(data: dict[str, Any]) -> Any:
    """Return the ``headroom`` provider entry of a parsed config."""
    providers = data.get("provider")
    return providers.get("headroom") if isinstance(providers, dict) else None


def _stored_options(manifest: DeploymentManifest) -> dict[str, Any]:
    """Return the provider options kept in the deployment state."""
    store = manifest.provider_options
    if not isinstance(store, dict):
        return {}
    options = store.get(ToolTarget.OPENCODE.value)
    return dict(options) if isinstance(options, dict) else {}


def _store_options(manifest: DeploymentManifest, options: dict[str, Any]) -> None:
    """Keep the provider options in the deployment state."""
    store = manifest.provider_options
    if not isinstance(store, dict):
        store = {}
        manifest.provider_options = store
    store[ToolTarget.OPENCODE.value] = dict(options)


def _capture_options(path: Path, manifest: DeploymentManifest) -> None:
    """Store the credential from the config in the deployment state.

    Called before the provider entry is removed: the config is rewritten
    without it, so a key the user configured would be gone for the next
    activation.
    """
    if not _named_upstream(manifest) or not path.exists():
        return
    try:
        data = _parse_json_loose(fsutil.read_text(path))
    except OSError:
        return
    options = _user_options(_headroom_entry(data))
    if options:
        _store_options(manifest, {**_stored_options(manifest), **options})


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for OpenCode."""
    del backend
    del port
    return {}


def apply_provider_scope(manifest: DeploymentManifest) -> ManagedMutation | None:
    """Apply OpenCode provider-scope configuration when requested."""
    if manifest.scope != ConfigScope.PROVIDER.value:
        return None

    config_file = opencode_config_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file = config_file.with_name(config_file.name + ".headroom-backup")
    migrate_legacy_opencode_jsonc_backup(config_file, backup_file)
    snapshot_opencode_config_if_unwrapped(config_file, backup_file)

    if config_file.exists():
        content = fsutil.read_text(config_file)
        data = _parse_json_loose(content)
    else:
        data = {}

    # Extra models come from the manifest, not the file: every re-apply first
    # reverts (restoring the pre-install backup) and then replaces the whole
    # `headroom` entry, so ids hand-added there do not survive.
    extra_models = extra_models_from_env(manifest.base_env)
    entry = headroom_provider_entry(manifest.port, extra_models)

    # OpenCode sends the provider's apiKey to the proxy, which forwards it to
    # the upstream. The credential belongs to the user, so keep it -- but only
    # when the deployment names the upstream, and only the credential: Headroom
    # owns npm, name, baseURL and models. It is read from the config and from
    # the deployment state, which holds it across the deactivate/activate
    # cycle that rewrites the config without it.
    if _named_upstream(manifest):
        preserved = {**_stored_options(manifest), **_user_options(_headroom_entry(data))}
        if preserved:
            entry["options"] = {**preserved, **entry["options"]}
            _store_options(manifest, preserved)

    provider = {"headroom": entry}
    data = _inject_key_into_json(data, "provider", provider)

    config_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return ManagedMutation(
        target=ToolTarget.OPENCODE.value,
        kind="json-block",
        path=str(config_file),
    )


def restore_opencode_backup(config_path: str, backup_path: str) -> None:
    """Restore the owned OpenCode snapshot, if it still exists."""
    path = Path(config_path)
    backup_file = Path(backup_path)
    migrate_legacy_opencode_jsonc_backup(path, backup_file)
    if backup_file.exists():
        shutil.copy2(backup_file, path)
        backup_file.unlink()


def revert_provider_scope(
    mutation: ManagedMutation,
    manifest: DeploymentManifest,
    *,
    restore_backup: bool = True,
) -> None:
    """Undo OpenCode provider-scope configuration.

    Final removal restores the pre-install snapshot when available. Temporary
    deactivation removes only the managed provider and retains that snapshot.
    """
    if not mutation.path:
        return
    path = Path(mutation.path)
    backup_file = path.with_name(path.name + ".headroom-backup")
    migrate_legacy_opencode_jsonc_backup(path, backup_file)
    # Keep the credential before the provider entry goes away.
    _capture_options(path, manifest)
    if restore_backup and backup_file.exists():
        try:
            restore_opencode_backup(str(path), str(backup_file))
            return
        except OSError:
            pass
    if not path.exists():
        return
    content = fsutil.read_text(path)
    data = _parse_json_loose(content)
    providers = data.get("provider")
    if isinstance(providers, dict) and "headroom" in providers:
        providers.pop("headroom")
        if not providers:
            data.pop("provider", None)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
        return
    cleaned = strip_opencode_headroom_blocks(content)
    if cleaned != content.strip():
        path.write_text(cleaned + "\n", encoding="utf-8", newline="\n")
    elif not data and not content.strip():
        path.unlink(missing_ok=True)
