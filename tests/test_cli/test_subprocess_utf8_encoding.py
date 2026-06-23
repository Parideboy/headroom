"""Guard: every text-mode subprocess call in the shipped package pins encoding.

On Windows, text-mode ``subprocess`` defaults to the locale codec (cp1252) when
``encoding=`` is omitted. Child output that is UTF-8 (e.g. a repo index printing
symbol names with ``↔``/``—``) then raises ``UnicodeDecodeError: 'charmap'`` in the
reader thread and aborts startup. The fix is ``encoding="utf-8", errors="replace"``.

This can't be reproduced on a UTF-8 CI box, so we assert the invariant on the
source instead: a `subprocess` call passing ``text=True`` (or
``universal_newlines=True``) must also pass ``encoding=``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "headroom"
_SUBPROCESS_FUNCS = {"run", "Popen", "check_output", "check_call", "call"}


def _kwarg(call: ast.Call, name: str) -> ast.keyword | None:
    return next((k for k in call.keywords if k.arg == name), None)


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_subprocess_call(call: ast.Call) -> bool:
    func = call.func
    # subprocess.run(...) / sp.run(...) — match on the attribute name.
    return isinstance(func, ast.Attribute) and func.attr in _SUBPROCESS_FUNCS


def _offenders() -> list[str]:
    bad: list[str] = []
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_subprocess_call(node):
                continue
            text_kw = _kwarg(node, "text")
            un_kw = _kwarg(node, "universal_newlines")
            text_mode = (text_kw is not None and _is_true(text_kw.value)) or (
                un_kw is not None and _is_true(un_kw.value)
            )
            if text_mode and _kwarg(node, "encoding") is None:
                rel = path.relative_to(_PACKAGE.parent)
                bad.append(f"{rel}:{node.lineno}")
    return bad


def test_text_mode_subprocess_calls_pin_encoding() -> None:
    offenders = _offenders()
    assert not offenders, (
        "text-mode subprocess calls missing encoding= (Windows cp1252 decode crash):\n"
        + "\n".join(offenders)
    )


if __name__ == "__main__":  # pragma: no cover - manual run
    test_text_mode_subprocess_calls_pin_encoding()
    print("ok: all text-mode subprocess calls pin encoding")
