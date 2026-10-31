"""#4548: reasoning dropped on a cold turn stays dropped on the next warm turn."""

from types import SimpleNamespace

from headroom.transforms.thinking_compactor import compact_reasoning_openai_chat


class _HalfKompress:
    def compress(self, text, allow_download=False):
        words = text.split()
        return SimpleNamespace(compressed=" ".join(words[: len(words) // 2]))


def _history(n, tag):
    msgs = []
    for k in range(n):
        msgs.append({"role": "user", "content": f"q{k}"})
        reasoning = " ".join(f"{tag}{k}w{j}" for j in range(60))
        msgs.append({"role": "assistant", "content": f"a{k}", "reasoning_content": reasoning})
    return msgs


def test_cold_drop_reports_dropped_upto():
    out, stats = compact_reasoning_openai_chat(
        _history(3, "c"), kompress=None, keep_last_turns=1, drop=True
    )
    assert [m.get("reasoning_content") for m in out[1::2]][:2] == ["", ""]
    assert stats["dropped_upto"] == 2


def test_warm_turn_after_cold_reuses_cold_prefix():
    cold, stats = compact_reasoning_openai_chat(
        _history(3, "w"), kompress=None, keep_last_turns=1, drop=True
    )
    warm, _ = compact_reasoning_openai_chat(
        _history(4, "w"),
        kompress=_HalfKompress(),
        keep_last_turns=1,
        drop=False,
        drop_first_turns=stats["dropped_upto"],
    )
    assert warm[:4] == cold[:4]  # previously dropped turns byte-identical
    assert 0 < len(warm[5]["reasoning_content"].split()) < 60  # newer turn Kompressed
    assert len(warm[7]["reasoning_content"].split()) == 60  # active turn kept


def test_no_sticky_floor_is_unchanged():
    out, stats = compact_reasoning_openai_chat(
        _history(3, "u"), kompress=_HalfKompress(), keep_last_turns=1
    )
    assert all(len(m["reasoning_content"].split()) == 30 for m in out[1:4:2])
    assert stats["dropped_upto"] == 0


def test_inline_think_spans_stay_dropped():
    think = "<think>" + " ".join(f"t{j}" for j in range(60)) + "</think>"
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": think + "a0"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": think + "a1"},
    ]
    out, _ = compact_reasoning_openai_chat(
        msgs, kompress=_HalfKompress(), keep_last_turns=1, drop_first_turns=1
    )
    assert out[1]["content"] == "a0"
    assert out[3]["content"] == think + "a1"
