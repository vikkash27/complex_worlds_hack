from scripts import run_demo


def test_sample_transition_uses_symbolic_fallback_unless_gemini_vision_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOCEREBRA_USE_GEMINI_VISION", raising=False)

    def fail_if_called():
        raise AssertionError("Gemini scorer should be opt-in for reproducible demo artifact generation")

    monkeypatch.setattr(run_demo, "gemini_reward_scorer", fail_if_called)

    score = run_demo.score_sample_transition(tmp_path / "cache.json")

    assert "Cached symbolic fallback" in score["rationale"]
