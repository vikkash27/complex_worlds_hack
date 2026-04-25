from scripts import run_demo


def test_sample_transition_uses_symbolic_fallback_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ROBOCEREBRA_FORCE_SYMBOLIC_VLM", raising=False)

    score = run_demo.score_sample_transition(tmp_path / "cache.json")

    assert (tmp_path / "gemini_demo_frame.png").exists()
    assert "Cached symbolic fallback" in score["rationale"]
