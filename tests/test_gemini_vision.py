import google.genai
import pytest

from robocerebra_rl.rewards import build_gemini_reward_contents, gemini_reward_scorer, parse_gemini_score


def test_gemini_reward_contents_include_image_bytes_when_available(tmp_path):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    contents = build_gemini_reward_contents(
        {
            "task_id": "countertop-cleanup-1",
            "subgoal": "sort_recyclables",
            "action": "sort_recyclables",
            "progress_delta": 0.25,
            "image_path": str(image_path),
        }
    )

    assert contents["image"] is not None
    assert contents["image"]["mime_type"] == "image/png"
    assert contents["image"]["bytes"] == image_path.read_bytes()
    assert "strict JSON" in contents["prompt"]


def test_parse_gemini_score_recovers_json_wrapped_in_markdown():
    score = parse_gemini_score(
        '```json\n{"progress_delta": 0.5, "subgoal_complete": true, '
        '"irreversible_error": false, "confidence": 0.82, "rationale": "Mug reached tray."}\n```',
        fallback_progress_delta=0.0,
    )

    assert score["progress_delta"] == 0.5
    assert score["subgoal_complete"] is True
    assert score["confidence"] == 0.82


def test_gemini_reward_scorer_without_api_key_is_symbolic(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    scorer = gemini_reward_scorer()
    out = scorer(
        {
            "task_id": "breakfast-tray-1",
            "state_hash": "abc",
            "subgoal": "locate_items",
            "action": "locate_items",
            "progress_delta": 0.142857,
            "image_path": None,
        }
    )

    assert "symbolic fallback" in out["rationale"].lower()
    assert out["progress_delta"] == 0.142857


def test_gemini_reward_scorer_with_api_key_calls_genai_client(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    captured: dict = {}

    class FakeModels:
        def generate_content(self, model, contents):
            captured["model"] = model
            captured["contents_len"] = len(contents)

            class FakeResponse:
                text = (
                    '{"progress_delta": 0.41, "subgoal_complete": true, '
                    '"irreversible_error": false, "confidence": 0.87, "rationale": "Tray visible."}'
                )

            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str = ""):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(google.genai, "Client", FakeClient)

    png = tmp_path / "frame.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    scorer = gemini_reward_scorer(model="gemini-test-model")
    out = scorer(
        {
            "task_id": "breakfast-tray-1",
            "state_hash": "deadbeef",
            "subgoal": "locate_items",
            "action": "locate_items",
            "progress_delta": 0.1,
            "image_path": str(png),
        }
    )

    assert captured["api_key"] == "test-api-key"
    assert captured["model"] == "gemini-test-model"
    assert captured["contents_len"] >= 2
    assert out["progress_delta"] == 0.41
    assert out["subgoal_complete"] is True
    assert out["confidence"] == 0.87


@pytest.mark.asyncio
async def test_env_uses_gemini_scorer_when_flag_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCEREBRA_USE_GEMINI_VISION", "1")
    monkeypatch.setenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOCEREBRA_REWARD_CACHE", str(tmp_path / "vlm_cache.json"))
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    from robocerebra_rl.env import ExecuteSkillInput, RoboCerebraRewardLabEnv, ScoreProgressInput

    def stub_scorer(payload):
        return {
            "progress_delta": 0.99,
            "subgoal_complete": True,
            "irreversible_error": False,
            "confidence": 0.66,
            "rationale": "Stub VLM: integration path exercised.",
        }

    monkeypatch.setattr("robocerebra_rl.env.gemini_reward_scorer", lambda model=None: stub_scorer)

    env = RoboCerebraRewardLabEnv(RoboCerebraRewardLabEnv.list_tasks("train")[0])
    await env.execute_skill(ExecuteSkillInput(action="inspect_scene"))
    await env.execute_skill(ExecuteSkillInput(action="locate_items"))
    result = await env.score_progress(ScoreProgressInput(subgoal="locate_items"))

    assert result.metadata["progress_delta"] == 0.99
    assert result.metadata["confidence"] == 0.66
    assert "Stub VLM" in result.metadata["rationale"]
