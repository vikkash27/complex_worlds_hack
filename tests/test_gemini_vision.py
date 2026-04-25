from robocerebra_rl.rewards import build_gemini_reward_contents, parse_gemini_score


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
