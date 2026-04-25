from scripts.benchmark_openreward import summarize_policy_comparison


def test_policy_comparison_reports_lifts_and_claim_boundary():
    comparison = summarize_policy_comparison(
        baseline_name="reactive_script",
        improved_name="dense_trained",
        baseline_metrics={"success_rate": 0.35, "mean_reward": 1.2, "mean_tool_calls": 12.0},
        improved_metrics={"success_rate": 0.75, "mean_reward": 2.8, "mean_tool_calls": 9.0},
        gemini_metrics={"mean_confidence": 0.81, "agreement_rate": 0.68},
    )

    assert comparison["success_lift"] == 0.4
    assert comparison["mean_reward_lift"] == 1.6
    assert comparison["tool_call_delta"] == -3.0
    assert comparison["gemini_vision"]["mean_confidence"] == 0.81
    assert "macro-policy" in comparison["claim_boundary"]


def test_policy_comparison_reports_total_tool_call_volume():
    comparison = summarize_policy_comparison(
        baseline_name="reactive_script",
        improved_name="expert",
        baseline_metrics={"success_rate": 0.4, "mean_reward": 1.0, "mean_tool_calls": 55.0, "total_tool_calls": 110},
        improved_metrics={"success_rate": 0.8, "mean_reward": 2.0, "mean_tool_calls": 48.0, "total_tool_calls": 96},
    )

    assert comparison["baseline_total_tool_calls"] == 110
    assert comparison["improved_total_tool_calls"] == 96
    assert comparison["aggregate_tool_calls"] == 206
