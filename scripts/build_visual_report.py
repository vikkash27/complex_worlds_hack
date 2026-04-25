from __future__ import annotations

import html
import json
from pathlib import Path


def build_visual_report(artifacts_dir: Path, output_path: Path) -> None:
    metrics_path = artifacts_dir / "metrics" / "leaderboard.json"
    leaderboard = json.loads(metrics_path.read_text(encoding="utf-8"))
    randomized = leaderboard.get("randomized_heldout", {})
    baseline_metrics = randomized.get(
        "reactive_script",
        randomized.get("fixed_script", leaderboard.get("random_baseline", {})),
    )
    trained_metrics = randomized.get("dense_trained", leaderboard.get("dense_trained_policy", {}))
    headline = leaderboard["headline"]
    trace_preview = _trace_preview(artifacts_dir / "traces" / "dense_trained.jsonl")

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboCerebra Reward Lab Demo</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f7f5; color: #171717; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ font-size: 30px; margin-bottom: 6px; }}
    h2 {{ margin-top: 34px; }}
    .subtitle {{ color: #555; margin-top: 0; max-width: 820px; line-height: 1.5; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 24px 0; }}
    .stat {{ background: white; border: 1px solid #d9d9d4; border-radius: 12px; padding: 18px; }}
    .stat strong {{ display: block; font-size: 28px; margin-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .panel {{ background: white; border: 1px solid #d9d9d4; border-radius: 12px; padding: 18px; }}
    img {{ max-width: 100%; border: 1px solid #d9d9d4; border-radius: 10px; background: white; }}
    code {{ background: #ececea; padding: 2px 5px; border-radius: 5px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e3e3df; padding: 9px 7px; text-align: left; }}
    th {{ color: #555; font-weight: 600; }}
    @media (max-width: 800px) {{ .stats, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>RoboCerebra Reward Lab</h1>
  <p class="subtitle">Dense Gemini-style subgoal rewards are evaluated on randomized held-out physical-AI scenes. The old deterministic 100% result is now treated as a smoke test; the headline below compares the dense policy against a stronger fixed-script baseline under distractors, action failures, and disturbance variations.</p>

  <section class="stats">
    <div class="stat"><strong>{headline["progress_lift"]:+.3f}</strong><span>progress_lift</span></div>
    <div class="stat"><strong>{headline["success_lift"]:+.3f}</strong><span>success_lift</span></div>
    <div class="stat"><strong>{headline["reward_auc_lift"]:+.3f}</strong><span>reward_auc_lift</span></div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Reactive-Script Baseline</h2>
      <p>Success rate: <strong>{baseline_metrics["success_rate"]:.1%}</strong><br />
      Mean progress: <strong>{baseline_metrics["mean_progress"]:.1%}</strong><br />
      Disturbance recovery: <strong>{baseline_metrics["disturbance_recovery_rate"]:.1%}</strong><br />
      Tool calls: <strong>{baseline_metrics.get("mean_tool_calls", 0):.1f}</strong></p>
      <img src="../replays/baseline_random.gif" alt="Random baseline replay" />
    </div>
    <div class="panel">
      <h2>Dense Reward Policy</h2>
      <p>Success rate: <strong>{trained_metrics["success_rate"]:.1%}</strong><br />
      Mean progress: <strong>{trained_metrics["mean_progress"]:.1%}</strong><br />
      Disturbance recovery: <strong>{trained_metrics["disturbance_recovery_rate"]:.1%}</strong><br />
      Tool calls: <strong>{trained_metrics.get("mean_tool_calls", 0):.1f}</strong></p>
      <img src="../replays/dense_trained.gif" alt="Dense trained replay" />
    </div>
  </section>

  <h2>Training Curve</h2>
  <img src="../plots/training_curve.png" alt="Training curve" />

  <h2>Side-by-Side Replay</h2>
  <img src="../replays/side_by_side_before_after.gif" alt="Side-by-side before and after replay" />

  <h2>Tool-Call Trace Preview</h2>
  <div class="panel">
    <pre>{html.escape(trace_preview)}</pre>
  </div>

  <h2>Leaderboard JSON</h2>
  <div class="panel">
    <pre>{html.escape(json.dumps(leaderboard, indent=2, sort_keys=True))}</pre>
  </div>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _trace_preview(path: Path, max_lines: int = 8) -> str:
    if not path.exists():
        return "Trace not generated yet. Run scripts/run_demo.py first."
    lines = path.read_text(encoding="utf-8").splitlines()[:max_lines]
    pretty = []
    for line in lines:
        try:
            pretty.append(json.dumps(json.loads(line), indent=2, sort_keys=True))
        except json.JSONDecodeError:
            pretty.append(line)
    return "\n\n".join(pretty)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build_visual_report(root / "artifacts", root / "artifacts" / "visual_report" / "index.html")
    print(root / "artifacts" / "visual_report" / "index.html")


if __name__ == "__main__":
    main()
