from __future__ import annotations

import html
import json
from pathlib import Path


def build_visual_report(artifacts_dir: Path, output_path: Path) -> None:
    metrics_path = artifacts_dir / "metrics" / "leaderboard.json"
    leaderboard = json.loads(metrics_path.read_text(encoding="utf-8"))
    policies = leaderboard.get("policies", {})
    headline = leaderboard.get("headline", {})
    test_expert = policies.get("test::expert", {})
    test_reactive = policies.get("test::reactive_script", {})
    test_random = policies.get("test::random", {})
    val_expert = policies.get("validation::expert", {})
    train_expert = policies.get("train::expert", {})
    trace_preview = _trace_preview(artifacts_dir / "traces" / "dense_trained.jsonl")

    report = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>RoboCerebra Reward Lab — Shift Mode</title>
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
  <h1>RoboCerebra Reward Lab — Shift Mode</h1>
  <p class=\"subtitle\">
    Long-horizon multi-job hospitality shifts run on persistent inventory,
    memory, a clock, ticket queue, and deterministic non-stationary events.
    Headline below compares the capability-rich expert oracle to a strong
    reactive-script baseline; both run hundreds-to-thousands of OpenReward
    tool calls per episode, but only the expert handles events and
    summarizes memory at the end. The legacy <code>baseline_random</code>
    and <code>dense_trained</code> single-job rollouts are kept as smoke
    tests for the per-job engine; see <code>artifacts/replays/</code>.
  </p>

  <section class=\"stats\">
    <div class=\"stat\"><strong>{int(headline.get('p50_tool_calls_test_expert', 0))}</strong><span>median tool calls (test, expert)</span></div>
    <div class=\"stat\"><strong>{headline.get('success_rate_test_expert', 0):.0%}</strong><span>expert success vs {headline.get('success_rate_test_reactive', 0):.0%} reactive</span></div>
    <div class=\"stat\"><strong>{headline.get('tool_call_factor_expert_vs_reactive', 1.0):.2f}×</strong><span>tool-call ratio expert / reactive</span></div>
  </section>

  <section class=\"grid\">
    <div class=\"panel\">
      <h2>Reactive-Script Baseline (test)</h2>
      <p>Success rate: <strong>{test_reactive.get('success_rate', 0):.0%}</strong><br />
      Mean tool calls: <strong>{test_reactive.get('mean_tool_calls', 0):.1f}</strong><br />
      Events handled: <strong>{test_reactive.get('mean_events_handled', 0)}</strong><br />
      Memory recalls: <strong>{test_reactive.get('mean_memory_recalls', 0)}</strong><br />
      Tool diversity: <strong>{test_reactive.get('mean_tool_diversity', 0)}</strong></p>
      <img src=\"../replays/baseline_random.gif\" alt=\"Random baseline replay\" />
    </div>
    <div class=\"panel\">
      <h2>Expert Oracle (test)</h2>
      <p>Success rate: <strong>{test_expert.get('success_rate', 0):.0%}</strong><br />
      Mean tool calls: <strong>{test_expert.get('mean_tool_calls', 0):.1f}</strong><br />
      Events handled: <strong>{test_expert.get('mean_events_handled', 0)}</strong><br />
      Memory recalls: <strong>{test_expert.get('mean_memory_recalls', 0)}</strong><br />
      Tool diversity: <strong>{test_expert.get('mean_tool_diversity', 0)}</strong></p>
      <img src=\"../replays/dense_trained.gif\" alt=\"Dense trained replay\" />
    </div>
  </section>

  <h2>Per-Split Expert Profile</h2>
  <div class=\"panel\">
    <table>
      <thead><tr><th>Split</th><th>Episodes</th><th>Mean tool calls</th><th>Median</th><th>Min</th><th>Max</th><th>Success</th></tr></thead>
      <tbody>
        <tr><td>train</td><td>{train_expert.get('episodes', 0)}</td><td>{train_expert.get('mean_tool_calls', 0)}</td><td>{train_expert.get('median_tool_calls', 0)}</td><td>{train_expert.get('min_tool_calls', 0)}</td><td>{train_expert.get('max_tool_calls', 0)}</td><td>{train_expert.get('success_rate', 0):.0%}</td></tr>
        <tr><td>validation</td><td>{val_expert.get('episodes', 0)}</td><td>{val_expert.get('mean_tool_calls', 0)}</td><td>{val_expert.get('median_tool_calls', 0)}</td><td>{val_expert.get('min_tool_calls', 0)}</td><td>{val_expert.get('max_tool_calls', 0)}</td><td>{val_expert.get('success_rate', 0):.0%}</td></tr>
        <tr><td>test</td><td>{test_expert.get('episodes', 0)}</td><td>{test_expert.get('mean_tool_calls', 0)}</td><td>{test_expert.get('median_tool_calls', 0)}</td><td>{test_expert.get('min_tool_calls', 0)}</td><td>{test_expert.get('max_tool_calls', 0)}</td><td>{test_expert.get('success_rate', 0):.0%}</td></tr>
      </tbody>
    </table>
  </div>

  <h2>Random Sanity Check (test)</h2>
  <div class=\"panel\">
    <p>Random policy gives up almost immediately and never satisfies the shift
    success contract. Mean tool calls: <strong>{test_random.get('mean_tool_calls', 0):.1f}</strong>;
    success rate: <strong>{test_random.get('success_rate', 0):.0%}</strong>.</p>
  </div>

  <h2>Side-by-Side Replay (legacy per-job demo)</h2>
  <img src=\"../replays/side_by_side_before_after.gif\" alt=\"Side-by-side before and after replay\" />

  <h2>Tool-Call Trace Preview</h2>
  <div class=\"panel\">
    <pre>{html.escape(trace_preview)}</pre>
  </div>

  <h2>Leaderboard JSON</h2>
  <div class=\"panel\">
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
