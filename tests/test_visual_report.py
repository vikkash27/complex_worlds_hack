from pathlib import Path

from scripts.build_visual_report import build_visual_report


def test_visual_report_embeds_core_metrics(tmp_path):
    output = tmp_path / "index.html"

    build_visual_report(Path("artifacts"), output)

    html = output.read_text(encoding="utf-8")
    assert "RoboCerebra Reward Lab" in html
    assert "progress_lift" in html
    assert "baseline_random.gif" in html
    assert "dense_trained.gif" in html
