from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def build_side_by_side(
    baseline_dir: Path,
    trained_dir: Path,
    output_path: Path,
) -> None:
    baseline_frames = sorted(baseline_dir.glob("frame_*.png"))
    trained_frames = sorted(trained_dir.glob("frame_*.png"))
    frame_count = max(len(baseline_frames), len(trained_frames))
    frames: list[Image.Image] = []

    for index in range(frame_count):
        left = Image.open(baseline_frames[min(index, len(baseline_frames) - 1)]).convert("RGB")
        right = Image.open(trained_frames[min(index, len(trained_frames) - 1)]).convert("RGB")
        canvas = Image.new("RGB", (left.width + right.width, left.height + 44), "white")
        canvas.paste(left, (0, 44))
        canvas.paste(right, (left.width, 44))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 14), "Before training: fixed/reactive baseline", fill="black")
        draw.text((left.width + 18, 14), "After training: dense reward policy", fill="black")
        frames.append(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=350, loop=0)


def main() -> None:
    build_side_by_side(
        ROOT / "artifacts" / "replays" / "baseline_frames",
        ROOT / "artifacts" / "replays" / "trained_frames",
        ROOT / "artifacts" / "replays" / "side_by_side_before_after.gif",
    )
    print(ROOT / "artifacts" / "replays" / "side_by_side_before_after.gif")


if __name__ == "__main__":
    main()
