from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from robocerebra_rl.world import BreakfastTrayWorld


def render_world(world: BreakfastTrayWorld, path: str | Path | None = None) -> Image.Image:
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((30, 40, 610, 320), outline="black", width=3)
    draw.text((48, 55), f"RoboCerebra Reward Lab: {world.task.label}", fill="black")
    draw.text((48, 82), f"Ticks: {world.ticks}/{world.horizon_ticks}", fill="black")
    draw.text((48, 106), f"Progress: {world.progress_fraction:.0%}", fill="black")
    draw.text((48, 130), f"Next: {world.expected_action}", fill="black")

    x = 52
    y = 190
    for index, subgoal in enumerate(world.task.subgoals):
        completed = index < world.progress_index
        fill = (92, 156, 89) if completed else (220, 220, 220)
        draw.rounded_rectangle((x, y, x + 70, y + 42), radius=8, fill=fill, outline="black")
        draw.text((x + 8, y + 12), str(index + 1), fill="black")
        draw.text((x, y + 50), subgoal.replace("_", "\n"), fill="black")
        x += 82

    tray_fill = (126, 174, 214) if world.success else (245, 245, 245)
    draw.rounded_rectangle((430, 68, 570, 152), radius=12, fill=tray_fill, outline="black")
    draw.text((455, 98), "TRAY", fill="black")
    if world.disturbance_recovered:
        draw.text((438, 124), "bump recovered", fill="black")

    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    return image


def save_replay(frames: list[Image.Image], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
    )
