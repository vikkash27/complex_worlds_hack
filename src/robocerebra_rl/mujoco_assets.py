from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
TASK_SHOWCASE_SCENE_NAME = "robocerebra_task_showcase.xml"

# scene_with_hands + free-joint task tray (with child mug geom) and weld to right_wrist_yaw_link,
# mocap spill + tote, and contact excludes so the arm does not throw the free body before grasp.
# Weld relpose (pos wxyz) must stay in sync with WELD_REL_* in :mod:`robocerebra_rl.mujoco_pick_place`.
_TASK_EXCLUDE_TRAY = (
    ("rc_task_tray", "right_shoulder_pitch_link"),
    ("rc_task_tray", "right_shoulder_roll_link"),
    ("rc_task_tray", "right_shoulder_yaw_link"),
    ("rc_task_tray", "right_elbow_link"),
    ("rc_task_tray", "right_wrist_roll_link"),
    ("rc_task_tray", "right_wrist_pitch_link"),
    ("rc_task_tray", "right_wrist_yaw_link"),
    ("rc_task_tray", "right_hand_thumb_0_link"),
    ("rc_task_tray", "right_hand_thumb_1_link"),
    ("rc_task_tray", "right_hand_thumb_2_link"),
    ("rc_task_tray", "right_hand_middle_0_link"),
    ("rc_task_tray", "right_hand_middle_1_link"),
    ("rc_task_tray", "right_hand_index_0_link"),
    ("rc_task_tray", "right_hand_index_1_link"),
)
_TASK_EXCLUDE_XML = "\n    ".join(
    f'<exclude body1="{a}" body2="{b}"/>' for a, b in _TASK_EXCLUDE_TRAY
)
_TASK_SHOWCASE_MJCF = f"""<mujoco model="robocerebra_g1_task_showcase">
  <include file="scene_with_hands.xml"/>
  <default>
    <default class="rc_task">
      <geom solimp="0.99 0.99 0.001" solref="0.02 1" condim="3" friction="0.8 0.05 0.001" margin="0.001"/>
    </default>
  </default>
  <asset>
    <material name="rc_wood" rgba="0.52 0.34 0.18 1"/>
    <material name="rc_porcelain" rgba="0.94 0.94 0.97 1"/>
    <material name="rc_spill" rgba="0.15 0.38 0.82 0.88"/>
    <material name="rc_tote" rgba="0.38 0.28 0.2 1"/>
  </asset>
  <worldbody>
    <body name="rc_task_tray" pos="-0.2 0.45 0.9">
      <freejoint name="rc_tray_free"/>
      <inertial pos="0 0 0" quat="1 0 0 0" mass="0.35" diaginertia="0.02 0.02 0.04"/>
      <geom name="rc_tray_surf" class="rc_task" type="box" size="0.12 0.08 0.01" material="rc_wood"/>
      <geom name="rc_mug_on_tray" class="rc_task" type="cylinder" pos="0.1 0.04 0.07" size="0.04 0.04" material="rc_porcelain"/>
    </body>
    <body name="rc_mocap_spill" mocap="true" pos="0 0 0.87">
      <geom name="rc_geom_spill" type="ellipsoid" size="0.1 0.1 0.012" material="rc_spill" contype="0" conaffinity="0"/>
    </body>
    <body name="rc_mocap_tote" mocap="true" pos="-0.4 0.5 0.45">
      <geom name="rc_geom_tote" type="box" size="0.11 0.08 0.12" material="rc_tote" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality>
    <weld name="rc_weld_tray_palm" body1="right_wrist_yaw_link" body2="rc_task_tray"
      active="false" torquescale="1" solref="0.01 0.6" solimp="0.99 0.999 0.0001"
      relpose="0.14 0.02 -0.04 1 0 0 0"/>
  </equality>
  <contact>
    {_TASK_EXCLUDE_XML}
  </contact>
</mujoco>
"""


def write_task_showcase_scene(g1_dir: Path) -> Path:
    """Write the hospitality-artifact mocap scene next to the Menagerie G1 assets."""
    d = g1_dir.expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    out = d / TASK_SHOWCASE_SCENE_NAME
    out.write_text(_TASK_SHOWCASE_MJCF, encoding="utf-8")
    return out


MENAGERIE_G1_SCENE_RELATIVE_PATHS = (
    "mujoco_menagerie/unitree_g1/scene_with_hands.xml",
    "mujoco_menagerie/unitree_g1/scene.xml",
    "unitree_g1/scene_with_hands.xml",
    "unitree_g1/scene.xml",
)


@dataclass(frozen=True)
class MenagerieFetchPlan:
    target_dir: Path
    manifest_path: Path
    commands: tuple[str, ...]


@dataclass(frozen=True)
class MenagerieG1Manifest:
    root: Path
    scene_relative_path: Path

    @property
    def scene_path(self) -> Path:
        return self.root / self.scene_relative_path

    @property
    def manifest_path(self) -> Path:
        return self.root / "menagerie_g1_manifest.json"

    @classmethod
    def from_scene(cls, *, root: Path, scene_path: Path) -> "MenagerieG1Manifest":
        return cls(root=root, scene_relative_path=scene_path.relative_to(root))

    def require_existing_scene(self) -> Path:
        if not self.scene_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo Menagerie Unitree G1 scene not found at {self.scene_path}. "
                "Run scripts/mujoco/fetch_menagerie_g1.py first."
            )
        return self.scene_path

    def ensure_task_showcase_scene(self) -> Path:
        """Path to the wrap scene (Unitree G1 + mocap breakfast/lab props), written beside Menagerie files."""
        self.require_existing_scene()
        return write_task_showcase_scene(self.scene_path.parent)

    def as_json(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "scene_relative_path": self.scene_relative_path.as_posix(),
        }


def build_menagerie_fetch_plan(target_dir: Path) -> MenagerieFetchPlan:
    target = target_dir.expanduser()
    repo_dir = target / "mujoco_menagerie"
    return MenagerieFetchPlan(
        target_dir=target,
        manifest_path=target / "menagerie_g1_manifest.json",
        commands=(
            f"git clone --filter=blob:none --sparse {MENAGERIE_REPO} {repo_dir}",
            f"cd {repo_dir} && git sparse-checkout set unitree_g1",
        ),
    )


def find_menagerie_g1_scene(root: Path) -> Path | None:
    base = root.expanduser()
    for relative in MENAGERIE_G1_SCENE_RELATIVE_PATHS:
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def write_menagerie_g1_manifest(manifest: MenagerieG1Manifest) -> Path:
    manifest.root.mkdir(parents=True, exist_ok=True)
    path = manifest.manifest_path
    path.write_text(json.dumps(manifest.as_json(), indent=2), encoding="utf-8")
    return path


def load_menagerie_g1_manifest(path: Path) -> MenagerieG1Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = MenagerieG1Manifest(
        root=path.parent,
        scene_relative_path=Path(payload["scene_relative_path"]),
    )
    manifest.require_existing_scene()
    return manifest
