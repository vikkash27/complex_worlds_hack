# Isaac Asset Inventory

This project uses a local-first asset strategy for the hackathon demo:

1. Fetch official Unitree assets into `artifacts/isaac/vendor/unitree/`.
2. Validate the local G1 asset before generating the humanoid replay.
3. Use built-in Isaac/Omniverse paths only as non-primary backup candidates.

Candidate categories are encoded in `src/robocerebra_rl/isaac_scene.py`:

- `mobile_base`: Carter or Jackal-style mobile robot assets.
- `manipulator`: Franka or UR10-style manipulator assets.
- `humanoid`: local official Unitree **G1** manifest first, then `/Isaac/Robots/Unitree/G1/g1.usd`, H1, and IsaacSim Humanoid. `scripts/isaac/run_humanoid_replay_in_container.sh` exports `ROBOCEREBRA_HUMANOID_USD` from the validated manifest.
- `kitchen_props`: mug, food box, tray/countertop props.

The humanoid block proxy is disabled by default because the hackathon replay should show the real Unitree G1. Enable `ROBOCEREBRA_HUMANOID_PROXY=1` only for debugging asset failures.
