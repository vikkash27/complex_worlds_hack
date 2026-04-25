# Isaac Asset Inventory

This project uses a hybrid asset strategy for the hackathon demo:

1. Try built-in Isaac/Omniverse assets from the container or streamed content browser.
2. Fall back to colored USD proxy geometry when those assets are unavailable.

Candidate categories are encoded in `src/robocerebra_rl/isaac_scene.py`:

- `mobile_base`: Carter or Jackal-style mobile robot assets.
- `manipulator`: Franka or UR10-style manipulator assets.
- `humanoid`: Unitree H1 and IsaacSim Humanoid assets.
- `kitchen_props`: mug, food box, tray/countertop props.

The fallback geometry is intentional: it keeps the replay deterministic on CI and on machines where the Isaac content cache differs. The Brev/Isaac smoke test should confirm whether the container can resolve the candidate assets and visually inspect the fallback quality.
