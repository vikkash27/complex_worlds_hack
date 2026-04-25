"""Compare extended (D+B) RoboCerebra vs old single-tray baseline.

Prints per-policy metrics under both regimes and computes numerical lifts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.eval import evaluate_policy, randomized_world  # noqa: E402
from robocerebra_rl.train import train_tabular_policy  # noqa: E402
from robocerebra_rl.world import BreakfastTrayWorld, SceneConfig, iter_policy_actions  # noqa: E402


def baseline_world(seed: int) -> BreakfastTrayWorld:
    """Old single-tray, single-disturbance regime (tray_count=1, default schedule)."""
    scene = SceneConfig.from_seed(seed)
    if seed >= 1000:
        scene = SceneConfig(
            mug_position=scene.mug_position,
            snack_position=scene.snack_position,
            tray_position=scene.tray_position,
            disturbance_tick=scene.disturbance_tick,
            distractor_count=max(2, scene.distractor_count),
            action_failure_prob=max(0.18, scene.action_failure_prob),
            disturbance_severity=max(0.7, scene.disturbance_severity),
        )
    return BreakfastTrayWorld(
        seed=seed,
        horizon_ticks=1000 + (seed % 3) * 250,
        max_macro_steps=18 if seed >= 1000 else 30,
        tray_count=1,
        disturbances_per_tray=(1,),
        scene=scene,
    )


def evaluate_world_factory(world_factory, policy, episodes: int, seed: int) -> dict:
    from robocerebra_rl.rewards import symbolic_dense_reward
    successes, progresses, rewards, ticks, recoveries, calls = [], [], [], [], [], []
    for i in range(episodes):
        w = world_factory(seed + i)
        total = 0.0
        while not w.done:
            a = iter_policy_actions(policy, w)
            t = w.step(a)
            total += symbolic_dense_reward(t)
        successes.append(1.0 if w.success else 0.0)
        progresses.append(w.progress_fraction)
        rewards.append(total)
        ticks.append(w.ticks)
        denom = sum(w.task.disturbances_per_tray) or 1
        recoveries.append(w.disturbances_recovered_total / denom)
        calls.append(w.macro_steps)

    def stats(xs):
        return {"mean": round(mean(xs), 4), "std": round(stdev(xs) if len(xs) > 1 else 0.0, 4)}

    return {
        "success_rate": round(mean(successes), 4),
        "progress": stats(progresses),
        "reward": stats(rewards),
        "ticks": stats(ticks),
        "recovery": stats(recoveries),
        "tool_calls": stats(calls),
    }


def main() -> None:
    EPISODES = 80
    EVAL_SEED = 2000

    print("Training dense+sparse on extended (D+B) regime...")
    dense_ext, _ = train_tabular_policy(episodes=200, seed=13, reward_mode="dense", randomized=True)
    sparse_ext, _ = train_tabular_policy(episodes=200, seed=13, reward_mode="sparse", randomized=True)

    print("Training dense+sparse on baseline (single-tray) regime...")

    def baseline_train(reward_mode: str):
        from robocerebra_rl.train import TabularPolicy, ensure_state, policy_state, ACTIONS
        from robocerebra_rl.rewards import symbolic_dense_reward, sparse_success_reward
        import random as _r
        rng = _r.Random(13)
        policy = TabularPolicy()
        for ep in range(200):
            w = baseline_world(13 + ep)
            eps_e = max(0.05, 0.75 * (1.0 - ep / 199))
            while not w.done:
                s = policy_state(w)
                vals = ensure_state(policy, s)
                if rng.random() < eps_e:
                    a = rng.choice(ACTIONS)
                else:
                    a = max(ACTIONS, key=lambda c: vals[c])
                tr = w.step(a)
                r = symbolic_dense_reward(tr) if reward_mode == "dense" else sparse_success_reward(tr)
                ns = policy_state(w)
                nv = ensure_state(policy, ns)
                vals[a] = vals[a] + 0.4 * (r + 0.8 * max(nv.values()) - vals[a])
        return policy

    dense_base = baseline_train("dense")
    sparse_base = baseline_train("sparse")

    print(f"\nEvaluating policies @ {EPISODES} episodes each...\n")

    regimes = {
        "baseline_single_tray": baseline_world,
        "extended_multi_tray_density": randomized_world,
    }
    policies = {
        "expert": "expert",
        "random": "random",
        "fixed_script": "fixed_script",
        "reactive_script": "reactive_script",
    }

    results: dict[str, dict] = {}
    for regime_name, factory in regimes.items():
        results[regime_name] = {}
        for pname, p in policies.items():
            results[regime_name][pname] = evaluate_world_factory(factory, p, EPISODES, EVAL_SEED)
        # trained policies per regime
        if regime_name == "baseline_single_tray":
            results[regime_name]["dense_trained"] = evaluate_world_factory(factory, dense_base, EPISODES, EVAL_SEED)
            results[regime_name]["sparse_trained"] = evaluate_world_factory(factory, sparse_base, EPISODES, EVAL_SEED)
        else:
            results[regime_name]["dense_trained"] = evaluate_world_factory(factory, dense_ext, EPISODES, EVAL_SEED)
            results[regime_name]["sparse_trained"] = evaluate_world_factory(factory, sparse_ext, EPISODES, EVAL_SEED)

    print("=" * 92)
    print(f"{'regime':<28} {'policy':<18} {'success':>8} {'progress':>9} {'reward':>9} {'calls':>7} {'ticks':>8} {'recov':>7}")
    print("=" * 92)
    for regime_name, by_policy in results.items():
        for pname, m in by_policy.items():
            print(
                f"{regime_name:<28} {pname:<18} "
                f"{m['success_rate']:>8.3f} "
                f"{m['progress']['mean']:>9.3f} "
                f"{m['reward']['mean']:>9.2f} "
                f"{m['tool_calls']['mean']:>7.1f} "
                f"{m['ticks']['mean']:>8.0f} "
                f"{m['recovery']['mean']:>7.3f}"
            )
        print("-" * 92)

    print("\n=== Numerical Improvement: Extended vs Baseline ===\n")
    base = results["baseline_single_tray"]
    ext = results["extended_multi_tray_density"]

    def lift(field_path, pname):
        b = base[pname]
        e = ext[pname]
        for k in field_path.split("."):
            b = b[k]
            e = e[k]
        return e, b, e - b, (e / b if b not in (0, 0.0) else float("inf"))

    print(f"{'metric':<35} {'policy':<14} {'baseline':>10} {'extended':>10} {'delta':>10} {'ratio':>10}")
    print("-" * 92)
    for metric_path in ["tool_calls.mean", "ticks.mean", "reward.mean", "recovery.mean"]:
        for pname in ["dense_trained", "expert", "random", "fixed_script", "reactive_script"]:
            e, b, d, r = lift(metric_path, pname)
            r_str = f"{r:>10.2f}x" if r != float("inf") else f"{'inf':>10}"
            print(f"{metric_path:<35} {pname:<14} {b:>10.2f} {e:>10.2f} {d:>+10.2f} {r_str}")
        print()

    print("=== Headline gap: dense_trained vs strongest scripted (reactive_script) ===\n")
    for regime_name, by_policy in results.items():
        d = by_policy["dense_trained"]
        b = by_policy["reactive_script"]
        gap_success = d["success_rate"] - b["success_rate"]
        gap_progress = d["progress"]["mean"] - b["progress"]["mean"]
        gap_reward = d["reward"]["mean"] - b["reward"]["mean"]
        gap_recovery = d["recovery"]["mean"] - b["recovery"]["mean"]
        print(f"{regime_name}:")
        print(f"  success_lift:  {gap_success:+.3f}")
        print(f"  progress_lift: {gap_progress:+.3f}")
        print(f"  reward_lift:   {gap_reward:+.2f}")
        print(f"  recovery_lift: {gap_recovery:+.3f}")
        print()

    out = ROOT / "artifacts" / "metrics" / "extended_vs_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
