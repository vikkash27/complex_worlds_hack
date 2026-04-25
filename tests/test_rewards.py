from robocerebra_rl.rewards import GeminiRewardCache, symbolic_dense_reward
from robocerebra_rl.world import BreakfastTrayWorld


def test_symbolic_dense_reward_values_progress_and_recovery():
    world = BreakfastTrayWorld(seed=11, horizon_ticks=800)

    locate = world.step("locate_items")
    clear = world.step("clear_workspace")

    assert symbolic_dense_reward(locate) > 0.0
    assert symbolic_dense_reward(clear) > symbolic_dense_reward(locate)


def test_gemini_reward_cache_reuses_state_action_scores(tmp_path):
    calls = []

    def scorer(payload):
        calls.append(payload)
        return {
            "progress_delta": 0.25,
            "subgoal_complete": True,
            "irreversible_error": False,
            "confidence": 0.91,
            "rationale": "The mug is now on the tray.",
        }

    cache = GeminiRewardCache(tmp_path / "cache.json", scorer=scorer)

    first = cache.score("task-a", "state-hash", "pick_mug", "pick_mug")
    second = cache.score("task-a", "state-hash", "pick_mug", "pick_mug")

    assert first == second
    assert len(calls) == 1
