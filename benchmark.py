"""Small, reproducible performance and policy benchmark for the project.

Examples:
    python benchmark.py
    python benchmark.py --actions 20000 --repeats 7 --episodes 50
    python benchmark.py --skip-agent
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from constants import DT, MAX_CURRENT
from train_agent import MagneticDipoleEnv
from world import create_default_world


def set_currents(simulation, currents):
    for coil, current in zip(simulation.electromagnets, currents):
        coil.current = float(current)


def run_physics(currents, substeps, compiled):
    simulation = create_default_world(currents=np.zeros(6))
    start = time.perf_counter()
    for action_currents in currents:
        set_currents(simulation, action_currents)
        if compiled:
            simulation.step_many(DT, substeps)
        else:
            for _ in range(substeps):
                simulation.step(DT)
    elapsed = time.perf_counter() - start
    return elapsed, simulation


def benchmark_physics(actions, substeps, repeats, seed):
    rng = np.random.default_rng(seed)
    currents = rng.uniform(-MAX_CURRENT, MAX_CURRENT, size=(actions, 6))

    # Trigger JIT compilation before timing it.
    warmup = create_default_world(currents=np.zeros(6))
    warmup.step_many(DT, substeps)

    python_times = []
    compiled_times = []
    for _ in range(repeats):
        elapsed, _ = run_physics(currents, substeps, compiled=False)
        python_times.append(elapsed)
        elapsed, _ = run_physics(currents, substeps, compiled=True)
        compiled_times.append(elapsed)

    # Check both paths on a shorter identical trajectory. Chaotic long-running
    # trajectories can diverge after tiny floating-point differences.
    check_currents = currents[: min(actions, 100)]
    _, python_sim = run_physics(check_currents, substeps, compiled=False)
    _, compiled_sim = run_physics(check_currents, substeps, compiled=True)
    state_errors = {
        "position": float(
            np.max(np.abs(python_sim.object.position - compiled_sim.object.position))
        ),
        "velocity": float(
            np.max(np.abs(python_sim.object.velocity - compiled_sim.object.velocity))
        ),
        "orientation": float(
            np.max(
                np.abs(
                    python_sim.object.magnetic_moment_direction
                    - compiled_sim.object.magnetic_moment_direction
                )
            )
        ),
        "angular_velocity": float(
            np.max(
                np.abs(
                    python_sim.object.angular_velocity
                    - compiled_sim.object.angular_velocity
                )
            )
        ),
    }

    python_median = statistics.median(python_times)
    compiled_median = statistics.median(compiled_times)
    physics_steps = actions * substeps
    return {
        "actions": actions,
        "substeps_per_action": substeps,
        "physics_steps": physics_steps,
        "repeats": repeats,
        "python_seconds_median": python_median,
        "compiled_seconds_median": compiled_median,
        "python_physics_steps_per_second": physics_steps / python_median,
        "compiled_physics_steps_per_second": physics_steps / compiled_median,
        "speedup": python_median / compiled_median,
        "max_state_error_after_check": state_errors,
    }


def benchmark_environment(actions, seed):
    env = MagneticDipoleEnv(max_steps=actions + 1)
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    action_values = rng.uniform(-1.0, 1.0, size=(actions, 6)).astype(np.float32)

    # Warm the compiled kernel, then reset so timing starts from a clean state.
    env.step(np.zeros(6, dtype=np.float32))
    observation, _ = env.reset(seed=seed)
    start = time.perf_counter()
    completed = 0
    for action in action_values:
        observation, _, terminated, truncated, _ = env.step(action)
        completed += 1
        if terminated or truncated:
            observation, _ = env.reset(seed=seed + completed)
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "actions": completed,
        "seconds": elapsed,
        "environment_steps_per_second": completed / elapsed,
        "simulated_seconds_per_wall_second": completed
        * env.action_time
        / elapsed,
    }


def evaluate_agent(model_path, episodes, seed):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    policy_results = []
    baseline_results = []

    for use_policy, results in ((True, policy_results), (False, baseline_results)):
        env = MagneticDipoleEnv()
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            initial_distance = info["distance"]
            terminated = truncated = False
            while not (terminated or truncated):
                if use_policy:
                    action, _ = model.predict(observation, deterministic=True)
                else:
                    action = np.zeros(6, dtype=np.float32)
                observation, _, terminated, truncated, info = env.step(action)
            results.append(
                {
                    "success": info["success"],
                    "initial_distance": initial_distance,
                    "final_distance": info["distance"],
                    "wall_contact": info["wall_contact"],
                }
            )
        env.close()

    def summarize(results):
        return {
            "episodes": len(results),
            "success_rate": statistics.mean(r["success"] for r in results),
            "median_initial_distance_m": statistics.median(
                r["initial_distance"] for r in results
            ),
            "median_final_distance_m": statistics.median(
                r["final_distance"] for r in results
            ),
            "episodes_ending_at_wall": sum(r["wall_contact"] for r in results),
        }

    return {
        "checkpoint": str(model_path),
        "training_timesteps": int(model.num_timesteps),
        "policy": summarize(policy_results),
        "zero_current_baseline": summarize(baseline_results),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=int, default=5_000)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--model", type=Path, default=Path("magnetic_dipole_realistic_ppo.zip")
    )
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--json", type=Path, help="optionally save raw results")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.actions <= 0 or args.substeps <= 0 or args.repeats <= 0:
        raise SystemExit("actions, substeps, and repeats must be positive")

    results = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "numpy": np.__version__,
        },
        "physics": benchmark_physics(
            args.actions, args.substeps, args.repeats, args.seed
        ),
        "environment": benchmark_environment(args.actions, args.seed),
    }
    if not args.skip_agent and args.model.exists():
        results["agent"] = evaluate_agent(args.model, args.episodes, args.seed)
    elif not args.skip_agent:
        results["agent"] = {"skipped": f"checkpoint not found: {args.model}"}

    output = json.dumps(results, indent=2)
    print(output)
    if args.json:
        args.json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
