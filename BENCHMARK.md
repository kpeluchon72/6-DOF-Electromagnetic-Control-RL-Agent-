# Benchmark results

Measured on November 16, 2026 with `python benchmark.py` on Windows 11,
Python 3.13, NumPy 2.5.2, and a 12-logical-core Intel CPU (Family 6, Model 186).
The raw output is saved in `benchmark_results.json`.

## Physics performance

The benchmark applies the same 5,000 pre-generated six-coil current commands to
both simulation paths, with 5 physics substeps per command. Reported timings are
the median of 3 runs and exclude Numba compilation time.

| Path | Physics steps/s | Time for 25,000 steps |
| --- | ---: | ---: |
| Python/NumPy `Simulation.step` | 13,423 | 1.862 s |
| Numba `Simulation.step_many` | 1,753,402 | 0.0143 s |

The compiled path was **130.6x faster**. After an identical 500-step validation
trajectory, the largest absolute difference between the two implementations was
`1.03e-13` (angular velocity); position differed by at most `1.67e-16` m.

The complete Gymnasium environment, including action processing, reward
calculation, observations, and 5 compiled physics substeps per action, sustained
**27,376 environment steps/s**. That is 136.9 simulated seconds per wall-clock
second for the configured 5 ms action interval.

## Checkpoint evaluation

`magnetic_dipole_realistic_ppo.zip` contains a PPO policy trained for 5,013,504
environment timesteps. It was evaluated deterministically on 20 seeded targets
and compared with a zero-current controller on the same targets.

| Controller | Strict successes | Median initial error | Median final error | Episodes ending at a wall |
| --- | ---: | ---: | ---: | ---: |
| PPO checkpoint | 0/20 | 8.02 cm | 2.19 cm | 1/20 |
| Zero-current baseline | 0/20 | 8.02 cm | 19.65 cm | 20/20 |

The PPO policy reduced median terminal target error by **88.8% versus the
zero-current baseline** (and 72.6% from its own initial error), but did not meet
the environment's combined 5 mm position, velocity, angular-velocity, and
10-degree orientation success criteria. Treat this as evidence of useful control
learning, not a solved-task success rate.

## Reproduce

```powershell
.\.venv\Scripts\python.exe benchmark.py --json benchmark_results.json
```

For a stronger checkpoint evaluation, increase the sample size:

```powershell
.\.venv\Scripts\python.exe benchmark.py --episodes 100 --json benchmark_results.json
```

Hardware, background load, package versions, and power settings affect throughput.

## Resume-ready points

Use the first two bullets now; use the third only if you are comfortable explaining
the strict-success caveat in an interview.

- Built a 6-DOF electromagnetic-control simulator in Python with six independently
  actuated coils, dipole-dipole forces and torques, gravity, damping, collision
  constraints, and semi-implicit Euler integration.
- Optimized the reinforcement-learning physics loop with a Numba-compiled batched
  kernel, improving median throughput **130.6x** (13.4K to 1.75M physics steps/s)
  while matching the reference implementation within `1.1e-13` in validation.
- Designed a 21-dimensional Gymnasium environment and trained a six-action PPO
  controller for **5.0M timesteps**, reducing median final target error **88.8%**
  versus a zero-current baseline across 20 seeded trials (19.65 cm to 2.19 cm).
