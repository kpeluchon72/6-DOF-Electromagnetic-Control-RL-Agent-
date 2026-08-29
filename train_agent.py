"""Goal-conditioned Gymnasium environment and PPO training entry point."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from constants import DT, MAX_CURRENT, MAX_CURRENT_CHANGE
from world import create_default_world


TARGET_LOW = np.array([-0.05, 0.02, -0.05], dtype=np.float32)
TARGET_HIGH = np.array([0.05, 0.10, 0.05], dtype=np.float32)
VELOCITY_SCALE = 1.0
ANGULAR_VELOCITY_SCALE = 100.0
SUCCESS_DISTANCE = 0.005
SUCCESS_SPEED = 0.02
SUCCESS_ANGULAR_SPEED = 2.0
SUCCESS_ANGLE_DEGREES = 10.0


class MagneticDipoleEnv(gym.Env):
    """Move the dipole to a position goal and leave it pointing upward.

    Observation (21 normalized values): position, linear velocity, dipole
    orientation, angular velocity, target-position error, and actual currents.

    Action (6 values in [-1, 1]): normalized currents for the coils in
    +x, -x, +y, -y, +z, -z order.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_position=None,
        target_orientation=(0.0, 1.0, 0.0),
        max_steps=1000,
        sim_steps_per_action=5,
        max_current=MAX_CURRENT,
        max_current_change=MAX_CURRENT_CHANGE,
    ):
        super().__init__()
        self.fixed_target_position = (
            None
            if target_position is None
            else np.asarray(target_position, dtype=np.float32)
        )
        if self.fixed_target_position is not None and (
            self.fixed_target_position.shape != (3,)
        ):
            raise ValueError("target_position must contain exactly three values.")

        self.target_orientation = np.asarray(target_orientation, dtype=np.float32)
        orientation_length = np.linalg.norm(self.target_orientation)
        if self.target_orientation.shape != (3,) or orientation_length == 0:
            raise ValueError("target_orientation must be a nonzero 3D vector.")
        self.target_orientation /= orientation_length

        self.max_steps = int(max_steps)
        self.sim_steps_per_action = int(sim_steps_per_action)
        self.max_current = float(max_current)
        self.max_normalized_current_change = float(max_current_change) / self.max_current
        self.action_time = DT * self.sim_steps_per_action

        self.sim = create_default_world(currents=np.zeros(6))
        self.position_scale = float(np.max(np.abs(self.sim.box_max)))
        self.target_position = np.zeros(3, dtype=np.float32)
        self.current_step = 0
        self.previous_distance = 0.0
        self.previous_alignment = 0.0
        self.total_energy = 0.0
        self.normalized_currents = np.zeros(6, dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(21,), dtype=np.float32
        )

    def _get_observation(self):
        magnetic_object = self.sim.object
        orientation = magnetic_object.magnetic_moment_direction.copy()
        orientation /= np.linalg.norm(orientation) + 1e-8
        target_error = self.target_position - magnetic_object.position

        observation = np.concatenate(
            (
                magnetic_object.position / self.position_scale,
                np.tanh(magnetic_object.velocity / VELOCITY_SCALE),
                orientation,
                np.tanh(
                    magnetic_object.angular_velocity / ANGULAR_VELOCITY_SCALE
                ),
                target_error / (2.0 * self.position_scale),
                self.normalized_currents,
            )
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _choose_target(self, options):
        if "target_position" in options:
            target = np.asarray(options["target_position"], dtype=np.float32)
        elif self.fixed_target_position is not None:
            target = self.fixed_target_position.copy()
        else:
            target = self.np_random.uniform(TARGET_LOW, TARGET_HIGH).astype(np.float32)

        if target.shape != (3,):
            raise ValueError("target_position must contain exactly three values.")
        if np.any(target < self.sim.box_min) or np.any(target > self.sim.box_max):
            raise ValueError("target_position must be inside the simulation box.")
        self.target_position = target

    def _alignment(self):
        return float(
            np.dot(
                self.sim.object.magnetic_moment_direction,
                self.target_orientation,
            )
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = create_default_world(currents=np.zeros(6))
        self._choose_target(options or {})
        self.current_step = 0
        self.total_energy = 0.0
        self.normalized_currents.fill(0.0)
        self.previous_distance = float(
            np.linalg.norm(self.target_position - self.sim.object.position)
        )
        self.previous_alignment = self._alignment()
        return self._get_observation(), self._info(np.zeros(6, dtype=np.float32))

    def step(self, action):
        self.current_step += 1
        desired_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        current_change = np.clip(
            desired_action - self.normalized_currents,
            -self.max_normalized_current_change,
            self.max_normalized_current_change,
        )
        self.normalized_currents += current_change
        currents = self.normalized_currents * self.max_current

        for coil, current in zip(self.sim.electromagnets, currents):
            coil.current = float(current)
        self.sim.step_many(DT, self.sim_steps_per_action)

        distance = float(
            np.linalg.norm(self.target_position - self.sim.object.position)
        )
        alignment = self._alignment()
        speed = float(np.linalg.norm(self.sim.object.velocity))
        angular_speed = float(np.linalg.norm(self.sim.object.angular_velocity))

        progress = self.previous_distance - distance
        alignment_progress = alignment - self.previous_alignment
        energy = float(np.mean(self.normalized_currents**2) * self.action_time)
        self.total_energy += energy

        reward = (
            40.0 * progress
            + 1.0 * alignment_progress
            - 0.5 * energy
            - 0.02 * (1.0 - alignment)
            - 0.001
        )

        # Progress teaches the magnet how to approach. This continuing penalty
        # makes remaining close better than turning the coils off and drifting.
        normalized_distance = distance / self.position_scale
        reward -= 0.5 * normalized_distance

        # Give a smooth, increasingly valuable signal in the final 2 cm. Unlike
        # a threshold bonus, this still tells the policy which way is better.
        proximity_bonus = np.exp(-((distance / 0.02) ** 2))
        reward += 0.1 * proximity_bonus

        distance_to_wall = float(
            np.min(self.sim.box_max - np.abs(self.sim.object.position))
        )
        wall_margin = 0.03
        if distance_to_wall < wall_margin:
            wall_proximity = (wall_margin - distance_to_wall) / wall_margin
            reward -= 0.2 * wall_proximity**2

        wall_contact = distance_to_wall <= 1e-8
        if wall_contact:
            reward -= 1.0

        # Smoothly require slower motion as the magnet enters the final 3 cm.
        # A smooth weight avoids creating a reward cliff at a fixed distance.
        settling_weight = max(0.0, 1.0 - distance / 0.03)
        if settling_weight > 0.0:
            normalized_speed = np.tanh(speed / 0.1)
            normalized_angular_speed = np.tanh(angular_speed / 10.0)
            reward -= 0.05 * settling_weight * (
                normalized_speed**2 + normalized_angular_speed**2
            )

            # Near the goal, discourage repeatedly commanding the maximum
            # allowed current change. Long-range steering remains unaffected.
            normalized_current_change = current_change / max(
                self.max_normalized_current_change, 1e-8
            )
            reward -= 0.01 * settling_weight * float(
                np.mean(normalized_current_change**2)
            )

        success = (
            distance < SUCCESS_DISTANCE
            and speed < SUCCESS_SPEED
            and angular_speed < SUCCESS_ANGULAR_SPEED
            and alignment > np.cos(np.deg2rad(SUCCESS_ANGLE_DEGREES))
        )
        if success:
            reward += 10.0

        self.previous_distance = distance
        self.previous_alignment = alignment
        # Contact remains heavily penalized, but it must not end the episode;
        # otherwise the policy can learn to crash to escape future penalties.
        terminated = bool(success)
        truncated = self.current_step >= self.max_steps and not terminated
        observation = self._get_observation()
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            self._info(currents, wall_contact=wall_contact, success=success),
        )

    def _info(self, currents, wall_contact=False, success=False):
        magnetic_object = self.sim.object
        distance = float(np.linalg.norm(self.target_position - magnetic_object.position))
        distance_to_wall = float(
            np.min(self.sim.box_max - np.abs(magnetic_object.position))
        )
        return {
            "distance": distance,
            "alignment": self._alignment(),
            "speed": float(np.linalg.norm(magnetic_object.velocity)),
            "angular_speed": float(np.linalg.norm(magnetic_object.angular_velocity)),
            "energy_used": self.total_energy,
            "currents": np.asarray(currents, dtype=np.float32).copy(),
            "target_position": self.target_position.copy(),
            "distance_to_wall": distance_to_wall,
            "wall_contact": bool(wall_contact),
            "success": bool(success),
        }


if __name__ == "__main__":
    import multiprocessing

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    multiprocessing.freeze_support()
    torch.set_num_threads(1)

    # Compile and cache the physics kernel before worker processes start.
    warmup_env = MagneticDipoleEnv(max_steps=1)
    warmup_env.reset(seed=0)
    warmup_env.step(np.zeros(6, dtype=np.float32))

    # Four workers measured fastest with the compiled physics on this CPU. Each
    # worker owns an independent simulation and contributes to one policy.
    env = make_vec_env(
        MagneticDipoleEnv,
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    model_path = Path("magnetic_dipole_realistic_ppo")
    saved_model_path = model_path.with_suffix(".zip")
    try:
        if saved_model_path.exists():
            model = PPO.load(model_path, env=env, device="cpu")
            fine_tune_learning_rate = 1e-4
            model.learning_rate = fine_tune_learning_rate
            model.lr_schedule = lambda _: fine_tune_learning_rate
            for parameter_group in model.policy.optimizer.param_groups:
                parameter_group["lr"] = fine_tune_learning_rate
            training_steps = 1_000_000
            reset_num_timesteps = False
            print(
                f"Continuing {saved_model_path} from "
                f"{model.num_timesteps:,} timesteps for {training_steps:,} more."
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cpu",
                n_steps=1024,
                batch_size=512,
                n_epochs=5,
            )
            training_steps = 2_000_000
            reset_num_timesteps = True
            print(f"No saved model found; training a fresh model for {training_steps:,} steps.")

        model.learn(
            total_timesteps=training_steps,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(model_path)
    finally:
        env.close()
