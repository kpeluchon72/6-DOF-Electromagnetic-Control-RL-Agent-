from stable_baselines3 import PPO
from vpython import rate

from train_agent import MagneticDipoleEnv
from visualization import SimulationViewer


env = MagneticDipoleEnv()
observation, _ = env.reset()

model = PPO.load("magnetic_dipole_realistic_ppo")
viewer = SimulationViewer(
    env.sim,
    max_current=env.max_current,
    target_position=env.target_position,
    target_orientation=env.target_orientation,
)

while True:
    rate(60)

    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    viewer.update()

    if terminated or truncated:
        print(
            f"distance={info['distance']:.4f}, "
            f"speed={info['speed']:.4f}, "
            f"angular_speed={info['angular_speed']:.3f}, "
            f"alignment={info['alignment']:.3f}, "
            f"wall={info['wall_contact']}, "
            f"success={info['success']}"
        )
        observation, _ = env.reset()
        viewer.simulation = env.sim
        viewer.set_target(env.target_position, env.target_orientation)
