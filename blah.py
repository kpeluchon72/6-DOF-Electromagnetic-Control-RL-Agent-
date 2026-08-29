from pathlib import Path

from stable_baselines3 import PPO


MODEL_PATH = Path("magnetic_dipole_safe_goal_fresh_ppo.zip")

model = PPO.load(MODEL_PATH, device="cpu")
total_parameters = sum(parameter.numel() for parameter in model.policy.parameters())
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.policy.parameters()
    if parameter.requires_grad
)

print(model.policy)
print(f"\nObservation space: {model.observation_space}")
print(f"Action space:      {model.action_space}")
print(f"Parameters:        {total_parameters:,}")
print(f"Trainable:         {trainable_parameters:,}")
print(f"Saved file size:   {MODEL_PATH.stat().st_size / 1024:.1f} KiB")
print(f"Parameter memory:  {total_parameters * 4 / 1024:.1f} KiB (float32)")
