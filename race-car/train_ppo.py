# train_ppo.py
# Trains a quick PPO on the RaceCarEnv and saves it to models/ppo_racecar.zip

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from racecar_env import RaceCarEnv

def make_env():
    return RaceCarEnv(max_steps=1800)

if __name__ == "__main__":
    # Single-process vectorized env keeps SB3 happy
    env = DummyVecEnv([make_env])

    # Small MLP, reasonable defaults for a quick first pass
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=200_000)

    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_racecar"
    model.save(model_path)
    print(f"Saved to {model_path}.zip")
