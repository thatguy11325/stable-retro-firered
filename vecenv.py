import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env():
    import retro

    retro.data.Integrations.add_custom_path(SCRIPT_DIR)
    return retro.make(
        "Emerald-GBAdvance", inttype=retro.data.Integrations.ALL, render_mode="human"
    )


if __name__ == "__main__":
    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4)
    )
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
    )
