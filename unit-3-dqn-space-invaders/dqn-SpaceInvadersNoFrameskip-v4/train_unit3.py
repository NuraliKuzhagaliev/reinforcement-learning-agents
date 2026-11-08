import json
import os
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

ENV_ID = "SpaceInvadersNoFrameskip-v4"
TRAIN_SEED = 42
MAX_TIMESTEPS = 400_000
STOP_AFTER = int(os.environ.get("UNIT3_STOP_AFTER", MAX_TIMESTEPS))
OUTPUT = Path("unit3-run")
SCREEN_SIZE = 42
DEVICE = os.environ.get("UNIT3_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("UNIT3_THREADS", "1")))

class CompactAtariCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, kernel_size=6, stride=3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(flat_size, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def make_env(training=False, render=False):
    def init_env():
        env = gym.make(ENV_ID, render_mode="rgb_array" if render else None)
        return Monitor(AtariWrapper(env, screen_size=SCREEN_SIZE, terminal_on_life_loss=training, clip_reward=training))
    return VecTransposeImage(VecFrameStack(DummyVecEnv([init_env]), n_stack=4))

def evaluate(model, seeds, render=False, max_steps=None):
    env = make_env(render=render)
    frames, returns, lengths = [], [], []
    try:
        for seed in seeds:
            env.seed(seed)
            obs = env.reset()
            done, total, length = False, 0.0, 0
            if render:
                frames.append(env.render())
            while not done and (max_steps is None or length < max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                total += float(rewards[0])
                length += 1
                if render and not done:
                    frames.append(env.render())
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    return np.asarray(returns), lengths, frames

def make_model(env):
    return DQN(
        "CnnPolicy", env, learning_rate=1e-4, buffer_size=60_000,
        learning_starts=8_000, batch_size=32, gamma=0.99, train_freq=4,
        gradient_steps=1, target_update_interval=1000, exploration_fraction=0.35,
        exploration_final_eps=0.02, optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination":False},
        policy_kwargs={"features_extractor_class":CompactAtariCNN, "net_arch":[64]},
        seed=TRAIN_SEED, device=DEVICE, verbose=0,
    )

class CheckpointCallback(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.best = float("-inf")

    def _on_step(self):
        if self.num_timesteps % 20_000:
            return True
        scores, _, _ = evaluate(self.model, range(1100, 1110))
        mean, std = float(scores.mean()), float(scores.std())
        score = mean - std
        print(f"steps={self.num_timesteps} validation={mean:.2f} +/- {std:.2f}; score={score:.2f}", flush=True)
        if score > self.best:
            self.best = score
            self.model.save(self.directory / "best")
            (self.directory / "validation.json").write_text(json.dumps({"steps":self.num_timesteps,"returns":scores.tolist(),"score":score}))
        return score < 245 and self.num_timesteps < STOP_AFTER

def train():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = make_env(training=True)
    try:
        model = make_model(env)
        model.learn(total_timesteps=MAX_TIMESTEPS, callback=CheckpointCallback(OUTPUT))
    finally:
        env.close()

if __name__ == "__main__":
    train()
