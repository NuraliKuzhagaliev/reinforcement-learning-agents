---
library_name: stable-baselines3
tags:
- LunarLander-v2
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
- ppo
model-index:
- name: PPO LunarLander-v2
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      type: LunarLander-v2
      name: LunarLander-v2
    metrics:
    - type: mean_reward
      name: mean_reward
      value: "247.41 +/- 20.43"
---

# PPO LunarLander-v2

A Stable-Baselines3 PPO agent trained from scratch for **1,015,808 timesteps** using Gymnasium 0.29.1 and Stable-Baselines3 2.3.2 for [Deep Reinforcement Learning Course Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/hands-on).

The model achieved **247.41 ± 20.43** across 10 deterministic evaluation episodes with evaluation seed 1000. The course score (mean minus standard deviation) is **226.98**. The replay shows the first evaluation episode.

## Load the model

Install Gymnasium 0.29.1 with Box2D, Stable-Baselines3 2.3.2, and huggingface_hub.

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO

checkpoint = hf_hub_download("LordN77/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip")
model = PPO.load(checkpoint, device="cpu")
```
