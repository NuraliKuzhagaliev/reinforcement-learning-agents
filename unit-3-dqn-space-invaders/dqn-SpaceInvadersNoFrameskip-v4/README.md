---
library_name: stable-baselines3
tags:
- SpaceInvadersNoFrameskip-v4
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
- dqn
model-index:
- name: dqn-SpaceInvadersNoFrameskip-v4
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      type: SpaceInvadersNoFrameskip-v4
      name: SpaceInvadersNoFrameskip-v4
    metrics:
    - type: mean_reward
      name: mean_reward
      value: "210.00 +/- 0.00"
---

# DQN for Space Invaders

Vanilla Stable-Baselines3 DQN trained from random initialization on `SpaceInvadersNoFrameskip-v4` for the [Hugging Face Deep RL Course, Unit 3](https://huggingface.co/learn/deep-rl-course/unit3/hands-on).

The selected checkpoint was saved at **20,000 agent timesteps**. The training run used a compact CNN, 42×42 grayscale observations, four stacked frames, and a fixed training seed. Evaluation uses full games and unclipped rewards. Checkpoint selection and evaluation use separate seeds.

| Evaluation | Mean reward | Population standard deviation | Score |
| --- | ---: | ---: | ---: |
| 10 episodes | 210.00 | 0.00 | **210.00** |
| Additional 30 episodes | 210.00 | 0.00 | 210.00 |

The score is mean reward minus standard deviation. `results.json` also includes a separate 5,000-agent-step evaluation. `replay.mp4` shows the first of the 10 evaluation games.

Use the evaluation environment settings in `config.json` or run the included training code. Only load model archives from sources you trust.
