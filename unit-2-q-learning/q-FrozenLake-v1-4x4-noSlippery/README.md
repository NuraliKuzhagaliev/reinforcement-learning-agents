---
tags:
- FrozenLake-v1-4x4-no_slippery
- q-learning
- reinforcement-learning
- custom-implementation
model-index:
- name: q-FrozenLake-v1-4x4-noSlippery
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      type: FrozenLake-v1-4x4-no_slippery
      name: FrozenLake-v1-4x4-no_slippery
    metrics:
    - type: mean_reward
      name: mean_reward
      value: "1.00 +/- 0.00"
---

# q-FrozenLake-v1-4x4-noSlippery

Tabular Q-learning agent trained from scratch for the [Hugging Face Deep RL Course, Unit 2](https://huggingface.co/learn/deep-rl-course/unit2/hands-on).

- Environment: `FrozenLake-v1`
- Training: 10,000 episodes, seed 42
- Q-table: 16 states × 4 actions
- Evaluation: 100 episodes using the greedy policy
- Mean reward: 1.00
- Population standard deviation: 0.00
- Course score (mean minus standard deviation): **1.00**

The first evaluation episode is recorded in `replay.mp4`. The checkpoint `q-learning.pkl` contains the Q-table and settings. Use Gymnasium 0.29.1 and load pickle files only from sources you trust.
