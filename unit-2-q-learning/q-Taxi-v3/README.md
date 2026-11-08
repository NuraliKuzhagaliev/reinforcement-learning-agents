---
tags:
- Taxi-v3
- q-learning
- reinforcement-learning
- custom-implementation
model-index:
- name: q-Taxi-v3
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      type: Taxi-v3
      name: Taxi-v3
    metrics:
    - type: mean_reward
      name: mean_reward
      value: "7.50 +/- 2.71"
---

# q-Taxi-v3

Tabular Q-learning agent trained from scratch for the [Hugging Face Deep RL Course, Unit 2](https://huggingface.co/learn/deep-rl-course/unit2/hands-on).

- Environment: `Taxi-v3`
- Training: 25,000 episodes, seed 42
- Q-table: 500 states × 6 actions
- Evaluation: 100 episodes using the greedy policy
- Mean reward: 7.50
- Population standard deviation: 2.71
- Course score (mean minus standard deviation): **4.79**

The first evaluation episode is recorded in `replay.mp4`. The checkpoint `q-learning.pkl` contains the Q-table and settings. Use Gymnasium 0.29.1 and load pickle files only from sources you trust.
