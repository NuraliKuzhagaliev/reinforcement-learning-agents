# Reinforcement Learning Agents

Four trained agents exploring **policy optimization, tabular Q-learning, and deep Q-networks**, built for Units 1–3 of the [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course).

**Author:** [Nurali Kuzhagaliev](https://github.com/NuraliKuzhagaliev) · **Model Hub:** [LordN77](https://huggingface.co/LordN77)

This repository collects the submitted notebooks, trained checkpoints, configurations, evaluation records, and gameplay replays. These are course-based implementations and experiments using established algorithms.

## Agents and recorded results

| Unit | Algorithm | Environment | Training | Mean reward ± std | Evaluation episodes |
| --- | --- | --- | --- | ---: | ---: |
| 1 | PPO / Stable-Baselines3 | LunarLander-v2 | 1,015,808 timesteps | 247.41 ± 20.43 | 10 |
| 2 | Tabular Q-learning | Taxi-v3 | 25,000 episodes | 7.50 ± 2.71 | 100 |
| 2 | Tabular Q-learning | FrozenLake-v1, 4×4, non-slippery | 10,000 episodes | 1.00 ± 0.00 | 100 |
| 3 | DQN / Stable-Baselines3 | SpaceInvadersNoFrameskip-v4 | 20,000 agent timesteps | 210.00 ± 0.00 | 10 + an additional 30 |

Results are taken from the supplied `results.json` files; they have not been re-evaluated as part of publishing this repository. Standard deviations are population standard deviations. Scores across different environments are not directly comparable. FrozenLake uses a deterministic, non-slippery map. The reported Space Invaders score is an early-training baseline, not a claim that the game is solved.

## Explore the projects

- **[Unit 1 — PPO LunarLander](unit-1-ppo-lunarlander/):** checkpoint, model card, configuration, evaluation summary, and [gameplay replay](unit-1-ppo-lunarlander/replay.mp4). The original training notebook is not included in the supplied materials. [Published model on Hugging Face](https://huggingface.co/LordN77/ppo-LunarLander-v2).
- **[Unit 2 — Q-learning](unit-2-q-learning/Unit_2_Q_Learning.ipynb):** training and evaluation notebook, epsilon-greedy exploration, Bellman updates, two saved Q-tables, episode-level results, and replays for [Taxi](unit-2-q-learning/q-Taxi-v3/replay.mp4) and [FrozenLake](unit-2-q-learning/q-FrozenLake-v1-4x4-noSlippery/replay.mp4).
- **[Unit 3 — DQN Space Invaders](unit-3-dqn-space-invaders/Unit_3_Space_Invaders_DQN.ipynb):** notebook, [training script](unit-3-dqn-space-invaders/dqn-SpaceInvadersNoFrameskip-v4/train_unit3.py), compact CNN checkpoint, evaluation configuration, raw episode results, and [gameplay replay](unit-3-dqn-space-invaders/dqn-SpaceInvadersNoFrameskip-v4/replay.mp4).

## Run the notebooks

Use a separate virtual environment for each unit. The notebooks include their dependency installation cells. Start Jupyter with the unit directory as the working directory and run cells in order.

```bash
git clone https://github.com/NuraliKuzhagaliev/reinforcement-learning-agents.git
cd reinforcement-learning-agents
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install jupyterlab
python -m jupyter lab
```

- **Unit 2:** running all cells retrains both Q-tables and writes new artifacts under `unit2_outputs/`. The supplied checkpoints remain in their existing model folders. Hugging Face publication is disabled by default.
- **Unit 3:** the notebook creates `train_unit3.py` and `package_unit3.py` in the working directory. With its default flags, it evaluates the supplied checkpoint; training and Hub publication are disabled. Atari requires the appropriate environment registration and lawfully obtained ROMs. The supplied `dqn.yml` describes an optional RL Zoo run; the compact CNN checkpoint was trained with `train_unit3.py`, whose settings differ.

The supplied deep-RL artifacts use Gymnasium **0.29.1** and Stable-Baselines3 **2.3.2**. Unit 3 records ALE-Py **0.10.2**. These are historical environment IDs and dependency versions: retain the recorded setup when reproducing these results rather than silently substituting newer environments.

## Load the LunarLander checkpoint

In an environment with the recorded Gymnasium/Box2D and Stable-Baselines3 dependencies installed:

```python
from stable_baselines3 import PPO

model = PPO.load(
    "unit-1-ppo-lunarlander/ppo-LunarLander-v2.zip",
    device="cpu",
)
```

For Q-learning, the notebook contains the checkpoint-loading and evaluation logic. For DQN, use the notebook so the custom feature extractor and preprocessing are available.

Only load pickle files and model checkpoints from sources you trust. Atari ROMs are not included.

## Attribution and scope

Based on the Hugging Face Deep RL Course; uses Gymnasium, Stable-Baselines3, PyTorch, and the Arcade Learning Environment. Original supplied model cards and notebook contents are preserved. Upstream course materials, libraries, and environment assets remain subject to their respective licenses. No new repository-wide license is asserted over third-party material.
