# Recurrent SAC for the Blind Key Door Environment

This repository provides a minimal, fully runnable implementation of a recurrent Soft Actor-Critic (RNN-SAC) agent for a partially observable "Blind Key Door" environment. The observation available to the agent is restricted to `[door_open, has_key, last_action]`, so the recurrent state must track the agent's belief about its location and progress.

## Project Structure

- `blind_key_door_env.py` – Lightweight environment that exposes the key-door challenge with discrete actions and minimal observations.
- `rnn_sac_agent.py` – Implementation of the belief encoder (GRU), recurrent SAC policy, critics, and sequence replay buffer.
- `train.py` – Training and evaluation loop that ties the agent and environment together.

## Dependencies

Install the required packages before running the training script:

```bash
pip install torch gymnasium numpy
```

The code automatically uses a CUDA device if one is available; otherwise it falls back to CPU.

## Training

To train the agent with default hyperparameters (15,000 episodes, sequence length of 20):

```bash
python train.py
```

During training the script periodically evaluates the policy in a deterministic version of the environment (no action slip). After training completes, the script renders one evaluation episode in the terminal.

To experiment with different settings (e.g., fewer episodes during development), import the helper function and override the defaults:

```python
from train import train_rnn_sac

agent, train_returns, eval_returns = train_rnn_sac(episodes=2000, batch_size=32, seq_len=16)
```

## Notes

- Intrinsic rewards such as curiosity or boredom can be incorporated in `train.py` by augmenting the reward signal before it is stored in the replay buffer.
- The environment uses a configurable `slip_prob` parameter to introduce stochasticity during training while keeping evaluation deterministic.
