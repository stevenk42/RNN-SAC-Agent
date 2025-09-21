import numpy as np

from blind_key_door_env import BlindKeyDoorEnv
from rnn_sac_agent import RNNSACAgent


def train_rnn_sac(episodes: int = 15000, batch_size: int = 64, seq_len: int = 20):
    env = BlindKeyDoorEnv(size=5, max_steps=100, slip_prob=0.1)
    eval_env = BlindKeyDoorEnv(size=5, max_steps=100, slip_prob=0.0)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = RNNSACAgent(obs_dim, action_dim, hidden_dim=64, seq_len=seq_len)

    returns = []
    eval_returns = []

    for i_episode in range(episodes):
        episode_record = {"obs": [], "act": [], "rew": [], "done": []}
        obs, _ = env.reset()
        agent.reset_hidden()
        ep_reward = 0.0

        for _ in range(env.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_record["obs"].append(np.array(obs, dtype=np.float32))
            episode_record["act"].append(int(action))
            episode_record["rew"].append(float(reward))
            episode_record["done"].append(float(done))

            obs = next_obs
            ep_reward += reward

            if done:
                break

        agent.buffer.push(episode_record)

        if len(agent.buffer) >= batch_size:
            agent.train(batch_size=batch_size)

        returns.append(ep_reward)

        if i_episode % 250 == 0:
            eval_reward = evaluate_agent(agent, eval_env)
            eval_returns.append(eval_reward)
            print(f"Episode {i_episode}, Last Reward: {ep_reward:.1f}, Eval Reward: {eval_reward:.1f}")

    env.close()
    eval_env.close()

    return agent, returns, eval_returns


def evaluate_agent(agent: RNNSACAgent, eval_env: BlindKeyDoorEnv, num_episodes: int = 5) -> float:
    total_reward = 0.0
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        agent.reset_hidden()
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / num_episodes


if __name__ == "__main__":
    agent, train_returns, eval_returns = train_rnn_sac()

    print("\n--- Final Evaluation Run ---")
    env = BlindKeyDoorEnv(size=5, max_steps=100, slip_prob=0.0, render_mode="ansi")
    obs, _ = env.reset()
    agent.reset_hidden()
    done = False
    print(env.render())

    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"\nAction: {['U', 'D', 'L', 'R', 'N'][action]}, Reward: {reward:.2f}")
        print(env.render())

    env.close()
