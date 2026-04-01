import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sinergym_env import SinergymGCNEnv
from ppo_agent_ds import PPOAgent
from meta_agent import OnlineMetaAgent
from collections import deque



def create_adjacency_matrix():
    adj_matrix = np.array([
        [1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1]
    ])
    return adj_matrix


def adj_matrix_to_edge_index(adj_matrix):
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1 and i != j:  # 排除自环
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


class SinergymTester:
    def __init__(self, env, agent, edge_index, test_episodes=10):
        self.env = env
        self.agent = agent
        self.edge_index = edge_index
        self.test_episodes = test_episodes

        # 测试结果存储
        self.episode_rewards = []
        self.episode_energies = []
        self.temperature_records = []
        self.action_records = []
        self.outdoor_temp_records = []
        self.energy_records = []
        self.prediction_records = []
        self.prediction_accuracy_records = []
        self.sigma_records = []
        self.meta_reward_records = []  # 单步meta奖励
        self.meta_total_reward_records = []  # meta累计总奖励
        self.meta_reward_per_update = []  # 每次meta更新周期内的平均奖励

        self.result_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"Testing started. Results will be saved to: {self.result_dir}")

        self.target_temp = 24.0
        # Meta-Agent 初始化
        self.history_len = 3
        self.meta_agent = OnlineMetaAgent(history_len=self.history_len, lr=0.01)
        self.update_freq = 3
        self.avg_window = 20

        # 队列
        self.scaled_action_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]
        self.prev_scaled_action = np.full(5, 24.0)
        self.temp_error_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]

    def scale_actions_with_meta(self, base_mu, sigmas):
        scaled_actions = []
        for i, mu in enumerate(base_mu):
            sigma = float(sigmas[i])
            scaled_action = (4 - sigma) * mu + 24.0
            scaled_actions.append(scaled_action)

        return np.array(scaled_actions)

    def _get_last_n(self, data, n=168):
        if len(data) <= n:
            return data
        return data[-n:]

    def _moving_avg(self, data, window=4):
        if len(data) < window:
            return [np.mean(data)] if data else []
        return np.convolve(data, np.ones(window)/window, mode='valid').tolist()

    def test(self, deterministic=True):
        print(f"Starting testing with {self.test_episodes} episodes...")

        for episode in range(self.test_episodes):
            state, info = self.env.reset()
            pred_sequence = info['pred_sequence']
            episode_reward = 0
            episode_energy = 0
            step_count = 0

            # 初始化队列
            self.scaled_action_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]
            self.temp_error_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]
            self.prev_scaled_action = np.full(5, 24.0)
            self.meta_agent.log_probs = []
            self.meta_agent.rewards = []

            # 存储每步数据
            episode_temperatures = []
            episode_actions = []
            episode_rewards = []
            episode_outdoor_temps = []
            episode_energies = []
            episode_predictions = []
            episode_prediction_accuracy = []
            episode_sigmas = []
            episode_meta_buffer = []  # 暂存当前更新周期的meta奖励
            episode_meta_updates = [] # 每次meta更新的平均奖励
            total_meta = 0.0          # meta累计总奖励

            done = False
            initial_temps = [state[i, 0].item() for i in range(5)]
            for i in range(5):
                for _ in range(self.history_len):
                    self.temp_error_history[i].append(initial_temps[i] - self.target_temp)

            while not done:
                # 构造Meta状态
                action_hist = np.array([list(q) for q in self.scaled_action_history])
                temp_hist = np.array([list(q) for q in self.temp_error_history])
                meta_state = np.concatenate([action_hist, temp_hist], axis=1)

                sigmas = self.meta_agent.get_action(meta_state)
                base_mu, _, _ = self.agent.get_action(state.numpy(), pred_sequence, self.edge_index, deterministic=deterministic)
                current_scaled_action = self.scale_actions_with_meta(base_mu, sigmas)

                delta_u = current_scaled_action - self.prev_scaled_action
                for i in range(5):
                    self.scaled_action_history[i].append(delta_u[i])
                self.prev_scaled_action = current_scaled_action.copy()

                next_state, reward, terminated, truncated, info = self.env.step(current_scaled_action.tolist())
                next_pred_sequence = info['pred_sequence']
                done = terminated or truncated

                current_temperatures = np.array([next_state[i, 0].item() for i in range(5)])
                temp_errors = current_temperatures - self.target_temp
                for i in range(5):
                    self.temp_error_history[i].append(temp_errors[i])
                meta_rewards = -np.abs(temp_errors)  # Meta奖励
                mean_meta_step = np.mean(meta_rewards) # 当前步平均meta奖励
                episode_meta_buffer.append(mean_meta_step)
                total_meta += mean_meta_step # 累计总数
                self.meta_agent.store_reward(meta_rewards)

                current_temperatures = [next_state[i, 0].item() for i in range(self.env.num_zones)]
                current_energy = info.get('total_energy_kwh', 0.0)
                current_outdoor_temp = next_state[0, 3].item()
                episode_temperatures.append(current_temperatures)
                episode_actions.append(current_scaled_action)
                episode_rewards.append(reward)
                episode_outdoor_temps.append(current_outdoor_temp)
                episode_energies.append(current_energy)
                episode_predictions.append(pred_sequence.copy())
                episode_sigmas.append(sigmas.copy())



                episode_reward += reward
                episode_energy += current_energy
                step_count += 1

                # Meta更新：每6步一次，记录本次更新周期的平均奖励
                if step_count % self.update_freq == 0:
                    if episode_meta_buffer:
                        avg_update = np.mean(episode_meta_buffer)
                        episode_meta_updates.append(avg_update)
                        episode_meta_buffer = []
                    print(f"  [Online Meta-Learning] Updating policy at step {step_count}...")
                    self.meta_agent.update_policy()

                state = next_state
                pred_sequence = next_pred_sequence

                if step_count >= 2000:
                    break

            if episode_meta_buffer:
                avg_update = np.mean(episode_meta_buffer)
                episode_meta_updates.append(avg_update)
            # 残余数据更新
            if len(self.meta_agent.rewards) > 0:
                self.meta_agent.update_policy()

            # 保存所有数据
            self.episode_rewards.append(episode_reward)
            self.episode_energies.append(episode_energy)
            self.temperature_records.append(episode_temperatures)
            self.action_records.append(episode_actions)
            self.outdoor_temp_records.append(episode_outdoor_temps)
            self.energy_records.append(episode_energies)
            self.prediction_records.append(episode_predictions)
            self.prediction_accuracy_records.append(episode_prediction_accuracy)
            self.sigma_records.append(episode_sigmas)
            self.meta_reward_per_update.append(episode_meta_updates)
            self.meta_total_reward_records.append(total_meta)

            print(f"Episode {episode + 1:2d}: "
                  f"Total Reward: {episode_reward:8.2f}, "
                  f"Total Energy: {episode_energy:8.2f} kWh, "
                  f"Total Meta Reward: {total_meta:8.2f}, "
                  f"Steps: {step_count:3d}")

            

        self.save_test_results()
        return self.episode_rewards, self.episode_energies

    def save_test_results(self):
        last_n_rewards = []
        last_n_energies = []
        for episode in range(self.test_episodes):
            n_steps = len(self.temperature_records[episode])
            start_idx = max(0, n_steps - 168)
            ep_rewards = np.sum(self.energy_records[episode][start_idx:]) if self.energy_records[episode] else 0
            ep_energy = np.sum(self.energy_records[episode][start_idx:]) if self.energy_records[episode] else 0
            last_n_rewards.append(ep_rewards)
            last_n_energies.append(ep_energy)

        overall_results = pd.DataFrame({
            'episode': range(1, self.test_episodes + 1),
            'total_reward_last168': last_n_rewards,
            'total_energy_last168': last_n_energies,
            'total_meta_reward': self.meta_total_reward_records
        })
        overall_results.to_csv(os.path.join(self.result_dir, 'overall_results_last168.csv'), index=False)

        print(f"Test results (last 168 steps) saved to: {self.result_dir}")


def main():
    """主测试函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_model_path = "policy_net_episode_final.pth"
    value_model_path = "value_net_episode_final.pth"
    idf_path = "/opt/project/osmfile/env1.epJSON"
    epw_path = "/opt/project/weatherfile/CHN_Sichuan.Mianyang.561960_CSWD.epw"

    env = SinergymGCNEnv(idf_file=idf_path, epw_file=epw_path, pred_seq_len=8)
    adj_matrix = create_adjacency_matrix()
    edge_index = adj_matrix_to_edge_index(adj_matrix)

    agent = PPOAgent(node_num=5, state_dim=7, action_dim=5, device=device, pred_seq_len=8)
    agent.load_model(policy_model_path, value_model_path)
    print("Model loaded successfully!")

    # 测试
    tester = SinergymTester(env=env, agent=agent, edge_index=edge_index, test_episodes=1)
    rewards, energies = tester.test(deterministic=True)

    env.close()
    print("Testing completed!")


if __name__ == "__main__":
    main()