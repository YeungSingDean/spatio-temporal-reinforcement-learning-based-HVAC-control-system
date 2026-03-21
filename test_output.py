import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sinergym_env import SinergymGCNEnv
from ppo_agent_ds import PPOAgent
from meta_agent import OnlineMetaAgent
from collections import deque


# 邻接矩阵处理函数
def create_adjacency_matrix():
    """根据文档创建邻接矩阵 - 用于策略网络的图结构"""
    adj_matrix = np.array([
        [1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1]
    ])
    return adj_matrix


def adj_matrix_to_edge_index(adj_matrix):
    """将邻接矩阵转换为edge_index格式 - 用于策略网络的图结构"""
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1 and i != j:  # 排除自环
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


class SinergymTester:
    """Sinergym环境测试器"""

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
        self.prediction_accuracy_records = []  # 新增：记录预测准确性
        self.sigma_records = []  # 新增

        # 创建结果目录
        self.result_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"Testing started. Results will be saved to: {self.result_dir}")

        self.target_temp = 24.0
        # ★ Meta-Agent 初始化
        self.history_len = 3
        self.meta_agent = OnlineMetaAgent(history_len=self.history_len, lr=0.01)
        self.update_freq = 6  # 每 24 步（即每天）更新一次策略

        # ★ 队列 1：记录动作 Delta
        self.scaled_action_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]
        self.prev_scaled_action = np.full(5, 24.0)

        # ★ 队列 2：新增记录室温跟踪误差 (T_actual - 24.0)
        self.temp_error_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]

    def scale_actions_with_meta(self, base_mu, sigmas):
        scaled_actions = []
        for i, mu in enumerate(base_mu):
            sigma = float(sigmas[i])
            scaled_action = (3 - sigma) * mu + 24.0
            scaled_actions.append(scaled_action)

        return np.array(scaled_actions)

    def _get_last_n(self, data, n=168):
        """截取数据最后 n 步，如果数据长度不足则返回全部"""
        if len(data) <= n:
            return data
        return data[-n:]

    def test(self, deterministic=True):
        """测试循环"""
        print(f"Starting testing with {self.test_episodes} episodes...")

        for episode in range(self.test_episodes):
            state, info = self.env.reset()
            pred_sequence = info['pred_sequence']  # 获取预测序列
            episode_reward = 0
            episode_energy = 0
            step_count = 0

            # 回合初清空两个历史队列
            self.scaled_action_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]
            self.temp_error_history = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(5)]
            self.prev_scaled_action = np.full(5, 24.0)

            # 确保上一个 Episode 遗留的梯度数据被清空
            self.meta_agent.log_probs = []
            self.meta_agent.rewards = []

            # 存储每步数据用于绘图
            episode_temperatures = []
            episode_actions = []
            episode_rewards = []
            episode_outdoor_temps = []
            episode_energies = []
            episode_predictions = []
            episode_prediction_accuracy = []  # 新增：记录预测准确性
            episode_sigmas = []

            done = False
            # 获取初始室温，填充到初始温度队列中 (避免第一步全0)
            initial_temps = [state[i, 0].item() for i in range(5)]
            for i in range(5):
                for _ in range(self.history_len):
                    self.temp_error_history[i].append(initial_temps[i] - self.target_temp)

            while not done:
                # 1. 提取合并状态：动作历史 + 温度历史 -> shape (5, 6)
                action_hist = np.array([list(q) for q in self.scaled_action_history])
                temp_hist = np.array([list(q) for q in self.temp_error_history])
                meta_state = np.concatenate([action_hist, temp_hist], axis=1)  # 沿着特征维度拼接

                # 2. Meta-Agent 根据拼接状态输出 Sigma
                sigmas = self.meta_agent.get_action(meta_state)

                # 3. 获取底层动作并映射
                base_mu, _, _ = self.agent.get_action(state.numpy(), pred_sequence, self.edge_index,
                                                      deterministic=deterministic)
                current_scaled_action = self.scale_actions_with_meta(base_mu, sigmas)

                # 缩放动作到实际范围
                # scaled_action = self.scale_actions(action)

                # 4. 更新动作历史
                delta_u = current_scaled_action - self.prev_scaled_action
                for i in range(5):
                    self.scaled_action_history[i].append(delta_u[i])
                self.prev_scaled_action = current_scaled_action.copy()

                # 5. 与环境交互
                next_state, reward, terminated, truncated, info = self.env.step(current_scaled_action.tolist())
                next_pred_sequence = info['pred_sequence']  # 获取下一步的预测序列
                done = terminated or truncated

                # 6. 获取新室温，并更新温度历史队列
                current_temperatures = np.array([next_state[i, 0].item() for i in range(5)])
                temp_errors = current_temperatures - self.target_temp  # 注意这里保留正负号，表示偏冷还是偏热
                for i in range(5):
                    self.temp_error_history[i].append(temp_errors[i])

                # 7. 计算 Meta-Agent 的 Reward (绝对值越小越好)
                meta_rewards = -np.abs(temp_errors)
                self.meta_agent.store_reward(meta_rewards)

                # 记录数据
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

                # 计算预测准确性（比较预测值与实际未来值）
                if step_count > 0 and step_count < len(episode_outdoor_temps):
                    # 计算当前预测序列的第一个值（下一步预测）与下一步实际值的差异
                    pred_accuracy = abs(pred_sequence[0] - current_outdoor_temp)
                    episode_prediction_accuracy.append(pred_accuracy)

                episode_reward += reward
                episode_energy += current_energy
                step_count += 1

                # ★ 7. 固定步长 (24步) 在线更新 Meta-Agent
                if step_count % self.update_freq == 0:
                    print(f"  [Online Meta-Learning] Updating policy at step {step_count}...")
                    self.meta_agent.update_policy()

                state = next_state
                pred_sequence = next_pred_sequence

                if step_count >= 672:  # 限制测试步数（一周：7天×24小时=168步）
                    break

            # 回合结束，处理未满 24 步的残余数据
            if len(self.meta_agent.rewards) > 0:
                self.meta_agent.update_policy()

            # 记录回合结果
            self.episode_rewards.append(episode_reward)
            self.episode_energies.append(episode_energy)
            self.temperature_records.append(episode_temperatures)
            self.action_records.append(episode_actions)
            self.outdoor_temp_records.append(episode_outdoor_temps)
            self.energy_records.append(episode_energies)
            self.prediction_records.append(episode_predictions)
            self.prediction_accuracy_records.append(episode_prediction_accuracy)
            self.sigma_records.append(episode_sigmas)

            print(f"Episode {episode + 1:2d}: "
                  f"Total Reward: {episode_reward:8.2f}, "
                  f"Total Energy: {episode_energy:8.2f} kWh, "
                  f"Steps: {step_count:3d}")

            # 每回合保存图表
            self.plot_episode_results(episode, episode_temperatures, episode_actions,
                                      episode_rewards, episode_outdoor_temps, episode_energies,
                                      episode_predictions, episode_prediction_accuracy, episode_sigmas)

        # 保存总体测试结果
        self.save_test_results()

        return self.episode_rewards, self.episode_energies

    def plot_temperature_subplot(self, episode, temperatures):
        """单独绘制温度曲线子图，只显示最后168步"""
        # 截取最后168步
        temperatures = self._get_last_n(temperatures)
        if not temperatures:
            return

        plt.figure(figsize=(24, 8))
        time_steps = range(len(temperatures))

        # 温度曲线
        temperatures = np.array(temperatures)
        for i in range(5):
            plt.plot(time_steps, temperatures[:, i], label=f'Room {i + 1}', linewidth=2)

        # 目标温度范围
        plt.axhline(y=23.0, color='r', linestyle='--', alpha=0.7, label='Comfort Range (23-25°C)')
        plt.axhline(y=25.0, color='r', linestyle='--', alpha=0.7)
        plt.fill_between(time_steps, 23.0, 25.0, alpha=0.1, color='red')

        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3)

        # 保存单独的温度图
        plt.savefig(os.path.join(self.result_dir, f'episode_{episode + 1}_temperature_only.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sigma_subplot(self, episode, sigmas):
        """单独绘制Sigma值曲线子图，只显示最后168步，且只画平均Sigma"""
        # 截取最后168步
        sigmas = self._get_last_n(sigmas)
        if not sigmas:
            return

        plt.figure(figsize=(24, 8))
        time_steps = range(len(sigmas))

        # 转换为numpy数组便于处理
        sigmas = np.array(sigmas)

        # 绘制平均Sigma值（黑色粗线突出显示）
        mean_sigmas = np.mean(sigmas, axis=1)
        plt.step(time_steps, mean_sigmas, 'k-', linewidth=3, where='post', label='Mean σ (All Rooms)', alpha=0.8)

        # Sigma的取值范围参考线
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='σ = 0 (Max Shrinkage)')
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='σ = 2 (Min Shrinkage)')

        # 设置Y轴刻度（匹配Sigma的离散取值）
        plt.yticks([0, 1, 2], ['0 (Max)', '1 (Medium)', '2 (Min)'])
        plt.ylim(-0.2, 2.2)  # 留出边距

        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Action Shrinkage (σ)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)

        # 保存单独的Sigma图
        plt.savefig(os.path.join(self.result_dir, f'episode_{episode + 1}_sigma_only.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_episode_results(self, episode, temperatures, actions, rewards, outdoor_temps, energies, predictions,
                             prediction_accuracy, sigmas):
        """绘制单回合结果 - 所有子图只显示最后168步"""
        # 先绘制单独的温度图（已在函数内截取）
        self.plot_temperature_subplot(episode, temperatures)

        # 绘制单独的Sigma图（已在函数内截取）
        self.plot_sigma_subplot(episode, sigmas)

        # 截取所有数据最后168步
        temperatures = self._get_last_n(temperatures)
        actions = self._get_last_n(actions)
        rewards = self._get_last_n(rewards)
        outdoor_temps = self._get_last_n(outdoor_temps)
        energies = self._get_last_n(energies)
        predictions = self._get_last_n(predictions)
        prediction_accuracy = self._get_last_n(prediction_accuracy)
        sigmas = self._get_last_n(sigmas)

        if not temperatures:
            return

        # 创建4x2的子图布局
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()

        time_steps = range(len(temperatures))

        # 1. 温度曲线
        temperatures_arr = np.array(temperatures)
        for i in range(5):
            axes[0].plot(time_steps, temperatures_arr[:, i], label=f'Room {i + 1}', linewidth=1.5)

        # 目标温度范围
        axes[0].axhline(y=23.0, color='r', linestyle='--', alpha=0.7, label='Comfort Range')
        axes[0].axhline(y=25.0, color='r', linestyle='--', alpha=0.7)
        axes[0].fill_between(time_steps, 23.0, 25.0, alpha=0.1, color='red')
        axes[0].set_title('Room Temperatures')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 动作设定点曲线与 Sigma 动态变化 (双轴图)
        actions_arr = np.array(actions)
        sigmas_arr = np.array(sigmas) if sigmas else np.zeros((len(temperatures), 5))

        # 左轴：画底层 PPO 下发的物理设定点 (Setpoint)
        for i in range(5):
            axes[1].plot(time_steps, actions_arr[:, i], label=f'Room {i + 1} Setpoint', linewidth=1.5, alpha=0.8)
        axes[1].set_title('Setpoints & Adaptive Compression (\u03c3)')
        axes[1].set_ylabel('Setpoint (\u00b0C)')
        axes[1].grid(True, alpha=0.3)

        # 右轴：画高层 Meta-Agent 的内缩系数 Sigma (仅平均)
        ax1_twin = axes[1].twinx()
        mean_sigmas = np.mean(sigmas_arr, axis=1)
        ax1_twin.step(time_steps, mean_sigmas, 'k-', linewidth=2.5, where='post', label='Mean \u03c3 (0, 1, 2)')

        # 设置右轴的刻度，明确显示离散值
        ax1_twin.set_ylabel('Action Shrinkage (\u03c3)')
        ax1_twin.set_yticks([0, 1, 2])
        ax1_twin.set_ylim(-0.2, 2.2)  # 留出上下边距

        # 合并图例
        lines_1, labels_1 = axes[1].get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)

        # 3. 奖励曲线
        axes[2].plot(time_steps, rewards, 'g-', linewidth=1.5)
        axes[2].set_title('Step Rewards')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Reward')
        axes[2].grid(True, alpha=0.3)

        # 添加累计奖励
        cumulative_rewards = np.cumsum(rewards)
        ax3_twin = axes[2].twinx()
        ax3_twin.plot(time_steps, cumulative_rewards, 'r--', linewidth=1.5, alpha=0.7, label='Cumulative Reward')
        ax3_twin.set_ylabel('Cumulative Reward')
        ax3_twin.legend(loc='upper right')

        # 4. 温度分布直方图
        all_temperatures = temperatures_arr.flatten()
        axes[3].hist(all_temperatures, bins=20, alpha=0.7, edgecolor='black', density=True)
        axes[3].axvline(x=23.0, color='r', linestyle='--', label='Target Range')
        axes[3].axvline(x=25.0, color='r', linestyle='--')
        axes[3].set_title('Temperature Distribution')
        axes[3].set_xlabel('Temperature (°C)')
        axes[3].set_ylabel('Density')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # 5. 室外温度曲线
        axes[4].plot(time_steps, outdoor_temps, 'b-', linewidth=2, label='Outdoor Temperature')
        axes[4].set_title('Outdoor Temperature')
        axes[4].set_xlabel('Time Step')
        axes[4].set_ylabel('Temperature (°C)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        # 添加室内外温度对比
        mean_indoor_temp = np.mean(temperatures_arr, axis=1)
        axes[4].plot(time_steps, mean_indoor_temp, 'r-', linewidth=1.5, alpha=0.7, label='Mean Indoor Temperature')
        axes[4].legend()

        # 6. 能耗曲线
        axes[5].plot(time_steps, energies, 'orange', linewidth=2, label='Step Energy Consumption')
        axes[5].set_title('Energy Consumption')
        axes[5].set_xlabel('Time Step')
        axes[5].set_ylabel('Energy (kWh)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

        # 添加累计能耗
        cumulative_energy = np.cumsum(energies)
        ax6_twin = axes[5].twinx()
        ax6_twin.plot(time_steps, cumulative_energy, 'purple', linewidth=1.5, alpha=0.7, label='Cumulative Energy')
        ax6_twin.set_ylabel('Cumulative Energy (kWh)')
        ax6_twin.legend(loc='upper right')

        # 7. 预测序列与实际温度对比
        axes[6].set_title('Prediction vs Actual Temperature')

        # 绘制实际室外温度
        axes[6].plot(time_steps, outdoor_temps, 'k-', linewidth=2, label='Actual Outdoor Temp')

        # 绘制几个时间点的预测序列
        selected_steps = [0, len(time_steps) // 4, len(time_steps) // 2, 3 * len(time_steps) // 4, len(time_steps) - 1]
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, step in enumerate(selected_steps):
            if step < len(predictions):
                pred_steps = range(step, step + len(predictions[step]))
                axes[6].plot(pred_steps, predictions[step], color=colors[i],
                             marker='o', markersize=3, linewidth=1.5, alpha=0.7,
                             label=f'Step {step} Prediction')

        axes[6].set_xlabel('Time Step')
        axes[6].set_ylabel('Temperature (°C)')
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)

        # 8. 预测准确性分析
        if prediction_accuracy:
            axes[7].plot(range(len(prediction_accuracy)), prediction_accuracy, 'r-', linewidth=1.5,
                         label='Prediction Error')
            axes[7].axhline(y=np.mean(prediction_accuracy), color='b', linestyle='--',
                            label=f'Mean Error: {np.mean(prediction_accuracy):.3f}°C')
            axes[7].set_title('Prediction Accuracy (1-step ahead)')
            axes[7].set_xlabel('Time Step')
            axes[7].set_ylabel('Absolute Error (°C)')
            axes[7].legend()
            axes[7].grid(True, alpha=0.3)

            # 在图表上方添加统计信息
            stats_text = f"Prediction Stats:\nMax Error: {np.max(prediction_accuracy):.3f}°C\nMin Error: {np.min(prediction_accuracy):.3f}°C\nStd: {np.std(prediction_accuracy):.3f}°C"
            axes[7].text(0.02, 0.98, stats_text, transform=axes[7].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[7].text(0.5, 0.5, 'No prediction accuracy data',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[7].transAxes)
            axes[7].set_title('Prediction Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, f'episode_{episode + 1}_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_test_results(self):
        """保存测试结果为CSV格式，只保留最后168步的数据"""
        # 计算每个 episode 最后 168 步的奖励和能耗
        last_n_rewards = []
        last_n_energies = []
        for episode in range(self.test_episodes):
            # 获取该 episode 的步数
            n_steps = len(self.temperature_records[episode])
            # 取最后 168 步，若不足则取全部
            start_idx = max(0, n_steps - 168)
            # 计算最后 168 步的累计奖励
            ep_rewards = np.sum(self.energy_records[episode][start_idx:]) if self.energy_records[episode] else 0
            # 计算最后 168 步的累计能耗（假设 energy_records 是每步能耗）
            ep_energy = np.sum(self.energy_records[episode][start_idx:]) if self.energy_records[episode] else 0
            last_n_rewards.append(ep_rewards)
            last_n_energies.append(ep_energy)

        # 保存总体结果（基于最后168步）
        overall_results = pd.DataFrame({
            'episode': range(1, self.test_episodes + 1),
            'total_reward_last168': last_n_rewards,
            'total_energy_last168': last_n_energies
        })
        overall_results.to_csv(os.path.join(self.result_dir, 'overall_results_last168.csv'), index=False)

        # 保存详细数据（只保留最后168步）
        for episode in range(self.test_episodes):
            n_steps = len(self.temperature_records[episode])
            start_idx = max(0, n_steps - 168)

            episode_data = []
            # 仅遍历最后168步
            for step in range(start_idx, n_steps):
                row = {
                    'episode': episode + 1,
                    'step': step + 1,  # 原始步号，便于追溯
                    'relative_step': step - start_idx + 1,  # 相对步号
                    'reward': self.energy_records[episode][step] if step < len(self.energy_records[episode]) else 0,
                    'outdoor_temp': self.outdoor_temp_records[episode][step] if step < len(
                        self.outdoor_temp_records[episode]) else 0,
                    'energy': self.energy_records[episode][step] if step < len(self.energy_records[episode]) else 0,
                }

                # 添加房间温度
                for room in range(5):
                    row[f'room_{room + 1}_temp'] = self.temperature_records[episode][step][room] if step < len(
                        self.temperature_records[episode]) else 0

                # 添加房间设定点
                for room in range(5):
                    row[f'room_{room + 1}_setpoint'] = self.action_records[episode][step][room] if step < len(
                        self.action_records[episode]) else 0

                # 添加Sigma值
                if step < len(self.sigma_records[episode]):
                    for room in range(5):
                        row[f'room_{room + 1}_sigma'] = self.sigma_records[episode][step][room]
                else:
                    for room in range(5):
                        row[f'room_{room + 1}_sigma'] = np.nan

                # 添加预测序列
                if step < len(self.prediction_records[episode]):
                    for i, pred in enumerate(self.prediction_records[episode][step]):
                        row[f'pred_{i + 1}'] = pred

                # 添加预测准确性
                if step < len(self.prediction_accuracy_records[episode]):
                    row['prediction_accuracy'] = self.prediction_accuracy_records[episode][step]
                else:
                    row['prediction_accuracy'] = np.nan

                episode_data.append(row)

            # 保存为CSV
            episode_df = pd.DataFrame(episode_data)
            episode_df.to_csv(os.path.join(self.result_dir, f'episode_{episode + 1}_detailed_last168.csv'), index=False)

        # 保存统计摘要（基于最后168步）
        summary = {
            'mean_reward_last168': np.mean(last_n_rewards),
            'std_reward_last168': np.std(last_n_rewards),
            'mean_energy_last168': np.mean(last_n_energies),
            'std_energy_last168': np.std(last_n_energies),
            'min_reward_last168': np.min(last_n_rewards),
            'max_reward_last168': np.max(last_n_rewards),
            'min_energy_last168': np.min(last_n_energies),
            'max_energy_last168': np.max(last_n_energies),
            # 预测误差（仅最后168步）
            'mean_prediction_error_last168': np.mean(
                [np.mean(acc[-168:]) if len(acc) > 168 else np.mean(acc) for acc in self.prediction_accuracy_records if
                 len(acc) > 0]) if any(
                len(acc) > 0 for acc in self.prediction_accuracy_records) else 0,
            # Sigma统计（仅最后168步）
            'mean_sigma_last168': np.mean(
                [np.mean(np.array(s)[-168:].flatten()) if len(s) > 168 else np.mean(np.array(s).flatten()) for s in
                 self.sigma_records if len(s) > 0]) if any(
                len(s) > 0 for s in self.sigma_records) else 0,
            'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        summary_path = os.path.join(self.result_dir, 'test_summary_last168.txt')
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        # 保存预测准确性详细分析（仅最后168步）
        if any(len(acc) > 0 for acc in self.prediction_accuracy_records):
            pred_analysis_path = os.path.join(self.result_dir, 'prediction_analysis_last168.txt')
            with open(pred_analysis_path, 'w') as f:
                f.write("Prediction Accuracy Analysis (Last 168 steps)\n")
                f.write("=" * 50 + "\n")
                for i, acc in enumerate(self.prediction_accuracy_records):
                    if len(acc) > 0:
                        acc_last = acc[-168:] if len(acc) > 168 else acc
                        f.write(f"Episode {i + 1}:\n")
                        f.write(f"  Mean Error: {np.mean(acc_last):.4f}°C\n")
                        f.write(f"  Std Error: {np.std(acc_last):.4f}°C\n")
                        f.write(f"  Max Error: {np.max(acc_last):.4f}°C\n")
                        f.write(f"  Min Error: {np.min(acc_last):.4f}°C\n")
                        f.write(f"  Samples: {len(acc_last)}\n\n")

        # 保存Sigma详细分析（仅最后168步）
        if any(len(s) > 0 for s in self.sigma_records):
            sigma_analysis_path = os.path.join(self.result_dir, 'sigma_analysis_last168.txt')
            with open(sigma_analysis_path, 'w') as f:
                f.write("Sigma (Action Shrinkage) Analysis (Last 168 steps)\n")
                f.write("=" * 50 + "\n")
                for i, sigma_ep in enumerate(self.sigma_records):
                    if len(sigma_ep) > 0:
                        sigma_arr = np.array(sigma_ep)
                        if len(sigma_arr) > 168:
                            sigma_arr = sigma_arr[-168:]
                        f.write(f"Episode {i + 1}:\n")
                        f.write(f"  Mean Sigma (All Rooms): {np.mean(sigma_arr):.4f}\n")
                        f.write(f"  Std Sigma: {np.std(sigma_arr):.4f}\n")
                        f.write(f"  Max Sigma: {np.max(sigma_arr):.4f}\n")
                        f.write(f"  Min Sigma: {np.min(sigma_arr):.4f}\n")
                        for room in range(5):
                            room_sigma = sigma_arr[:, room]
                            f.write(f"  Room {room + 1} Mean Sigma: {np.mean(room_sigma):.4f}\n")
                        f.write(f"  Samples: {len(sigma_arr)}\n\n")

        print(f"Test results (last 168 steps) saved to: {self.result_dir}")
        print(
            f"Average Reward (last 168 steps): {summary['mean_reward_last168']:.2f} ± {summary['std_reward_last168']:.2f}")
        print(
            f"Average Energy (last 168 steps): {summary['mean_energy_last168']:.2f} kWh ± {summary['std_energy_last168']:.2f} kWh")
        if summary['mean_prediction_error_last168'] > 0:
            print(f"Average Prediction Error (last 168 steps): {summary['mean_prediction_error_last168']:.4f}°C")
        if summary['mean_sigma_last168'] > 0:
            print(f"Average Sigma (last 168 steps): {summary['mean_sigma_last168']:.4f}")


def main():
    """主测试函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型路径
    policy_model_path = "policy_net_episode_final.pth"
    value_model_path = "value_net_episode_final.pth"

    # 文件路径
    idf_path = "/opt/project/osmfile/env1.epJSON"
    epw_path = "/opt/project/weatherfile/CHN_Sichuan.Mianyang.561960_CSWD.epw"

    # 创建测试环境
    env = SinergymGCNEnv(
        idf_file=idf_path,
        epw_file=epw_path,
        pred_seq_len=8
    )

    # 创建图结构
    adj_matrix = create_adjacency_matrix()
    edge_index = adj_matrix_to_edge_index(adj_matrix)

    # 创建PPO智能体
    node_num = 5
    state_dim = 7
    action_dim = 5

    agent = PPOAgent(
        node_num=node_num,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        pred_seq_len=8
    )

    # 加载训练好的模型
    agent.load_model(policy_model_path, value_model_path)
    print("Model loaded successfully!")

    # 创建测试器
    tester = SinergymTester(
        env=env,
        agent=agent,
        edge_index=edge_index,
        test_episodes=1
    )

    # 开始测试
    rewards, energies = tester.test(deterministic=True)

    # 关闭环境
    env.close()

    print("Testing completed!")


if __name__ == "__main__":
    main()