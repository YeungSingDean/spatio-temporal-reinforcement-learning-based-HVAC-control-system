import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sinergym_env import SinergymGCNEnv
from ppo_agent_ds import PPOAgent



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

        # 获取动作边界
        self.min_actions = [20] * 5
        self.max_actions = [28] * 5

        # 测试结果存储
        self.episode_rewards = []
        self.episode_energies = []
        self.temperature_records = []
        self.action_records = []
        self.outdoor_temp_records = []
        self.energy_records = []
        self.prediction_records = []
        self.prediction_accuracy_records = []  # 新增：记录预测准确性

        # 创建结果目录
        self.result_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"Testing started. Results will be saved to: {self.result_dir}")

    def scale_actions(self, actions):
        """将动作从[-1, 1]范围缩放到实际动作范围"""
        scaled_actions = []
        for i, action in enumerate(actions):
            # 从[-1, 1]映射到[min_action, max_action]
            scaled_action = (action + 1) / 2 * (self.max_actions[i] - self.min_actions[i]) + self.min_actions[i]
            scaled_actions.append(scaled_action)
        return np.array(scaled_actions)

    def test(self, deterministic=True):
        """测试循环"""
        print(f"Starting testing with {self.test_episodes} episodes...")

        for episode in range(self.test_episodes):
            state, info = self.env.reset()
            pred_sequence = info['pred_sequence']  # 获取预测序列
            episode_reward = 0
            episode_energy = 0
            step_count = 0

            # 存储每步数据用于绘图
            episode_temperatures = []
            episode_actions = []
            episode_rewards = []
            episode_outdoor_temps = []
            episode_energies = []
            episode_predictions = []
            episode_prediction_accuracy = []  # 新增：记录预测准确性

            done = False
            while not done:
                # 选择动作（需要传入预测序列）
                action, _, _ = self.agent.get_action(state.numpy(), pred_sequence, self.edge_index,
                                                     deterministic=deterministic)

                # 缩放动作到实际范围
                scaled_action = self.scale_actions(action)

                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(scaled_action.tolist())
                next_pred_sequence = info['pred_sequence']  # 获取下一步的预测序列
                done = terminated or truncated

                # 记录数据
                current_temperatures = [next_state[i, 0].item() for i in range(self.env.num_zones)]
                current_energy = info.get('total_energy_kwh', 0.0)
                current_outdoor_temp = next_state[0, 3].item()

                episode_temperatures.append(current_temperatures)
                episode_actions.append(scaled_action)
                episode_rewards.append(reward)
                episode_outdoor_temps.append(current_outdoor_temp)
                episode_energies.append(current_energy)
                episode_predictions.append(pred_sequence.copy())

                # 计算预测准确性（比较预测值与实际未来值）
                if step_count > 0 and step_count < len(episode_outdoor_temps):
                    # 计算当前预测序列的第一个值（下一步预测）与下一步实际值的差异
                    pred_accuracy = abs(pred_sequence[0] - current_outdoor_temp)
                    episode_prediction_accuracy.append(pred_accuracy)

                episode_reward += reward
                episode_energy += current_energy
                step_count += 1

                state = next_state
                pred_sequence = next_pred_sequence

                if step_count >= 168:  # 限制测试步数（一周：7天×24小时=168步）
                    break

            # 记录回合结果
            self.episode_rewards.append(episode_reward)
            self.episode_energies.append(episode_energy)
            self.temperature_records.append(episode_temperatures)
            self.action_records.append(episode_actions)
            self.outdoor_temp_records.append(episode_outdoor_temps)
            self.energy_records.append(episode_energies)
            self.prediction_records.append(episode_predictions)
            self.prediction_accuracy_records.append(episode_prediction_accuracy)

            print(f"Episode {episode + 1:2d}: "
                  f"Total Reward: {episode_reward:8.2f}, "
                  f"Total Energy: {episode_energy:8.2f} kWh, "
                  f"Steps: {step_count:3d}")

            # 每回合保存图表
            self.plot_episode_results(episode, episode_temperatures, episode_actions,
                                      episode_rewards, episode_outdoor_temps, episode_energies,
                                      episode_predictions, episode_prediction_accuracy)

        # 保存总体测试结果
        self.save_test_results()

        return self.episode_rewards, self.episode_energies

    def plot_temperature_subplot(self, episode, temperatures):
        """单独绘制温度曲线子图"""

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

        #plt.title(f'Room Temperatures - Episode {episode + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3)

        # 保存单独的温度图
        plt.savefig(os.path.join(self.result_dir, f'episode_{episode + 1}_temperature_only.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_episode_results(self, episode, temperatures, actions, rewards, outdoor_temps, energies, predictions,
                             prediction_accuracy):
        """绘制单回合结果 - 添加了预测序列显示和准确性分析"""
        # 先绘制单独的温度图
        self.plot_temperature_subplot(episode, temperatures)

        # 创建4x2的子图布局
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()

        time_steps = range(len(temperatures))

        # 1. 温度曲线（已在单独的函数中绘制，这里保留用于综合图）
        temperatures = np.array(temperatures)
        for i in range(5):
            axes[0].plot(time_steps, temperatures[:, i], label=f'Room {i + 1}', linewidth=1.5)

        # 目标温度范围
        axes[0].axhline(y=23.0, color='r', linestyle='--', alpha=0.7, label='Comfort Range')
        axes[0].axhline(y=25.0, color='r', linestyle='--', alpha=0.7)
        axes[0].fill_between(time_steps, 23.0, 25.0, alpha=0.1, color='red')
        axes[0].set_title('Room Temperatures')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 动作曲线
        actions = np.array(actions)
        for i in range(5):
            axes[1].plot(time_steps, actions[:, i], label=f'Room {i + 1} Setpoint', linewidth=1.5)
        axes[1].set_title('Temperature Setpoints')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Setpoint (°C)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

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
        all_temperatures = temperatures.flatten()
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
        mean_indoor_temp = np.mean(temperatures, axis=1)
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
        """保存测试结果为CSV格式"""
        # 保存总体结果
        overall_results = pd.DataFrame({
            'episode': range(1, self.test_episodes + 1),
            'total_reward': self.episode_rewards,
            'total_energy': self.episode_energies
        })
        overall_results.to_csv(os.path.join(self.result_dir, 'overall_results.csv'), index=False)

        # 保存详细数据
        for episode in range(self.test_episodes):
            # 创建详细数据DataFrame
            episode_data = []
            for step in range(len(self.temperature_records[episode])):
                row = {
                    'episode': episode + 1,
                    'step': step + 1,
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
            episode_df.to_csv(os.path.join(self.result_dir, f'episode_{episode + 1}_detailed.csv'), index=False)

        # 保存统计摘要
        summary = {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_energy': np.mean(self.episode_energies),
            'std_energy': np.std(self.episode_energies),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_energy': np.min(self.episode_energies),
            'max_energy': np.max(self.episode_energies),
            'mean_prediction_error': np.mean(
                [np.mean(acc) for acc in self.prediction_accuracy_records if len(acc) > 0]) if any(
                len(acc) > 0 for acc in self.prediction_accuracy_records) else 0,
            'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        summary_path = os.path.join(self.result_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        # 保存预测准确性详细分析
        if any(len(acc) > 0 for acc in self.prediction_accuracy_records):
            pred_analysis_path = os.path.join(self.result_dir, 'prediction_analysis.txt')
            with open(pred_analysis_path, 'w') as f:
                f.write("Prediction Accuracy Analysis\n")
                f.write("=" * 40 + "\n")
                for i, acc in enumerate(self.prediction_accuracy_records):
                    if len(acc) > 0:
                        f.write(f"Episode {i + 1}:\n")
                        f.write(f"  Mean Error: {np.mean(acc):.4f}°C\n")
                        f.write(f"  Std Error: {np.std(acc):.4f}°C\n")
                        f.write(f"  Max Error: {np.max(acc):.4f}°C\n")
                        f.write(f"  Min Error: {np.min(acc):.4f}°C\n")
                        f.write(f"  Samples: {len(acc)}\n\n")

        print(f"Test results saved to: {self.result_dir}")
        print(f"Average Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Average Energy: {summary['mean_energy']:.2f} kWh ± {summary['std_energy']:.2f} kWh")
        if 'mean_prediction_error' in summary and summary['mean_prediction_error'] > 0:
            print(f"Average Prediction Error: {summary['mean_prediction_error']:.4f}°C")


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