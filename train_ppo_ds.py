import torch
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from environment_ds import DataDrivenHVACEnv, load_and_preprocess_data, adj_matrix_to_edge_index, \
    create_adjacency_matrix
from ppo_agent_ds import PPOAgent


class PPOTrainer:
    """PPO训练器"""

    def __init__(self, env, agent, edge_index, max_episodes=1000, max_steps_per_episode=1000,
                 update_frequency=2048, save_interval=50, log_interval=10):
        self.env = env
        self.agent = agent
        self.edge_index = edge_index
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        self.save_interval = save_interval
        self.log_interval = log_interval

        # 获取动作边界
        self.min_actions, self.max_actions = env.get_action_bounds()
        print(f"Action bounds - Min: {self.min_actions}, Max: {self.max_actions}")

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.action_clipping_stats = []  # 记录动作裁剪统计

        # 创建保存目录
        self.save_dir = f"ppo_models_{datetime.now().strftime('%Y%m%d')}"
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Training started. Models will be saved to: {self.save_dir}")
        print(f"Device: {agent.device}")

    def scale_actions(self, actions):
        """将动作从[-1, 1]范围缩放到实际动作范围"""
        scaled_actions = []
        for i, action in enumerate(actions):
            # 从[-1, 1]映射到[min_action, max_action]
            scaled_action = (action + 1) / 2 * (self.max_actions[i] - self.min_actions[i]) + self.min_actions[i]
            scaled_actions.append(scaled_action)
        return np.array(scaled_actions)

    def train(self):
        """训练循环"""
        start_time = time.time()
        total_steps = 0

        for episode in range(self.max_episodes):
            state, pred_sequence = self.env.reset()  # 现在返回状态和预测序列
            episode_reward = 0
            episode_steps = 0
            episode_clipping_count = 0

            for step in range(self.max_steps_per_episode):
                # 选择动作 - 现在传入预测序列
                action, log_prob, value = self.agent.get_action(state.numpy(), pred_sequence, self.edge_index)

                # 缩放动作到实际范围
                scaled_action = self.scale_actions(action)

                # 执行动作 - 现在返回下一状态、下一预测序列、奖励等
                next_state, next_pred_sequence, reward, done, _ = self.env.step(scaled_action.tolist())

                # 检查动作是否被裁剪
                min_actions, max_actions = self.env.get_action_bounds()
                for i, a in enumerate(scaled_action):
                    if a <= min_actions[i] or a >= max_actions[i]:
                        episode_clipping_count += 1
                        break

                # 存储转移 - 现在包含预测序列
                self.agent.store_transition(
                    state.numpy(), pred_sequence, action, reward,
                    next_state.numpy(), next_pred_sequence, done, log_prob, value
                )

                state = next_state
                pred_sequence = next_pred_sequence
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # 定期更新网络
                if len(self.agent.states) >= self.update_frequency:
                    policy_loss, value_loss, entropy = self.agent.update(self.edge_index)

                    if policy_loss != 0:  # 只有当有更新时才记录
                        self.policy_losses.append(policy_loss)
                        self.value_losses.append(value_loss)
                        self.entropies.append(entropy)

                if done:
                    break

            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            clipping_percentage = episode_clipping_count / episode_steps * 100 if episode_steps > 0 else 0
            self.action_clipping_stats.append(clipping_percentage)

            # 记录和保存
            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                avg_clipping = np.mean(self.action_clipping_stats[-self.log_interval:])

                print(f"Episode {episode:4d}, "
                      f"Steps: {total_steps:6d}, "
                      f"Reward: {episode_reward:8.2f}, "
                      f"Avg Reward: {avg_reward:8.2f}, "
                      f"Length: {episode_steps:3d}, "
                      f"Avg Length: {avg_length:5.1f}, "
                      f"Clipping: {clipping_percentage:5.1f}%")

            if episode % self.save_interval == 0 and episode > 0:
                self.save_models(episode)
                self.plot_training_progress()

        # 训练结束
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # 保存最终模型和图表
        self.save_models('final')
        self.plot_training_progress()

        return self.episode_rewards, self.policy_losses, self.value_losses

    def save_models(self, episode):
        """保存模型"""
        policy_path = os.path.join(self.save_dir, f"policy_net_episode_{episode}.pth")
        value_path = os.path.join(self.save_dir, f"value_net_episode_{episode}.pth")
        self.agent.save_model(policy_path, value_path)

        # 保存训练统计
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies,
            'action_clipping_stats': self.action_clipping_stats,
            'action_bounds': {
                'min': self.min_actions,
                'max': self.max_actions
            }
        }
        stats_path = os.path.join(self.save_dir, f"training_stats_episode_{episode}.npy")
        np.save(stats_path, stats, allow_pickle=True)

    def plot_training_progress(self):
        """绘制训练进度图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 奖励曲线
        if self.episode_rewards:
            ax1.plot(self.episode_rewards)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)

            # 添加移动平均
            window = min(50, len(self.episode_rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
                ax1.plot(range(window - 1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2,
                         label=f'MA({window})')
                ax1.legend()

        # 策略损失
        if self.policy_losses:
            ax2.plot(self.policy_losses)
            ax2.set_title('Policy Loss')
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.grid(True)

        # 价值损失
        if self.value_losses:
            ax3.plot(self.value_losses)
            ax3.set_title('Value Loss')
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.grid(True)

        # 动作裁剪统计
        if self.action_clipping_stats:
            ax4.plot(self.action_clipping_stats)
            ax4.set_title('Action Clipping Percentage')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Clipping %')
            ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # 加载数据
    data_path = r"C:\Users\TRY\Desktop\simdata\traindata_pred.csv"
    dataset = load_and_preprocess_data(data_path)

    # 创建环境
    env = DataDrivenHVACEnv(
        dataset=dataset,
        num_rooms=5,
        lookback=24,
        control_horizon=24*7,  # 一周的控制时域
        target_temp_range=(23, 25),
        energy_penalty_threshold=0.1
    )

    # 创建图结构
    adj_matrix = create_adjacency_matrix()
    edge_index = adj_matrix_to_edge_index(adj_matrix)

    # 创建PPO智能体
    node_num = 5
    state_dim = 7  # 7维状态
    action_dim = 5  # 5个房间的动作
    pred_seq_len = 8  # 预测序列长度

    agent = PPOAgent(
        node_num=node_num,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=3e-4,
        gamma=0.9,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        pred_seq_len=pred_seq_len  # 新增参数
    )

    # 创建训练器
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        edge_index=edge_index,
        max_episodes=500,
        max_steps_per_episode=1000,
        update_frequency=2048,
        save_interval=50,
        log_interval=10
    )

    # 开始训练
    rewards, policy_losses, value_losses = trainer.train()

    print("Training completed!")


if __name__ == "__main__":
    main()