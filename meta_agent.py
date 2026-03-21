import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class OnlineMetaAgent(nn.Module):
    """
    状态增强版在线元智能体
    输入: [过去 3 步设定点 Delta, 过去 3 步室温误差] -> 维度 6
    """

    def __init__(self, history_len=3, num_actions=3, lr=0.01):
        super(OnlineMetaAgent, self).__init__()
        self.history_len = history_len

        # 输入维度翻倍：history_len (动作差) + history_len (温度差)
        input_dim = history_len * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),  # 稍微增加一点神经元容量以处理更多状态
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.95

    def get_action(self, state_history):
        """
        state_history: shape (5, 6)
        """
        state_tensor = torch.FloatTensor(state_history)
        logits = self.net(state_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        self.log_probs.append(dist.log_prob(actions))
        return actions.numpy()

    def store_reward(self, reward_array):
        self.rewards.append(torch.FloatTensor(reward_array))

    def update_policy(self):
        if len(self.rewards) == 0:
            return

        discounted_rewards = []
        R = torch.zeros(5)
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.stack(discounted_rewards)

        if discounted_rewards.shape[0] > 1:
            mean = discounted_rewards.mean(dim=0)
            std = discounted_rewards.std(dim=0) + 1e-9
            discounted_rewards = (discounted_rewards - mean) / std

        loss = 0
        for log_prob, G in zip(self.log_probs, discounted_rewards):
            loss -= (log_prob * G).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []