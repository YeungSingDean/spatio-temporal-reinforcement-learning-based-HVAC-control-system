import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
import torch.optim as optim


class GCNPolicyNetwork(nn.Module):
    """带GCN嵌入和LSTM预测的策略网络"""

    def __init__(self, node_num, state_dim, action_dim, pred_seq_len=8, hidden_dim=64, hidden_dim2=256,
                 lstm_hidden_dim=64):
        super(GCNPolicyNetwork, self).__init__()
        self.node_num = node_num
        self.state_dim = state_dim
        self.pred_seq_len = pred_seq_len

        # GCN层用于提取空间特征
        self.gcn1 = GCNConv(state_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # LSTM层用于处理预测序列
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=1)

        # 全连接层 - 输入维度变为 GCN输出维度 + LSTM隐藏层维度
        self.fc = nn.Sequential(
            nn.Linear(node_num * hidden_dim + lstm_hidden_dim, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim * 2)  # 输出均值和标准差
        )

        # 使用tanh激活函数来限制动作范围
        self.tanh = nn.Tanh()

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)

    def forward(self, state, pred_sequence, edge_index):
        """
        Args:
            state: [batch_size, node_num, state_dim] or [node_num, state_dim]
            pred_sequence: [batch_size, pred_seq_len] or [pred_seq_len] - 未来室外温度预测序列
            edge_index: [2, num_edges]
        """
        batch_size = state.shape[0] if state.dim() == 3 else 1
        node_num = self.node_num
        state_dim = self.state_dim

        # ===== GCN部分 =====
        # 重塑状态为 [batch_size * node_num, state_dim]
        if state.dim() == 3:
            state_reshaped = state.view(-1, state_dim)
        else:
            state_reshaped = state.view(-1, state_dim)

        # GCN前向传播
        x_gcn = F.relu(self.gcn1(state_reshaped, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))

        # 重塑为 [batch_size, node_num * hidden_dim]
        x_gcn = x_gcn.view(batch_size, -1)

        # ===== LSTM预测部分 =====
        # 处理预测序列输入形状
        if pred_sequence.dim() == 1:
            pred_sequence = pred_sequence.unsqueeze(0)  # [pred_seq_len] -> [1, pred_seq_len]

        # 重塑为LSTM输入格式: [batch_size, seq_len, input_size=1]
        pred_sequence = pred_sequence.unsqueeze(-1)  # [batch_size, pred_seq_len, 1]

        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(pred_sequence)
        # 取最后一个时间步的隐藏状态
        x_lstm = h_n[-1]  # [batch_size, lstm_hidden_dim]

        # ===== 特征拼接 =====
        # 将GCN输出和LSTM输出拼接
        x_combined = torch.cat([x_gcn, x_lstm], dim=-1)  # [batch_size, node_num * hidden_dim + lstm_hidden_dim]

        # 全连接层
        output = self.fc(x_combined)

        # 分割为均值和标准差
        mean, std = torch.chunk(output, 2, dim=-1)

        # 使用tanh限制均值范围在[-1, 1]之间
        mean = self.tanh(mean)

        std = F.softplus(std) + 1e-4  # 确保标准差为正

        return mean, std


class ValueNetwork(nn.Module):
    """价值网络，结构与策略网络类似"""

    def __init__(self, node_num, state_dim, pred_seq_len=8, hidden_dim=64, hidden_dim2=256, lstm_hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.node_num = node_num
        self.state_dim = state_dim
        self.pred_seq_len = pred_seq_len

        # GCN层
        self.gcn1 = GCNConv(state_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(node_num * hidden_dim + lstm_hidden_dim, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # 输出单个价值
        )

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)

    def forward(self, state, pred_sequence, edge_index):
        batch_size = state.shape[0] if state.dim() == 3 else 1
        node_num = self.node_num
        state_dim = self.state_dim

        # ===== GCN部分 =====
        # 重塑状态
        if state.dim() == 3:
            state_reshaped = state.view(-1, state_dim)
        else:
            state_reshaped = state.view(-1, state_dim)

        # GCN前向传播
        x_gcn = F.relu(self.gcn1(state_reshaped, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))

        # 重塑
        x_gcn = x_gcn.view(batch_size, -1)

        # ===== LSTM预测部分 =====
        # 处理预测序列输入形状
        if pred_sequence.dim() == 1:
            pred_sequence = pred_sequence.unsqueeze(0)
        pred_sequence = pred_sequence.unsqueeze(-1)

        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(pred_sequence)
        x_lstm = h_n[-1]

        # ===== 特征拼接 =====
        x_combined = torch.cat([x_gcn, x_lstm], dim=-1)

        # 全连接层
        value = self.fc(x_combined)

        return value


class PPOAgent:
    """PPO智能体"""

    def __init__(self, node_num, state_dim, action_dim, device,
                 lr=3e-4, gamma=0.9, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 pred_seq_len=8):

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.pred_seq_len = pred_seq_len

        # 创建策略网络和价值网络
        self.policy_net = GCNPolicyNetwork(node_num, state_dim, action_dim, pred_seq_len).to(device)
        self.value_net = ValueNetwork(node_num, state_dim, pred_seq_len).to(device)

        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr)

        # 存储经验
        self.states = []
        self.pred_sequences = []  # 新增：存储预测序列
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_pred_sequences = []  # 新增：存储下一状态的预测序列
        self.dones = []
        self.log_probs = []
        self.values = []

    def get_action(self, state, pred_sequence, edge_index, deterministic=False):
        """根据状态和预测序列选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        pred_sequence_tensor = torch.FloatTensor(pred_sequence).to(self.device)
        edge_index_tensor = edge_index.to(self.device)

        with torch.no_grad():
            mean, std = self.policy_net(state_tensor, pred_sequence_tensor, edge_index_tensor)
            value = self.value_net(state_tensor, pred_sequence_tensor, edge_index_tensor)

            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                action = action.squeeze(0).cpu().numpy()
                log_prob = log_prob.item()  # 确保是标量
                value = value.squeeze().item()  # 确保是标量

                return action, log_prob, value

        return action.squeeze(0).cpu().numpy(), 0, value.squeeze().item()

    def store_transition(self, state, pred_sequence, action, reward, next_state, next_pred_sequence, done, log_prob,
                         value):
        """存储转移经验"""
        self.states.append(state)
        self.pred_sequences.append(pred_sequence)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.next_pred_sequences.append(next_pred_sequence)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, next_value):
        """计算广义优势估计"""
        values = self.values + [next_value]
        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns

    def update(self, edge_index, epochs=10, batch_size=64):
        """更新网络参数"""
        if len(self.states) < batch_size:
            return 0, 0, 0

        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        pred_sequences = torch.FloatTensor(np.array(self.pred_sequences)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        edge_index_tensor = edge_index.to(self.device)

        # 计算GAE和回报
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(self.next_states[-1]).unsqueeze(0).to(self.device)
            next_pred_sequence_tensor = torch.FloatTensor(self.next_pred_sequences[-1]).to(self.device)
            next_value = self.value_net(next_state_tensor, next_pred_sequence_tensor,
                                        edge_index_tensor).squeeze().cpu().item()

        advantages, returns = self.compute_gae(next_value)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮更新
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_pred_sequences = pred_sequences[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]

                # 计算新策略
                mean, std = self.policy_net(batch_states, batch_pred_sequences, edge_index_tensor)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # 策略损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                new_values = self.value_net(batch_states, batch_pred_sequences, edge_index_tensor)
                new_values = new_values.squeeze(-1)  # 从 [batch_size, 1] 变为 [batch_size]

                if batch_returns.dim() > 1:
                    batch_returns = batch_returns.squeeze(-1)

                value_loss_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )

                if value_loss_clipped.dim() > 1:
                    value_loss_clipped = value_loss_clipped.squeeze(-1)

                value_loss1 = F.mse_loss(new_values, batch_returns)
                value_loss2 = F.mse_loss(value_loss_clipped, batch_returns)
                value_loss = torch.max(value_loss1, value_loss2)

                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        # 清空经验缓冲区
        self.clear_memory()

        # 返回平均损失
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_value_loss = np.mean(value_losses) if value_losses else 0
        avg_entropy = np.mean(entropy_losses) if entropy_losses else 0

        return avg_policy_loss, avg_value_loss, avg_entropy

    def clear_memory(self):
        """清空经验缓冲区"""
        self.states = []
        self.pred_sequences = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_pred_sequences = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def save_model(self, policy_path, value_path):
        """保存模型"""
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.value_net.state_dict(), value_path)

    def load_model(self, policy_path, value_path):
        """加载模型"""
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))