import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import random


# 1. 环境GCN模型定义
class EnvironmentGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(EnvironmentGCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = self.activation(self.gcn3(x, edge_index))
        x = self.output_layer(x)
        return x


# 2. 数据归一化类
class Normalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, data):
        self.scaler.fit(data)
        self.is_fitted = True

    def transform(self, data):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first")
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first")
        return self.scaler.inverse_transform(data)


# 3. 邻接矩阵处理函数
def create_adjacency_matrix():
    """根据文档创建邻接矩阵"""
    adj_matrix = np.array([
        [1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1]
    ])
    return adj_matrix


def adj_matrix_to_edge_index(adj_matrix):
    """将邻接矩阵转换为edge_index格式"""
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1 and i != j:  # 排除自环
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


# 4. 数据加载函数
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    print("Loading data...")
    data = pd.read_csv(file_path)

    # 更新列名映射
    column_mapping = {
        'outdoor_temp': 'Environment_SiteOutdoorAirDrybulbTemperature_C_',
        'total_energy': 'CHILLER_WATERCOOLED_ChillerElectricityEnergy_J_',
        'diffuse_solar': 'Environment_SiteDiffuseSolarRadiationRatePerArea_W_m2_',
        'direct_solar': 'Environment_SiteDirectSolarRadiationRatePerArea_W_m2_'
    }

    # 更新房间相关的列
    room_columns = {}
    for i in range(1, 6):  # 5个房间
        room_columns[f'temperature_room_{i}'] = f'THERMALZONE{i}_ZoneAirTemperature_C_'
        room_columns[f'humidity_room_{i}'] = f'THERMALZONE{i}_ZoneAirRelativeHumidity___'
        room_columns[f'occupancy_heat_room_{i}'] = f'THERMALZONE{i}_ZonePeopleSensibleHeatingEnergy_J_'
        room_columns[f'cooling_setpoint_room_{i}'] = f'THERMALZONE{i}_ZoneThermostatCoolingSetpointTemperature_C_'

    # 添加预测序列列
    pred_columns = {}
    for i in range(1, 9):  # 8个预测步
        pred_columns[f'step{i}_OutdoorTemp_C_'] = f'step{i}_OutdoorTemp_C_'

    # 选择需要的列
    all_columns = list(column_mapping.values()) + list(room_columns.values()) + list(pred_columns.values())
    available_columns = [col for col in all_columns if col in data.columns]

    if len(available_columns) != len(all_columns):
        print("Warning: Some columns are missing from the dataset")
        print("Available columns:", data.columns.tolist())

    data_subset = data[available_columns]

    # 重命名列以便于处理
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    reverse_room_mapping = {v: k for k, v in room_columns.items()}
    reverse_pred_mapping = {v: k for k, v in pred_columns.items()}
    all_reverse_mapping = {**reverse_mapping, **reverse_room_mapping, **reverse_pred_mapping}

    data_renamed = data_subset.rename(columns=all_reverse_mapping)

    return data_renamed


# 5. 完整的强化学习环境
class DataDrivenHVACEnv:
    def __init__(self, dataset, num_rooms=5, lookback=24, control_horizon=8,
                 target_temp_range=(23, 25), energy_penalty_threshold=0.4):
        """
        多房间HVAC强化学习环境

        Args:
            dataset: 包含完整时序数据的DataFrame
            num_rooms: 房间数量
            lookback: 初始时间步的回看窗口
            control_horizon: 控制时域长度
            target_temp_range: 目标温度范围
            energy_penalty_threshold: 能耗惩罚阈值
        """
        self.dataset = dataset
        self.num_rooms = num_rooms
        self.lookback = lookback
        self.control_horizon = control_horizon
        self.target_temp_range = target_temp_range
        self.energy_penalty_threshold = energy_penalty_threshold

        # 环境参数
        self.current_step = None
        self.max_steps = len(dataset) - 1
        self.edge_index = adj_matrix_to_edge_index(create_adjacency_matrix())

        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练的环境GCN模型
        self.env_gcn, self.input_normalizer, self.output_normalizer = self._load_environment_gcn('environment_gcn.pth')
        self.env_gcn.eval()  # 设置为评估模式
        self.env_gcn = self.env_gcn.to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        # 动作空间和状态空间
        self.action_dim = num_rooms  # 每个房间一个温度设定点动作
        self.state_dim = 7  # 7维状态特征
        self.pred_seq_len = 8  # 预测序列长度

        # 奖励参数
        self.temp_weight = 1.0
        self.energy_weight = 1.0
        self.penalty_weight = 1.0

    def _load_environment_gcn(self, model_path):
        """加载预训练的环境GCN模型"""
        # 使用weights_only=False解决PyTorch 2.6的兼容性问题
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        input_dim = 8  # 7状态 + 1动作
        output_dim = 3  # 温度、湿度、总能耗
        model = EnvironmentGCN(input_dim, output_dim)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 重建归一化器
        input_normalizer = Normalizer()
        input_normalizer.scaler.mean_ = checkpoint['input_normalizer_mean']
        input_normalizer.scaler.scale_ = checkpoint['input_normalizer_scale']
        input_normalizer.is_fitted = True

        output_normalizer = Normalizer()
        output_normalizer.scaler.mean_ = checkpoint['output_normalizer_mean']
        output_normalizer.scaler.scale_ = checkpoint['output_normalizer_scale']
        output_normalizer.is_fitted = True

        return model, input_normalizer, output_normalizer

    def reset(self):
        """重置环境状态"""
        # 随机选择起始时间步（在lookback和max_steps-control_horizon之间）
        self.current_step = random.randint(
            self.lookback,
            min(self.max_steps - self.control_horizon, len(self.dataset) - self.control_horizon - 1)
        )

        # 重置步数计数器
        self.step_count = 0

        return self._get_observation()

    def _get_observation(self):
        """获取当前时刻的观察状态和预测序列"""
        current_data = self.dataset.iloc[self.current_step]

        # 构建GCN节点特征 (7维状态)
        features = []
        for i in range(self.num_rooms):
            room_features = [
                current_data[f'temperature_room_{i + 1}'],
                current_data[f'humidity_room_{i + 1}'],
                current_data[f'occupancy_heat_room_{i + 1}'],
                current_data['outdoor_temp'],
                current_data['total_energy'],
                current_data['diffuse_solar'],
                current_data['direct_solar']
            ]
            features.append(room_features)

        # 获取未来8个时间步的室外温度预测
        pred_sequence = [
            current_data[f'step{i}_OutdoorTemp_C_'] for i in range(1, 9)
        ]

        return torch.FloatTensor(features), np.array(pred_sequence)

    def _prepare_gcn_input(self, state, actions):
        """准备环境GCN的输入数据"""
        # 将动作添加到状态特征中，形成8维输入
        gcn_input = []
        for i in range(self.num_rooms):
            state_features = state[i].tolist()
            action_feature = actions[i]
            gcn_input.append(state_features + [action_feature])

        gcn_input = np.array(gcn_input)

        # 归一化输入
        gcn_input_normalized = self.input_normalizer.transform(gcn_input.reshape(-1, 8)).reshape(self.num_rooms, 8)

        return torch.FloatTensor(gcn_input_normalized)

    def _parse_gcn_output(self, gcn_output):
        """解析环境GCN的输出"""
        # 反归一化输出
        gcn_output_denormalized = self.output_normalizer.inverse_transform(
            gcn_output.detach().cpu().numpy().reshape(-1, 3)
        ).reshape(self.num_rooms, 3)

        # 提取预测值
        next_temperatures = gcn_output_denormalized[:, 0]  # 温度
        next_humidities = gcn_output_denormalized[:, 1]  # 湿度
        next_total_energy = gcn_output_denormalized[0, 2]  # 总能耗（所有房间相同）

        return next_temperatures, next_humidities, next_total_energy

    def calculate_reward(self, temperatures, total_energy):
        """计算奖励函数"""
        reward = 0

        # 温度奖励：越接近目标温度范围，奖励越高
        for temp in temperatures:
            if self.target_temp_range[0] <= temp <= self.target_temp_range[1]:
                # 在目标范围内，给予正奖励
                distance_to_center = abs(temp - np.mean(self.target_temp_range))
                max_distance = (self.target_temp_range[1] - self.target_temp_range[0]) / 2
                temp_reward = (1 - distance_to_center / max_distance) * self.temp_weight
                reward += temp_reward
            else:
                # 超出目标范围，给予惩罚
                if temp < self.target_temp_range[0]:
                    penalty = (self.target_temp_range[0] - temp) * self.penalty_weight
                else:
                    penalty = (temp - self.target_temp_range[1]) * self.penalty_weight
                reward -= penalty

        # 能耗奖励：能耗越低，奖励越高
        energy_reward = (self.energy_penalty_threshold - total_energy) * self.energy_weight
        reward += energy_reward

        # 能耗惩罚：超过阈值给予额外惩罚
        if total_energy > self.energy_penalty_threshold:
            reward -= self.penalty_weight * (total_energy - self.energy_penalty_threshold)

        return reward

    def step(self, actions):
        """
        执行动作并返回下一状态、奖励、完成标志

        Args:
            actions: 每个房间的温度设定点动作值列表
        """
        # 获取动作边界并裁剪动作
        min_actions, max_actions = self.get_action_bounds()
        clipped_actions = []
        for i, a in enumerate(actions):
            if a < min_actions[i]:
                clipped_actions.append(min_actions[i])
            elif a > max_actions[i]:
                clipped_actions.append(max_actions[i])
            else:
                clipped_actions.append(a)

        # 获取当前状态和预测序列
        current_state, current_pred_sequence = self._get_observation()

        # 准备环境GCN输入
        gcn_input = self._prepare_gcn_input(current_state, clipped_actions)  # 使用裁剪后的动作
        gcn_input = gcn_input.to(self.device)

        # 使用环境GCN预测下一状态
        with torch.no_grad():
            gcn_output = self.env_gcn(gcn_input, self.edge_index)

        # 解析GCN输出
        next_temperatures, next_humidities, next_total_energy = self._parse_gcn_output(gcn_output)

        # 获取下一时刻的真实外生变量
        next_step_data = self.dataset.iloc[self.current_step + 1]
        next_outdoor_temp = next_step_data['outdoor_temp']
        next_occupancy_heat = [
            next_step_data[f'occupancy_heat_room_{i + 1}'] for i in range(self.num_rooms)
        ]
        next_diffuse_solar = next_step_data['diffuse_solar']
        next_direct_solar = next_step_data['direct_solar']

        # 构建下一状态：预测的受控变量 + 真实的外生变量
        next_state = []
        for i in range(self.num_rooms):
            room_features = [
                next_temperatures[i],  # 预测的温度
                next_humidities[i],  # 预测的湿度
                next_occupancy_heat[i],  # 真实的人员显热
                next_outdoor_temp,  # 真实的外界温度
                next_total_energy,  # 预测的总能耗
                next_diffuse_solar,  # 真实的散射太阳辐射
                next_direct_solar  # 真实的直射太阳辐射
            ]
            next_state.append(room_features)

        # 获取下一时刻的预测序列
        next_pred_sequence = [
            next_step_data[f'step{i}_OutdoorTemp_C_'] for i in range(1, 9)
        ]

        # 计算奖励
        reward = self.calculate_reward(next_temperatures, next_total_energy)

        # 更新步数
        self.current_step += 1
        self.step_count += 1

        # 检查是否完成
        done = (self.step_count >= self.control_horizon) or (self.current_step >= self.max_steps)

        return torch.FloatTensor(next_state), np.array(next_pred_sequence), reward, done, {}

    def get_action_bounds(self):
        """获取动作空间的上下界"""
        # 温度设定点的合理范围（根据实际情况调整）
        # 通常温度设定点在20-28°C之间
        min_actions = [20.0] * self.num_rooms  # 最低温度设定点
        max_actions = [28.0] * self.num_rooms  # 最高温度设定点

        return min_actions, max_actions