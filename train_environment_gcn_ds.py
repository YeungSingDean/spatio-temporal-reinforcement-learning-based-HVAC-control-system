import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备：优先使用GPU，没有则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 1. 数据加载和预处理
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
        room_columns[f'cooling_setpoint_room_{i}'] = f'THERMALZONE{i}_ZoneThermostatCoolingSetpointTemperature_C_'  # 动作变为温度设定点

    # 选择需要的列（确保表头存在）
    all_columns = list(column_mapping.values()) + list(room_columns.values())
    available_columns = [col for col in all_columns if col in data.columns]

    if len(available_columns) != len(all_columns):
        print("Warning: 部分列在数据集中不存在")
        print("存在的列:", data.columns.tolist())

    data_subset = data[available_columns]

    # 重命名列以便于处理
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    reverse_room_mapping = {v: k for k, v in room_columns.items()}
    all_reverse_mapping = {**reverse_mapping, **reverse_room_mapping}

    data_renamed = data_subset.rename(columns=all_reverse_mapping)

    return data_renamed


# 2. 构建邻接矩阵和边索引（确保在GPU上）
def create_adjacency_matrix():
    """创建邻接矩阵"""
    adj_matrix = np.array([
        [1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1]
    ])
    return adj_matrix


def adj_matrix_to_edge_index(adj_matrix):
    """将邻接矩阵转换为edge_index并强制移到设备"""
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1 and i != j:  # 排除自环
                edge_index.append([i, j])
    # 转换为张量并移到设备（关键修正：确保edge_index在GPU上）
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    return edge_index


# 3. 环境GCN模型定义
class EnvironmentGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(EnvironmentGCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        # 确保x和edge_index在同一设备（额外检查）
        x = x.to(edge_index.device)
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = self.activation(self.gcn3(x, edge_index))
        x = self.output_layer(x)
        return x


# 4. 数据归一化类
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


# 5. 准备训练数据
def prepare_training_data(data, lookback=1):
    """准备GCN训练数据"""
    num_rooms = 5
    num_samples = len(data) - lookback

    X = []
    y = []

    for i in range(num_samples):
        current_data = data.iloc[i]
        next_data = data.iloc[i + 1]

        # 构建节点特征 (8维: 7状态 + 1动作)
        node_features = []
        for room_idx in range(1, 6):
            features = [
                current_data[f'temperature_room_{room_idx}'],
                current_data[f'humidity_room_{room_idx}'],
                current_data[f'occupancy_heat_room_{room_idx}'],
                current_data['outdoor_temp'],
                current_data['total_energy'],
                current_data['diffuse_solar'],
                current_data['direct_solar'],
                current_data[f'cooling_setpoint_room_{room_idx}']  # 动作变为温度设定点
            ]
            node_features.append(features)

        # 构建目标 (3维: 下一时刻温度、湿度、总能耗)
        targets = []
        for room_idx in range(1, 6):
            target = [
                next_data[f'temperature_room_{room_idx}'],
                next_data[f'humidity_room_{room_idx}'],
                next_data['total_energy']
            ]
            targets.append(target)

        X.append(node_features)
        y.append(targets)

    return np.array(X), np.array(y)


# 6. 绘制预测对比图
def plot_predictions_comparison(model, test_dataset, input_normalizer, output_normalizer,
                                edge_index, start_idx=752, num_points=168):
    """绘制真实值与预测值的对比图"""
    model.eval()
    model = model.to('cpu')  # 可视化时移回CPU
    edge_index = edge_index.to('cpu')

    end_idx = min(start_idx + num_points, len(test_dataset))
    indices = range(start_idx, end_idx)

    true_temperatures = [[] for _ in range(5)]
    pred_temperatures = [[] for _ in range(5)]
    true_humidities = [[] for _ in range(5)]
    pred_humidities = [[] for _ in range(5)]
    true_energies = []
    pred_energies = []

    with torch.no_grad():
        for idx in indices:
            data_point = test_dataset[idx]
            x = data_point.x.to('cpu')  # 移到CPU处理
            output = model(x, edge_index)

            # 反归一化
            pred_denormalized = output_normalizer.inverse_transform(
                output.numpy().reshape(-1, 3)
            ).reshape(5, 3)
            true_denormalized = output_normalizer.inverse_transform(
                data_point.y.cpu().numpy().reshape(-1, 3)  # 确保y在CPU
            ).reshape(5, 3)

            # 存储数据
            for room_idx in range(5):
                true_temperatures[room_idx].append(true_denormalized[room_idx, 0])
                pred_temperatures[room_idx].append(pred_denormalized[room_idx, 0])
                true_humidities[room_idx].append(true_denormalized[room_idx, 1])
                pred_humidities[room_idx].append(pred_denormalized[room_idx, 1])
            true_energies.append(true_denormalized[0, 2])
            pred_energies.append(pred_denormalized[0, 2])

    # 绘制温度对比图
    plt.figure(figsize=(30, 10))
    for room_idx in range(5):
        plt.subplot(2, 3, room_idx + 1)
        plt.plot(true_temperatures[room_idx], 'b-', label='真实值')
        plt.plot(pred_temperatures[room_idx], 'r--', label='预测值')
        plt.title(f'房间 {room_idx + 1} 温度对比')
        plt.xlabel('时间步')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('temperature_comparison.png', dpi=300)
    plt.show()

    # 绘制湿度对比图
    plt.figure(figsize=(15, 10))
    for room_idx in range(5):
        plt.subplot(2, 3, room_idx + 1)
        plt.plot(true_humidities[room_idx], 'b-', label='真实值')
        plt.plot(pred_humidities[room_idx], 'r--', label='预测值')
        plt.title(f'房间 {room_idx + 1} 湿度对比')
        plt.xlabel('时间步')
        plt.ylabel('湿度 (%)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('humidity_comparison.png', dpi=300)
    plt.show()

    # 绘制总能耗对比图
    plt.figure(figsize=(10, 6))
    plt.plot(true_energies, 'b-', label='真实能耗')
    plt.plot(pred_energies, 'r--', label='预测能耗')
    plt.title('总能耗对比')
    plt.xlabel('时间步')
    plt.ylabel('能耗 (J)')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_comparison.png', dpi=300)
    plt.show()

    # 计算评估指标
    print("\n模型预测性能评估:")
    print("=" * 50)
    for room_idx in range(5):
        temp_mae = np.mean(np.abs(np.array(true_temperatures[room_idx]) - np.array(pred_temperatures[room_idx])))
        temp_rmse = np.sqrt(np.mean((np.array(true_temperatures[room_idx]) - np.array(pred_temperatures[room_idx]))**2))
        humidity_mae = np.mean(np.abs(np.array(true_humidities[room_idx]) - np.array(pred_humidities[room_idx])))
        humidity_rmse = np.sqrt(np.mean((np.array(true_humidities[room_idx]) - np.array(pred_humidities[room_idx]))**2))
        print(f"房间 {room_idx + 1}:")
        print(f"  温度 - MAE: {temp_mae:.4f}°C, RMSE: {temp_rmse:.4f}°C")
        print(f"  湿度 - MAE: {humidity_mae:.4f}%, RMSE: {humidity_rmse:.4f}%")
    energy_mae = np.mean(np.abs(np.array(true_energies) - np.array(pred_energies)))
    energy_rmse = np.sqrt(np.mean((np.array(true_energies) - np.array(pred_energies))** 2))
    print(f"总能耗 - MAE: {energy_mae:.2f}J, RMSE: {energy_rmse:.2f}J")


# 7. 训练函数
def train_environment_gcn(data_path, model_save_path='environment_gcn.pth'):
    """训练环境GCN模型"""
    # 加载数据
    data = load_and_preprocess_data(data_path)
    print(f"数据形状: {data.shape}")

    # 创建邻接矩阵和edge_index（强制在GPU上）
    adj_matrix = create_adjacency_matrix()
    edge_index = adj_matrix_to_edge_index(adj_matrix)
    print(f"边索引形状: {edge_index.shape}, 设备: {edge_index.device}")  # 检查设备

    # 准备训练数据
    X, y = prepare_training_data(data)
    print(f"训练数据 - X形状: {X.shape}, y形状: {y.shape}")

    # 重塑数据用于归一化
    X_reshaped = X.reshape(-1, X.shape[-1])  # (samples * nodes, features)
    y_reshaped = y.reshape(-1, y.shape[-1])  # (samples * nodes, targets)

    # 数据归一化
    input_normalizer = Normalizer()
    output_normalizer = Normalizer()
    input_normalizer.fit(X_reshaped)
    output_normalizer.fit(y_reshaped)

    X_normalized = input_normalizer.transform(X_reshaped).reshape(X.shape)
    y_normalized = output_normalizer.transform(y_reshaped).reshape(y.shape)

    # 转换为PyTorch张量并移到设备（关键：确保x和y在GPU上）
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    y_tensor = torch.FloatTensor(y_normalized).to(device)
    print(f"X张量设备: {X_tensor.device}, y张量设备: {y_tensor.device}")  # 检查设备

    # 创建数据集（确保所有张量在同一设备）
    dataset = []
    for i in range(len(X_tensor)):
        data_point = Data(
            x=X_tensor[i],  # 已在GPU
            y=y_tensor[i],  # 已在GPU
            edge_index=edge_index  # 已在GPU（关键修正）
        )
        dataset.append(data_point)

    # 划分训练测试集
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型并移到设备
    input_dim = 8  # 7状态 + 1动作
    output_dim = 3  # 温度、湿度、总能耗
    model = EnvironmentGCN(input_dim, output_dim).to(device)
    print(f"模型设备: {next(model.parameters()).device}")  # 检查模型设备

    # 损失函数和优化器
    criterion = nn.MSELoss().to(device)  # 损失函数移到GPU
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    # 训练循环
    num_epochs = 200
    train_losses = []
    test_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            # 确保batch中所有数据在设备上（双重保险）
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)  # x和edge_index均在GPU
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index)
                loss = criterion(output, batch.y)
                test_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d}, 训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}')

    # 保存模型（移回CPU兼容保存）
    torch.save({
        'model_state_dict': model.cpu().state_dict(),
        'input_normalizer_mean': input_normalizer.scaler.mean_,
        'input_normalizer_scale': input_normalizer.scaler.scale_,
        'output_normalizer_mean': output_normalizer.scaler.mean_,
        'output_normalizer_scale': output_normalizer.scaler.scale_,
        'edge_index': edge_index.cpu()
    }, model_save_path)
    print(f"模型已保存至 {model_save_path}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('GCN训练曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=300)
    plt.show()

    # 绘制预测对比图
    plot_predictions_comparison(model, test_dataset, input_normalizer, output_normalizer, edge_index)

    return model, input_normalizer, output_normalizer, train_dataset, test_dataset


# 8. 加载训练好的模型
def load_environment_gcn(model_path):
    """加载训练好的环境GCN模型"""
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = 8
    output_dim = 3
    model = EnvironmentGCN(input_dim, output_dim).to(device)
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

    edge_index = checkpoint['edge_index'].to(device)  # 加载后移到设备

    return model, input_normalizer, output_normalizer, edge_index


# 主执行函数
if __name__ == "__main__":
    data_path = r"C:\Users\TRY\Desktop\simdata\2\traindata.csv"
    model_save_path = "environment_gcn.pth"

    try:
        # 训练环境GCN
        model, input_normalizer, output_normalizer, train_dataset, test_dataset = train_environment_gcn(data_path,
                                                                                                        model_save_path)

        # 测试加载功能
        loaded_model, loaded_input_norm, loaded_output_norm, edge_index = load_environment_gcn(model_save_path)
        print("GCN模型训练和评估完成!")

    except Exception as e:
        print(f"训练过程出错: {e}")
        # 打印设备信息用于调试
        print(f"当前设备: {device}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数: {torch.cuda.device_count()}")