import torch
import numpy as np
import os
import gymnasium as gym
import yaml
from sinergym.envs.eplus_env import EplusEnv

class CustomReward:
    """自定义奖励函数"""

    def __init__(self, num_zones=5, comfort_range=(23.0, 25.0), energy_weight=0.01, comfort_weight=1.0):
        self.num_zones = num_zones
        self.comfort_range = comfort_range
        self.energy_weight = energy_weight
        self.comfort_weight = comfort_weight

    def calculate(self, temperatures, total_energy_kwh):
        """计算奖励"""
        reward = 0.0

        # 舒适度奖励
        for temp in temperatures:
            if self.comfort_range[0] <= temp <= self.comfort_range[1]:
                # 在舒适范围内，给予正奖励
                distance_to_center = abs(temp - np.mean(self.comfort_range))
                max_distance = (self.comfort_range[1] - self.comfort_range[0]) / 2
                if max_distance > 0:
                    comfort_reward = (1 - distance_to_center / max_distance) * self.comfort_weight
                else:
                    comfort_reward = self.comfort_weight
                reward += comfort_reward
            else:
                # 超出舒适范围，给予惩罚
                if temp < self.comfort_range[0]:
                    penalty = (self.comfort_range[0] - temp) * self.comfort_weight
                else:
                    penalty = (temp - self.comfort_range[1]) * self.comfort_weight
                reward -= penalty

        # 能耗奖励（能耗越低，奖励越高）
        energy_reward = -total_energy_kwh * self.energy_weight
        reward += energy_reward

        return reward

    def __call__(self, obs):
        """
        使该类实例可调用，适配 Sinergym 的 reward_fn(obs) 接口。
        假设 obs 是字典类型，包含 Zone{i}_Air_Temperature 和 Chiller_Electricity。
        """
        temperatures = []
        for i in range(1, self.num_zones + 1):
            key = f'Zone{i}_Air_Temperature'
            if key not in obs:
                raise KeyError(f"Observation missing required key: {key}")
            temperatures.append(obs[key])

        # 获取本步冷水机能耗（J），转换为 kWh
        chiller_joules = obs.get('Chiller_Electricity', 0.0)
        total_energy_kwh = chiller_joules / 3_600_000.0  # J → kWh

        reward = self.calculate(temperatures, total_energy_kwh)

        # 返回 (reward, reward_components_dict)
        return reward, {
            'comfort_component': reward + total_energy_kwh * self.energy_weight,
            'energy_component': -total_energy_kwh * self.energy_weight,
        }

class SinergymGCNEnv(EplusEnv):
    def __init__(self, idf_file, epw_file, pred_seq_len=8):
        if not os.path.exists(idf_file):
            raise FileNotFoundError(f"IDF file not found: {idf_file}")
        if not os.path.exists(epw_file):
            raise FileNotFoundError(f"EPW file not found: {epw_file}")

        # 定义观测变量
        variables = {
            # 区域温度
            'Zone1_Air_Temperature': ('Zone Air Temperature', 'THERMAL ZONE 1'),
            'Zone2_Air_Temperature': ('Zone Air Temperature', 'THERMAL ZONE 2'),
            'Zone3_Air_Temperature': ('Zone Air Temperature', 'THERMAL ZONE 3'),
            'Zone4_Air_Temperature': ('Zone Air Temperature', 'THERMAL ZONE 4'),
            'Zone5_Air_Temperature': ('Zone Air Temperature', 'THERMAL ZONE 5'),
            # 区域湿度
            'Zone1_Relative_Humidity': ('Zone Air Relative Humidity', 'THERMAL ZONE 1'),
            'Zone2_Relative_Humidity': ('Zone Air Relative Humidity', 'THERMAL ZONE 2'),
            'Zone3_Relative_Humidity': ('Zone Air Relative Humidity', 'THERMAL ZONE 3'),
            'Zone4_Relative_Humidity': ('Zone Air Relative Humidity', 'THERMAL ZONE 4'),
            'Zone5_Relative_Humidity': ('Zone Air Relative Humidity', 'THERMAL ZONE 5'),
            # 人员感热
            'Zone1_People_Heating': ('Zone People Sensible Heating Energy', 'THERMAL ZONE 1'),
            'Zone2_People_Heating': ('Zone People Sensible Heating Energy', 'THERMAL ZONE 2'),
            'Zone3_People_Heating': ('Zone People Sensible Heating Energy', 'THERMAL ZONE 3'),
            'Zone4_People_Heating': ('Zone People Sensible Heating Energy', 'THERMAL ZONE 4'),
            'Zone5_People_Heating': ('Zone People Sensible Heating Energy', 'THERMAL ZONE 5'),
            # 环境变量
            'Outdoor_Drybulb_Temp': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            'Diffuse_Solar_Radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
            'Direct_Solar_Radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
            # 冷水机能耗
            'Chiller_Electricity': ('Chiller Electricity Energy', 'CHILLER - WATER COOLED'),
        }

        # 定义执行器
        actuators = {
            'Zone1_ClgSP': ('Schedule:Year', 'Schedule Value', 'ZONE1_COOLING_SP'),
            'Zone2_ClgSP': ('Schedule:Year', 'Schedule Value', 'ZONE2_COOLING_SP'),
            'Zone3_ClgSP': ('Schedule:Year', 'Schedule Value', 'ZONE3_COOLING_SP'),
            'Zone4_ClgSP': ('Schedule:Year', 'Schedule Value', 'ZONE4_COOLING_SP'),
            'Zone5_ClgSP': ('Schedule:Year', 'Schedule Value', 'ZONE5_COOLING_SP'),
        }

        action_space = gym.spaces.Box(
            low=20, high=28, shape=(5,), dtype=np.float32
        )

        building_config = {
            'runperiod': (1, 8, 2005, 7, 8, 2005),  # 8月1日到8月7日
            'timesteps_per_hour': 1,
        }

        self._config = {
            'env_name': 'CustomGCNEnv-v0',
            'idf': os.path.basename(idf_file),
            'epw': [os.path.basename(epw_file)],
            'variables': variables,
            'actuators': actuators,
            'action_space': action_space,
            'building_config': building_config,
            'timesteps_per_hour': building_config['timesteps_per_hour'],
            'runperiod': building_config['runperiod'],
        }

        # 初始化父类：传入 CustomReward 类（非实例！）或实例？
        # Sinergym expects a callable, so we pass an INSTANCE that is callable.
        reward_kwargs = {
            'num_zones': 5,
            'comfort_range': (23.0, 25.0),
            'energy_weight': 0.01,
            'comfort_weight': 1.0
        }

        # 初始化父类
        super().__init__(
            env_name='CustomGCNEnv-v0',
            building_file=idf_file,
            weather_files=epw_file,
            action_space=action_space,
            variables=variables,
            actuators=actuators,
            building_config=building_config,
            reward=CustomReward,  # 传类，不是实例
            reward_kwargs=reward_kwargs,
        )

        self.num_zones = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_seq_len = pred_seq_len

        # 预加载EPW文件中的天气数据用于完美预测
        self.weather_data = self._load_weather_data(epw_file)
        self.current_step = 5088-24*31
        self.total_steps = 7 * 24  # 7天，每小时1步

    def _load_weather_data(self, epw_file):
        """从EPW文件加载天气数据"""
        weather_data = []
        try:
            with open(epw_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # EPW文件格式：跳过前8行头部信息
                for i, line in enumerate(lines[8:]):
                    if line.strip():
                        data = line.split(',')
                        if len(data) > 6:
                            # 提取干球温度（第6列，索引5）
                            dry_bulb_temp = float(data[6])
                            weather_data.append(dry_bulb_temp)
        except Exception as e:
            print(f"Warning: Could not load EPW file properly: {e}")
            # 如果加载失败，使用默认温度
            weather_data = [20.0] * 1000  # 足够长的默认数据

        return weather_data

    def get_observation(self, obs_dict):
        """构建每个房间的 7 维状态向量"""
        if isinstance(obs_dict, dict):
            obs_list = []
            for i in range(1, self.num_zones + 1):
                zone_temp = obs_dict[f'Zone{i}_Air_Temperature']
                zone_humidity = obs_dict[f'Zone{i}_Relative_Humidity']
                zone_people = obs_dict[f'Zone{i}_People_Heating'] / 3600000.0
                outdoor_temp = obs_dict['Outdoor_Drybulb_Temp']
                diffuse_solar = obs_dict['Diffuse_Solar_Radiation']
                direct_solar = obs_dict['Direct_Solar_Radiation']
                chiller_energy = obs_dict['Chiller_Electricity'] / 3600000.0
                node_state = [
                    float(zone_temp),
                    float(zone_humidity),
                    float(zone_people),
                    float(outdoor_temp),
                    float(chiller_energy),
                    float(diffuse_solar),
                    float(direct_solar)
                ]
                obs_list.append(node_state)
            return torch.FloatTensor(obs_list)
        elif isinstance(obs_dict, np.ndarray):
            obs_list = []
            for i in range(self.num_zones):
                zone_temp = obs_dict[i]
                zone_humidity = obs_dict[i + self.num_zones]
                zone_people = obs_dict[i + 2 * self.num_zones] / 3600000.0
                outdoor_temp = obs_dict[3 * self.num_zones]
                diffuse_solar = obs_dict[3 * self.num_zones + 1]
                direct_solar = obs_dict[3 * self.num_zones + 2]
                chiller_energy = obs_dict[3 * self.num_zones + 3] / 3600000.0
                node_state = [
                    float(zone_temp),
                    float(zone_humidity),
                    float(zone_people),
                    float(outdoor_temp),
                    float(chiller_energy),
                    float(diffuse_solar),
                    float(direct_solar)
                ]
                obs_list.append(node_state)
            return torch.FloatTensor(obs_list)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs_dict)}")

    def get_prediction_sequence(self):
        """获取真实的未来温度预测（完美预测）"""
        # 从预加载的天气数据中获取未来温度
        future_temps = []
        for i in range(self.pred_seq_len):
            future_step = self.current_step + i + 1
            if future_step < len(self.weather_data):
                future_temps.append(self.weather_data[future_step])
            else:
                # 如果超出数据范围，使用最后一个温度
                future_temps.append(self.weather_data[-1])

        return np.array(future_temps, dtype=np.float32)

    def step(self, action):
        """执行一步动作"""
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 20, 28)

        obs_dict, reward, terminated, truncated, info = super().step(action)

        # 获取观测和预测序列
        obs = self.get_observation(obs_dict)
        pred_sequence = self.get_prediction_sequence()

        # 更新当前步数
        self.current_step += 1

        # 将总能耗和预测序列存入 info
        if isinstance(obs_dict, dict):
            chiller_energy_j = obs_dict.get('Chiller_Electricity', 0.0)
        elif isinstance(obs_dict, np.ndarray):
            chiller_energy_j = obs_dict[18]
        else:
            chiller_energy_j = 0.0

        info['total_energy_kwh'] = chiller_energy_j / 3600000.0
        info['pred_sequence'] = pred_sequence

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs_dict, info = super().reset(**kwargs)
        print(f"Observation type: {type(obs_dict)}")
        if isinstance(obs_dict, dict):
            print(f"Available observation keys: {list(obs_dict.keys())}")
        else:
            print(f"Observation shape: {obs_dict.shape}")

        # 重置步数计数器
        self.current_step = 5088-24*31

        obs = self.get_observation(obs_dict)
        pred_sequence = self.get_prediction_sequence()
        info['pred_sequence'] = pred_sequence

        return obs, info

    def save_config(self):
        """保存配置"""
        config_path = os.path.join(self.workspace_path, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f)