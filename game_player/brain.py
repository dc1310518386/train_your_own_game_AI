# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3
import numpy as np
import os
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

# ---------- 以下根据 control_keyboard_keys.py 里定义的函数来导入 ----------
from game_player.control_keyboard_keys import W, S, A
# ---------- 以上根据 control_keyboard_keys.py 里定义的函数来导入 ----------

# ---*---

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['action', 'reward', 'observation_idx', 'next_observation_idx']
        )
        self.observations = []  # 用于存储 observation
        self.next_observations = []  # 用于存储 next_observation
        self.capacity = capacity
        self.i = 0
        self.count = 0

    def store(self, *args):
        # 将新的 observation 和 next_observation 添加到列表中
        self.observations.append(args[0])
        self.next_observations.append(args[3])
        
        # 确保存储索引不会超出有效范围
        observation_idx = len(self.observations) - 1
        next_observation_idx = len(self.next_observations) - 1

        # 更新 memory
        self.memory.loc[self.i] = {
            'action': args[1],
            'reward': args[2],
            'observation_idx': observation_idx,
            'next_observation_idx': next_observation_idx,
        }
        
        # 更新索引和计数
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)


    def sample(self, size):

        

        invalid_indices = self.memory['observation_idx'][self.memory['observation_idx'] >= len(self.observations)]
        print(f"Invalid observation indices: {invalid_indices}")
        indices = np.random.choice(self.count, size=size, replace=False).astype(int)

        observations = np.array([self.observations[idx] for idx in self.memory.loc[indices, 'observation_idx']])
        next_observations = np.array([self.next_observations[idx] for idx in self.memory.loc[indices, 'next_observation_idx']])

        actions = self.memory.loc[indices, 'action'].values
        rewards = self.memory.loc[indices, 'reward'].values

        return observations, actions, rewards, next_observations

class MyLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x, tf.float32)

# DoubleDQN
class DoubleDQN:
    def __init__(
        self,
        in_height,      # 图像高度
        in_width,       # 图像宽度
        in_channels,    # 颜色通道数量
        outputs,        # 动作数量
        lr,             # 学习率
        gamma,    # 奖励衰减
        replay_memory_size,     # 记忆容量
        replay_start_size,      # 开始经验回放时存储的记忆量，到达最终探索率后才开始
        batch_size,             # 样本抽取数量
        update_freq,                   # 训练评估网络的频率
        target_network_update_freq,    # 更新目标网络的频率
        save_weights_path,    # 指定模型权数保存的路径。
        load_weights_path     # 指定模型权重加载的路径。
    ):
        self.in_height   = in_height
        self.in_width    = in_width
        self.in_channels = in_channels
        self.outputs     = outputs
        self.lr          = lr,

        self.gamma = gamma

        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.update_freq = update_freq
        self.target_network_update_freq = target_network_update_freq

        self.save_weights_path = save_weights_path
        self.load_weights_path = load_weights_path

        self.min_epsilon = 0.1    # 最终探索率

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

        self.step = 0    # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        input_shape = [self.in_height, self.in_width, self.in_channels]

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        x = MyLayer()(inputs)
        outputs = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, weights=None, classes=self.outputs)(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(float(self.lr[0])),    
            loss=tf.keras.losses.CategoricalCrossentropy(),    
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

        if self.load_weights_path:
            if os.path.exists(self.load_weights_path):
                model.load_weights(self.load_weights_path)
                print('Load ' + self.load_weights_path)
            else:
                print('Nothing to load ')

        return model

    # 行为选择方法
    def choose_action(self, observation):

        # 先看运行的步数(self.step)有没有达到开始回放经验的要求(self.replay_start_size)，没有就随机探索
                                                  # 如果已经达到了，就再看随机数在不在最终探索率范围内，在的话也是随机探索
        if self.step <= self.replay_start_size or np.random.rand() < self.min_epsilon:
            q_values = np.random.rand(self.outputs)
            self.who_play = '随机探索'
        else:
            observation = observation.reshape(-1, self.in_height, self.in_width, self.in_channels)
            q_values = self.evaluate_net.predict(observation)[0]
            self.who_play = '模型预测'

        action = np.argmax(q_values)

        # ---------- 以下根据 control_keyboard_keys.py 里定义的函数来修改 ----------




        # 执行动作
        if   action == 0:
            W()
        elif action == 1:
            S()
        elif action == 2:
            A()

        # 不够可以添加，注意，一定要是正整数，还要和上一个相邻
        # ---------- 以上根据 control_keyboard_keys.py 里定义的函数来修改 ----------

        return action

    # 学习方法
    def learn(self, verbose=0):

        self.step += 1
        print(f"\r Learning step: {self.step}", end='')
        # 当前步数满足更新评估网络的要求
        if self.step % self.update_freq == 0:

            # 当前步数满足更新目标网络的要求
            if self.step % self.target_network_update_freq == 0:
                self.update_target_network() 

            # 经验回放
            observations, actions, rewards, next_observations = self.replayer.sample(self.batch_size)

            # 数据预处理
            observations = observations.reshape(-1, self.in_height, self.in_width, self.in_channels)
            actions = actions.astype(np.int8)
            next_observations = next_observations.reshape(-1, self.in_height, self.in_width, self.in_channels)


            next_eval_qs = self.evaluate_net.predict(next_observations)
            next_actions = next_eval_qs.argmax(axis=-1)

            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]

            us = rewards + self.gamma * next_max_qs
            targets = self.evaluate_net.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us


            self.evaluate_net.fit(observations, targets, batch_size=1, verbose=verbose)

            self.save_evaluate_network()

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)