import gymnasium as gym
import numpy as np

class DoublePendulumEnvironment:
    def __init__(self) -> None:
        """コンストラクタ"""
        self.env = gym.make('InvertedDoublePendulum-v4', render_mode='rgb_array')

    def reset(self, seed: int=123) -> np.ndarray:
        """エージェントのリセット

        Args:
            seed (int): シード値, Defaults to 123 

        Returns:
            np.ndarray: 観測
        """
        observation, info = self.env.reset(seed=seed)
        return observation

    def step(self, action: list) -> tuple:
        """エージェントの状態を更新

        Args:
            action (list): 制御信号，形状(1, )で，[0, 1]の範囲の値

        Returns:
            tuple: 観測, 報酬, 終了フラグ
        """
        observation, reward, terminated, _, _ = self.env.step(action)
        return observation, reward, terminated

    def render(self) -> np.ndarray:
        """環境のrgb_arrayを取得

        Returns:
            np.ndarray: 環境のrgb_array
        """
        rgd_array = self.env.render()
        return rgd_array

    def close(self):
        """環境を閉じる"""
        self.env.close()

    def get_info(self) -> dict:
        """環境情報を取得

        Returns:
            dict: 環境情報
        """
        action_space_size = self.env.action_space.shape[0]
        observation_space_size = self.env.observation_space.shape[0]
        max_episode_steps = self.env.spec.max_episode_steps
        info = {
            'output_size': action_space_size,
            'input_size': observation_space_size,
            'timestep': max_episode_steps
        }
        return info


if __name__ == '__main__':
    env = DoublePendulumEnvironment()
    env_info = env.get_info()
    print(env_info)
    observation = env.reset()
    print(observation)
    for i in range(11):
        action = [0.5]
        observation, reward, terminated = env.step(action)
        print("Step: ", i)
        print("Observation: ", observation)
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        if terminated:
            break
    env.close()