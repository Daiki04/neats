import numpy as np
import matplotlib.pyplot as plt
import os

def load_flappy(ROOT_DIR: str, data_name: str) -> dict:
    """目標のフラッピーのデータを読み込む

    Args:
        ROOT_DIR (str): ルートディレクトリ
        data_name (str): データ名

    Returns:
        dict: 環境情報: 入力サイズ, 出力サイズ, 上限値, 下限値, 重力加速度, タイムステップ

    Raises:
        AssertionError: 読み込んだ入出力データのサイズが異なる場合

    Examples:
        env_info = load_flappy(ROOT_DIR, 'normal')
    """
    data_file = os.path.join(ROOT_DIR, 'envs', 'flappy', 'flappy_files', f'{data_name}.txt') # データファイルのパス
    
    index = 0 # 行番号
    input_size = None # 入力サイズ
    output_size = None # 出力サイズ
    upper_bound = None # 上限値
    lower_bound = None # 下限値
    g = None # 重力加速度
    timestep = None # タイムステップ

    ### データの読み込み ###
    with open(data_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if len(line) == 0:
                continue

            elif index == 0:
                input_size = int(line)
            elif index == 1:
                output_size = int(line)
            elif index == 2:
                upper_bound = float(line)
            elif index == 3:
                lower_bound = float(line)
            elif index == 4:
                g = float(line)
            elif index == 5:
                timestep = int(line)

            index += 1 # 行番号を更新

    env_info = {
        'input_size': input_size,
        'output_size': output_size,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'g': g,
        'timestep': timestep
    }

    return env_info

class Agent:
    """Flappy agent class"""
    def __init__(self, env_info: dict, seed: int = 123) -> None:
        """コンストラクタ

        Args:
            env_info (dict): 環境情報：入力サイズ, 出力サイズ, 上限値, 下限値, 重力加速度，タイムステップ
            seed (int): シード値, Defaults to 123
        """
        self.env_info = env_info
        self.g = -1 * env_info['g'] # 重力加速度
        self.v_x = 1.0 # 横方向の速度
        self.v_y_sigma = 3.0 # 初期速度の標準偏差
        self.v_jump = 3.0
        self.reset(seed)

    def reset(self, seed: int) -> None:
        """エージェントのリセット
        
        Args:
            seed (int): シード値
        """
        np.random.seed(seed)
        self.x = 0.0 # 初期x座標
        self.y = 0.0 # 初期y座標
        self.v_y = self.v_y_sigma * np.random.randn() # 初期y方向の速度

    def get_obs(self) -> np.ndarray:
        """エージェントの観測を取得

        Returns:
            np.ndarray: 観測
        """
        return list([self.y, self.v_y])
    
    def apply_control_signal(self, control_signal: np.ndarray) -> None:
        """制御信号を適用

        Args:
            control_signal (np.ndarray): 制御信号
        """
        if control_signal[0] < 0.5:
            self.v_y += self.g
        else:
            self.v_y = self.v_jump

    def update_state(self) -> None:
        """エージェントの状態を更新"""
        self.y += self.v_y
        self.x += self.v_x
        
    
class FlappyEnvironment:
    """Flappy bird environment class"""
    def __init__(self, env_info: dict) -> None:
        """コンストラクタ

        Args:
            env_info (dict): 環境情報：入力サイズ, 出力サイズ, 上限値, 下限値, タイムステップ
        """
        self.env_info = env_info
        self.agent = Agent(env_info)
        self.timestep = env_info['timestep']
        self.upper_bound = env_info['upper_bound']
        self.lower_bound = env_info['lower_bound']

    def reset(self, seed: int) -> np.ndarray:
        """環境のリセット

        Args:
            seed (int): シード値
        Returns:
            np.ndarray: 観測
        """
        self.agent.reset(seed)
        self.done = False # エピソード終了フラグ
        return self.agent.get_obs()

    def get_observation(self) -> np.ndarray:
        """観測の取得

        Returns:
            np.ndarray: 観測
        """
        return self.agent.get_obs()
    
    def step(self, control_signal: np.ndarray) -> np.ndarray:
        """エージェントの状態を更新

        Args:
            control_signal (np.ndarray): 制御信号，[0, 1]の範囲の値

        Returns:
            np.ndarray: 観測
        """
        info = {}
        self.agent.apply_control_signal(control_signal)
        self.agent.update_state()
        if self.agent.y > self.upper_bound or self.agent.y < self.lower_bound:
            self.done = True
            info['status'] = 'out of bounds'
            info['reward'] = self.agent.x
        if self.agent.x >= self.timestep:
            self.done = True
            info['status'] = 'goal'
            info['reward'] = self.agent.x

        return self.agent.get_obs(), self.done, info
    
if __name__ == '__main__':
    env_info = {
        'input_size': 2,
        'output_size': 1,
        'upper_bound': 10.0,
        'lower_bound': -10.0,
        'g': 0.0,
        'timestep': 100
    }
    env = FlappyEnvironment(env_info)
    obs = env.reset(0)
    done = False
    while not done:
        action = np.random.rand(1)
        obs, done, info = env.step(action)
        print(obs)
