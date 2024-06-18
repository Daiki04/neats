import os
import numpy as np
import matplotlib.pyplot as plt

from flappy_environment import FlappyEnvironment, load_flappy

class FlappyControllerEvaluator:
    """フラッピーのcontrollerの評価を行うクラス"""

    def __init__(self, env: object, num_trial: int=10) -> None:
        """コンストラクタ

        Args:
            env (object): 環境
            num_trial (int, optional): 試行回数. Defaults to 10.
        """
        self.env = env
        self.num_trial = num_trial

    def get_observation(self) -> np.ndarray:
        """観測の取得

        Returns:
            np.ndarray: 観測
        """
        return self.agent.get_obs()
    
    def evaluate_controller(self, key: int,  controller: object, generation: int) -> dict:
        """controllerの評価

        Args:
            key (int): キー
            controller (object): controller
            generation (int): 世代数

        Returns:
            dict: 評価結果
        """
        rewards = []
        for nt in range(self.num_trial):
            obs = self.env.reset(nt)
            done = False
            while not done:
                action = controller.activate(obs)
                obs, done, info = self.env.step(action)
            rewards.append(info['reward'])
        results = {
            'fitness': np.mean(rewards)
        }

        return results
    
    def print_result(self,  controller: object) -> None:
        """評価結果の表示

        Args:
            controller (object): controller
        """
        results = self.evaluate_controller(0, controller, 0)
        print('fitness:', results['fitness'])

    def make_result_img(self, controller: object, save_path: str) -> None:
        """評価結果のGIF作成

        Args:
            controller (object): controller
            save_path (str): 保存先
        """
        obs = self.env.reset(0)
        done = False
        lacations = [[self.env.agent. x, self.env.agent.y]]
        while not done:
            action = controller.activate(obs)
            obs, done, info = self.env.step(action)
            lacations.append([self.env.agent.x, self.env.agent.y])

        lacations = np.array(lacations)
        plt.plot(lacations[:, 0], lacations[:, 1])
        plt.xlim(0, self.env.agent.env_info['timestep'])
        plt.ylim(self.env.agent.env_info['upper_bound'], self.env.agent.env_info['lower_bound'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Flappy')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.grid(which='major', color='gray', linestyle='--')
        plt.savefig(save_path)
        plt.close()
    
if __name__ == '__main__':
    ROOT_DIR = CURR_DIR = os.path.dirname(os.path.abspath("__file__"))
    data_name = 'normal'
    env.agent.env_info = load_flappy(ROOT_DIR, data_name)
    env = FlappyEnvironment(env.agent.env_info)
    evaluator = FlappyControllerEvaluator(env)
    results = evaluator.evaluate_controller(0, lambda obs: 0.7, 0)
    print(results)
