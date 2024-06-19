import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os

from doublepole_environment import run_markov_simulation

class DoublePoleControllerEvaluator:
    """DoublePoleのcontrollerの評価を行うクラス"""

    def __init__(self, num_trial: int=1) -> None:
        """コンストラクタ

        Args:
            num_trial (int, optional): 試行回数. Defaults to 1.
        """
        self.num_trial = num_trial
    
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
        for _ in range(self.num_trial):
            reward = run_markov_simulation(controller)
            rewards.append(reward)
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

    def make_img(self, obs: list, save_path: str, num: int) -> None:
        """画像の作成

        Args:
            obs (list): 観測
            save_path (str): 保存先
            num (int): 画像番号
        """
        SCARE = 2
    
        x = obs[0]
        theta1 = obs[2]
        theta2 = obs[4]

        # 環境の描画
        field_width = 2.4 # -2.4m ~ 2.4m
        field_height = 4 # 0m ~ 4m

        # 座標軸の設定
        plt.xlim(-field_width, field_width)
        plt.ylim(0, field_height)

        # カートの描画
        CART_WIDTH = 0.2
        CART_HEIGHT = 0.3
        cart_x = x # カートの中心のx座標
        cart_y = 0
        plt.gca().add_patch(plt.Rectangle((cart_x - CART_WIDTH / 2, cart_y - CART_HEIGHT / 2), CART_WIDTH, CART_HEIGHT, fill=True, color='gray'))

        # ポールの描画
        POLE_1_LENGTH = 1.0 * SCARE
        pole1_x = cart_x
        pole1_y = cart_y + CART_HEIGHT / 2
        pole1_end_x = pole1_x + POLE_1_LENGTH * math.sin(theta1)
        pole1_end_y = pole1_y + POLE_1_LENGTH * math.cos(theta1)
        plt.plot([pole1_x, pole1_end_x], [pole1_y, pole1_end_y], color='red', linewidth=2, label='pole1', linestyle='dashed')

        POLE_2_LENGTH = 0.1 * SCARE
        pole2_x = cart_x
        pole2_y = cart_y + CART_HEIGHT / 2
        pole2_end_x = pole2_x + POLE_2_LENGTH * math.sin(theta2)
        pole2_end_y = pole2_y + POLE_2_LENGTH * math.cos(theta2)
        plt.plot([pole2_x, pole2_end_x], [pole2_y, pole2_end_y], color='blue', linewidth=2, label='pole2', linestyle='dashed')

        plt.title(f'Double Pole: time={num}\nx={x:.2f}, theta1={theta1:.2f}, theta2={theta2:.2f}')

        plt.legend()
        plt.savefig(os.path.join(save_path, f'{num}.png'))
        plt.close()

    def make_imgs(self, obss: list, save_path: str) -> None:
        """画像の作成

        Args:
            obs (list): 観測のリスト
            save_path (str): 保存先
        """
        for i, obs in enumerate(obss):
            self.make_img(obs, save_path, i)

    def get_gif(self, controller: object, save_path: str) -> None:
        """GIFの作成

        Args:
            controller (object): controller
            save_path (str): 保存先
        """
        _, obs = run_markov_simulation(controller, get_obs=True)

        temp_path = os.path.join(save_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)
        self.make_imgs(obs, temp_path)
        images_path = [os.path.join(temp_path, f'{i}.png') for i in range(len(obs))]
        images = [Image.open(img_path) for img_path in images_path]
        images[0].save(os.path.join(save_path, 'result.gif'), save_all=True, append_images=images[1:], duration=25, loop=0, optimize=True)
        for img_path in images_path:
            os.remove(img_path)
        os.rmdir(temp_path)

if __name__ == '__main__':
    evaluator = DoublePoleControllerEvaluator()
    class controller:
        def activate(self, obs):
            return [1]
    dummy_controller = controller()
    results = evaluator.evaluate_controller(0, dummy_controller, 0)
    print(results)