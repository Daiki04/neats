import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from pendulum_environment import PendulumEnvironment

class PendulumControllerEvaluator:
    """Pendulumのcontrollerの評価を行うクラス"""

    def __init__(self, env: object, num_trial: int=10) -> None:
        """コンストラクタ

        Args:
            env (object): 環境
            num_trial (int, optional): 試行回数. Defaults to 10.
        """
        self.env = env
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
        timesteps = []
        for nt in range(self.num_trial):
            total_reward = 0
            obs = self.env.reset(nt)
            done = False
            time = 0
            while not done:
                action = controller.activate(obs)
                action[0] = action[0] * 6 - 3
                obs, reward, done = self.env.step(action)
                total_reward += reward
                time += 1
                if time >= 1000:
                    break
            rewards.append(total_reward)
            timesteps.append(time)
        # print(rewards)
        # print(timesteps)
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

    def make_img(self, frames: list, observations: list, save_path: str) -> None:
        """評価結果のGIF作成

        Args:
            frames (list): フレームのリスト
            observations (list): 観測のリスト
            save_path (str): 保存先
        """
        for i, frame in enumerate(frames):

            # カートポールを描画
            plt.figure(figsize=(9, 7), facecolor='white')
            plt.suptitle(f' Pendulum\nTime step: {i}', fontsize=20)
            plt.imshow(frame)
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.imshow(frame)
            plt.savefig(os.path.join(save_path, f'{i}.png'))
            plt.close()
        

    def get_gif(self, controller: object, save_path: str) -> None:
        """GIFの作成

        Args:
            controller (object): controller
            save_path (str): 保存先
        """
        obs = self.env.reset(1000)
        done = False
        time = 0
        frames = []
        observations = []
        while not done:
            rgb_array = self.env.render()
            observations.append(obs)
            frames.append(rgb_array)
            action = controller.activate(obs)
            action[0] = action[0] * 6 - 3
            obs, _, done = self.env.step(action)
            time += 1
            if time >= 1000 or done:
                rgb_array = self.env.render()
                frames.append(rgb_array)
                observations.append(obs)
                break
        
        # 一時的にフォルダを作成し，その中に画像を保存．その後，GIFに変換し，保存先に移動．一時的に作成したフォルダは削除
        temp_path = os.path.join(save_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)
        self.make_img(frames, observations, temp_path)
        images_path = [os.path.join(temp_path, f'{i}.png') for i in range(len(frames))]
        images = [Image.open(img) for img in images_path]
        images[0].save(os.path.join(save_path, 'result.gif'), save_all=True, append_images=images[1:], duration=50, loop=0, optimize=True)
        for img_path in images_path:
            os.remove(img_path)
        os.rmdir(temp_path)


if __name__ == '__main__':
    env = PendulumEnvironment()
    evaluator = PendulumControllerEvaluator(env)
    class controller:
        def activate(self, obs):
            return [0.5]
    dummy_controller = controller()
    results = evaluator.evaluate_controller(0, dummy_controller, 0)
    print(results)