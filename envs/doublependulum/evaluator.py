import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.special import expit

from doublependulum_environment import DoublePendulumEnvironment

class DoublePendulumControllerEvaluator:
    """DoublePendulumのcontrollerの評価を行うクラス"""

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
                action[0] = action[0] * 2 - 1
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
            'fitness': np.mean(rewards),
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
            plt.suptitle(f'Double Pendulum\nTime step: {i}', fontsize=20)
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
            action[0] = action[0] * 2 - 1
            obs, _, done = self.env.step(action)
            time += 1
            if time >= 500 or done:
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


class DoublePendulumControllerEvaluatorNS2:
    """DoublePendulumのcontrollerの評価を行うクラス"""

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
        obs_data = []
        act_data = []

        trial_data = []
        for nt in range(self.num_trial):
            total_reward = 0
            obs = self.env.reset(nt)
            done = False
            time = 0
            while not done:
                action = controller.activate(obs)
                action[0] = action[0] * 2 - 1
                obs, reward, done = self.env.step(action)
                total_reward += reward
                time += 1
                obs_data.append(np.array(obs))
                act_data.append(np.array(action))
                if time >= 1000:
                    break
            rewards.append(total_reward)
            timesteps.append(time)

            obs_data = np.vstack(obs_data)
            obs_cov = self.calc_covar(obs_data, print_flag=True)

            act_data = np.vstack(act_data)
            act_cov = self.calc_covar(act_data, align=False)

            data = np.hstack([obs_cov, act_cov])
            trial_data.append(data)

            obs_data = []
            act_data = []
        # print(rewards)
        # print(timesteps)
        # results = {
        #     'fitness': np.mean(rewards),
        #     "score": np.mean(rewards),
        #     'data': np.mean(rewards, axis=0).reshape(1, -1)
        # }

        results = {
            'fitness': np.mean(rewards),
            "score": np.mean(rewards),
            'data': np.mean(trial_data, axis=0).reshape(1, -1)
        }

        return results
    
    @staticmethod
    def calc_covar(vec, align=True, print_flag=False) -> np.ndarray:
        """共分散を計算する

        Args:
            vec (_type_): 観測または行動データのベクトル，観測は(n, s), 行動は(n, a)の形式, aは行動の次元数
            align (bool, optional): 平均を引くかどうか. Defaults to True.

        Returns:
            np.ndarray: 共分散行列
        """
        if print_flag:
            print(vec.shape)
        ave = np.mean(vec,axis=0) # 平均を計算, (s,)または(a,)
        if print_flag:
            print(ave.shape)
        if align:
            vec_align = (vec-ave).T # 平均を引いて転置, (s,n)または(a,n)
        else:
            vec_align = vec.T # 転置, (s,n)または(a,n)
        
        if print_flag:
            print(vec_align.shape)
        # vec.shape[1]は特徴量の次元数
        comb_indices = np.tril_indices(vec.shape[1],k=0) # 特徴慮の全組み合わせのインデックスを取得, (2, s*(s+1)/2)or(2, a*(a+1)/2)
        if print_flag:
            print(len(comb_indices[0]))

        # 共分散を計算
        # vec_alignの各行は各特徴量の時系列データ
        # vec_align[comb_indices[0]]: (s*(s+1)/2, n) or (a*(a+1)/2, n)
        # vec_align[comb_indices[1]]: (s*(s+1)/2, n) or (a*(a+1)/2, n)
        # それぞれの行同士の要素積を取り，その平均を取ることで各特徴量の共分散を計算
        # 積は(s*(s+1)/2, n) or (a*(a+1)/2, n)
        # covar: (s*(s+1)/2) or (a*(a+1)/2)
        covar = np.mean(vec_align[comb_indices[0]]*vec_align[comb_indices[1]],axis=1)
        return covar
    
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
            plt.suptitle(f'Double Pendulum\nTime step: {i}', fontsize=20)
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
            action[0] = action[0] * 2 - 1
            obs, _, done = self.env.step(action)
            time += 1
            if time >= 500 or done:
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

class DoublePendulumControllerEvaluator_KAN:
    """DoublePendulumのcontrollerの評価を行うクラス"""

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
                action[0] = expit(action[0]) # シグモイド関数を通す
                action[0] = action[0] * 2 - 1
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
            plt.suptitle(f'Double Pendulum\nTime step: {i}', fontsize=20)
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
            action[0] = action[0] * 2 - 1
            obs, _, done = self.env.step(action)
            time += 1
            if time >= 500 or done:
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
    env = DoublePendulumEnvironment()
    evaluator = DoublePendulumControllerEvaluatorNS2(env, num_trial=1)
    class controller:
        def activate(self, obs):
            return [0.5]
    dummy_controller = controller()
    results = evaluator.evaluate_controller(0, dummy_controller, 0)
    print(results["data"].shape)