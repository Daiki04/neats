import numpy as np

class MazeControllerEvaluator:
    def __init__(self, maze, timesteps):
        self.maze = maze
        self.timesteps = timesteps

    def evaluate_agent(self, key, controller, generation):
        self.maze.reset()

        done = False
        t = 0
        for i in range(self.timesteps):
            t += 1
            obs = self.maze.get_observation()
            action = controller.activate(obs)
            done = self.maze.update(action)
            if done:
                break

        if done:
            score = 1.0
        else:
            distance = self.maze.get_distance_to_exit()
            score = (self.maze.initial_distance - distance) / self.maze.initial_distance

        last_loc = self.maze.get_agent_location()
        results = {
            'fitness': score,
            'data': last_loc
        }
        return results


class MazeControllerEvaluatorNS:
    def __init__(self, maze, timesteps):
        self.maze = maze
        self.timesteps = timesteps

    def evaluate_agent(self, key, controller, generation):
        self.maze.reset()

        obs_data = []
        act_data = []
        rewards = []

        trial_data = []
        for __ in range(1):
            done = False
            for i in range(self.timesteps):
                obs = self.maze.get_observation()
                action = controller.activate(obs)

                obs_data.append(obs.copy())
                act_data.append(action.copy())

                done = self.maze.update(action)
                if done:
                    break

            if done:
                score = 1.0
                rewards.append(1.0)
            else:
                distance = self.maze.get_distance_to_exit()
                score = (self.maze.initial_distance - distance) / self.maze.initial_distance
                rewards.append(score)

            obs_data = np.vstack(obs_data)
            obs_cov = self.calc_covar(obs_data)

            act_data = np.vstack(act_data)
            act_cov = self.calc_covar(act_data, align=False)

            loc_data = self.maze.get_agent_location()

            data = np.hstack([obs_cov, act_cov, loc_data])
            trial_data.append(data)
            last_loc = self.maze.get_agent_location()

        results = {
            'fitness': np.mean(rewards),
            'score': np.mean(rewards),
            'data': np.mean(np.vstack(trial_data), axis=0),
            'loc': last_loc
        }

        return results
    
    @staticmethod
    def calc_covar(vec, align=True) -> np.ndarray:
        """共分散を計算する

        Args:
            vec (_type_): 観測または行動データのベクトル，観測は(n, s), 行動は(n, a)の形式, aは行動の次元数
            align (bool, optional): 平均を引くかどうか. Defaults to True.

        Returns:
            np.ndarray: 共分散行列
        """
        ave = np.mean(vec,axis=0) # 平均を計算, (s,)または(a,)
        if align:
            vec_align = (vec-ave).T # 平均を引いて転置, (s,n)または(a,n)
        else:
            vec_align = vec.T # 転置, (s,n)または(a,n)
        # vec.shape[1]は特徴量の次元数
        comb_indices = np.tril_indices(vec.shape[1],k=0) # 特徴慮の全組み合わせのインデックスを取得, (2, s*(s+1)/2)or(2, a*(a+1)/2)

        # 共分散を計算
        # vec_alignの各行は各特徴量の時系列データ
        # vec_align[comb_indices[0]]: (s*(s+1)/2, n) or (a*(a+1)/2, n)
        # vec_align[comb_indices[1]]: (s*(s+1)/2, n) or (a*(a+1)/2, n)
        # それぞれの行同士の要素積を取り，その平均を取ることで各特徴量の共分散を計算
        # 積は(s*(s+1)/2, n) or (a*(a+1)/2, n)
        # covar: (s*(s+1)/2) or (a*(a+1)/2)
        covar = np.mean(vec_align[comb_indices[0]]*vec_align[comb_indices[1]],axis=1)
        return covar




# class MazeControllerEvaluatorNS:
#     def __init__(self, maze, timesteps):
#         self.maze = maze
#         self.timesteps = timesteps

#     def evaluate_agent(self, key, controller, generation):
#         self.maze.reset()

#         done = False
#         for i in range(self.timesteps):
#             obs = self.maze.get_observation()
#             action = controller.activate(obs)
#             done = self.maze.update(action)
#             if done:
#                 break

#         if done:
#             score = 1.0
#         else:
#             distance = self.maze.get_distance_to_exit()
#             score = (self.maze.initial_distance - distance) / self.maze.initial_distance

#         last_loc = self.maze.get_agent_location()
#         results = {
#             'score': score,
#             'data': last_loc
#         }
#         return results
