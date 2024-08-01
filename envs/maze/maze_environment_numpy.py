import os
import numpy as np

MAX_AGENT_SPEED = 3.0  # エージェントの最大直線速度
MAX_ANGULAR_VELOCITY = 3.0  # エージェントの最大角速度


class Agent:
    """迷路内を移動するエージェントを表すクラス"""

    def __init__(self, location: np.ndarray, heading: float = 0, radius: float = 8.0, range_finder_range: float = 100.0,
                 max_speed: float = 3.0, max_angular_vel: float = 3.0, speed_scale: float = 1.0, angular_scale: float = 1.0) -> None:
        """エージェントの初期化

        Args:
            location (np.ndarray): エージェントの初期位置
            heading (float, optional): エージェントの初期進行方向. Defaults to 0.
            radius (float, optional): エージェントの半径. Defaults to 8.0.
            range_finder_range (float, optional): 距離計センサーの最大検出距離. Defaults to 100.0.
            max_speed (float, optional): エージェントの最大直線速度. Defaults to 3.0.[m/s].
            max_angular_vel (float, optional): エージェントの最大角速度. Defaults to 3.0.[度/秒].
            speed_scale (float, optional): 直線速度制御信号の倍率. Defaults to 1.0.
            angular_scale (float, optional): 角速度制御信号の倍率. Defaults to 1.0.

        Note:
            - 入力の制御信号は0.0～1.0の範囲にクリップされているため，speed_scaleとangular_scaleで倍率を調整する
            - センサ10個：与えられた方向で最も近い障害物までの距離を示す6つの距離センサーとゴールの方向を示す4つのレーダセンサーを持つ
            - 制御2個：直線速度と角速度を加える2つのアクチュエータを持つ
        """
        self.heading = heading
        self.radius = radius
        self.range_finder_range = range_finder_range
        self.location = location
        self.max_speed = max_speed
        self.max_angular_vel = max_angular_vel
        self.speed_scale = speed_scale
        self.angular_scale = angular_scale

        self.speed = 0  # エージェントの直線速度[m/s]
        self.angular_vel = 0  # エージェントの角速度[度/秒]

        # 距離センサの監視方向(角度)
        self.range_finder_angles = np.array(
            [-90.0, -45.0, 0.0, 45.0, 90.0, -180.0])

        # ゴールまでの方向を監視するレーダセンサの視野角(角度)
        self.radar_angles = np.array(
            [[315.0, 405.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]])

        # 距離センサの検出結果を格納するリスト
        self.range_finders = None
        # レーダセンサの検出結果を格納するリスト
        self.radar = None

    def get_obs(self) -> np.ndarray:
        """エージェントのセンサ情報を取得する

        Returns:
            np.ndarray: エージェントのセンサ情報（[距離センサの検出結果, レーダセンサの検出結果]）
        """
        obs = list(self.range_finders) + list(self.radar)
        return obs

    def apply_control_signals(self, control_signals: np.ndarray) -> None:
        """制御信号を計算

        Args:
            control_signals (np.ndarray): 制御信号（[直線速度, 角速度]）

        Note:
            - 入力の制御信号は0.0から1.0の範囲にクリップされている
        """
        self.angular_vel += (control_signals[0] - 0.5)*self.angular_scale
        self.speed += (control_signals[1] - 0.5)*self.speed_scale

        # エージェントの最大（最小）直線速度を超えないようにクリップ
        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed)
        # エージェントの最大（最小）角速度を超えないようにクリップ
        self.angular_vel = np.clip(
            self.angular_vel, -self.max_angular_vel, self.max_angular_vel)

    def distance_to_exit(self, exit_point: np.ndarray) -> float:
        """エージェントとゴールとの距離を計算

        Args:
            exit_point (np.ndarray): ゴールの位置

        Returns:
            float: エージェントとゴールとの距離

        Note:
            - L2ノルム（ユークリッド距離）でエージェントとゴールとの距離を計算
        """
        return np.linalg.norm(self.location-exit_point)  # L2ノルム（ユークリッド距離）でエージェントとゴールとの距離を計算

    def update_rangefinder_sensors(self, walls: np.ndarray) -> None:
        """距離センサの検出結果を更新

        Args:
            walls (np.ndarray): 迷路の壁の情報（[壁の始点, 壁の終点]）

        Note:
            - センサの検出結果は0～1の範囲に正規化されている
            - それぞれのセンサの検出結果は最も近い壁までの距離を示す，配列の形状は(6,)
        """

        # センサの角度をラジアンに変換，エージェントの進行方向を考慮
        range_finder_angles = (self.range_finder_angles +
                               self.heading) / 180 * np.pi

        # 壁の始点と終点のリストをそれぞれA, Bとする
        # (N, 2) -> (1, N, 2), N: 壁の数
        A = np.expand_dims(walls[:, 0, :], axis=0)
        B = np.expand_dims(walls[:, 1, :], axis=0)  # (N, 2) -> (1, N, 2)

        # (2,) -> (1, 2), エージェントの位置を2次元に変換，計算をしやすくするため
        location = np.expand_dims(self.location, axis=0)
        # センサの最大検出距離からセンサの検出距離の終点を計算
        finder_points = location + self.range_finder_range * \
            np.vstack([np.cos(range_finder_angles),
                      np.sin(range_finder_angles)]).T

        # センサの検出距離の始点と終点をそれぞれC, Dとする
        C = np.expand_dims(location, axis=1)  # (1, 2) -> (1, 1, 2)
        D = np.expand_dims(finder_points, axis=1)  # (6, 2) -> (1, 6, 2)

        AC = A-C  # 距離センサの始点と壁の始点との差分（Agentを原点とした座標ベクトルの計算），ベクトルの表記的にはCAが正しい
        DC = D-C  # 距離センサの終点と壁の始点との差分（Agentを原点とした座標ベクトルの計算），ベクトルの表記的にはCDが正しい
        BA = B-A  # 壁の終点と始点との差分，ベクトルの表記的にはABが正しい

        ### センサベクトルCDと壁ベクトルABの交点が線分AB，CD上に存在するかの判定（交差判定） ###
        # 参考：https://qiita.com/zu_rin/items/09876d2c7ec12974bc0f
        rTop = AC[:, :, 1] * DC[:, :, 0] - AC[:, :, 0] * \
            DC[:, :, 1]  # |CA x CD|，外積の大きさ（正しいベクトルの表記で記載）
        sTop = AC[:, :, 1] * BA[:, :, 0] - AC[:, :, 0] * \
            BA[:, :, 1]  # |CA x AB|，外積の大きさ（正しいベクトルの表記で記載）
        Bot = BA[:, :, 0] * DC[:, :, 1] - BA[:, :, 1] * \
            DC[:, :, 0]  # |AB x CD|，外積の大きさ（正しいベクトルの表記で記載）

        # errstate: 0除算や無効な演算を無視する
        with np.errstate(divide='ignore', invalid='ignore'):
            # np.where(condition, x, y): conditionがTrueの場合はx, Falseの場合はyを返す
            # rTop / Bot, 0除算の場合は0を返す，rTop / Botは交点がベクトルAB上に存在するための媒介変数
            r = np.where(Bot == 0, 0, rTop / Bot)
            # sTop / Bot, 0除算の場合は0を返す，sTop / Botは交点がベクトルCD上に存在するための媒介変数
            s = np.where(Bot == 0, 0, sTop / Bot)

        # 交点とエージェントの距離を計算
        # 媒介変数が0～1の範囲内に存在する場合は交点が線分AB，CD上に存在する→壁との距離は検出できる，交点（壁）とエージェントの距離をセット
        # エージェントと壁の距離を計算：交点A + r * BAとエージェント位置CのL2ノルム（ユークリッド距離）
        # 媒介変数が0～1の範囲外に存在する場合は交点が線分AB，CD上に存在しない→壁との距離は検出できない，最大検出距離をセット
        distances = np.where((Bot != 0) & (r > 0) & (r < 1) & (s > 0) & (s < 1),  # 交点が線分AB，CD上に存在するかの条件の確認
                             np.linalg.norm((A + np.expand_dims(r, axis=-1) * BA) - C, axis=-1), self.range_finder_range)  # (N, 6) -> (N, 6), N: 壁の数
        # センサの検出結果をセット：それぞれのセンサが検出した最も近い壁までの距離をそのセンサの検出結果とする．値は0～1の範囲に正規化，(N, 6) -> (6,)，N: 壁の数
        self.range_finders = np.min(
            distances, axis=1) / self.range_finder_range

    def update_radars(self, exit_point: np.ndarray) -> None:
        """レーダセンサの検出結果を更新

        Args:
            exit_point (np.ndarray): ゴールの位置
        """
        # エージェントからゴールまでの角度
        # 参考：https://univ-study.net/arctan/
        exit_angle = np.arctan2(
            exit_point[0]-self.location[0], exit_point[1]-self.location[1]) % np.pi
        radar_angles = (self.radar_angles + self.heading) / \
            180 * np.pi  # レーダセンサの角度をラジアンに変換，エージェントの進行方向を考慮

        radar_range = radar_angles[:, 1]-radar_angles[:, 0]  # レーダセンサの視野角
        # エージェントからゴールまでの角度とレーダセンサの始点の角度の差分
        radar_diff = (exit_angle-radar_angles[:, 0]) % (2*np.pi)
        radar = np.zeros(self.radar_angles.shape[0])  # レーダセンサの検出結果を格納するリスト
        radar[radar_diff < radar_range] = 1  # ゴールが視野角内に存在するレーダーの検出結果を1に，それ以外は0
        self.radar = radar  # レーダセンサの検出結果をセット


class MazeEnvironment:
    """迷路環境を表すクラス"""

    def __init__(self, init_location: np.ndarray, walls: np.ndarray, exit_point: np.ndarray, init_heading: float = 180, exit_range: float = 5.0, agent_kwargs: dict = {}) -> None:
        """迷路環境の初期化

        Args:
            init_location (np.ndarray): エージェントの初期位置
            walls (np.ndarray): 迷路の壁の情報（[壁の始点, 壁の終点]）
            exit_point (np.ndarray): ゴールの位置
            init_heading (float, optional): エージェントの初期進行方向. Defaults to 180.
            exit_range (float, optional): ゴールの許容距離. ゴール位置の一定範囲に入ればゴールと判定する．Defaults to 5.0.
            agent_kwargs (dict, optional): エージェントの初期化パラメータ. Defaults to {}.
        """
        self.walls = walls
        self.exit_point = exit_point
        self.exit_range = exit_range
        self.init_location = init_location
        self.init_heading = init_heading
        self.agent_kwargs = agent_kwargs
        self.agent = None  # エージェント
        self.exit_found = None  # ゴールが見つかったかどうか

    def reset(self) -> None:
        """エージェントのリセット
        """
        self.agent = Agent(location=self.init_location,
                           heading=self.init_heading, **self.agent_kwargs)

        self.exit_found = False
        self.initial_distance = self.agent.distance_to_exit(
            self.exit_point)  # エージェントとゴールとの初期距離(L2ノルム)

        # 距離センサとレーダセンサの初期位置での値をセット
        self.agent.update_rangefinder_sensors(self.walls)
        self.agent.update_radars(self.exit_point)

    def get_distance_to_exit(self) -> float:
        """エージェントとゴールとの距離を取得

        Returns:
            float: エージェントとゴールとの距離
        """
        return self.agent.distance_to_exit(self.exit_point)

    def get_agent_location(self) -> np.ndarray:
        """エージェントの位置を取得

        Returns:
            np.ndarray: エージェントの現在位置
        """
        return self.agent.location.copy()

    def get_observation(self) -> np.ndarray:
        """エージェントのセンサ情報を取得

        Returns:
            np.ndarray: エージェントのセンサ情報([6つの距離センサの検出結果, 4つのレーダセンサの検出結果])
        """
        return self.agent.get_obs()

    def test_wall_collision(self, location: np.ndarray) -> bool:
        """エージェントが壁と衝突しているかどうかを判定

        Args:
            location (np.ndarray): エージェントの位置

        Returns:
            bool: エージェントが壁と衝突しているかどうか
        """

        A = self.walls[:, 0, :]  # 壁の始点, (N, 2), N: 壁の数
        B = self.walls[:, 1, :]  # 壁の終点, (N, 2), N: 壁の数
        C = np.expand_dims(location, axis=0)  # エージェントの位置, (2,) -> (1, 2)
        BA = B-A  # 壁の終点と始点との差分，ベクトルの表記的にはABが正しい

        ### エージェントと壁の線分との最小距離を計算 ###
        # 参考：https://qiita.com/deltaMASH/items/e7ffcca78c9b75710d09
        uTop = np.sum((C - A) * BA, axis=1)  # 内積計算，ベクトルACとベクトルABの内積
        # ベクトルABの長さの2乗，2点間の距離の2乗，内分点の大きさ計算と正規化をまとめて計算しているのでABの大きさを2回掛けている
        uBot = np.sum(np.square(BA), axis=1)

        # ベクトルAB上の点Cとの最近傍点との内分点の位置：
        # u < 0：内分点が線分ABの始点Aよりも前に存在する，線分ABと点Cの最短距離は点Aと点Cの距離
        # u > 1：内分点が線分ABの終点Bよりも後ろに存在する，線分ABと点Cの最短距離は点Bと点Cの距離
        # 0 < u < 1：内分点が線分AB上に存在する，線分ABと点Cの最短距離は内分点と点Cの距離
        u = uTop / uBot  # 内分点の位置
        dist1 = np.minimum(
            np.linalg.norm(A - C, axis=1),
            np.linalg.norm(B - C, axis=1))  # CAとCBの距離の小さい方を取得
        dist2 = np.linalg.norm(A + np.expand_dims(u, axis=-1)
                               * BA - C, axis=1)  # 内分点と点Cの距離

        # 内分点が線分AB上に存在しない場合はdist1，存在する場合はdist2をセット
        distances = np.where((u < 0) | (u > 1), dist1, dist2)

        # エージェントの半径と壁との距離がエージェントの半径より小さいかどうか：小さい場合は衝突している
        return np.min(distances) < self.agent.radius

    def update(self, control_signals: np.ndarray) -> bool:
        """エージェントの状態を更新

        Args:
            control_signals (np.ndarray): 制御信号([直線速度, 角速度])

        Returns:
            bool: ゴールが見つかったかどうか
        """

        # ゴールが見つかっている場合はTrueを返す
        if self.exit_found:
            return True

        # 増幅して制御信号を適用
        self.agent.apply_control_signals(control_signals)

        # X, Y方向の速度を計算
        vel = np.array([np.cos(self.agent.heading/180*np.pi) * self.agent.speed,
                        np.sin(self.agent.heading/180*np.pi) * self.agent.speed])

        # エージェントの向きを更新，角速度[度/秒]を加算
        self.agent.heading = (self.agent.heading +
                              self.agent.angular_vel) % 360

        # エージェントの位置を更新
        new_loc = self.agent.location + vel

        # エージェントが壁と衝突していない場合はエージェントの位置を更新
        if not self.test_wall_collision(new_loc):
            self.agent.location = new_loc

        # 距離センサとレーダセンサの検出結果を更新
        self.agent.update_rangefinder_sensors(self.walls)
        self.agent.update_radars(self.exit_point)

        # ゴールが見つかったかどうかを判定
        distance = self.get_distance_to_exit()
        self.exit_found = (distance < self.exit_range)

        return self.exit_found  # ゴールが見つかったかどうかを返す（タスクが終了したかどうか）

    @staticmethod
    def read_environment(ROOT_DIR: str, maze_name: str, maze_kwargs: dict = {}, agent_kwargs: dict = {}) -> "MazeEnvironment":
        """迷路環境を読み込んで作成

        Args:
            ROOT_DIR (str): ルートディレクトリのパス
            maze_name (str): 使用する迷路の名前
            maze_kwargs (dict, optional): 迷路環境の初期化パラメータ. Defaults to {}.
            agent_kwargs (dict, optional): エージェントの初期化パラメータ. Defaults to {}.

        Returns:
            MazeEnvironment: 迷路環境のオブジェクト
        """
        # 迷路ファイルのパス
        maze_file = os.path.join(
            ROOT_DIR, 'envs', 'maze', 'maze_files', f'{maze_name}.txt')

        ### 迷路ファイルの読み込み ###
        index = 0
        walls = []
        maze_agent, maze_exit = None, None
        with open(maze_file, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                if len(line) == 0:
                    # 空白行はスキップ
                    continue

                elif index == 0:
                    # 1行目はエージェントの初期位置：(x, y)
                    loc = np.array(list(map(float, line.split(' '))))
                elif index == 1:
                    # 2行目はゴールの位置：(x, y)
                    maze_exit = np.array(list(map(float, line.split(' '))))
                else:
                    # 3行目以降は壁の始点(x1,y1)と終点(x2,y2)：(x1, y1, x2, y2)
                    wall = np.array(list(map(float, line.split(' '))))
                    walls.append(wall)

                # インデックスを更新
                index += 1

        # 壁の情報を整形，(N, 4) -> (N, 2, 2), N: 壁の数，[[[x1, y1], [x2, y2]], ...]
        walls = np.reshape(np.vstack(walls), (-1, 2, 2))

        # 迷路環境を初期化して，そのオブジェクトを返す
        return MazeEnvironment(
            init_location=loc,
            walls=walls,
            exit_point=maze_exit,
            **maze_kwargs,
            agent_kwargs=agent_kwargs)

    @staticmethod
    def make_environment(start_point: list, walls: list, exit_point: list,
                         maze_kwargs: dict = {}, agent_kwargs: dict = {}) -> "MazeEnvironment":
        """迷路環境を引数から作成

        Args:
            start_point (list): エージェントの初期位置([x, y])
            walls (list): 迷路の壁の情報（[壁の始点, 壁の終点]）(N, 4), N: 壁の数
            exit_point (list): ゴールの位置([x, y])
            maze_kwargs (dict, optional): 迷路環境の初期化パラメータ. Defaults to {}.
            agent_kwargs (dict, optional): エージェントの初期化パラメータ. Defaults to {}.

        Returns:
            MazeEnvironment: 迷路環境のオブジェクト
        """
        return MazeEnvironment(
            init_location=np.array(start_point),
            walls=np.reshape(np.vstack(walls), (-1, 2, 2)),
            exit_point=np.array(exit_point),
            **maze_kwargs,
            agent_kwargs=agent_kwargs
        )
