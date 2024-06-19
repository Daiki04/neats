"""
システムダイナミクスの数値近似にルンゲクッタ4次法を用いた
ニュートン則に基づく2極のカートポール装置のシミュレーション
"""

import math
import random

# Double-pole balancerの定数を定義
GRAVITY = -9.8          # 重力加速度[m/s^2]
MASS_CART = 1.0         # カートの質量[kg]
FORCE_MAG = 10.0        # 力の大きさ[N]
MASS_POLE_1 = 1.0       # 1番目のポールの質量[kg]
LENGTH_1 = 1.0 / 2.0    # 1番目のポールの支点から重心までの長さ[m]（ポールの長さの半分）
MASS_POLE_2 = 0.1       # 2番目のポールの質量[kg]
LENGTH_2 = 0.1 / 2.0    # 2番目のポールの支点から重心までの長さ[m]（ポールの長さの半分）
MUP = 0.000002          # ポールの支点におけるカートとの摩擦係数
MAX_BAL_STEPS = 2000    # シミュレーションの最大時間ステップ数

# ポールと垂直線との許容最大角度の絶対値[rad]（度数法で36度）
THIRTY_SIX_DEG_IN_RAD = (36 * math.pi) / 180.0


def calc_step(action: int, x: float, x_dot: float, theta1: float, theta1_dot: float, theta2: float, theta2_dot: float) -> tuple:
    """シミュレーションの1ステップについてシステムダイナミクスの計算

    Args:
        action (int): 0または1の値をとる制御入力
        x (float): カートの位置[m]
        x_dot (float): カートの速度[m/s]
        theta1 (float): 1番目のポールの角度[rad]
        theta1_dot (float): 1番目のポールの角速度[rad/s]
        theta2 (float): 2番目のポールの角度[rad]
        theta2_dot (float): 2番目のポールの角速度[rad/s]

    Returns:
        tuple: カートの加速度[m/s^2], 1番目のポールの角加速度[rad/s^2], 2番目のポールの角加速度[rad/s^2]
    """
    # 制御入力を適用
    force = -FORCE_MAG if action == 0 else FORCE_MAG  # 制御入力を力の方向に変換：0 -> -10, 1 -> 10

    # システムの状態変数から基本的な値を計算
    cos_theta_1 = math.cos(theta1)      # 1番目のポールの角度の余弦
    sin_theta_1 = math.sin(theta1)      # 1番目のポールの角度の正弦
    g_sin_theta_1 = GRAVITY * sin_theta_1  # 1番目のポールに垂直に働く重力の成分
    cos_theta_2 = math.cos(theta2)      # 2番目のポールの角度の余弦
    sin_theta_2 = math.sin(theta2)      # 2番目のポールの角度の正弦
    g_sin_theta_2 = GRAVITY * sin_theta_2  # 2番目のポールに垂直に働く重力の成分

    # カートの加速度，ポールの角速度を計算するための中間値の計算
    ml_1 = LENGTH_1 * MASS_POLE_1
    ml_2 = LENGTH_2 * MASS_POLE_2
    temp_1 = MUP * theta1_dot / ml_1
    temp_2 = MUP * theta2_dot / ml_2
    fi_1 = (ml_1 * theta1_dot * theta1_dot * sin_theta_1) + \
        (0.75 * MASS_POLE_1 * cos_theta_1 * (temp_1 + g_sin_theta_1))
    fi_2 = (ml_2 * theta2_dot * theta2_dot * sin_theta_2) + \
        (0.75 * MASS_POLE_2 * cos_theta_2 * (temp_2 + g_sin_theta_2))
    mi_1 = MASS_POLE_1 * (1 - (0.75 * cos_theta_1 * cos_theta_1))
    mi_2 = MASS_POLE_2 * (1 - (0.75 * cos_theta_2 * cos_theta_2))

    # カートの加速度，ポールの角加速度を計算
    x_ddot = (force + fi_1 + fi_2) / (mi_1 + mi_2 +
                                      MASS_CART)                    # カートの加速度
    theta_1_ddot = -0.75 * (x_ddot * cos_theta_1 +
                            g_sin_theta_1 + temp_1) / LENGTH_1   # 1番目のポールの角加速度
    theta_2_ddot = -0.75 * (x_ddot * cos_theta_2 +
                            g_sin_theta_2 + temp_2) / LENGTH_2   # 2番目のポールの角加速度

    return x_ddot, theta_1_ddot, theta_2_ddot


def outside_bounds(x: float, theta1: float, theta2: float) -> bool:
    """状態が制約を満たしているかどうかを判定

    Args:
        x (float): カートの位置[m]
        theta1 (float): 1番目のポールの角度[rad]
        theta2 (float): 2番目のポールの角度[rad]

    Returns:
        bool: 状態が制約を満たしている場合はFalse，満たしていない場合はTrue
    """
    if x < -2.4 or x > 2.4:
        return True
    if theta1 < -THIRTY_SIX_DEG_IN_RAD or theta1 > THIRTY_SIX_DEG_IN_RAD:
        return True
    if theta2 < -THIRTY_SIX_DEG_IN_RAD or theta2 > THIRTY_SIX_DEG_IN_RAD:
        return True
    return False


def rk4(f: int, y: list, dydx: list, tau: float) -> None:
    """ルンゲクッタ4次法による2極カートシステムダイナミクスの数値近似

    Args:
        f (int): 制御入力
        y (list): 状態変数のリスト[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        dydx (list): 状態変数の導関数のリスト[x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
        tau (float): シミュレーションの時間ステップ幅

    Note:
        4次のルンゲクッタ法：https://qiita.com/kaityo256/items/e3428deb394b3ad1e739
        ここではdydx = f(y) = k1 となっている．dydxの中見は状態変数の時刻tにおける導関数の値
    """
    # k2 = f(y + tau * k1 / 2)を計算
    # k1 = [x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
    hh = tau / 2.0
    yt = [None] * 6

    # y + tau * k1 / 2を計算
    for i in range(6):
        yt[i] = y[i] + hh * dydx[i]

    # k2[1], k2[3], k2[5]を計算．k2[0], k2[2], k2[4]はyt[1], yt[3], yt[5]と等しい
    x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=f,
                                                   x=yt[0],
                                                   x_dot=yt[1],
                                                   theta1=yt[2],
                                                   theta1_dot=yt[3],
                                                   theta2=yt[4],
                                                   theta2_dot=yt[5])

    # k2 = dyt
    dyt = [yt[1], x_ddot, yt[3], theta_1_ddot, yt[5], theta_2_ddot]

    # k3 = f(y + tau * k2 / 2)を計算

    # y + tau * k2 / 2を計算
    for i in range(6):
        yt[i] = y[i] + hh * dyt[i]

    # k3[1], k3[3], k3[5]を計算．k3[0], k3[2], k3[4]はyt[1], yt[3], yt[5]と等しい
    x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=f,
                                                   x=yt[0],
                                                   x_dot=yt[1],
                                                   theta1=yt[2],
                                                   theta1_dot=yt[3],
                                                   theta2=yt[4],
                                                   theta2_dot=yt[5])
    # k3 = dym
    dym = [yt[1], x_ddot, yt[3], theta_1_ddot, yt[5], theta_2_ddot]

    # k4を計算
    # (y + tau * k3)と(k2 + k3)を計算
    for i in range(6):
        yt[i] = y[i] + tau * dym[i]  # y + tau * k3
        dym[i] += dyt[i]  # dym = k2 + k3

    # k4[1], k4[3], k4[5]を計算．k4[0], k4[2], k4[4]はyt[1], yt[3], yt[5]と等しい
    x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=f,
                                                   x=yt[0],
                                                   x_dot=yt[1],
                                                   theta1=yt[2],
                                                   theta1_dot=yt[3],
                                                   theta2=yt[4],
                                                   theta2_dot=yt[5])
    # k4 = dyt
    dyt = [yt[1], x_ddot, yt[3], theta_1_ddot, yt[5], theta_2_ddot]

    # ルンゲクッタ4次法による数値近似
    h6 = tau / 6.0
    # y(t + tau) = y(t) + h6 * (k1 + 2 * k2 + 2 * k3 + k4) = y(t) + h6 * (k1 + 2*(k2 + k3) + k4) = y(t) + h6 * (dydx + dyt + 2 * dym)
    for i in range(6):
        y[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i])


def apply_action(action: int, state: list, step_number: int) -> list:
    """カートポールシミュレーションに制御アクションを適用

    Args:
        action (int): 制御入力
        state (list): 状態変数のリスト[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        step_number (int): シミュレーションのステップ番号

    Returns:
        list: シミュレーション後の状態変数のリスト[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    """
    # シミュレーションの時間ステップ幅
    TAU = 0.01

    # 制御入力は2TAUごとに与えられるため，2ステップ分のシミュレーションを行う
    # 状態変数の導関数のリスト[x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
    dydx = [None] * 6
    for _ in range(2):
        dydx[0] = state[1]  # x_dot
        dydx[2] = state[3]  # theta1_dot
        dydx[4] = state[5]  # theta2_dot
        # システムダイナミクスの計算：x_ddot, theta_1_ddot, theta_2_ddotを計算
        x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=action,
                                                       x=state[0], x_dot=state[1],
                                                       theta1=state[2], theta1_dot=state[3],
                                                       theta2=state[4], theta2_dot=state[5])
        dydx[1] = x_ddot
        dydx[3] = theta_1_ddot
        dydx[5] = theta_2_ddot
        # 状態stateを更新：k1 = dydxとして，ルンゲクッタ4次法による数値近似
        rk4(f=action, y=state, dydx=dydx, tau=TAU)

    return state


def run_markov_simulation(net: object, max_bal_steps: int = MAX_BAL_STEPS, get_obs=False) -> int:
    """Double-pole balancerのシミュレーションを任意の時間ステップ数だけ実行

    Args:
        net (object): ネットワーク
        max_bal_steps (int): シミュレーションの最大時間ステップ数

    Returns:
        int: シミュレーション中に制約を満たす時間ステップ数
    """
    # 指定された時間ステップ数だけシミュレーションを実行
    input = [None] * 6  # 入力ベクトル
    # 状態変数のリスト[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]を初期化
    state = reset_state([None] * 6)
    observations = [state.copy()]
    for steps in range(max_bal_steps):
        # 入力ベクトルを設定：状態変数を0~1に正規化
        input[0] = (state[0] + 2.4) / 4.8
        input[1] = (state[1] + 1.5) / 3.0
        input[2] = (state[2] + THIRTY_SIX_DEG_IN_RAD) / \
            (THIRTY_SIX_DEG_IN_RAD * 2.0)
        input[3] = (state[3] + 2.0) / 4.0
        input[4] = (state[4] + THIRTY_SIX_DEG_IN_RAD) / \
            (THIRTY_SIX_DEG_IN_RAD * 2.0)
        input[5] = (state[5] + 2.0) / 4.0

        # ニューラルネットワークに入力ベクトルを与えて出力を取得
        output = net.activate(input)
        # 出力が0.5未満なら0，そうでなければ1をactionに設定
        action = 0 if output[0] < 0.5 else 1

        # 状態stateを更新
        state = apply_action(action=action, state=state, step_number=steps)
        observations.append(state.copy())

        # 状態が制約を満たしていない場合はシミュレーションを終了
        if outside_bounds(x=state[0], theta1=state[2], theta2=state[4]):
            if get_obs:
                return steps, observations
            return steps
    if get_obs:
        return max_bal_steps, observations
    return max_bal_steps


def reset_state(state: list) -> list:
    """状態配列を初期値にリセットする関数

    Args:
        state (list): 状態変数のリスト[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]

    Returns:
        list: リセット後の状態変数のリスト[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    """
    state[0], state[1], state[3], state[4], state[5] = 0, 0, 0, 0, 0
    state[2] = math.pi / 180.0  # the one_degree
    return state

def get_envinfo():
    envinfo = {
            'output_size': 1,
            'input_size': 6,
            'timestep': MAX_BAL_STEPS
        }
    return envinfo

if __name__ == '__main__':
    class dummy_network:
        def activate(self, input):
            return [1]
        
    net = dummy_network()
    max_bal_steps = 1000
    steps = run_markov_simulation(net, max_bal_steps)
    print(f"steps: {steps}")