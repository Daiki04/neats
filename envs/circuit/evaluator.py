### 回路の読み込みと評価を行う ###

import os
import numpy as np


def load_circuit(ROOT_DIR: str, data_name: str) -> tuple:
    """目標の回路データを読み込む

    Args:
        ROOT_DIR (str): ルートディレクトリ
        data_name (str): データ名

    Returns:
        tuple: 入力データ, 出力データ

    Raises:
        AssertionError: 読み込んだ入出力データのサイズが異なる場合

    Examples:
        input_data, output_data = load_circuit(ROOT_DIR, 'xor')
    """
    data_file = os.path.join(ROOT_DIR, 'envs', 'circuit',
                             'circuit_files', f'{data_name}.txt')  # データファイルのパス

    index = 0  # 行番号
    input_size = None  # 入力サイズ
    output_size = None  # 出力サイズ
    input_data = []  # 入力データ
    output_data = []  # 出力データ

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
            else:
                data = list(map(float, line.split(' ')))
                assert len(data) == input_size+output_size

                input_data.append(data[:input_size])
                output_data.append(data[input_size:])

            index += 1  # 行番号を更新

    input_data = np.vstack(input_data)
    output_data = np.vstack(output_data)

    return input_data, output_data


class CircuitEvaluator:
    """回路の評価を行うクラス"""

    def __init__(self, input_data: np.ndarray, output_data: np.ndarray, error_type: str = 'mse') -> None:
        """コンストラクタ

        Args:
            input_data (np.ndarray): 入力データ
            output_data (np.ndarray): 出力データ
            error_type (str, optional): 評価指標. Defaults to 'mse'.

        Raises:
            AssertionError: error_typeが['mse', 'mae']のどちらでもない場合

        Examples:
            input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            output_data = np.array([[0], [1], [1], [0]])
            evaluator = CircuitEvaluator(input_data, output_data, error_type='mse')
        """

        assert error_type in [
            'mse', 'mae'], 'choise error_type from [mse, mae].'

        self.input_data = input_data
        self.output_data = output_data
        self.error_type = error_type

    def evaluate_circuit(self, key: str, circuit: object, generation: int) -> dict:
        """回路の評価を行う

        Args:
            key (str): ゲノム番号
            circuit (object): 回路（ネットワーク）
            generation (int): 世代

        Returns:
            dict: 評価結果
        """

        ### 予測値を計算 ###
        output_pred = []
        for inp in self.input_data:
            pred = circuit.activate(inp)
            output_pred.append(pred)

        output_pred = np.vstack(output_pred)

        ### 評価指標を計算 ###
        if self.error_type == 'mae':
            error = np.mean(np.abs(self.output_data - output_pred))
        else:
            error = np.mean(np.square(self.output_data - output_pred))

        results = {
            'fitness': 1.0 - error  # 適応度: 1.0 - 誤差（最大化）
        }
        return results

    def print_result(self, circuit):

        output_pred = []
        for inp, out in zip(self.input_data, self.output_data):
            pred = circuit.activate(inp)
            output_pred.append(pred)

            print('input: ', inp, end='  ')
            print('label: ', out, end='  ')
            print('predict: ',
                  '[' + ' '.join(map(lambda z: f'{z: =.2f}', pred)) + ']')

        output_pred = np.vstack(output_pred)
        if self.error_type == 'mae':
            error = np.mean(np.abs(self.output_data - output_pred))
        else:
            error = np.mean(np.square(self.output_data - output_pred))

        print(f'error: {error: =.5f}')
