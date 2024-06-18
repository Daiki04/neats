import os
import shutil
import json
import argparse


def initialize_experiment(experiment_name: str, save_path: str, args: argparse.Namespace) -> None:
    """実験の初期化

    Args:
        experiment_name (str): 実験名
        save_path (str): 結果の保存先
        args (argparse.Namespace): コマンドライン引数

    Examples:
        initialize_experiment('xor', 'out/circuit_neat/xor', args)
    """
    try:
        os.makedirs(save_path)
    except:
        # 既にディレクトリが存在する場合
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            quit()
        print()

    argument_file = os.path.join(save_path, 'arguments.json') # 引数の保存先

    # argsオブジェクトが持つ属性を辞書に変換してjson形式で保存
    with open(argument_file, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_experiment(expt_path):
    with open(os.path.join(expt_path, 'arguments.json'), 'r') as f:
        expt_args = json.load(f)
    return expt_args
