import argparse


def get_args() -> argparse.Namespace:
    """コマンドライン引数を取得する

    Returns:
        argparse.Namespace: コマンドライン引数
    """
    # パーサーの作成
    # パーサー：コマンドライン引数を解析するためのクラス
    parser = argparse.ArgumentParser(
        description='Circuit NEAT experiment'
    )

    ### 引数の追加 ###
    parser.add_argument(
        '-n', '--name',
        type=str,
        help='experiment name (default: "{task}")'
    )
    parser.add_argument(
        '-t', '--task',
        default='and', type=str,
        help='circuit name (default: and, built on "envs/circuit/circuit_files/")'
    )

    parser.add_argument(
        '-p', '--pop-size',
        default=150, type=int,
        help='population size of NEAT (default: 150)'
    )
    parser.add_argument(
        '-g', '--generation',
        default=300, type=int,
        help='iterations of NEAT (default: 300)'
    )
    
    parser.add_argument(
        '-e', '--error',
        default='mse', type=str,
        help='error function (default: mse, options: [mse, mae])'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=4, type=int,
        help='number of parallel evaluation processes (default: 4)'
    )

    ### 引数の解析 ###
    args = parser.parse_args()

    # 実験名が指定されていない場合はタスク名を実験名とする
    if args.name is None:
        args.name = args.task

    return args


def get_figure_args():
    parser = argparse.ArgumentParser(
        description='make circuit figures'
    )

    parser.add_argument(
        'name',
        type=str,
        help='nam of experiment for making figures'
    )
    parser.add_argument(
        '-s', '--specified',
        type=int,
        help='input id, make figure for the only specified circuit (usage: "-s {id}")'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=1, type=int,
        help='number of parallel making processes (default: 1)'
    )
    parser.add_argument(
        '--not-overwrite',
        action='store_true', default=False,
        help='skip process if already gif exists (default: False)'
    )
    parser.add_argument(
        '--no-multi',
        action='store_true', default=False,
        help='do without using multiprocessing. if error occur, try this option. (default: False)'
    )

    args = parser.parse_args()

    assert args.name is not None, 'argumented error: input "{experiment name}"'

    return args
