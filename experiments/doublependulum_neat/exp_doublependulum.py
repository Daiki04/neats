import os
import sys
import random
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath("__file__"))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))
LIB_DIR = os.path.join(ROOT_DIR, 'libs') # ライブラリのディレクトリ
ARGUMENTS_DIR = os.path.join(ROOT_DIR, 'arguments') # 引数のディレクトリ
ENV_DIR = os.path.join(ROOT_DIR, 'envs', 'doublependulum') # 環境のディレクトリ
sys.path.append(LIB_DIR)
sys.path.append(ARGUMENTS_DIR)
sys.path.append(ENV_DIR)

import neat_test
from parallel import EvaluatorParallel # 並列評価用クラス
from experiment_utils import initialize_experiment # 実験用関数
from evaluator import DoublePendulumControllerEvaluator # 評価器
from doublependulum_environment import DoublePendulumEnvironment
from doublependulum_neat import get_args

def main():
    ### 準備 ###
    args = get_args() # コマンドライン引数の取得

    for trial in range(args.trials):
        # シード値を固定
        np.random.seed(trial)
        random.seed(trial)

        save_path = os.path.join(CURR_DIR, 'out', 'doublependulum_exp', args.name+'_'+str(trial)) # 結果の保存先
        initialize_experiment(args.name, save_path, args) # 実験の初期化
        decode_function = neat_test.FeedForwardNetwork.create # ネットワークのデコード関数

        env = DoublePendulumEnvironment() # 環境の作成
        env_info = env.get_info() # 環境情報の取得
        evaluator = DoublePendulumControllerEvaluator(env) # 評価器
        evaluate_function = evaluator.evaluate_controller # 評価関数

        # 並列評価器の初期化
        parallel = EvaluatorParallel(
            num_workers=args.num_cores,
            evaluate_function=evaluate_function,
            decode_function=decode_function
        )

        # NEATの設定ファイルの作成
        # custom_config: NEATの設定ファイルに記述されているパラメータを上書き
        config_file = os.path.join(CURR_DIR, 'config', 'doublependulum.cfg')
        custom_config = [
            ('NEAT', 'pop_size', args.pop_size),
            ('NEAT', 'fitness_threshold', env_info['timestep']*10),
            ('DefaultGenome', 'num_inputs', env_info['input_size']),
            ('DefaultGenome', 'num_outputs', env_info['output_size']),
        ]
        config = neat_test.make_config(config_file, custom_config=custom_config)
        config_out_file = os.path.join(save_path, 'doublependulum.cfg')
        config.save(config_out_file) # 設定ファイルの保存

        ### NEATの実行 ###
        pop = neat_test.Population(config) # NEATの初期化
        
        figure_path = os.path.join(save_path, 'figure') # 進化の様子の保存先
        reporters = [
            neat_test.SaveResultReporter(save_path),
            neat_test.StdOutReporter(True),
        ] # レポーターの設定

        # レポーターの追加
        for reporter in reporters:
            pop.add_reporter(reporter)

        # 進化の実行
        try:
            best_genome = pop.run(fitness_function=parallel.evaluate, n=args.generation)

            print()
            print('best doublependulum result:')
            evaluator.print_result(decode_function(best_genome, config.genome_config)) # 最良個体の評価結果の表示
        finally:
            neat_test.figure.make_species(save_path) # 種の進化の様子の保存
            evaluator.get_gif(decode_function(best_genome, config.genome_config), save_path)

if __name__ == '__main__':
    main()