import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath("__file__"))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))
LIB_DIR = os.path.join(ROOT_DIR, 'libs') # ライブラリのディレクトリ
ARGUMENTS_DIR = os.path.join(ROOT_DIR, 'arguments') # 引数のディレクトリ
ENV_DIR = os.path.join(ROOT_DIR, 'envs', 'maze') # 環境のディレクトリ
sys.path.append(LIB_DIR)
sys.path.append(ARGUMENTS_DIR)
sys.path.append(ENV_DIR)

import neat_test as neat
from experiment_utils import initialize_experiment
from parallel import EvaluatorParallel
from evaluator import MazeControllerEvaluator
from maze_drawer import MazeReporterNEAT
from maze_environment_numpy import MazeEnvironment
from maze_neat import get_args


def main():
    args = get_args()

    save_path = os.path.join(CURR_DIR, 'out', 'maze_neat', args.name)

    initialize_experiment(args.name, save_path, args)


    maze_env = MazeEnvironment.read_environment(ROOT_DIR, args.task)


    decode_function = neat.FeedForwardNetwork.create

    evaluator = MazeControllerEvaluator(maze_env, args.timesteps)
    evaluate_function = evaluator.evaluate_agent

    parallel = EvaluatorParallel(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode_function
    )


    config_file = os.path.join(CURR_DIR, 'config', 'maze_neat.cfg')
    custom_config = [
        ('NEAT', 'pop_size', args.pop_size),
    ]
    config = neat.make_config(config_file, custom_config=custom_config)
    config_out_file = os.path.join(save_path, 'maze_neat.cfg')
    config.save(config_out_file)


    pop = neat.Population(config)

    figure_path = os.path.join(save_path, 'progress')
    reporters = [
        neat.SaveResultReporter(save_path),
        neat.StdOutReporter(True),
        MazeReporterNEAT(maze_env, args.timesteps, figure_path, decode_function, args.generation, no_plot=args.no_plot)
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)


    try:
        pop.run(fitness_function=parallel.evaluate, n=args.generation)
    finally:
        neat.figure.make_species(save_path)

if __name__=='__main__':
    main()
