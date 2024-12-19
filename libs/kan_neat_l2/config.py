import neat
from .genome import DefaultGenome
from neat.config import *
from .reproduction import DefaultReproduction

def make_config(config_file: str, extra_info: list=None, custom_config: list=None) -> neat.Config:
    """NEATの設定ファイルを作成

    Args:
        config_file (str): 設定ファイルのパス
        extra_info (list, optional): 追加情報. Defaults to None.
        custom_config (list, optional): カスタム設定. Defaults to None.

    Returns:
        neat.Config: NEATの設定ファイル

    Examples:
        config = make_config('config/circuit_neat.cfg', custom_config=[('NEAT', 'pop_size', 100)])

    ※custom_configはNEAT-Pythonの最新バージョンでは存在しない
    """
    config = neat.Config(DefaultGenome,
                         DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file,
                         extra_info=extra_info,
                         custom_config=custom_config)
    return config