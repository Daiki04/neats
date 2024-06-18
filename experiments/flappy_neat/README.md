# NEAT-Pythonのテスト
このフォルダをひな形として，NEATの改良実験を行う．

## コマンドライン引数
| オプション | 項目 | デフォルト値 | 備考 |
| --- | --- | --- | --- |
| `-n`, `--name` | 実験名 | タスク名 ||
| `-t`, `--task` | タスク名 | `normal` ||
| `-p`, `--pop-size` | 個体数 | 150 ||
| `-g`, `--generations` | 世代数 | 500 ||
| `-c`, `--num-cores` | 並列評価に使用するCPUコア数 | 4 ||

## 実行例
```sh
cd ./experiments/flappy_neat
python flappt.py -n normal -t normal -p 150 -g 500 -c 4
```