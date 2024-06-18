# NEAT-Pythonのテスト
このフォルダをひな形として，NEATの改良実験を行う．

## コマンドライン引数
| オプション | 項目 | デフォルト値 | 備考 |
| --- | --- | --- | --- |
| `-n`, `--name` | 実験名 | タスク名 ||
| `-t`, `--task` | タスク名 | `and` ||
| `-p`, `--pop-size` | 個体数 | 150 ||
| `-g`, `--generations` | 世代数 | 300 ||
| `-e`, `--error` | 損失関数 | `mse` |選択可能な損失関数は[`mse`, `mae`]|
| `-c`, `--num-cores` | 並列評価に使用するCPUコア数 | 4 ||

## 実行例
```sh
cd C:\Users\daiki\Documents\neats\experiments\test_neat
python and.py -n test -t and -p 100 -g 200 -e mae -c 2
```