import os
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def make_species(expt_path: str):
    """種分化の過程を可視化

    Args:
        expt_path (str): 実験ディレクトリのパス
    """

    history_file = os.path.join(expt_path, 'history_pop.csv') # 種分化の履歴ファイル[generation, id, fitness, species, parent1, parent2]

    data = pd.read_csv(history_file) # データ読み込み(pd.DataFrame)

    max_generation = data['generation'].max() # 終了世代
    generation_pop_size = dict(data['generation'].value_counts()) # 世代ごとの個体数

    species_data = {} # 種ごとのデータ
    for key in data['species'].unique():
        sp_data = data.query(f'species=={key}') # speciesがkeyの行のみ抽出

        ### 種の生存期間
        created = sp_data['generation'].min()   # 種の生成世代
        extinct = sp_data['generation'].max()+1 # 種の絶滅世代

        ### 祖先の特定
        first_genome_parent = sp_data['parent1'].iloc[0] # 種に属する最初の個体の親から派生元の種を特定
        if first_genome_parent==-1:
            # 初期個体の場合は祖先なし
            ancestor = -1
        else:
            # 祖先の種を特定
            ancestor = data['species'].iloc[(data['id']==first_genome_parent).idxmax()] # 種に属する最初の個体の親idが最後に所属した種id

        ### 種の個体数の推移
        pop_history = dict(sp_data['generation'].value_counts()) # 種の世代ごとの個体数
        # 種の個体数の推移を世代ごとの割合に変換，描画のために消滅or未生成の世代も含める
        # 消滅or未生成の世代genではdict.get(gen, 0)でデフォルト値0を設定
        pop_history = np.array([pop_history.get(gen, 0)/generation_pop_size[gen] for gen in range(max_generation+1)])

        ### 種ごとのデータを保存
        species_data[key] = {
            'created': created,
            'extinct': extinct,
            'ancestor': ancestor,
            'pop_history': pop_history,
        }

    ### 種の描画順序を決定
    order = []
    stack = [-1] # 祖先なしの種にはancestor=-1が設定されているため，-1を初期値としてstackに追加
    while len(stack)>0:
        k = stack.pop(0)
        for key,species in species_data.items():
            if species['ancestor']==k:
                stack.insert(0, key) # stackの先頭にkeyを挿入
        order.append(k) # stackから取り出したkeyをorderに追加
    order = order[1:]  # 最初の-1を削除

    ### 種の描画
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    base_y = {order[0]: 0} # 種ごとのy座標
    for i in range(len(order)):

        key = order[i] # 種のid
        ### 種ごとのデータ
        ancestor = species_data[key]['ancestor'] # 祖先の種id
        created = species_data[key]['created']   # 生成世代
        extinct = species_data[key]['extinct']   # 絶滅世代
        gen = np.arange(created, extinct)        # 生成から絶滅までの世代
        pop = species_data[key]['pop_history']   # 世代ごとの個体数の総数に対する割合

        # 種ごとのy座標を設定
        if i>0:
            base_y[key] = base_y[prev_key] + max(0.2, np.max(np.where((prev_pop>0) & (pop>0), prev_pop, 0))+0.1)

        # 線の幅を個体数の割合に応じて変更
        upper = base_y[key] + pop[created:extinct] # 上限：種ごとのy座標+個体数の割合, [created:extinct]
        bottom = np.full(len(gen), base_y[key]) # 下限：種ごとのy座標, [created:extinct]

        # 種ごとのy座標に対して，世代ごとの個体数の割合を描画
        ax.plot(gen, bottom, color=colors[i%10], alpha=0.8)
        ax.plot(gen, upper, color=colors[i%10], alpha=0.2)
        ax.fill_between(gen, bottom, upper, alpha=0.5, color=colors[i%10])

        # 祖先が存在する場合（ancestor!=1）は祖先と種を繋ぐ点線を描画
        if ancestor!=-1:
            ax.plot([created]*2, [base_y[ancestor], base_y[key]], color='k', ls=':')

        # 次の種のために保存
        prev_key = key
        prev_created = created
        prev_extinct = extinct
        prev_pop = pop

    fig_height = base_y[prev_key] + np.max(prev_pop)+0.1

    fig.set_figheight(fig_height)
    fig.set_figwidth(max_generation/30)
    fig.set_figwidth(15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlim([-max_generation*0.05, max_generation*1.05])
    ax.set_yticks([])

    ax.set_xlabel('generation')
    ax.set_ylabel('species')

    filename = os.path.join(expt_path, 'species.jpg')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # plt.show()
