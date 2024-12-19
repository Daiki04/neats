from . import metrices
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# エンコーダの定義
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.layers.append(nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            # x = torch.relu(layer(x))
            x = torch.nn.functional.leaky_relu(layer(x))
        return x

class BaseArchive:
    def __init__(self, config, size=50):
        self.config = config
        self.archive = {}
        self.size = size

    def update_archive(self):
        raise NotImplementedError


class NoveltyArchive(BaseArchive):
    def __init__(self, config, size=50):
        super().__init__(config, size)
        self.novelty_threshold = None

        # 行動空間次元、隠れ層次元、エンコーダ出力次元の設定
        input_dim = 67
        output_dim = 16
        hidden_dim_rand = [32, 32]
        hidden_dim_learned = [32, 32, 32]
        # hidden_dim_learned = [46, 32, 24, 16]

        # ランダムエンコーダと学習エンコーダを初期化
        self.random_encoder = Encoder(input_dim, hidden_dim_rand, output_dim, num_layers=3)
        self.learned_encoder = Encoder(input_dim, hidden_dim_learned, output_dim, num_layers=4)

        # オプティマイザの設定
        self.optimizer = optim.Adam(self.learned_encoder.parameters(), lr=0.01)

        # ランダムエンコーダはランダムなパラメータで固定
        for param in self.random_encoder.parameters():
            param.requires_grad = False

    # 新奇性の計算
    def calculate_novelty(self, genome):
        behavior = genome.data
        with torch.no_grad():
            rand_embedding = self.random_encoder(torch.tensor(behavior, dtype=torch.float32))
        learned_embedding = self.learned_encoder(torch.tensor(behavior, dtype=torch.float32))
        novelty = torch.norm(rand_embedding - learned_embedding, p=2).item()  # ユークリッド距離
        return novelty

    def update_archive(self, population):
        new_archive = {}

        # 新規性の計算
        for key,genome in population.items():
            novelty = self.calculate_novelty(genome)
            genome.novelty = novelty

            if self.novelty_threshold is None or novelty > self.novelty_threshold:
                new_archive[key] = genome

        # アーカイブの更新(現在の新規性アーカイブ+新しく追加された個体)
        self.archive.update(new_archive)

        # アーカイブのサイズが上限を超えた場合、新規性が高い順にソートして上位の個体のみ残す
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True)[:self.size])

        # エンコーダの学習
        losses = []
        dataset = [(torch.tensor(agent.data, dtype=torch.float32), torch.tensor(agent.data, dtype=torch.float32)) for _, agent in population.items()]
        self.optimizer.zero_grad()
        for data, target in dataset:
            rand_embedding = self.random_encoder(data)
            learned_embedding = self.learned_encoder(data)
            loss = torch.nn.functional.mse_loss(rand_embedding, learned_embedding)
            losses.append(loss.detach().numpy())
            loss.backward()
        self.optimizer.step()
        print(f'loss: {np.mean(losses)}')

        for key, genome in self.archive.items():
            novelty = self.calculate_novelty(genome)
            genome.novelty = novelty

        self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True))

        self.archive_novelty = {key:genome.novelty for key,genome in self.archive.items()}

        self.archive_prob = {key:genome.novelty/sum(self.archive_novelty.values()) for key,genome in self.archive.items()}

        if len(self.archive) == self.size:
            self.novelty_threshold = min([genome.novelty for genome in self.archive.values()])

class FitnessArchive(BaseArchive):
    def __init__(self, config, size=100):
        super().__init__(config, size)

    def update_archive(self, population):
        self.archive.update(population)
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].fitness, reverse=True)[:self.size])
        
        self.archive_fitness = {key:genome.fitness for key,genome in self.archive.items()}
        self.archive_prob = {key:genome.fitness/sum(self.archive_fitness.values()) for key,genome in self.archive.items()}

    def get_best_genome(self):
        return self.archive[next(iter(self.archive))]
