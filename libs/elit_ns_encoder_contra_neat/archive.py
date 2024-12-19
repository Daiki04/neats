from . import metrices
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# エンコーダの定義
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.layers.append(nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            # x = torch.nn.functional.leaky_relu(layer(x))
        x = self.layers[-1](x)
        # for layer in self.layers:
        #     x = torch.relu(layer(x))
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
        input_dim = 58
        output_dim = 12
        hidden_dim_rand = [16, 12]
        hidden_dim_learned = [16, 12]
        # hidden_dim_rand = [36, 24, 24, 12]
        # hidden_dim_learned = [36, 24, 24, 12]
        # hidden_dim_learned = [46, 32, 24, 16]

        # ランダムエンコーダと学習エンコーダを初期化
        # ランダムエンコーダは対照学習
        # 学習エンコーダはランダムエンコーダの出力との差を最小化するように学習
        self.random_encoder = Encoder(input_dim, hidden_dim_rand, output_dim, num_layers=3)
        self.learned_encoder = Encoder(input_dim, hidden_dim_learned, output_dim, num_layers=3)

        # オプティマイザの設定
        self.optimizer_random = optim.Adam(self.random_encoder.parameters(), lr=1e-4)
        self.optimizer_learned = optim.Adam(self.learned_encoder.parameters(), lr=1e-4)

    # 新奇性の計算
    def calculate_novelty(self, genome):
        behavior = genome.data
        # print(np.array(genome.data).shape)
        with torch.no_grad():
            rand_embedding = self.random_encoder(torch.tensor(behavior, dtype=torch.float32))
            learned_embedding = self.learned_encoder(torch.tensor(behavior, dtype=torch.float32))
        novelty = torch.norm(rand_embedding - learned_embedding, p=2).item()  # ユークリッド距離
        return novelty
    
    def data_augmentation(self, x):
        noise = torch.randn_like(x) * 0.1  # ノイズの強度は調整可能
        return x + noise
    
    # 対照損失関数（NT-Xent Loss）
    def nt_xent_loss(self, z_i, z_j, temperature=0.6):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / temperature

        # ラベルの作成
        batch_size = z_i.size(0)
        labels = torch.arange(batch_size).to(z_i.device)
        labels = torch.cat([labels, labels], dim=0)

        # 対角成分を無視するマスク
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        similarity_matrix = similarity_matrix.masked_select(mask).view(2 * batch_size, -1)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

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

        for key, genome in self.archive.items():
            novelty = self.calculate_novelty(genome)
            genome.novelty = novelty

        self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True))

        self.archive_novelty = {key:genome.novelty for key,genome in self.archive.items()}

        self.archive_prob = {key:genome.novelty/sum(self.archive_novelty.values()) for key,genome in self.archive.items()}

        if len(self.archive) == self.size:
            self.novelty_threshold = min([genome.novelty for genome in self.archive.values()])

        # ランダムエンコーダの学習
        dataset = [agent.data for _, agent in population.items()]
        dataset = np.array(dataset)
        dataset = torch.tensor(dataset, dtype=torch.float32)
        x_i = self.data_augmentation(dataset)
        x_j = self.data_augmentation(dataset)
        self.optimizer_random.zero_grad()
        z_i = self.random_encoder(x_i)
        z_j = self.random_encoder(x_j)
        loss_random = self.nt_xent_loss(z_i, z_j)
        loss_random.backward()
        self.optimizer_random.step()


        # 学習エンコーダの学習
        self.optimizer_learned.zero_grad()
        random_embedding = self.random_encoder(dataset)
        learned_embedding = self.learned_encoder(dataset)
        loss_learned = torch.nn.functional.mse_loss(learned_embedding, random_embedding)
        loss_learned.backward()
        self.optimizer_learned.step()

        print(f'loss_random: {loss_random.item()}, loss_learned: {loss_learned.item()}')

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
