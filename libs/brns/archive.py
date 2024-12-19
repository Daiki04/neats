from . import metrices
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# エンコーダの定義
class Encoder(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 3 * input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])

        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.activation = torch.nn.functional.leaky_relu

    def forward(self, x):
        x = torch.Tensor(x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

class BaseArchive:
    def __init__(self, config, size=50):
        self.config = config
        self.archive = {}
        self.size = size

    def update_archive(self):
        raise NotImplementedError


class NoveltyArchive(BaseArchive):
    def __init__(self, config, size=500, pre_train=False):
        super().__init__(config, size)
        self.novelty_threshold = None
        self.batch_size = 128

        # 行動空間次元、隠れ層次元、エンコーダ出力次元の設定
        in_dim = 2
        out_dim = 2 * in_dim

        # ランダムエンコーダと学習エンコーダを初期化
        self.random_encoder = Encoder(in_dim, 3, out_dim)
        self.learned_encoder = Encoder(in_dim, 5, out_dim)

        # オプティマイザの設定
        # self.optimizer = optim.Adam(self.learned_encoder.parameters(), lr=0.01)
        self.optimizer = optim.RAdam(self.learned_encoder.parameters(), lr=0.01)

        # ランダムエンコーダはランダムなパラメータで固定
        for param in self.random_encoder.parameters():
            param.requires_grad = False

        self.random_encoder.eval()

        if pre_train:
            make_network_divergent(self.random_encoder, self.learned_encoder, [[0, 300], [0, 300]])

    # 新奇性の計算
    def calculate_novelty(self, genome):
        behavior = genome.data
        with torch.no_grad():
            rand_embedding = self.random_encoder(torch.tensor(behavior, dtype=torch.float32))
        learned_embedding = self.learned_encoder(torch.tensor(behavior, dtype=torch.float32))
        novelty = torch.norm(rand_embedding - learned_embedding, p=2).item()  # ユークリッド距離
        return novelty
    
    def update(self, population):
        self.pop = population
        self.pop_bds = [genome.data for _, genome in population.items()]
        self.pop_bds = np.array(self.pop_bds)
        # print(self.pop_bds.shape)

    def update_archive(self, population):
        population = {**self.archive, **population}
        self.update(population)

        pop_novs = []
        for i in range(0, len(self.pop_bds), self.batch_size):
            batch = torch.Tensor(self.pop_bds[i:i+self.batch_size])
            with torch.no_grad():
                latent_random = self.random_encoder(batch)
                self.learned_encoder.eval()
                latent_learned = self.learned_encoder(batch)
                diff = (latent_random - latent_learned)**2
                diff = diff.sum(dim=1)
                pop_novs += diff.cpu().detach().tolist()

        assert len(pop_novs) == len(self.pop_bds), f'{len(pop_novs)} != {len(self.pop_bds)}'

        for i, (key, genome) in enumerate(population.items()):
            genome.novelty = pop_novs[i]

        # if len(pop_novs) <= int(self.size*0.2):
        if len(pop_novs) <= int(self.size):
            self.archive = population
        else:
            # self.archive = dict(sorted(population.items(), key=lambda x: x[1].novelty, reverse=True)[:int(self.size*0.2)])
            self.archive = dict(sorted(population.items(), key=lambda x: x[1].novelty, reverse=True)[:int(self.size)])

        self.train()

    def train(self):
        self.update(self.archive)
        losses = []
        for epoch in range(3):
            for i in range(0, len(self.pop_bds), self.batch_size):
                batch = torch.Tensor(self.pop_bds[i:i+self.batch_size])
                with torch.no_grad():
                    latent_random = self.random_encoder(batch)
                self.learned_encoder.train()
                self.optimizer.zero_grad()
                latent_learned = self.learned_encoder(batch)
                l1 = (latent_random - latent_learned)**2
                l1 = l1.sum(dim=1)
                weights = torch.Tensor([1.0 for _ in range(batch.shape[0])])
                loss = l1 * weights
                loss = loss.mean()
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            print(f'epoch: {epoch+1}, loss: {np.mean(losses)}')
            losses = []

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
    
def make_network_divergent(frozen, learn, limits, iters=50):
    optimizer = optim.Adam(learn.parameters(), lr=0.001)
    batch_sz = 32

    for it in range(iters):
        learn.train()
        frozen.eval()

        batch = torch.zeros(batch_sz, frozen.input_dim)
        for d_i in range(frozen.input_dim):
            batch[:, d_i] = torch.rand(batch_sz) * (limits[d_i][1] - limits[d_i][0]) + limits[d_i][0]

        optimizer.zero_grad()
        latent_frozen = frozen(batch)
        latent_learn = learn(batch)
        loss = ((latent_frozen - latent_learn)**2).sum(dim=1).mean() * -1
        loss.backward()
        optimizer.step()

        print(f'iter: {it+1}, loss: {loss.item()}')