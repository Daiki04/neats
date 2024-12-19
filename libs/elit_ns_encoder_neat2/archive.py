from . import metrices
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

# エンコーダの定義
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=3):
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim * 3

        self.layers = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim)])
        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, output_dim))

        self.bn = torch.nn.BatchNorm1d(self.hidden_dim)

    def forward(self, x):
        x = torch.Tensor(x)
        for layer in self.layers[:-1]:
            # x = self.bn(torch.nn.functional.leaky_relu(layer(x)))
            x = torch.nn.functional.leaky_relu(layer(x))
        x = self.layers[-1](x)
        return x
    
    def weights_to_rand(self, d=5):
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.data = torch.randn_like(layer.weight.data) * d
                layer.bias.data = torch.randn_like(layer.bias.data) * d

class BaseArchive:
    def __init__(self, config, size=50):
        self.config = config
        self.archive = {}
        self.size = size

    def update_archive(self):
        raise NotImplementedError


class NoveltyArchive(BaseArchive):
    def __init__(self, config, size=200):
        super().__init__(config, size)
        self.novelty_threshold = None
        self.batch_size = 32

        # 行動空間次元、隠れ層次元、エンコーダ出力次元の設定
        input_dim = 2
        output_dim = 2 * input_dim

        # ランダムエンコーダと学習エンコーダを初期化
        self.random_encoder = Encoder(input_dim, output_dim)
        self.random_encoder.eval()
        self.learned_encoder = Encoder(input_dim, output_dim, num_hidden=5)

        # オプティマイザの設定
        self.optimizer = optim.Adam(self.learned_encoder.parameters(), lr=1e-2)

        # ランダムエンコーダはランダムなパラメータで固定
        for param in self.random_encoder.parameters():
            param.requires_grad = False

    def train(self, genome):
        genome_data = [x.data for x in genome.values()]
        genome_data = np.array(genome_data)
        losses = []

        for _ in range(3):
            for i in range(0, genome_data.shape[0], self.batch_size):
                batch = torch.tensor(genome_data[i:i+self.batch_size], dtype=torch.float32)
            with torch.no_grad():
                latent_random = self.random_encoder(batch)

            self.learned_encoder.train()
            self.optimizer.zero_grad()
            latent_learned = self.learned_encoder(batch)
            l1 = (latent_random - latent_learned) ** 2
            l1 = l1.sum(1)
            weights = torch.Tensor([1.0 for _ in range(batch.shape[0])])
            loss = l1 * weights
            loss = loss.mean().clone()
            losses.append(loss.item())

            if torch.isnan(loss).any():
                raise Exception("loss is Nan. Maybe tray reducing the learning rate.")
            
            loss.backward()
            self.optimizer.step()

        print(f"loss: {np.mean(losses)}")

    def update_archive(self, population):
        new_archive = {}
        # new_archive_data = []
        candidate_archive = {**self.archive, **population}
        candidate_data = [x.data for x in candidate_archive.values()]
        candidate_data = np.array(candidate_data)
        candidate_keys = list(candidate_archive.keys())
        candidate_novelty = []


        # 新規性の計算
        for i in range(0, candidate_data.shape[0], self.batch_size):
            batch = torch.tensor(candidate_data[i:i+self.batch_size], dtype=torch.float32)
            with torch.no_grad():
                latent_random = self.random_encoder(batch)
                self.learned_encoder.eval()
                latent_learned = self.learned_encoder(batch)
                diff = (latent_random - latent_learned) ** 2
                diff = diff.sum(1)
                diff = diff.cpu().detach().tolist()
                latent_learned = latent_learned.cpu().detach().numpy()

            for j, key in enumerate(candidate_keys[i:i+self.batch_size]):
                genome = candidate_archive[key]
                genome.novelty = diff[j]
                genome.latent = latent_learned[j]
                candidate_novelty.append(diff[j])

        # 新規性の閾値を設定
        mean_novelty = np.mean(candidate_novelty)
        std_novelty = np.std(candidate_novelty)
        self.novelty_threshold = mean_novelty + 3 * std_novelty
                
        # for key, genome in candidate_archive.items():
        #     if len(new_archive) == 0:
        #         new_archive[key] = genome
        #         new_archive_data.append(genome.latent)
        #     else:
        #         cos_sim = cosine_similarity([genome.latent], new_archive_data)
        #         max_sim = max(cos_sim[0])
        #         if max_sim < 0.8:
        #             new_archive[key] = genome
        #             new_archive_data.append(genome.latent)

        #     if len(new_archive) >= self.size:
        #         break
        
        # self.archive = new_archive
        
        for key, genome in candidate_archive.items():
            if genome.novelty > self.novelty_threshold:
                new_archive[key] = genome

            if len(new_archive) >= self.size:
                break

        # self.archive = dict(sorted(candidate_archive.items(), key=lambda x: x[1].novelty, reverse=True)[:self.size])

        self.archive = new_archive

        if len(self.archive) != 0:
            self.archive_novelty = {key:genome.novelty for key,genome in self.archive.items()}
            self.archive_prob = {key:genome.novelty/sum(self.archive_novelty.values()) for key,genome in self.archive.items()}
            self.archive_is_empty = False
        else:
            self.archive_novelty = {}
            self.archive_prob = {}
            self.archive_is_empty = True

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
