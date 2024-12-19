from . import metrices
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=3):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim * 3

        self.layers = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim)])
        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.leaky_relu(layer(x))
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
    def __init__(self, config, size=200):
        super().__init__(config, size)
        self.novelty_threshold = None
        self.batch_size = 32

        # 入力次元とエンコーダ次元の設定
        input_dim = 2
        output_dim = 2 * input_dim

        # ランダムエンコーダと学習エンコーダの初期化
        self.random_encoder = Encoder(input_dim, output_dim)
        self.learned_encoder = Encoder(input_dim, output_dim, num_hidden=5)

        # Optimizerの設定
        self.optimizer_learned = optim.Adam(self.learned_encoder.parameters(), lr=1e-2)
        self.optimizer_random = optim.Adam(self.random_encoder.parameters(), lr=1e-3)

        # データ拡張設定（SimCLR用）
        self.data_augmentation = transforms.Compose([
            transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)], p=0.5),
            transforms.RandomApply([transforms.Lambda(lambda x: x * torch.randn_like(x) * 0.1)], p=0.5)
        ])

    def simclr_loss(self, z_i, z_j, temperature=0.5):
        """SimCLRの対照学習用損失関数"""
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / temperature
        sim = sim - torch.diag_embed(torch.diag(sim))

        exp_sim = torch.exp(sim)
        exp_sum = exp_sim.sum(dim=1, keepdim=True)
        log_prob = sim - torch.log(exp_sum)

        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(z.device)
        loss = -log_prob[range(2 * batch_size), labels].mean()
        return loss

    def train_random_encoder(self, genome_data):
        """SimCLRでランダムエンコーダを学習"""
        losses = []
        for _ in range(3):
            for i in range(0, genome_data.shape[0], self.batch_size):
                batch = torch.tensor(genome_data[i:i+self.batch_size], dtype=torch.float32)

                # データ拡張
                x_i = self.data_augmentation(batch)
                x_j = self.data_augmentation(batch)

                # エンコード
                z_i = self.random_encoder(x_i)
                z_j = self.random_encoder(x_j)

                # SimCLR損失計算と学習
                loss = self.simclr_loss(z_i, z_j)
                self.optimizer_random.zero_grad()
                loss.backward()
                self.optimizer_random.step()
                losses.append(loss.item())

        print(f"Random encoder loss: {np.mean(losses)}")

    def train_learned_encoder(self, genome_data):
        # 学習エンコーダの学習
        losses = []
        for _ in range(3):
            for i in range(0, genome_data.shape[0], self.batch_size):
                batch = torch.tensor(genome_data[i:i+self.batch_size], dtype=torch.float32)
                with torch.no_grad():
                    latent_random = self.random_encoder(batch)

                self.learned_encoder.train()
                self.optimizer_learned.zero_grad()
                latent_learned = self.learned_encoder(batch)
                l1 = (latent_random - latent_learned) ** 2
                loss = l1.mean()
                losses.append(loss.item())

                if torch.isnan(loss).any():
                    raise Exception("Loss is NaN. Try reducing the learning rate.")

                loss.backward()
                self.optimizer_learned.step()

        print(f"Learned encoder loss: {np.mean(losses)}")

    def train(self, genome):
        genome_data = [x.data for x in genome.values()]
        genome_data = np.array(genome_data)

        # ランダムエンコーダの学習
        self.train_random_encoder(genome_data)

        # 学習エンコーダの学習
        self.train_learned_encoder(genome_data)

    def update_archive(self, population):
        new_archive = {}
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

        for key, genome in candidate_archive.items():
            if genome.novelty > self.novelty_threshold:
                new_archive[key] = genome

            if len(new_archive) >= self.size:
                break

        self.archive = new_archive

        if len(self.archive) != 0:
            self.archive_novelty = {key: genome.novelty for key, genome in self.archive.items()}
            self.archive_prob = {key: genome.novelty / sum(self.archive_novelty.values()) for key, genome in self.archive.items()}
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
