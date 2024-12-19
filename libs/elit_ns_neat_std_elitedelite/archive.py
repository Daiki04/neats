from . import metrices
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        self.metric_func = getattr(metrices, config.metric, None)
        self.novelty_threshold = config.threshold_init
        self.threshold_std_factor = 3  # 標準偏差に掛ける係数
        assert self.metric_func is not None, f'metric {config.metric} is not implemented in distances.py'

    def map_distance(self, key1, genome1, genomes):
        distances = {}
        for key2, genome2 in genomes.items():
            if key1 == key2:
                continue
            d = self.metric_func(genome1.data, genome2.data)
            distances[key2] = d
        return distances

    def update_archive(self, population):
        # すべての新規性スコアを収集して平均と標準偏差を用いて閾値を設定
        all_novelties = []
        for genome in population.values():
            distances = self.map_distance(None, genome, population)
            novelty = self.knn(list(distances.values()), k=self.config.neighbors)
            all_novelties.append(novelty)
        
        if all_novelties:
            mean_novelty = np.mean(all_novelties)
            std_novelty = np.std(all_novelties)
            self.novelty_threshold = mean_novelty + self.threshold_std_factor * std_novelty

        new_archive = {}

        candidates = population
        candidates.update(self.archive)

        for key, genome in population.items():
            distances_archive = self.map_distance(key, genome, self.archive)
            distances_new_archive = self.map_distance(key, genome, new_archive)
            distances_current = self.map_distance(key, genome, population)

            distances_archive.update(distances_new_archive)
            novelty_archive = self.knn(list(distances_archive.values()))

            distances_current.update(distances_archive)
            novelty = self.knn(
                list(distances_current.values()),
                k=self.config.neighbors
            )
            genome.novelty = novelty

            # 新規性の閾値で追加判定
            if novelty > self.novelty_threshold or self.novelty_threshold is None:
                new_archive[key] = genome

        # アーカイブの更新
        self.archive = new_archive
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True)[:self.size])

        for key, genome in self.archive.items():
            distances_archive = self.map_distance(key, genome, self.archive)
            novelty = self.knn(list(distances_archive.values()), k=self.config.neighbors)
            genome.novelty = novelty

        self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True))

        self.archive_novelty = {key: genome.novelty for key, genome in self.archive.items()}
        self.archive_prob = {key: genome.novelty / sum(self.archive_novelty.values()) for key, genome in self.archive.items()}

    def knn(self, distances, k=5):
        if len(distances) == 0:
            return float('inf')

        distances = sorted(distances)
        knn = distances[:k]
        density = sum(knn) / len(knn)
        return density

class FitnessArchive(BaseArchive):
    def __init__(self, config, size=100):
        super().__init__(config, size)

    def update_archive(self, population):
        self.archive.update(population)
        new_archive = {}
        new_archive_data = []
        self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].fitness, reverse=True))

        for key, genome in self.archive.items():
            if len(new_archive) == 0:
                new_archive[key] = genome
                new_archive_data.append(genome.data)
            else:
                sim_matrix = cosine_similarity([genome.data], new_archive_data)
                if sim_matrix.max() < 0.9:
                    new_archive[key] = genome
                    new_archive_data.append(genome.data)
        
        self.archive_fitness = {key:genome.fitness for key,genome in self.archive.items()}
        self.archive_prob = {key:genome.fitness/sum(self.archive_fitness.values()) for key,genome in self.archive.items()}

    def get_best_genome(self):
        return self.archive[next(iter(self.archive))]
