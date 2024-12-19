from . import metrices
import numpy as np

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
        candidates = population.copy()
        candidates.update(self.archive)
        for genome in candidates.values():
            distances = self.map_distance(None, genome, population)
            distances_archive = self.map_distance(None, genome, self.archive)
            distances.update(distances_archive)
            novelty = self.knn(list(distances.values()), k=self.config.neighbors)
            genome.novelty = novelty
            all_novelties.append(novelty)
        
        if all_novelties:
            mean_novelty = np.mean(all_novelties)
            std_novelty = np.std(all_novelties)
            self.novelty_threshold = mean_novelty + self.threshold_std_factor * std_novelty

        new_archive = {}

        for key, genome in candidates.items():

            # 新規性の閾値で追加判定
            if genome.novelty > self.novelty_threshold or self.novelty_threshold is None:
                new_archive[key] = genome

        # アーカイブの更新
        self.archive = new_archive
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True)[:self.size])
        else:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True))

        for key, genome in self.archive.items():
            distances = self.map_distance(key, genome, self.archive)
            genome.novelty_archive = self.knn(list(distances.values()), k=self.config.neighbors)

        self.archive_novelty = {key: genome.novelty_archive for key, genome in self.archive.items()}
        self.archive_prob = {key: genome.novelty_archive / sum(self.archive_novelty.values()) for key, genome in self.archive.items()}

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
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].fitness, reverse=True)[:self.size])
        
        self.archive_fitness = {key:genome.fitness for key,genome in self.archive.items()}
        self.archive_prob = {key:genome.fitness/sum(self.archive_fitness.values()) for key,genome in self.archive.items()}

    def get_best_genome(self):
        return self.archive[next(iter(self.archive))]
