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

    def update_archive(self, population, elite_population):
        # すべての新規性スコアを収集して平均と標準偏差を用いて閾値を設定
        all_novelties = []
        for genome in population.values():
            distances_elite = self.map_distance(None, genome, elite_population)
            novelty = self.knn(list(distances_elite.values()), k=self.config.neighbors)
            genome.novelty = novelty
            all_novelties.append(novelty)

        for genome in self.archive.values():
            distances_elite = self.map_distance(None, genome, elite_population)
            novelty = self.knn(list(distances_elite.values()), k=self.config.neighbors)
            genome.novelty = novelty
        
        if all_novelties:
            mean_novelty = np.mean(all_novelties)
            std_novelty = np.std(all_novelties)
            self.novelty_threshold = mean_novelty + self.threshold_std_factor * std_novelty

        new_archive = {}
        candidate_archive = {**self.archive, **population}

        for key, genome in candidate_archive.items():

            # 新規性の閾値で追加判定
            if genome.novelty > self.novelty_threshold or self.novelty_threshold is None:
                new_archive[key] = genome

        # アーカイブの更新
        self.archive = new_archive
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True)[:self.size])
        else:
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
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].fitness, reverse=True)[:self.size])
        
        self.archive_fitness = {key:genome.fitness for key,genome in self.archive.items()}
        self.archive_prob = {key:genome.fitness/sum(self.archive_fitness.values()) for key,genome in self.archive.items()}

    def get_best_genome(self):
        return self.archive[next(iter(self.archive))]
