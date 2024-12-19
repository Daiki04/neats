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
        # self.novelty_threshold = None
        assert self.metric_func is not None, f'metric {config.metric} is not impelemented in distances.py'

    def map_distance(self, key1, genome1, genomes):
        distances = {}
        for key2,genome2 in genomes.items():
            if key1==key2:
                continue
            
            # print(np.array(genome1.data).shape)
            d = self.metric_func(genome1.data, genome2.data)
            distances[key2] = d

        return distances

    def update_archive(self, population, elite_archive):
        new_archive = {}
        for key,genome in population.items():
            distances_archive = self.map_distance(key, genome, elite_archive)

            novelty = self.knn(list(distances_archive.values()), k=self.config.neighbors)

            if novelty > self.novelty_threshold or self.novelty_threshold is None:
                new_archive[key] = genome

            genome.novelty = novelty

        for key, genome in self.archive.items():
            distances_archive = self.map_distance(key, genome, self.archive)
            novelty = self.knn(list(distances_archive.values()), k=self.config.neighbors)

            genome.novelty = novelty

        # if len(new_archive)>0:
        #     self.time_out = 0
        # else:
        #     self.time_out += 1

        # if self.time_out >= 20:
        #     self.novelty_threshold *= 0.95
        #     if self.novelty_threshold < self.config.threshold_floor:
        #         self.novelty_threshold = self.config.threshold_floor
        #     self.time_out = 0

        # if len(new_archive) >= 5:
        #     self.novelty_threshold *= 1.2

        self.archive.update(new_archive)
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].novelty, reverse=True)[:self.size])

        self.archive_novelty = {key:genome.novelty for key,genome in self.archive.items()}

        self.archive_prob = {key:genome.novelty/sum(self.archive_novelty.values()) for key,genome in self.archive.items()}

        if len(self.archive) == self.size:
            self.novelty_threshold = min([genome.novelty for genome in self.archive.values()])

    def knn(self, distances, k=5):
        if len(distances)==0:
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
