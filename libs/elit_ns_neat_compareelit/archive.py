from . import metrices
import numpy as np
from sklearn.neighbors import NearestNeighbors
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
        # self.novelty_threshold = None
        assert self.metric_func is not None, f'metric {config.metric} is not impelemented in distances.py'

    def update_archive(self, population, elite_archive):
        new_archive = {}
        new_archive_data = []
        data_pop = [ genome.data for genome in population.values() ]
        data_elite = [ genome.data for genome in elite_archive.values() ]

        k = 10 if len(elite_archive) > 10 else 2
        nearest_neighbors = NearestNeighbors(metric="euclidean", n_neighbors=k)
        nearest_neighbors.fit(data_elite)

        distances_pop_elit, _ = nearest_neighbors.kneighbors(data_pop)
        # print(np.array(distances_pop_elit).shape)
        # print()
        distances_pop_elit = np.mean(distances_pop_elit, axis=1)
        # print(distances_pop_elit)
        for i, (key, genome) in enumerate(population.items()):
            genome.novelty = distances_pop_elit[i]

        if len(self.archive) != 0:
            data_novelty = [ genome.data for genome in self.archive.values() ]
            distances_novelty_elit, _ = nearest_neighbors.kneighbors(data_novelty, return_distance=True)
            distances_novelty_elit = np.mean(distances_novelty_elit, axis=1)
            for i, (key, genome) in enumerate(self.archive.items()):
                genome.novelty = distances_novelty_elit[i]

        # populationとarchiveを合わせて，noveltyが大きい順にソート
        archive_pop_and_novelty = {**population, **self.archive}
        archive_pop_and_novelty = dict(sorted(archive_pop_and_novelty.items(), key=lambda x: x[1].novelty, reverse=True))

        for key, genome in archive_pop_and_novelty.items():
            if len(new_archive) == 0:
                new_archive[key] = genome
                new_archive_data.append(genome.data)
                continue

            cos_sim = cosine_similarity([genome.data], new_archive_data)
            max_sim = np.max(cos_sim)
            if max_sim < 0.9:
                new_archive[key] = genome
                new_archive_data.append(genome.data)

            if len(new_archive) > self.size:
                break

        self.archive = new_archive

        self.archive_novelty = {key:genome.novelty for key,genome in self.archive.items()}

        self.archive_prob = {key:genome.novelty/sum(self.archive_novelty.values()) for key,genome in self.archive.items()}

class FitnessArchive(BaseArchive):
    def __init__(self, config, size=100):
        super().__init__(config, size)

    def update_archive(self, population):
        new_archive = {}
        new_archive_data = []
        self.archive.update(population)
        self.archive = dict(sorted(self.archive.items(), key=lambda x: x[1].fitness, reverse=True)[:self.size])

        for key, genome in self.archive.items():
            if len(new_archive) == 0:
                new_archive[key] = genome
                new_archive_data.append(genome.data)
                continue

            cos_sim = cosine_similarity([genome.data], new_archive_data)
            max_sim = np.max(cos_sim)
            if max_sim < 0.9:
                new_archive[key] = genome
                new_archive_data.append(genome.data)

            if len(new_archive) > self.size:
                break
        
        self.archive_fitness = {key:genome.fitness for key,genome in self.archive.items()}
        self.archive_prob = {key:genome.fitness/sum(self.archive_fitness.values()) for key,genome in self.archive.items()}

    def get_best_genome(self):
        return self.archive[next(iter(self.archive))]
