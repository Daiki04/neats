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
        self.novelty_threshold = None

    def calculate_novelty(self, data_now, data_fitness_archive):
        cos_sim = cosine_similarity(data_now, data_fitness_archive)
        min_dist = np.minimum.outer(np.linalg.norm(
            data_now, axis=1), np.linalg.norm(data_fitness_archive, axis=1))
        max_dist = np.maximum.outer(np.linalg.norm(
            data_now, axis=1), np.linalg.norm(data_fitness_archive, axis=1))
        ratio_dist = min_dist / max_dist
        novelty = 0.5 * (1 + cos_sim)/2 + 0.5 * ratio_dist

        # num_neighbours = 10 if len(
            # data_fitness_archive) > 10 else len(data_fitness_archive)
        num_neighbours = 10
        knn = np.sort(novelty, axis=1)[:, -num_neighbours:]
        knn = 1 - np.mean(knn, axis=1)
        return knn

    def update_archive(self, population, fitness_archive):
        new_archive = {}

        # 新規性の計算
        data_pops = [agent.data for _, agent in population.items()]
        data_fitness_archive = [agent.data for _,
                                agent in fitness_archive.archive.items()]
        data_novelty_archive = [agent.data for _,
                                agent in self.archive.items()]
        novelty_pops_list = self.calculate_novelty(
            data_pops, data_fitness_archive)
        for i, (key, genome) in enumerate(population.items()):
            novelty = novelty_pops_list[i]
            genome.novelty = novelty

            if self.novelty_threshold is None or novelty > self.novelty_threshold:
                new_archive[key] = genome

        if len(self.archive) != 0:
            novelty_archive_list = self.calculate_novelty(
                data_novelty_archive, data_fitness_archive)
            for i, (key, genome) in enumerate(self.archive.items()):
                novelty = novelty_archive_list[i]
                genome.novelty = novelty

                if self.novelty_threshold is None or novelty > self.novelty_threshold:
                    new_archive[key] = genome

        self.archive = new_archive

        # アーカイブのサイズが上限を超えた場合、新規性が高い順にソートして上位の個体のみ残す
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(
            ), key=lambda x: x[1].novelty, reverse=True)[:self.size])
            self.novelty_threshold = min([genome.novelty for genome in self.archive.values()])
        else:
            self.archive = dict(sorted(self.archive.items(),
                                key=lambda x: x[1].novelty, reverse=True))
        if len(self.archive) == 0:
            self.novelty_threshold = None
        self.archive_novelty = {
            key: genome.novelty for key, genome in self.archive.items()}
        self.archive_prob = {
            key: genome.novelty/sum(self.archive_novelty.values()) for key, genome in self.archive.items()}
        # print("Archive : ", self.archive_prob)


class FitnessArchive(BaseArchive):
    def __init__(self, config, size=30):
        super().__init__(config, size)

    def update_archive(self, population):
        self.archive.update(population)
        if len(self.archive) > self.size:
            self.archive = dict(sorted(self.archive.items(
            ), key=lambda x: x[1].fitness, reverse=True)[:self.size])

        self.archive_fitness = {
            key: genome.fitness for key, genome in self.archive.items()}
        self.archive_prob = {
            key: genome.fitness/sum(self.archive_fitness.values()) for key, genome in self.archive.items()}

    def get_best_genome(self):
        return self.archive[next(iter(self.archive))]


if __name__ == '__main__':
    novelty_archive = NoveltyArchive(None)
    archive = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [
        16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]]
    pop_now = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-28, -29, -30]]
    ans = novelty_archive.calculate_novelty(pop_now, archive)
    print(ans)
