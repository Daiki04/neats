import csv
import os
import pickle
import csv
import numpy as np

from neat_cppn import BaseReporter, StdOutReporter

class SaveResultReporter(BaseReporter):

    def __init__(self, save_path):
        self.generation = None

        self.save_path = save_path
        self.history_pop_file = os.path.join(self.save_path, 'history_pop.csv')
        self.history_pop_header = ['generation', 'id', 'novelty', 'fitness', 'parent1', 'parent2']
        self.history_novelty_file = os.path.join(self.save_path, 'history_novelty.csv')
        self.history_novelty_header = ['generation', 'id', 'novelty', 'fitness', 'parent1', 'parent2']
        self.history_fitness_file = os.path.join(self.save_path, 'history_fitness.csv')
        self.history_fitness_header = ['generation', 'id', 'novelty', 'fitness', 'parent1', 'parent2']

        self.genome_path = os.path.join(self.save_path, 'genome')
        os.makedirs(self.genome_path, exist_ok=True)

        with open(self.history_pop_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_pop_header)
            writer.writeheader()

        with open(self.history_novelty_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_novelty_header)
            writer.writeheader()

        with open(self.history_fitness_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_fitness_header)
            writer.writeheader()


    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, best_genome):
        with open(self.history_pop_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_pop_header)
            for key,genome in population.items():
                items = {
                    'generation': self.generation,
                    'id': genome.key,
                    'novelty': genome.novelty,
                    'fitness': genome.fitness,
                    'parent1': genome.parent1,
                    'parent2': genome.parent2
                }
                writer.writerow(items)

        current_novelty = max(population.values(), key=lambda z: z.novelty)
        items = {
            'generation': self.generation,
            'id': current_novelty.key,
            'novelty': current_novelty.novelty,
            'fitness': current_novelty.fitness,
            'parent1': current_novelty.parent1,
            'parent2': current_novelty.parent2
        }
        with open(self.history_novelty_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_novelty_header)
            writer.writerow(items)
        novelty_file = os.path.join(self.genome_path, f'{current_novelty.key}.pickle')
        with open(novelty_file, 'wb') as f:
            pickle.dump(current_novelty, f)

        current_fitness = max(population.values(), key=lambda z: z.fitness)
        items = {
            'generation': self.generation,
            'id': current_fitness.key,
            'novelty': current_fitness.novelty,
            'fitness': current_fitness.fitness,
            'parent1': current_fitness.parent1,
            'parent2': current_fitness.parent2
        }
        with open(self.history_fitness_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_fitness_header)
            writer.writerow(items)
        fitness_file = os.path.join(self.genome_path, f'{current_fitness.key}.pickle')
        with open(fitness_file, 'wb') as f:
            pickle.dump(current_fitness, f)


class NoveltySearchReporter(StdOutReporter):

    def post_evaluate(self, config, population, best_genome):
        noveltyes = [c.novelty for c in population.values()]
        fit_mean = np.mean(noveltyes)
        fit_std = np.std(noveltyes)
        print('Population\'s average novelty: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))

        fitnesss = [c.fitness for c in population.values()]
        rew_mean = np.mean(fitnesss)
        rew_std = np.std(fitnesss)
        print('Population\'s average fitness : {0:3.5f} stdev: {1:3.5f}'.format(rew_mean, rew_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - id {2}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_genome.key))
