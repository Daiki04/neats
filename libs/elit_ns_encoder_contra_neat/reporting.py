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
        self.history_pop_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2', "num_nodes", "num_connections", "num_enabled_connections", "num_disabled_connections"]
        self.history_novelty_file = os.path.join(self.save_path, 'history_novelty.csv')
        self.history_novelty_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2', "num_nodes", "num_connections", "num_enabled_connections", "num_disabled_connections"]
        self.history_score_file = os.path.join(self.save_path, 'history_score.csv')
        self.history_score_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2', "num_nodes", "num_connections", "num_enabled_connections", "num_disabled_connections"]

        self.genome_path = os.path.join(self.save_path, 'genome')
        os.makedirs(self.genome_path, exist_ok=True)

        with open(self.history_pop_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_pop_header)
            writer.writeheader()

        with open(self.history_novelty_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_novelty_header)
            writer.writeheader()

        with open(self.history_score_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_score_header)
            writer.writeheader()


    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        with open(self.history_pop_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_pop_header)
            for key,genome in population.items():
                items = {
                    'generation': self.generation,
                    'id': genome.key,
                    'novelty': genome.novelty,
                    'fitness': genome.fitness,
                    'species': species.get_species_id(genome.key),
                    'parent1': genome.parent1,
                    'parent2': genome.parent2,
                    "num_nodes": len(genome.nodes),
                    "num_connections": len(genome.connections),
                    "num_enabled_connections": len([c for c in genome.connections.values() if c.enabled]),
                    "num_disabled_connections": len([c for c in genome.connections.values() if not c.enabled]),
                }
                writer.writerow(items)

        current_novelty = max(population.values(), key=lambda z: z.novelty)
        items = {
            'generation': self.generation,
            'id': current_novelty.key,
            'novelty': current_novelty.novelty,
            'fitness': current_novelty.fitness,
            'species': species.get_species_id(current_novelty.key),
            'parent1': current_novelty.parent1,
            'parent2': current_novelty.parent2,
            "num_nodes": len(genome.nodes),
            "num_connections": len(genome.connections),
            "num_enabled_connections": len([c for c in genome.connections.values() if c.enabled]),
            "num_disabled_connections": len([c for c in genome.connections.values() if not c.enabled]),
        }
        with open(self.history_novelty_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_novelty_header)
            writer.writerow(items)
        # novelty_file = os.path.join(self.genome_path, f'{current_novelty.key}.pickle')
        # with open(novelty_file, 'wb') as f:
        #     pickle.dump(current_novelty, f)

        current_score = max(population.values(), key=lambda z: z.fitness)
        items = {
            'generation': self.generation,
            'id': current_score.key,
            'novelty': current_score.novelty,
            'fitness': current_score.fitness,
            'species': species.get_species_id(current_score.key),
            'parent1': current_score.parent1,
            'parent2': current_score.parent2,
            "num_nodes": len(genome.nodes),
            "num_connections": len(genome.connections),
            "num_enabled_connections": len([c for c in genome.connections.values() if c.enabled]),
            "num_disabled_connections": len([c for c in genome.connections.values() if not c.enabled]),
        }
        with open(self.history_score_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_score_header)
            writer.writerow(items)
        score_file = os.path.join(self.genome_path, f'{current_score.key}.pickle')
        with open(score_file, 'wb') as f:
            pickle.dump(current_score, f)
# class SaveResultReporter(BaseReporter):

#     def __init__(self, save_path):
#         self.generation = None

#         self.save_path = save_path
#         self.history_pop_file = os.path.join(self.save_path, 'history_pop.csv')
#         self.history_pop_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2']

#         self.history_novelty_file = os.path.join(self.save_path, 'history_novelty.csv')
#         self.history_novelty_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2']

#         self.history_score_file = os.path.join(self.save_path, 'history_score.csv')
#         self.history_score_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2']

#         self.history_novelty_archive_file = os.path.join(self.save_path, 'history_novelty_archive.csv')
#         self.history_novelty_archive_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2']

#         self.history_fitness_archive_file = os.path.join(self.save_path, 'history_fitness_archive.csv')
#         self.history_fitness_archive_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2']

#         self.genome_path = os.path.join(self.save_path, 'genome')
#         os.makedirs(self.genome_path, exist_ok=True)

#         with open(self.history_pop_file, 'w') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_pop_header)
#             writer.writeheader()

#         with open(self.history_novelty_file, 'w') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_novelty_header)
#             writer.writeheader()

#         with open(self.history_score_file, 'w') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_score_header)
#             writer.writeheader()

#         with open(self.history_novelty_archive_file, 'w') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_novelty_archive_header)
#             writer.writeheader()

#         with open(self.history_fitness_archive_file, 'w') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_fitness_archive_header)
#             writer.writeheader()


#     def start_generation(self, generation):
#         self.generation = generation

#     def post_evaluate(self, config, population, species, best_genome):
#         novelty_archive = population.novelty_archive.archive
#         fitness_archive = population.fitness_archive.archive
#         with open(self.history_pop_file, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_pop_header)
#             for key,genome in population.items():
#                 items = {
#                     'generation': self.generation,
#                     'id': genome.key,
#                     'novelty': genome.novelty,
#                     'fitness': genome.fitness,
#                     'species': species.get_species_id(genome.key),
#                     'parent1': genome.parent1,
#                     'parent2': genome.parent2
#                 }
#                 writer.writerow(items)

#         with open(self.history_novelty_archive_file, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_novelty_archive_header)
#             for key,genome in novelty_archive.archive.items():
#                 items = {
#                     'generation': self.generation,
#                     'id': genome.key,
#                     'novelty': genome.novelty,
#                     'fitness': genome.fitness,
#                     'species': species.get_species_id(genome.key),
#                     'parent1': genome.parent1,
#                     'parent2': genome.parent2
#                 }
#                 writer.writerow(items)

#         with open(self.history_fitness_archive_file, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_fitness_archive_header)
#             for key,genome in fitness_archive.archive.items():
#                 items = {
#                     'generation': self.generation,
#                     'id': genome.key,
#                     'novelty': genome.novelty,
#                     'fitness': genome.fitness,
#                     'species': species.get_species_id(genome.key),
#                     'parent1': genome.parent1,
#                     'parent2': genome.parent2
#                 }
#                 writer.writerow(items)

#         current_novelty = max(population.values(), key=lambda z: z.novelty)
#         items = {
#             'generation': self.generation,
#             'id': current_novelty.key,
#             'novelty': current_novelty.novelty,
#             'fitness': current_novelty.fitness,
#             'species': species.get_species_id(current_novelty.key),
#             'parent1': current_novelty.parent1,
#             'parent2': current_novelty.parent2
#         }
#         with open(self.history_novelty_file, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_novelty_header)
#             writer.writerow(items)
#         novelty_file = os.path.join(self.genome_path, f'{current_novelty.key}.pickle')
#         with open(novelty_file, 'wb') as f:
#             pickle.dump(current_novelty, f)

#         current_score = max(population.values(), key=lambda z: z.fitness)
#         items = {
#             'generation': self.generation,
#             'id': current_score.key,
#             'novelty': current_score.novelty,
#             'fitness': current_score.fitness,
#             'species': species.get_species_id(current_score.key),
#             'parent1': current_score.parent1,
#             'parent2': current_score.parent2
#         }
#         with open(self.history_score_file, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=self.history_score_header)
#             writer.writerow(items)
#         score_file = os.path.join(self.genome_path, f'{current_score.key}.pickle')
#         with open(score_file, 'wb') as f:
#             pickle.dump(current_score, f)


class NoveltySearchReporter(StdOutReporter):

    def post_evaluate(self, config, population, species, best_genome):
        noveltyes = [c.novelty for c in population.values()]
        fit_mean = np.mean(noveltyes)
        fit_std = np.std(noveltyes)
        print('Population\'s average novelty: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))

        scores = [c.fitness for c in population.values()]
        rew_mean = np.mean(scores)
        rew_std = np.std(scores)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness : {0:3.5f} stdev: {1:3.5f}'.format(rew_mean, rew_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))