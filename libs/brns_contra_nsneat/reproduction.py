from __future__ import division

import math
import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig

from neat.math_util import mean
from neat import DefaultReproduction


# modified to incoporate constraint function
class DefaultReproduction(DefaultClassConfig):
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])
    
    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes, constraint_function=None):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)

            if constraint_function is not None:
                while not constraint_function(g, genome_config, 0):
                    g = genome_type(key)
                    g.configure_new(genome_config)

            setattr(g, 'parent1', -1)
            setattr(g, 'parent2', -1)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self, config, fitness_archive, novelty_archive, pop_size, generation, novelty_flag, constraint_function=None):
        new_population = {}

        # novelty_search_percent = max(0.3, 1 - generation / 500)
        if novelty_flag:
            novelty_search_percent = 0.9
        else:
            novelty_search_percent = 0.3

        for _ in range(pop_size):
            if random.random() < novelty_search_percent:
                parent1_id, parent1 = random.choices(list(novelty_archive.archive.items()))[0]

                gid = next(self.genome_indexer)

                valid = False
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent1, config.genome_config)
                child.mutate(config.genome_config)

                if constraint_function is not None:
                    while not constraint_function(child, config.genome_config, generation):
                        child = config.genome_type(gid)
                        child.configure_crossover(parent1, parent1, config.genome_config)
                        child.mutate(config.genome_config)

                setattr(child, 'parent1', parent1_id)
                setattr(child, 'parent2', parent1_id)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent1_id)
            else:
                parent1_id, parent1 = random.choices(list(fitness_archive.archive.items()))[0]

                if random.random() < 0.5:
                    parent2_id, parent2 = random.choices(list(fitness_archive.archive.items()))[0]
                else:
                    parent2_id, parent2 = random.choices(list(novelty_archive.archive.items()))[0]

                gid = next(self.genome_indexer)

                valid = False
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)

                if constraint_function is not None:
                    while not constraint_function(child, config.genome_config, generation):
                        child = config.genome_type(gid)
                        child.configure_crossover(parent1, parent2, config.genome_config)
                        child.mutate(config.genome_config)

                setattr(child, 'parent1', parent1_id)
                setattr(child, 'parent2', parent2_id)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
