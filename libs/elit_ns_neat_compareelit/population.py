from __future__ import print_function

import os
import csv
from neat_cppn import Population
from neat.math_util import mean
from neat.reporting import ReporterSet
from . import archive
from . import metrices
from .reproduction import DefaultReproduction


class CompleteExtinctionException(Exception):
    pass

class Population:
    def __init__(self, config, initial_state=None, constraint_function=None):
        self.reporters = ReporterSet()
        self.novelty_archive = archive.NoveltyArchive(config)
        self.fitness_archive = archive.FitnessArchive(config)
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = DefaultReproduction(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size,
                                                           constraint_function=constraint_function)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    

    def run(self, evaluate_function, constraint_function=None, n=None):

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        self.archive_reporter = archive_reporter(self.save_path)
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            evaluate_function(self.population, self.config, self.generation)

            self.fitness_archive.update_archive(self.population)
            self.novelty_archive.update_archive(self.population, self.fitness_archive.archive)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                fitness = getattr(g, 'fitness', None)
                if fitness is None:
                    raise RuntimeError("fitness not assigned to genome {}".format(g.key))

                if best is None or fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, None, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(
                self.config, self.fitness_archive, self.novelty_archive, self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type, self.config.genome_config, self.config.pop_size,
                        constraint_function=constraint_function)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.archive_reporter.post_evaluate(self.fitness_archive, self.novelty_archive, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)
            print(f'novelty size: {len(self.novelty_archive.archive)}, novelty threshold: {self.novelty_archive.novelty_threshold}')

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome

class archive_reporter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.history_novelty_archive_file = os.path.join(self.save_path, 'history_novelty_archive.csv')
        self.history_novelty_archive_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2', "num_nodes", "num_connections", "num_enabled_connections", "num_disabled_connections"]

        self.history_fitness_archive_file = os.path.join(self.save_path, 'history_fitness_archive.csv')
        self.history_fitness_archive_header = ['generation', 'id', 'novelty', 'fitness', 'species', 'parent1', 'parent2', "num_nodes", "num_connections", "num_enabled_connections", "num_disabled_connections"]

        with open(self.history_novelty_archive_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_novelty_archive_header)
            writer.writeheader()

        with open(self.history_fitness_archive_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_fitness_archive_header)
            writer.writeheader()

    def post_evaluate(self, fitness_archive, novelty_archive, generation):
        with open(self.history_novelty_archive_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_novelty_archive_header)
            for genome in novelty_archive.archive.values():
                items = {
                    'generation': generation,
                    'id': genome.key,
                    'novelty': genome.novelty,
                    'fitness': genome.fitness,
                    'parent1': genome.parent1,
                    'parent2': genome.parent2,
                    "num_nodes": len(genome.nodes),
                    "num_connections": len(genome.connections),
                    "num_enabled_connections": len([c for c in genome.connections.values() if c.enabled]),
                    "num_disabled_connections": len([c for c in genome.connections.values() if not c.enabled]),
                }
                writer.writerow(items)

        with open(self.history_fitness_archive_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history_fitness_archive_header)
            for genome in fitness_archive.archive.values():
                writer.writerow({
                    'generation': generation,
                    'id': genome.key,
                    'novelty': genome.novelty,
                    'fitness': genome.fitness,
                    'parent1': genome.parent1,
                    'parent2': genome.parent2,
                    "num_nodes": len(genome.nodes),
                    "num_connections": len(genome.connections),
                    "num_enabled_connections": len([c for c in genome.connections.values() if c.enabled]),
                    "num_disabled_connections": len([c for c in genome.connections.values() if not c.enabled]),
                })