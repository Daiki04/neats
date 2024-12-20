import math
import random

from neat.math_util import mean
from neat import DefaultReproduction


# modified to incoporate constraint function
class DefaultReproduction(DefaultReproduction):

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

    def reproduce(self, config, species, pop_size, generation, constraint_function=None):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        species_id_af = [(s.key, s.adjusted_fitness)
                         for s in remaining_species]
        species_id_af.sort(reverse=True, key=lambda x: x[1])
        species_af_runk = [sid for sid, _ in species_id_af]
        species_af_runk_index = {sid: idx for idx,
                                 sid in enumerate(species_af_runk)}

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        species_old_members = {}
        species_spawn_amounts = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            species_spawn_amounts[s.key] = spawn

            if spawn <= 0:
                # species_old_members[s.key] = old_members[:
                #                                          self.reproduction_config.elitism]
                species_old_members[s.key] = old_members
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            # repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            # repro_cutoff = max(repro_cutoff, 2)
            # old_members = old_members[:repro_cutoff]

            species_old_members[s.key] = old_members

        for s in remaining_species:
            spawn = species_spawn_amounts[s.key]
            if spawn <= 0:
                continue
            stag_rate = (generation - s.last_improved) / \
                self.stagnation.stagnation_config.max_stagnation
            old_members = species_old_members[s.key]
            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                if stag_rate < 0.5 and species_af_runk_index[s.key] > 0 and len(remaining_species) > 1:
                    repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *len(old_members)))
                    repro_cutoff = max(repro_cutoff, 2)
                    parent1_id, parent1 = random.choice(old_members[:repro_cutoff])
                    another_s = random.choice(
                        species_af_runk[:species_af_runk_index[s.key]])
                    while another_s == s:
                        another_s = random.choice(
                            species_af_runk[:species_af_runk_index[s.key]])
                    tournament_size = 3 if len(species_old_members[another_s]) > 10 else min(
                        len(species_old_members[another_s]), 2)

                    tournament2 = random.sample(
                        list(range(len(species_old_members[another_s]))), tournament_size)
                    parent2_id, parent2 = species_old_members[another_s][min(tournament2)]

                    # Note that if the parents are not distinct, crossover will produce a
                    # genetically identical clone of the parent (but with a different ID).
                    gid = next(self.genome_indexer)

                    valid = False
                    child = config.genome_type(gid)
                    child.configure_crossover_topo(
                        parent1, parent2, config.genome_config)
                    child.mutate(config.genome_config)

                    if constraint_function is not None:
                        while not constraint_function(child, config.genome_config, generation):
                            child = config.genome_type(gid)
                            child.configure_crossover_topo(
                                parent1, parent2, config.genome_config)
                            child.mutate(config.genome_config)

                else:
                    repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *len(old_members)))
                    repro_cutoff = max(repro_cutoff, 2)
                    parent1_id, parent1 = random.choice(old_members[:repro_cutoff])
                    parent2_id, parent2 = random.choice(old_members[:repro_cutoff])

                    # Note that if the parents are not distinct, crossover will produce a
                    # genetically identical clone of the parent (but with a different ID).
                    gid = next(self.genome_indexer)

                    valid = False
                    child = config.genome_type(gid)
                    child.configure_crossover(
                        parent1, parent2, config.genome_config)
                    child.mutate(config.genome_config)

                    if constraint_function is not None:
                        while not constraint_function(child, config.genome_config, generation):
                            child = config.genome_type(gid)
                            child.configure_crossover(
                                parent1, parent2, config.genome_config)
                            child.mutate(config.genome_config)

                setattr(child, 'parent1', parent1_id)
                setattr(child, 'parent2', parent2_id)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
