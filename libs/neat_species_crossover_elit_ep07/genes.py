from random import random

from neat.genes import DefaultConnectionGene, DefaultNodeGene

class DefaultConnectionGene(DefaultConnectionGene):

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.3:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene
    
class DefaultNodeGene(DefaultNodeGene):
    
        def crossover(self, gene2):
            """ Creates a new gene randomly inheriting attributes from its parents."""
            assert self.key == gene2.key
    
            # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
            # here because `choice` is substantially slower.
            new_gene = self.__class__(self.key)
            for a in self._gene_attributes:
                if random() > 0.3:
                    setattr(new_gene, a.name, getattr(self, a.name))
                else:
                    setattr(new_gene, a.name, getattr(gene2, a.name))
    
            return new_gene