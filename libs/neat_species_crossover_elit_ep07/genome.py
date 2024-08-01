import copy

from neat.genome import DefaultGenome, DefaultGenomeConfig
from .genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import required_for_output

class DefaultGenome(DefaultGenome):

    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, {k: g for k,g in self.connections.items() if g.enabled},
                                                                  genome_config.input_keys, genome_config.output_keys)
        new_genome = DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome
    
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)
    
    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node
    
    def add_connection(self, config, input_key, output_key, weight, enabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection


def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes
