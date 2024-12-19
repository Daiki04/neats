from neat.graphs import feed_forward_layers
from .nn import FeedForwardNetwork
from neat.activations import sigmoid_activation
from neat.aggregations import sum_aggregation

class FeedForwardNetwork(FeedForwardNetwork):

    # modified argument "config" to indice "genome_config"
    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.input_keys, config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.ws, cg.wb, cg.ctp))
                        node_expr.append("{:.10e}*F.silu(v[{}]) + {:.10e}*self.compute_bspline(v[{}], {})".format(cg.ws, inode, cg.wb, inode, cg.ctp))
                        # print(node_expr)

                ng = genome.nodes[node]
                aggregation_function = config.aggregation_function_defs.get(ng.aggregation)
                node_evals.append((node, aggregation_function, ng.response, inputs))

        return FeedForwardNetwork(config.input_keys, config.output_keys, node_evals)


    @staticmethod
    def create_from_weights(config, input_keys, output_keys, biases, weights, weight_thr=0.05, default_aggregation='sum', default_activation='sigmoid'):
        connections = [key for key,weight in weights.items() if abs(weight)>weight_thr]

        layers = feed_forward_layers(input_keys, output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        weight = weights[conn_key]
                        inputs.append((inode, weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, weight))

                bias = biases[node]
                aggregation_function = config.aggregation_function_defs.get(default_aggregation)
                activation_function = config.activation_defs.get(default_activation)
                node_evals.append((node, activation_function, aggregation_function, bias, 1, inputs))

        return FeedForwardNetwork(input_keys, output_keys, node_evals)
    
    def __str__(self) -> str:
        text = "FeedForwardNetwork\n"
        text += "  input_keys: {}\n".format(self.input_keys)
        text += "  output_keys: {}\n".format(self.output_keys)
        text += "  node_evals:\n"

        for node, activation, aggregation, bias, response, inputs in self.node_evals:
            text += "    node: {}\n".format(node)
            text += "      activation: {}\n".format(activation)
            text += "      aggregation: {}\n".format(aggregation)
            text += "      bias: {}\n".format(bias)
            text += "      response: {}\n".format(response)
            text += "      inputs:\n"
            for inode, weight in inputs:
                text += "        inode: {}, weight: {}\n".format(inode, weight)

        return text
    
