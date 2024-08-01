from neat.graphs import feed_forward_layers

import numpy as np
from scipy.interpolate import BSpline
import torch.nn.functional as F
import torch


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, agg_func, response, links in self.node_evals:
            node_inputs = []
            for i, ws, wg, ctp in links:
                node_inputs.append(ws * float(self.compute_silu(self.values[i])) + wg * self.compute_bspline(self.values[i], ctp))
            s = agg_func(node_inputs)
            self.values[node] = s

        return [self.values[i] for i in self.output_nodes]
    
    def compute_bspline(self, x, ctp, degree=3):
        if not hasattr(self, 'knots'):
            self.knots = np.linspace(0, 1, len(ctp) + degree + 1)

        bspline_value = BSpline(self.knots, ctp, degree)(x)
        return float(bspline_value)
    
    def compute_silu(self, x):
        x_tensor = torch.tensor(x)
        return float(F.silu(x_tensor).item())
        

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
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
                        node_expr.append("{:.10e}*F.silu(v[{}]) + {:.10e}*self.compute_spline(v[{}], {})".format(cg.ws, inode, cg.wb, inode, cg.ctp))
                        print(node_expr)

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                node_evals.append((node, aggregation_function, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
