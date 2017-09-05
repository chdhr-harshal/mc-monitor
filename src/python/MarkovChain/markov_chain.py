#!/usr/local/bin/python

from __future__ import division
import numpy as np
import networkx as nx

class MarkovChain(object):
    """
    Markov Chain instance
    """
    def __init__(self,
                num_nodes=None,                 # Number of nodes
                num_items=None,                 # Number of items
                item_distribution='uniform',    # Different item distributions
                                                # 1. uniform
                                                # 2. direct
                                                # 3. inverse
                                                # 4. custom
                current_time=1,                 # Total time
                G=None):                        # Networkx DiGraph such that:
                                                # 1. Each node has an attribute 'num_items'
                                                # 2. Each edge has an attribute 'weight'

        # np.random.seed(100)
        self.num_nodes = num_nodes
        self.num_items = num_items
        self.current_time = current_time
        self.item_distribution = item_distribution

        if G is None:   # No networkx graph provided
            while True:
                try:
                    self.initial_transition_matrix = self.initialize_transition_matrix()
                except:
                    continue
                break
            self.G = nx.from_numpy_matrix(self.initial_transition_matrix, create_using=nx.DiGraph())

            self.initial_item_distribution = self.initialize_item_distribution()
            nx.set_node_attributes(self.G, 'num_items', self.initial_item_distribution)
        else:
            self.G = G
            self.initial_transition_matrix = nx.to_numpy_recarray(self.G, dtype=[('weight',float)]).weight
            self.initial_item_distribution = nx.get_node_attributes(self.G, 'num_items')


    def initialize_transition_matrix(self):
        """
        Creates a random transition matrix with no self loops
        """

        transition_matrix = np.random.rand(self.num_nodes, self.num_nodes)
        # transition_matrix = np.ones((self.num_nodes, self.num_nodes))

        for i in xrange(self.num_nodes):
            transition_matrix[i][i] = 0

        # Randomly remove some edges in the matrix
        mask = np.random.randint(0, 2, size=transition_matrix.shape).astype(np.bool)
        zero_matrix = np.zeros((self.num_nodes, self.num_nodes))
        transition_matrix[mask] = zero_matrix[mask]

        # Randomly remove some edges in the matrix
        mask = np.random.randint(0, 2, size=transition_matrix.shape).astype(np.bool)
        zero_matrix = np.zeros((self.num_nodes, self.num_nodes))
        transition_matrix[mask] = zero_matrix[mask]

        # # Randomly remove some edges in the matrix
        # mask = np.random.randint(0, 2, size=transition_matrix.shape).astype(np.bool)
        # zero_matrix = np.zeros((self.num_nodes, self.num_nodes))
        # transition_matrix[mask] = zero_matrix[mask]

        # Divide each row by its sum to maintain row stochasticity
        transition_matrix = transition_matrix/transition_matrix.sum(axis=1)[:, None]

        # Test that resultant transition matrix is row stochastic because
        # masking can lead to all zeros rows
        absdiff = np.abs(transition_matrix.sum(axis=1)) - np.ones((self.num_nodes))
        assert absdiff.max() <= 10*np.spacing(np.float64(1)), "Not row stochastic"

        return transition_matrix

    def initialize_item_distribution(self):
        temp_dict = {}
        if self.item_distribution == 'uniform':
            items_per_node = self.num_items/self.num_nodes
            for i in self.G.nodes():
                temp_dict[i] = items_per_node

        elif self.item_distribution == 'direct':
            out_degs = self.G.out_degree()
            total = np.sum(out_degs.values())
            for i in self.G.nodes():
                temp_dict[i] = self.num_items * out_degs[i]/total

        elif self.item_distribution == 'inverse':
            out_degs = self.G.out_degree()
            for i in out_degs:
                out_degs[i] = (self.num_nodes - 1 - out_degs[i])/(self.num_nodes - 1)
            total = np.sum(out_degs.values())
            for i in self.G.nodes():
                temp_dict[i] = self.num_items * out_degs[i]/total
        else:
            # Do nothing
            pass

        return temp_dict


    def simulate_time_step(self):
        self.current_time += 1

        transition_matrix_power = np.linalg.matrix_power(self.initial_transition_matrix, self.current_time)
        item_distribution_list = np.array(self.initial_item_distribution.values()).dot(transition_matrix_power)
        self.G = nx.from_numpy_matrix(transition_matrix_power, create_using=nx.DiGraph())

        updated_item_distribution = {}
        for i in self.G.nodes():
            updated_item_distribution[i] = item_distribution_list[i]

        nx.set_node_attributes(self.G, 'num_items', updated_item_distribution)
