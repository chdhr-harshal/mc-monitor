#!/usr/local/bin/python

"""
Creates a markov chain given transition matrix and object distribution
"""
from __future__ import division
import numpy as np
import networkx as nx
import itertools
from math import e

class Node(object):
    def __init__(self, node_id, transition_vector, num_objects):
        self.transition_vector = transition_vector
        self.num_objects = num_objects
        self.node_id = node_id

    def update_num_objects(self, num_objects):
        self.num_objects = num_objects


class MarkovChain(object):
    def __init__(self, num_nodes, num_objects, total_time,
            transition_matrix=None, object_distribution=None, G=None):
        self.num_nodes = num_nodes
        self.num_objects = num_objects
        self.total_time = total_time
        self.current_time = 0
        self.time_left = self.total_time
        self.nodes_list = {}

        if G is None:
            self.transition_matrix = self.initialize_transition_matrix(transition_matrix)
            self.current_transition_matrix_power = self.transition_matrix
            self.G = nx.from_numpy_matrix(self.transition_matrix, create_using=nx.DiGraph())
        else:
            self.G = G
            self.transition_matrix = transition_matrix

        for i in self.G.nodes():
            self.nodes_list[i] = Node(i, None, 0)

        self.object_history = []
        self.object_distribution = self.initialize_object_distribution(object_distribution)
        self.object_assignment = self.update_object_assignment()

        # np.random.seed(100)

    def initialize_transition_matrix(self, transition_matrix):
        if transition_matrix is None:
            transition_matrix = np.random.rand(self.num_nodes, self.num_nodes)
            # Remove the self loops
            for row in xrange(self.num_nodes):
                transition_matrix[row][row] = 0
            # Randomly remove some edges in the matrix
            mask = np.random.randint(0, 2, size=transition_matrix.shape).astype(np.bool)
            zero_matrix = np.zeros((self.num_nodes, self.num_nodes))
            transition_matrix[mask] = zero_matrix[mask]
            # Divide each row by its sum to maintain stochasticity
            transition_matrix = transition_matrix/transition_matrix.sum(axis=1)[:,None]
        else:
            transition_matrix = np.array(transition_matrix)
        return transition_matrix

    def initialize_object_distribution(self, object_distribution):
        if object_distribution is None or object_distribution == 'uniform':
            object_distribution = {}
            for i in self.G.nodes():
                object_distribution[i] = 1.0/self.num_nodes
        elif object_distribution == 'direct':
            out_degrees = self.G.out_degree()
            total = sum(out_degrees.values())
            for x in out_degrees:
                out_degrees[x] /= total
            object_distribution = out_degrees
        elif object_distribution == 'inverse':
            out_degrees = self.G.out_degree()
            for x in out_degrees:
                out_degrees[x] = (self.num_nodes - out_degrees[x])/self.num_nodes
            total = sum(out_degrees.values())
            for x in out_degrees:
                out_degrees[x] /= total
            object_distribution = out_degrees
        else:
            object_distribution = object_distribution
        return object_distribution

#     def initialize_object_distribution(self, object_distribution):
#         lambda_val = 2
#         if object_distribution is None or object_distribution == 'uniform':
#             object_distribution = np.array([1.0/self.num_nodes for i in xrange(self.num_nodes)])
#         elif object_distribution == 'direct':
#             out_degrees = self.G.out_degree()
#             total = e**(lambda_val * sum(out_degrees.values()))
#             for x in out_degrees:
#                 out_degrees[x] = e**(lambda_val * out_degrees[x])/total
#             total = sum(out_degrees.values())
#             for x in out_degrees:
#                 out_degrees[x] /= total
#             object_distribution = np.array(out_degrees.values())
#         elif object_distribution == 'inverse':
#             out_degrees = self.G.out_degree()
#             total = sum(out_degrees.values())
#             for x in out_degrees:
#                 out_degrees[x] = e**(lambda_val * (total - out_degrees[x]))/e**(lambda_val * total)
#             total = sum(out_degrees.values())
#             for x in out_degrees:
#                 out_degrees[x] /= total
#             object_distribution = np.array(out_degrees.values())
#         else:
#             object_distribution = np.array(object_distribution)
#         return object_distribution

    def update_object_distribution(self):
        object_distribution_vector = np.array([])
        for x in self.G.nodes():
            object_distribution_vector.append(self.object_distribution[x])
        object_distribution_vector = object_distribution_vector.dot(self.current_transition_matrix_power)
        object_distribution = {}
        for i in xrange(len(object_distribution_vector)):
            node = self.G.nodes()[i]
            object_distribution[node] = object_distribution_vector[i]
        return object_distribution

    def update_object_assignment(self):
        object_assignment = {}
        for i in self.G.nodes():
            object_assignment[i] = self.object_distribution[i]*self.num_objects
        for i in self.G.nodes():
            self.nodes_list[i].update_num_objects(object_assignment[i])
        self.object_history.append(object_assignment)
        return object_assignment

    def simulate_time_step(self):
        # Update current transitition matrix power
        if self.transition_matrix is not None:
            self.transition_matrix = self.transition_matrix.dot(self.transition_matrix)

            # Update graph because in algorithms queries being made to graph, not transition matrix
            nodes = self.G.nodes()
            G = nx.DiGraph()
            for i in xrange(len(nodes)):
                node_i_id = nodes[i]
                for j in xrange(len(self.transition_matrix[i])):
                    node_j_id = nodes[j]
                    probability = self.transition_matrix[i][j]
                    G.add_edge(node_i_id, node_j_id, weight=probability)
            # self.G = nx.from_numpy_matrix(self.transition_matrix, create_using=nx.DiGraph())

        # Update time stuff
        self.current_time += 1
        self.time_left -= 1
