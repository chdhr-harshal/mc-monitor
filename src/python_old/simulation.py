#!/usr/local/bin/python

"""
Simulates a markov chain with given number of nodes and objects distributed
uniformly randomly over the nodes
"""

import numpy as np
import pandas as pd
import networkx as nx
import itertools

class Node(object):
    def __init__(self, node_id, transition_vector, num_objects, node_ids):
        self.transition_vector = transition_vector
        self.num_objects = num_objects
        self.object_history = {}
        self.node_ids = node_ids

    def simulate_transitions(self):
        updates = {}
        for node_id in self.node_ids:
            updates[node_id] = 0

        for i in xrange(self.num_objects):
            destination = np.where(np.random.multinomial(1, self.transition_vector))[0][0]
            self.remove_objects()
            updates[destination]+=1

        return updates

    def update_history(self, time_left):
        self.object_history[time_left] = self.num_objects

    def remove_objects(self, num_removals=1):
        self.num_objects-=num_removals

    def add_objects(self, num_additions):
        self.num_objects+=num_additions

class MarkovChain(object):
    def __init__(self, num_nodes=3, num_objects=30, time_left=10,
                transition_matrix=None, initial_object_distribution=None):
        self.num_nodes = num_nodes
        self.node_ids = [i for i in xrange(self.num_nodes)]
        self.num_objects = num_objects
        self.set_transition_matrix(transition_matrix)
        self.nodes_list = []
        self.time_left = time_left
        self.current_object_distribution = {}
        self.set_initial_object_distribution(initial_object_distribution)
        self.G = nx.from_numpy_matrix(self.transition_matrix, create_using=nx.DiGraph())

    def __str__(self):
        return """
--------------------------
Time left: {}
Current object distribution: {}\n """.format(self.time_left, self.current_object_distribution)


    def set_transition_matrix(self, transition_matrix):
        if transition_matrix is None:
            self.transition_matrix = np.random.rand(self.num_nodes, self.num_nodes)
            self.transition_matrix = self.transition_matrix/self.transition_matrix.sum(axis=1)[:,None]
        else:
            self.transition_matrix = np.array(transition_matrix)

    def set_initial_object_distribution(self, initial_object_distribution):
        if initial_object_distribution is None:
            for node_id in self.node_ids:
                self.nodes_list.append(Node(node_id, self.transition_matrix[node_id],
                    self.num_objects/self.num_nodes, self.node_ids))
                self.current_object_distribution[node_id] = self.nodes_list[node_id].num_objects
        else:
            for node_id in xrange(len(self.node_ids)):
                self.nodes_list.append(Node(node_id, self.transition_matrix[node_id],
                    initial_object_distribution[node_id], self.node_ids))
                self.current_object_distribution[node_id] = self.nodes_list[node_id].num_objects


    # Get current object distribution
    def get_object_distribution(self):
        return self.current_object_distribution

    # Markov Chain simulation methods
    def simulate_markov_chain(self):
        for i in reversed(xrange(self.time_left)):
            self.simulate_time_step()

    def simulate_time_step(self):
        for node in self.nodes_list:
            updates = node.simulate_transitions()

            for node_id in updates:
                self.nodes_list[node_id].add_objects(updates[node_id])

        self.time_left-=1

        for node_id in xrange(self.num_nodes):
            self.nodes_list[node_id].update_history(self.time_left)
            self.current_object_distribution[node_id] = self.nodes_list[node_id].num_objects

    ###############################################

    # Edge selection objectives

    ###############################################

    # Returns k highest probability edges
    def get_k_max_probability_edges(self, k):
        sorted_edges = sorted(self.G.edges(data=True), key=lambda(source, target, data): data)[-k:]
        return [(x[0],x[1]) for x in sorted_edges]

    # Returns all possible edge sets of size k for the brute force algorithm
    def get_all_possible_edge_sets(self, k):
        return itertools.combinations(self.G.edges(),k)

    # Objective function calculation methods
    def calculate_D_i(self, E, i):
        D = self.G.edges()
        D_minus_E = list(set(D) - set(E))
        D_i_edges = filter(lambda (x,y):x==i, D_minus_E)
        return [edge[1] for edge in D_i_edges]

    def calculate_S_i(self, E, i):
        S_i_edges = filter(lambda(x,y):x==i, E)
        S_i = 0

        for edge in S_i_edges:
            source = edge[0]
            target = edge[1]
            S_i+=self.G.get_edge_data(source, target)['weight']

        return S_i

    def calculate_f_i(self, E, i):
        num_objects_i = self.nodes_list[i].num_objects
        S_i = self.calculate_S_i(E, i)
        D_i = self.calculate_D_i(E, i)

        A = num_objects_i * (1 - S_i)
        B = 0.0

        for k in D_i:
            p_ik = self.G.get_edge_data(i,k)['weight']
            C = p_ik/(1 - S_i)
            B += C*(1-C)

        return A*B

    def calculate_f(self, E):
        f = 0.0
        for i in self.node_ids:
            f += self.calculate_f_i(E, i)

        return f

    # Greedy algorithm
    def greedy_algorithm(self, k):
        E = self.G.edges()
        picked_edges = []
        for i in xrange(k):
            objective_values = {}
            Fp = self.calculate_f(picked_edges)
            for e in E:
                objective_values[e] = Fp - self.calculate_f(picked_edges + [e])
            edge_to_pick = max(objective_values, key=objective_values.get)
            picked_edges.append(edge_to_pick)
            E.remove(edge_to_pick)

        return picked_edges

    # Dynamic programming algorithm
    def dynamic_programming_algorithm(self, k):
        self.SOL_matrix = np.full((self.num_nodes, k+1), np.nan, dtype=float)

        for i in reversed(xrange(self.num_nodes)):
            for k_prime in xrange(k+1):
                self.SOL_matrix[i][k_prime] = self.get_SOL(i, k_prime)
        return self.SOL_matrix

    def calculate_ISOL(self, i, k):
        E_i = filter(lambda(x,y):x==i, self.G.edges())
        candidate_sets = []
        candidate_sets.append([])

        for j in range(1,k+1):
            for x in itertools.combinations(E_i, j):
                candidate_sets.append(list(x))

        objective_values = []
        for edge_set in candidate_sets:
            objective_values.append(self.calculate_f_i(edge_set, i))

        ISOL = min(objective_values)
        return ISOL

    def get_SOL(self, i, k):
        if k < 0:
            return np.infty
        if i >= self.num_nodes:
            return 0

        if not np.isnan(self.SOL_matrix[i][k]):
            return self.SOL_matrix[i][k]

        candidate_values = []
        for k_i in xrange(k+1):
            candidate_values.append(self.calculate_ISOL(i,k_i) + self.get_SOL(i+1, k-k_i))

        return min(candidate_values)

    # Algorithm calls - Edges
    def get_max_probability_algorithm_value(self, k):
        max_probability_edge_set = self.get_k_max_probability_edges(k)
        return self.calculate_f(max_probability_edge_set)

    def get_greedy_algorithm_value(self, k):
        greedy_edge_set = self.greedy_algorithm(k)
        return self.calculate_f(greedy_edge_set)

    def get_dynamic_programming_algorithm_value(self, k):
        self.dynamic_programming_algorithm(k)
        return self.SOL_matrix[0][k]

    def get_brute_force_algorithm_value(self, k):
        brute_force_obj_values = [self.calculate_f(x) for x in self.get_all_possible_edge_sets(k)]
        return min(brute_force_obj_values)

    # Simulation method
    def edge_objective_simulation(self, k):
        for i in reversed(xrange(self.time_left)):
            self.simulate_time_step()
            print "Time left: {}".format(i)
            print "Current object distribution dictionary: {}".format(self.get_object_distribution())
            print "\n"

            # print "Brute force OPTIMAL objective value: {}".format(self.get_brute_force_algorithm_value(k))
            print "Max probability algorithm value: {}".format(self.get_max_probability_algorithm_value(k))
            print "Dynamic programming algorithm value: {}".format(self.get_dynamic_programming_algorithm_value(k))
            print "Greedy algorithm value: {}".format(self.get_greedy_algorithm_value(k))

            print "================================================================================"


    ###############################################

    # Node selection objectives

    ###############################################

    def get_all_possible_node_sets(self, k):
        return itertools.combinations(self.G.nodes(),k)

    def get_top_k_nodes(self, k):
        baseline_objective = self.calculate_node_f([])
        node_objectives = {}
        for x in self.node_ids:
            node_objectives[x] = baseline_objective - self.calculate_node_f([x])

        sorted_nodes = sorted(node_objectives, key=node_objectives.get)[-k:]
        return sorted_nodes

    def get_greedy_nodes(self, k):
        V = self.G.nodes()
        picked_nodes = []
        for i in xrange(k):
            objective_values = {}
            baseline_objective = self.calculate_node_f(picked_nodes)
            for v in V:
                objective_values[v] = baseline_objective - self.calculate_node_f(picked_nodes + [v])
            node_to_pick = max(objective_values, key=objective_values.get)
            picked_nodes.append(node_to_pick)
            V.remove(node_to_pick)

        return picked_nodes

    # Objective function calculation methods
    def calculate_P_i(self, V, i):
        P_i_edges = filter(lambda(x,y):x==i and y in V, self.G.edges())
        P_i = 0

        for edge in P_i_edges:
            source = edge[0]
            target = edge[1]
            P_i += self.G.get_edge_data(source, target)['weight']

        return P_i

    def calculate_Q_i(self, V, i):
        Q = self.G.nodes()
        return list(set(Q) - set(V))

    def calculate_node_f_i(self, V, i):
        num_objects_i = self.nodes_list[i].num_objects
        P_i = self.calculate_P_i(V, i)
        if P_i >= 1:
            return 0

        Q_i = self.calculate_Q_i(V, i)

        A = num_objects_i * (1 - P_i)
        B = 0.0

        for k in Q_i:
            try:
                p_ik = self.G.get_edge_data(i,k)['weight']
            except:
                p_ik = 0
            C = p_ik/(1 - P_i)
            B += C*(1-C)

        return A*B

    def calculate_node_f(self, V):
        f = 0.0
        for i in self.node_ids:
            f += self.calculate_node_f_i(V, i)

        return f

    # Algorithm calls - Nodes
    def get_top_k_nodes_algorithm_value(self, k):
        top_k_nodes_set = self.get_top_k_nodes(k)
        print "Top k nodes set: {}".format(top_k_nodes_set)
        return self.calculate_node_f(top_k_nodes_set)

    def get_greedy_nodes_algorithm_value(self, k):
        greedy_nodes_set = self.get_greedy_nodes(k)
        # print "Greedy nodes set: {}".format(greedy_nodes_set)
        return self.calculate_node_f(greedy_nodes_set)

    def get_brute_force_nodes_algorithm_value(self, k):
        candidate_sets = self.get_all_possible_node_sets(k)
        brute_force_obj_values = [self.calculate_node_f(x) for x in candidate_sets]
        return min(brute_force_obj_values)

    def get_brute_force_nodes_sets(self, k):
        candidate_sets = [x for x in self.get_all_possible_node_sets(k)]
        brute_force_obj_values = [self.calculate_node_f(x) for x in candidate_sets]

        min_val = np.min(brute_force_obj_values)
        min_indices = np.argwhere(brute_force_obj_values == min_val).flatten().tolist()
        return [candidate_sets[x] for x in min_indices]

    # Simulation method
    def node_objective_simulation(self, k):
        for i in reversed(xrange(self.time_left)):
            self.simulate_time_step()
            print "Time left: {}".format(i)
            print "Current object distribution dictionary: {}".format(self.get_object_distribution())
            print "\n"

            print "Brute force OPTIMAL objective value: {}".format(self.get_brute_force_nodes_algorithm_value(k))
            print "Top k algorithm value: {}".format(self.get_top_k_nodes_algorithm_value(k))
            print "Greedy algorithm value: {}".format(self.get_greedy_nodes_algorithm_value(k))

            print "================================================================================"


    ###############################################

    # Expected SV evolution

    ###############################################
    def sv_evolution(self):
        for i in reversed(xrange(self.time_left)):
            self.simulate_time_step()
            print "Time left: {}".format(i)
            print "Current object distribution dictionary: {}".format(self.get_object_distribution())
            print "\n"

            print "SV value: {}".format(self.calculate_node_f([]))

            print "================================================================================"

# Main method
if __name__ == "__main__":
    nodes = 6
    objects = 100
    time_steps = 10
    initial_object_distribution=[15,15,15,15,15,25]
    transition_matrix = [[0,0.2,0.8,0,0,0],
                        [0.2,0,0.2,0.6,0,0],
                        [0.8,0.2,0,0,0,0],
                        [0,0.6,0,0,0.3,0.1],
                        [0,0,0,0.3,0,0.7],
                        [0,0,0,0.1,0.7,0.2]]
    mc = MarkovChain(nodes, objects, time_steps, transition_matrix, initial_object_distribution)
    print "Created Markov Chain with {} nodes, {} objects distributed equally, {} time steps to go".format(nodes,
                                                                                                            objects,
                                                                                                            time_steps)
    k = 3
    print "Setting k = {}".format(k)
    print "================================================================================"
    mc.edge_objective_simulation(k)
    #mc.sv_evolution()
