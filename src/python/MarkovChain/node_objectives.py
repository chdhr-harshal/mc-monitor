#!/usr/local/bin/python

"""
Child class of MarkovChain for relevant methods
related to the node objective function and algorithms
"""

from markov_chain import *
import copy
import random
from scipy.misc import logsumexp
from scipy.sparse.linalg import eigs

ROUNDING_DIGITS = 20

class MCNodeObjectives(MarkovChain):
    def __init__(self,
            num_nodes,
            num_objects,
            total_time,
            transition_matrix=None,
            object_distribution=None,
            G=None):
        # Create Markov Chain with given arguments
        super(MCNodeObjectives, self).__init__(num_nodes,
                                            num_objects,
                                            total_time,
                                            transition_matrix,
                                            object_distribution,
                                            G)

    # Objective function calculation methods
    def calculate_P_i(self, V, i):
        neighbors = set(self.G[i].keys())
        P_i_edges = [(i,y) for y in set(V).intersection(neighbors)]
        P_i = 0

        for edge in P_i_edges:
            source = edge[0]
            target = edge[1]
            P_i += self.G.get_edge_data(source, target)['weight']

        return np.around(P_i, ROUNDING_DIGITS)

    def calculate_Q_i(self, V, i):
        Q = self.G.nodes()
        return set(Q) - set(V)

    def calculate_f_i(self, V, i):
        num_objects = self.nodes_list[i].num_objects
        P_i = self.calculate_P_i(V, i)

        if P_i >=1:
            return 0

        edges = filter(lambda(x,y): y not in V, self.G.edges(i))

        A = num_objects * (1 - P_i)
        B = 0.0

        for (u,v) in edges:
            p_uv = self.G.get_edge_data(u,v)['weight']
            C = p_uv/(1 - P_i)
            B += C * (1 - C)
        return np.around(A * B, ROUNDING_DIGITS)

    def calculate_f(self, V):
        f = 0.0
        nodes = copy.deepcopy(self.G.nodes())
        for i in nodes:
            f += self.calculate_f_i(V, i)

        return np.around(f, ROUNDING_DIGITS)

    # Naive implementation of Greedy algorithm
    def naive_greedy(self, k):
        V = self.G.nodes()
        picked_nodes = []
        for i in xrange(k):
            objective_values = {}
            for v in V:
                objective_values[v] = self.calculate_f(picked_nodes + [v])
            greedy_choice_node = min(objective_values, key=objective_values.get)
            picked_nodes.append(greedy_choice_node)
            V.remove(greedy_choice_node)

        greedy_nodes = picked_nodes
        greedy_objective_value = self.calculate_f(greedy_nodes)

        return (greedy_nodes, greedy_objective_value)

    # Smart implementation of Greedy algorithm

    # Auxiliary method for the second summation
    def calculate_B_i(self, V, i, P_i):
        neighbors = set(self.G[i].keys())
        Q_i = neighbors - set(V)

        B_i = 0.0

        for k in Q_i:
            try:
                p_ik = self.G.get_edge_data(i, k)['weight']
            except:
                p_ik = 0
            B_i += p_ik * (1 - P_i - p_ik)

        return np.around(B_i, ROUNDING_DIGITS)

    def calculate_log_B_i(self, V, i, P_i):
        neighbors = set(self.G[i].keys())
        Q_i = neighbors - set(V)
        if len(Q_i) == 0:
            return -1*np.inf
        a = []
        for k in Q_i:
            p_ik = self.G.get_edge_data(i, k)['weight']
            to_append = np.log(p_ik*(1 - P_i - p_ik))
            a.append(to_append)
        return logsumexp(a)

    def get_smart_greedy_choice(self, P_dict, B_dict, candidate_nodes, all_nodes, picked_nodes):
        objective_values = {}
        P_matrix = {}
        B_matrix = {}

        for node_a in candidate_nodes:
            node_a_id = self.nodes_list[node_a].node_id
            f = 0.0
            P_matrix[node_a] = {}
            B_matrix[node_a] = {}
            for node_i in all_nodes:
                num_objects = self.nodes_list[node_i].num_objects
                node_i_id = self.nodes_list[node_i].node_id

                try:
                    p_ia = self.G.get_edge_data(node_i_id, node_a_id)['weight']
                except:
                    p_ia = 0

                P_matrix[node_a][node_i] = np.around(P_dict[node_i] + p_ia, ROUNDING_DIGITS)
                B_matrix[node_a][node_i] = np.around(B_dict[node_i] - (2* p_ia * (1 - P_dict[node_i] - p_ia)), ROUNDING_DIGITS)

                if P_matrix[node_a][node_i] < 1:
                    f += np.around(num_objects * B_matrix[node_a][node_i] / (1 - P_matrix[node_a][node_i]), ROUNDING_DIGITS)
                else:
                    f += 0
            objective_values[node_a] = f

        greedy_choice_node = min(objective_values, key=objective_values.get)
        P_dict = P_matrix[greedy_choice_node]
        B_dict = B_matrix[greedy_choice_node]
        return (greedy_choice_node, P_dict, B_dict)

    def get_smart_greedy_choice_3(self, P_dict, B_dict, candidate_nodes, all_nodes, picked_nodes):
        objective_values = {}
        P_matrix = {}
        B_matrix = {}

        for node_a in candidate_nodes:
            node_a_id = self.nodes_list[node_a].node_id
            f = []
            P_matrix[node_a] = {}
            B_matrix[node_a] = {}
            for node_i in all_nodes:
                num_objects = self.nodes_list[node_i].num_objects
                node_i_id = self.nodes_list[node_i].node_id
                try:
                    p_ia = self.G.get_edge_data(node_i_id, node_a_id)['weight']
                    if P_dict[node_i] != 0 :
                        p_vals = [P_dict[node_i], np.log(p_ia)]
                        P_matrix[node_a][node_i] = logsumexp(p_vals)
                    temp = logsumexp([np.log(1), P_matrix[node_a][node_i]], b=[1,-1], return_sign=True)[0]
                    b_vals = [B_dict[node_i], np.log(2*p_ia) + temp]
                    b_val = logsumexp(b_vals, b=[1, -1], return_sign=True)[0]
                    B_matrix[node_a][node_i] = b_val
                except:
                    P_matrix[node_a][node_i] = P_dict[node_i]
                    B_matrix[node_a][node_i] = B_dict[node_i]

                temp_term = logsumexp([np.log(1), P_matrix[node_a][node_i]], b=[1,-1], return_sign=True)[0]
                if temp_term == -1*np.inf:
                    P_matrix[node_a][node_i] = 0
                    temp_term = 0
                to_append = np.log(num_objects) + B_matrix[node_a][node_i] - temp_term
                f.append(np.log(num_objects) +  B_matrix[node_a][node_i] - temp_term)

            objective_values[node_a] = logsumexp(f)

        greedy_choice_node = min(objective_values, key=objective_values.get)
        P_dict = P_matrix[greedy_choice_node]
        B_dict = B_matrix[greedy_choice_node]
        return (greedy_choice_node, P_dict, B_dict)

    def get_smart_greedy_choice_2(self, P_dict, B_dict, candidate_nodes, all_nodes, picked_nodes):
        objective_values = {}
        P_matrix = {}
        B_matrix = {}

        for node_a in candidate_nodes:
            objective_values[node_a] = 0.0
            P_matrix[node_a] = {}
            B_matrix[node_a] = {}

        for node_i in all_nodes:
            num_objects = self.nodes_list[node_i].num_objects
            neighbors = set(self.G[node_i].keys()) - set(picked_nodes)
            not_neighbors = set(all_nodes) - set(self.G[node_i].keys()) - set(picked_nodes)

            for node_a in not_neighbors:
                P_matrix[node_a][node_i] = P_dict[node_i]
                B_matrix[node_a][node_i] = B_dict[node_i]


            for node_a in neighbors:
                p_ia = self.G.get_edge_data(node_i, node_a)['weight']

                P_matrix[node_a][node_i] = P_dict[node_i] + p_ia
                B_matrix[node_a][node_i] = B_dict[node_i] - (2*p_ia*(1 - P_dict[node_i] - p_ia))

            if P_matrix[node_a][node_i] < 1:
                objective_values[node_a] += num_objects * B_matrix[node_a][node_i] / (1 - P_matrix[node_a][node_i])
            else:
                objective_values[node_a] += 0

        greedy_choice_node = min(objective_values, key=objective_values.get)
        P_dict = P_matrix[greedy_choice_node]
        B_dict = B_matrix[greedy_choice_node]
        return (greedy_choice_node, P_dict, B_dict)

    def smart_greedy(self, k):
        V = copy.deepcopy(self.nodes_list.keys())
        U = copy.deepcopy(self.nodes_list.keys())
        picked_nodes = []

        P_dict = {}
        B_dict = {}

        for i in U:
            P_dict[i] = 0.0
            B_dict[i] = 0.0
            # P_dict[i] = self.calculate_P_i([], self.nodes_list[i].node_id)
            # B_dict[i] = self.calculate_B_i([], self.nodes_list[i].node_id, P_dict[i])


        for (u,v) in self.G.edges():
            p_uv = self.G.get_edge_data(u,v, {'weight':0.0})['weight']
            B_dict[u] += np.around(p_uv*(1-p_uv), ROUNDING_DIGITS)

        while len(picked_nodes) < k:
            smart_greedy_choice = self.get_smart_greedy_choice(P_dict, B_dict, V, U, picked_nodes)
            picked_nodes.append(smart_greedy_choice[0])
            P_dict = smart_greedy_choice[1]
            B_dict = smart_greedy_choice[2]
            V.remove(smart_greedy_choice[0])

        greedy_nodes = [self.nodes_list[node].node_id for node in picked_nodes]
        greedy_objective_value = self.calculate_f(greedy_nodes)
        return (greedy_nodes, greedy_objective_value)

    def smart_greedy_2(self, k):
        V = copy.deepcopy(self.nodes_list.keys())
        U = copy.deepcopy(self.nodes_list.keys())
        picked_nodes = []

        P_dict = {}
        B_dict = {}

        for i in U:
            B_dict[i] = self.calculate_log_B_i([], self.nodes_list[i].node_id, 0)
            P_dict[i] = -1*np.inf

        while len(picked_nodes) < k:
            smart_greedy_choice = self.get_smart_greedy_choice_3(P_dict, B_dict, V, U, picked_nodes)
            picked_nodes.append(smart_greedy_choice[0])
            P_dict = smart_greedy_choice[1]
            B_dict = smart_greedy_choice[2]
            V.remove(smart_greedy_choice[0])

        greedy_nodes = [self.nodes_list[node].node_id for node in picked_nodes]
        greedy_objective_value = self.calculate_f(greedy_nodes)
        return (greedy_nodes, greedy_objective_value)

    # Other baseline to compare

    # Top k nodes with highest betweeness centrality
    def highest_betweenness_centrality_nodes(self, k):
        if self.num_nodes > 1000:
            pivots = 1000
        else:
            pivots = self.num_nodes
        betweenness_centrality = nx.betweenness_centrality(self.G, k=pivots)
        sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get)
        sorted_nodes = [x for x in reversed(sorted_nodes)]
        nodes_set = sorted_nodes[:k]
        objective_value = self.calculate_f(nodes_set)
        return (nodes_set, objective_value)

    # Top k nodes with highest incoming probability
    def highest_in_probability_nodes(self, k):
        incoming_probability = self.G.in_degree(weight='weight')
        sorted_nodes = sorted(incoming_probability, key=incoming_probability.get)
        sorted_nodes = [x for x in reversed(sorted_nodes)]
        nodes_set = sorted_nodes[:k]
        objective_value = self.calculate_f(nodes_set)
        return (nodes_set, objective_value)

    # Top k nodes with incoming edges from highest number of other nodes
    def highest_in_degree_centrality_nodes(self, k):
        in_deg_centrality = nx.in_degree_centrality(self.G)
        sorted_nodes = sorted(in_deg_centrality, key=in_deg_centrality.get)
        sorted_nodes = [x for x in reversed(sorted_nodes)]
        nodes_set = sorted_nodes[:k]
        objective_value = self.calculate_f(nodes_set)
        return (nodes_set, objective_value)

    # Top k nodes with highest closeness centrality
    def highest_closeness_centrality_nodes(self, k):
        closeness_centrality = nx.closeness_centrality(self.G)
        sorted_nodes = sorted(closeness_centrality, key=closeness_centrality.get)
        sorted_nodes = [x for x in reversed(sorted_nodes)]
        nodes_set = sorted_nodes[:k]
        objective_value = self.calculate_f(nodes_set)
        return (nodes_set, objective_value)

    # Top k nodes with highest items
    def highest_item_nodes(self, k):
        item_nodes = {}
        for node in self.nodes_list.keys():
            item_nodes[node] =self.nodes_list[node].num_objects
        sorted_nodes = sorted(item_nodes, key=item_nodes.get)
        sorted_nodes = [x for x in reversed(sorted_nodes)]
        nodes_set = sorted_nodes[:k]
        objective_value = self.calculate_f(nodes_set)
        return (nodes_set, objective_value)

    # Randomly chosen k nodes
    def random_nodes(self, k):
        random_nodes = np.random.choice(self.num_nodes, k, replace=False)
        nodes_set = [x for x in random_nodes]
        objective_value = self.calculate_f(nodes_set)
        return (nodes_set, objective_value)

    # Get evolution of objective with increasing k
    def get_evolution(self, method, k):
        dataframe = []
        if method == self.smart_greedy:
            greedy_nodes_set = method(k)[0]
            for i in xrange(k+1):
                row = {}
                row['k'] = i
                # row['nodes_set'] = greedy_nodes_set[:i]
                row['objective_value'] = self.calculate_f(greedy_nodes_set[:i])
                dataframe.append(row)
        else:
            for i in xrange(k+1):
                row = {}
                result =  method(i)
                row['k'] = i
                # row['nodes_set'] = result[0]
                row['objective_value'] = result[1]
                dataframe.append(row)
        return dataframe

