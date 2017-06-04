#!/usr/local/bin/python

"""
Child class of Markov Chain for relevant methods
related to the edge objective function and algorithms
"""

from markov_chain import *
import copy

ROUNDING_DIGITS = 10

class MCEdgeObjectives(MarkovChain):
    def __init__(self,
                num_nodes,
                num_objects,
                total_time,
                transition_matrix=None,
                object_distribution=None,
                G=None):
        # Create Markov Chain with given arguments
        super(MCEdgeObjectives, self).__init__(num_nodes,
                                        num_objects,
                                        total_time,
                                        transition_matrix,
                                        object_distribution,
                                        G)
        # Calculate betweenness centrality of edges at the beginning
        self.betweenness_centrality = nx.edge_betweenness_centrality(self.G, weight='weight')

    # Objective function calculation methods
    def calculate_D_i(self, E, i):
        D = self.G.edges()
        D_minus_E = list(set(D) - set(E))
        D_i_edges = filter(lambda (x,y):x==i, D_minus_E)
        return [edge[1] for edge in D_i_edges]

    def calculate_S_i(self, E, i):
        S_i_edges = filter(lambda (x,y):x==i, E)
        S_i = 0

        for edge in S_i_edges:
            source = edge[0]
            target = edge[1]
            S_i += self.G.get_edge_data(source, target)['weight']

        return S_i

    def calculate_f_i(self, E, i):
        num_objects = self.nodes_list[i].num_objects
        S_i = self.calculate_S_i(E, i)
        D_i = self.calculate_D_i(E, i)

        A = num_objects * (1 - S_i)
        B = 0.0

        for k in D_i:
            p_ik = self.G.get_edge_data(i, k)['weight']
            C = p_ik/(1 - S_i)
            B += C * (1 - C)

        return A * B

    def calculate_f(self, E):
        f = 0.0
        nodes = copy.deepcopy(self.G.nodes())
        for i in nodes:
            f += self.calculate_f_i(E, i)

        return f

    # Naive implementation of Greedy algorithm
    def naive_greedy(self, k):
        E = self.G.edges()
        picked_edges = []
        for i in xrange(k):
            objective_values = {}
            baseline_objective = self.calculate_f(picked_edges)
            for e in E:
                objective_values[e] = baseline_objective - self.calculate_f(picked_edges + [e])
            greedy_choice = max(objective_values, key=objective_values.get)
            picked_edges.append(greedy_choice)
            E.remove(greedy_choice)

        greedy_edges = picked_edges
        greedy_objective_value = self.calculate_f(greedy_edges)

        return (greedy_edges, greedy_objective_value)

    # Smart implementation of Greedy algorithm

    # Auxiliary method for the second summation
    def calculate_B_i(self, i, S_i, D_i):
        B_i = 0.0

        for k in D_i:
            try:
                p_ik = self.G.get_edge_data(i, k)['weight']
            except:
                p_ik = 0
            B_i += p_ik * (1 - S_i - p_ik)

        return B_i

    def calculate_C_i(self, i,  D_i):
        C_i = 0.0

        for k in D_i:
            try:
                p_ik = self.G.get_edge_data(i, k)['weight']
            except:
                p_ik = 0

        return C_i

    def get_smart_greedy_choice_2(self, S_dict, B_dict, candidate_edges, all_nodes, picked_edges):
        objective_values = {}
        S_matrix = {}
        B_matrix = {}
        for a in xrange(len(candidate_edges)):
            S_matrix[a] = copy.deepcopy(S_dict)
            B_matrix[a] = copy.deepcopy(B_dict)
            (x,y) = candidate_edges[a]
            p_xy = self.G.get_edge_data(x,y)['weight']
            num_objects = self.nodes_list[x].num_objects

            new_S = S_dict[x] + p_xy
            new_B = B_dict[x] - 2*p_xy*(1 - new_S)

            if new_S == 1:
                new = 0
            else:
                new = new_B/(1 - new_S)

            if S_dict[x] == 1:
                old = 0
            else:
                old = B_dict[x]/(1 - S_dict[x])

            S_matrix[a][x] = new_S
            B_matrix[a][x] = new_B

            change = num_objects*(new - old)
            objective_values[a] = change

        greedy_choice_index = min(objective_values, key=objective_values.get)
        greedy_choice_edge = candidate_edges[greedy_choice_index]
        return (greedy_choice_edge, S_matrix[greedy_choice_index], B_matrix[greedy_choice_index])

    def get_smart_greedy_choice(self, S_dict, D_dict, B_dict, candidate_edges, all_nodes, picked_edges):
        objective_values = {}
        S_matrix = {}
        D_matrix = {}
        B_matrix = {}

        for a in xrange(len(candidate_edges)):
            (x,y) = candidate_edges[a]
            f = 0.0
            S_matrix[a] = {}
            D_matrix[a] = {}
            B_matrix[a] = {}
            for node_i in all_nodes:
                num_objects = self.nodes_list[node_i].num_objects
                node_i_id = self.nodes_list[node_i].node_id
                I = int(x==node_i_id)
                try:
                    p_iy = self.G.get_edge_data(node_i_id, y)['weight']
                except:
                    p_iy = 0

                S_matrix[a][node_i] = S_dict[node_i] + I*p_iy
                if I:
                    D_matrix[a][node_i] = set(D_dict[node_i]) - set([y])
                else:
                    D_matrix[a][node_i] = D_dict[node_i]

                C_i = self.calculate_C_i(node_i_id, D_dict[node_i])
                B_matrix[a][node_i] = B_dict[node_i] - I*(2*p_iy*(1-S_dict[node_i] - p_iy))

                if S_matrix[a][node_i] < 1:
                    f += num_objects * B_matrix[a][node_i] / (1 - S_matrix[a][node_i])
                else:
                    f += 0

            objective_values[a] = f

        greedy_choice_index = min(objective_values, key=objective_values.get)
        greedy_choice_edge = candidate_edges[greedy_choice_index]
        S_dict = S_matrix[greedy_choice_index]
        D_dict = D_matrix[greedy_choice_index]
        B_dict = B_matrix[greedy_choice_index]
        return (greedy_choice_edge, S_dict, D_dict, B_dict)

    def smart_greedy(self, k):
        E = copy.deepcopy(self.G.edges())
        D = copy.deepcopy(self.G.edges(data=True))
        U = copy.deepcopy(self.nodes_list.keys())
        picked_edges = []

        S_dict = {}
        B_dict = {}
        D_dict = {}

        for i in U:
            S_dict[i] = self.calculate_S_i([], self.nodes_list[i].node_id)
            D_dict[i] = self.calculate_D_i([], self.nodes_list[i].node_id)
            B_dict[i] = self.calculate_B_i(self.nodes_list[i].node_id, S_dict[i], D_dict[i])

        while len(picked_edges) < k:
            smart_greedy_choice = self.get_smart_greedy_choice(S_dict, D_dict, B_dict, E, U, picked_edges)
            picked_edges.append(smart_greedy_choice[0])
            S_dict = smart_greedy_choice[1]
            D_dict = smart_greedy_choice[2]
            B_dict = smart_greedy_choice[3]
            E.remove(smart_greedy_choice[0])

        greedy_edges = picked_edges
        greedy_objective_value = self.calculate_f(greedy_edges)
        return (greedy_edges, greedy_objective_value)

    def smart_greedy_2(self, k):
        E = copy.deepcopy(self.G.edges())
        D = copy.deepcopy(self.G.edges(data=True))
        U = copy.deepcopy(self.nodes_list.keys())
        picked_edges = []

        S_dict = {}
        D_dict = {}
        B_dict = {}

        for i in U:
            S_dict[i] = self.calculate_S_i([], self.nodes_list[i].node_id)
            D_dict[i] = self.calculate_D_i([], self.nodes_list[i].node_id)
            B_dict[i] = self.calculate_B_i(self.nodes_list[i].node_id, S_dict[i], D_dict[i])

        while len(picked_edges) < k:
            smart_greedy_choice = self.get_smart_greedy_choice_2(S_dict, B_dict, E, U, picked_edges)
            picked_edges.append(smart_greedy_choice[0])
            S_dict = smart_greedy_choice[1]
            B_dict = smart_greedy_choice[2]
            E.remove(smart_greedy_choice[0])

        greedy_edges = picked_edges
        greedy_objective_value = self.calculate_f(greedy_edges)
        return (greedy_edges, greedy_objective_value)

    # Dynamic programming algorithm
    def calculate_ISOL(self, i, k):
        E_i = self.G.edges(i)
        candidate_sets = []
        candidate_sets.append([])

        for j in  range(1, k+1):
            for x in itertools.combinations(E_i, j):
                candidate_sets.append(list(x))

        objective_values = {}
        for j in xrange(len(candidate_sets)):
            objective_values[j] = self.calculate_f_i(candidate_sets[j], i)

        min_index = min(objective_values, key=objective_values.get)
        ISOL_edge_set = candidate_sets[min_index]
        ISOL_value = objective_values[min_index]
        return (ISOL_edge_set, ISOL_value)

    def calculate_SOL(self, i, k):
        if k < 0:
            return ([], np.infty)
        if i >= self.num_nodes:
            return ([], 0)

        if not np.isnan(self.SOL_matrix[i][k]):
            return (self.SOL_edge_set_matrix[i][k], self.SOL_matrix[i][k])

        objective_values = {}
        candidate_sets = [0 for x in xrange(k+1)]
        for k_i in xrange(k+1):
            ISOL = self.calculate_ISOL(i, k_i)
            ISOL_value = ISOL[1]
            ISOL_edge_set = ISOL[0]
            SOL = self.calculate_SOL(i+1, k-k_i)
            SOL_value = SOL[1]
            SOL_edge_set = SOL[0]
            candidate_sets[k_i] = ISOL_edge_set + SOL_edge_set
            objective_values[k_i] = ISOL_value + SOL_value

        min_index = min(objective_values, key=objective_values.get)
        SOL_value = objective_values[min_index]
        SOL_edge_set = candidate_sets[min_index]
        return (SOL_edge_set, SOL_value)

    def dp_algorithm(self, k):
        self.SOL_matrix = np.full((self.num_nodes, k+1), np.nan, dtype=float)
        self.SOL_edge_set_matrix = np.zeros((self.num_nodes, k+1), dtype=np.ndarray)

        for i in reversed(xrange(self.num_nodes)):
            for k_prime in xrange(k+1):
                SOL = self.calculate_SOL(i, k_prime)
                self.SOL_matrix[i][k_prime] = SOL[1]
                self.SOL_edge_set_matrix[i][k_prime] = SOL[0]
        return (self.SOL_edge_set_matrix[0][k], self.SOL_matrix[0][k])

    # Other baseline to compare

    # Top k edges with highest betweenness centrality
    def highest_betweenness_centrality_edges(self, k):
        sorted_edges = sorted(self.betweenness_centrality, key=self.betweenness_centrality.get)
        sorted_edges = [x for x in reversed(sorted_edges)][:k]
        edges_set = [(x[0],x[1]) for x in sorted_edges]
        objective_value = self.calculate_f(edges_set)
        return (edges_set, objective_value)

    # Top k edges with highest probability
    def highest_probability_edges(self, k):
        sorted_edges = sorted(self.G.edges(data=True), key=lambda(source, target, data): data)
        sorted_edges = [x for x in reversed(sorted_edges)][:k]
        edges_set = [(x[0],x[1]) for x in sorted_edges]
        objective_value = self.calculate_f(edges_set)
        return (edges_set, objective_value)

    # Top k edges with highest items crossing them
    def highest_item_edges(self, k):
        item_nodes = {}
        for node in self.nodes_list:
            item_nodes[node] = self.nodes_list[node].num_objects
        E = self.G.edges()
        item_edges = {}
        for e in E:
            source = e[0]
            target = e[1]
            item_edges[e] = item_nodes[source]*self.G.get_edge_data(source, target)['weight']
        item_edges = sorted(item_edges, key=item_edges.get)
        sorted_edges = [x for x in reversed(item_edges)][:k]
        edges_set = [(x[0],x[1]) for x in sorted_edges]
        objective_value = self.calculate_f(edges_set)
        return (edges_set, objective_value)

    # Randomly chosen k edges
    def random_edges(self, k):
        all_edges = self.G.edges()
        edges_set = [all_edges[x] for x in np.random.choice(len(all_edges), k, replace=False)]
        objective_value = self.calculate_f(edges_set)
        return (edges_set, objective_value)

    # Get evolution of objective with increasing k
    def get_evolution(self, method, k):
        dataframe = []
        if method == self.smart_greedy:
            greedy_edges_set = method(k)[0]
            for i in xrange(k+1):
                row = {}
                row['k'] = i
                # row['edges_set'] = greedy_edges_set[:i]
                row['objective_value'] = self.calculate_f(greedy_edges_set[:i])
                dataframe.append(row)
        else:
            for i in xrange(k+1):
                row = {}
                result = method(i)
                row['k'] = i
                # row['edges_set'] = result[0]
                row['objective_value'] = result[1]
                dataframe.append(row)
        return dataframe
