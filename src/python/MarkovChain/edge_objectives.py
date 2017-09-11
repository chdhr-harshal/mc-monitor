#!/usr/local/bin/python

"""
Edge objective functions

Algorithms implemented:
    1. naive_greedy_parallel - In each iteration pick 1 edge greedily.
    2. naive_greedy_heuristic - Pick all k edges at once greedily.
    3. smart_greedy_parallel - In each iteration pick 1 edge smart greedily.
    4. smart_greedy_heuristic - In each iteration pick "edges_per_step" number of edge smart greedily.
    5. dp_algorithm - Dynamic programming algorithm to pick k edges.
"""

from __future__ import division
from markov_chain import MarkovChain
import copy
import random
import numpy as np
import networkx as nx
from multiprocessing import Pool
from multiprocessing import cpu_count
from itertools import combinations

np.seterr(all="raise")
cores = cpu_count()

# ------------------------------------------------------------
# Naive Greedy algorithm
# ------------------------------------------------------------

def calculate_F(D):
    F = np.float128(0.0)

    for u in mc.G.nodes():
        F_u = np.float128(0.0)
        successors = mc.G[u]

        # Calculate rho
        rho = np.float128(0.0)
        for v in successors:
            if (u,v) in D:
                rho += successors[v]['weight']

        # Calcualte F_u
        x_dash = mc.G.node[u]['num_items'] * (1 - rho)

        for v in successors:
            if (u,v) not in D:
                P_dash = successors[v]['weight'] / (1 - rho)
                F_u += P_dash * (1 - P_dash)

        F_u = x_dash * F_u

        if np.abs(F_u) < 1e-5:
            F += 0
        else:
            F += F_u

    return (D, F)

def naive_greedy_parallel(k):
    pool = Pool(cores)
    picked_set = []

    for i in xrange(k):
        candidate_edges = set(mc.G.edges()) - set(picked_set)
        candidate_sets = [picked_set + [e] for e in candidate_edges]

        objective_values = pool.imap(calculate_F, candidate_sets)
        objective_values = sorted(objective_values, key=lambda x: x[1])
        picked_set = objective_values[0][0]

    pool.close()
    pool.join()

    return calculate_F(picked_set)

def naive_greedy_heuristic(k):
    pool = Pool(cores)
    picked_set = []

    candidate_edges = [[x] for x in set(mc.G.edges()) - set(picked_set)]

    objective_values = pool.imap(calculate_F, candidate_edges)
    objective_values = sorted(objective_values, key=lambda x: x[1])

    picked_set = [x[0][0] for x in objective_values[:k]]

    pool.close()
    pool.join()

    return calculate_F(picked_set)

# ------------------------------------------------------------
# Smart Greedy algorithm
# ------------------------------------------------------------

def calculate_smart_F(args):
    e = args[0]
    rho_dict = args[1]
    B_dict = args[2]
    picked_set = list(args[3]) + list([e])

    F = np.float128(0.0)
    rho_dash_dict = {}
    B_dash_dict = {}

    for u in mc.G.nodes():
        x = mc.G.node[u]['num_items']

        if u == e[0]:
            P = mc.G.edge[u][e[1]]['weight']
        else:
            P = 0

        successor_edges = [(u,v) for v in mc.G.successors(u)]
        if set(successor_edges) - set(picked_set) == set():
            rho_dash_dict[u] = 1
            B_dash_dict[u] = 0
            continue

        rho_dash_dict[u] = rho_dict[u] + P
        B_dash_dict[u] = B_dict[u] - (2 * P * (1 - rho_dict[u] - P))

        if rho_dash_dict[u] < 1:
            F_u = x * B_dash_dict[u] / (1 - rho_dash_dict[u])
            if np.abs(F_u) < 1e-5:
                F += 0
            else:
                F += F_u
        else:
            F += 0


    return (e, F, rho_dash_dict, B_dash_dict)

def smart_greedy_parallel(k):
    pool = Pool(cores)
    picked_set = []

    rho_dict = {}
    B_dict = {}

    for u in mc.G.nodes():
        rho_dict[u] = np.float128(0.0)
        B_dict[u] = np.float128(0.0)

        successors = mc.G[u]
        for v in successors:
            P = successors[v]['weight']
            B_dict[u] += P * (1 - P)

    while len(picked_set) < k:
        candidate_edges = set(mc.G.edges()) - set(picked_set)
        args = [(candidate_edge, rho_dict, B_dict, picked_set) for candidate_edge in candidate_edges]

        objective_values = pool.imap(calculate_smart_F, args)
        objective_values = sorted(objective_values, key=lambda x: x[1])

        picked_edge = objective_values[0][0]
        rho_dict = objective_values[0][2]
        B_dict = objective_values[0][3]
        picked_set.append(picked_edge)

    pool.close()
    pool.join()

    return calculate_F(picked_set)

def smart_greedy_heuristic(k, edges_per_step):
    pool = Pool(cores)
    picked_set = []

    rho_dict = {}
    B_dict = {}

    for u in mc.G.nodes():
        rho_dict[u] = np.float128(0.0)
        B_dict[u] = np.float128(0.0)

        successors = mc.G[u]
        for v in successors:
            P = successors[v]['weight']
            B_dict[u] += P * (1-P)

    while len(picked_set) < k:
        candidate_edges = set(mc.G.edges()) - set(picked_set)
        args = [(candidate_edge, rho_dict, B_dict, picked_set) for candidate_edge in candidate_edges]

        objective_values = pool.imap(calculate_smart_F, args)
        objective_values = sorted(objective_values, key=lambda x: x[1])

        for i in xrange(edges_per_step):
            picked_edge = objective_values[i][0]
            picked_set.append(picked_edge)

        # Update rho dict and B_dict
        for u in mc.G.nodes():
            rho_dict[u] = np.float128(0.0)
            B_dict[u] = np.float128(0.0)
            successor_edges = [(u,v) for v in mc.G[u]]
            for (u,v) in successor_edges:
                if (u,v) in picked_set:
                    rho_dict[u] += mc.G[u][v]['weight']

            for (u,v) in successor_edges:
                if (u,v) not in picked_set:
                    P = mc.G[u][v]['weight']
                    B_dict[u] += P * ( 1 - rho_dict[u] - P)

    pool.close()
    pool.join()

    return calculate_F(picked_set)

# ------------------------------------------------------------
# Dynamic programming algorithm
# ------------------------------------------------------------

def calculate_ISOL(i, k):
    E_i = mc.G.edges(i)
    candidate_sets = []
    candidate_sets.append([])

    for j in range(1, k+1):
        for x in combinations(E_i, j):
            candidate_sets.append(list(x))

    objective_values = {}
    for j in xrange(len(candidate_sets)):
        F_i = np.float128(0.0)
        successors = mc.G[i]
        # Calculate rho
        rho = np.float128(0.0)
        for v in successors:
            if (i,v) in candidate_sets[j]:
                rho += successors[v]['weight']
        # Calculate F_i
        x_dash = mc.G.node[i]['num_items'] * (1 - rho)
        for v in successors:
            if (i,v) not in candidate_sets[j]:
                P_dash = successors[v]['weight'] / (1 - rho)
                F_i += P_dash * (1 - P_dash)
        F_i = x_dash * F_i
        objective_values[j] = F_i


    min_index = min(objective_values, key=objective_values.get)
    ISOL_edge_set = candidate_sets[min_index]
    ISOL_value = objective_values[min_index]
    return (ISOL_edge_set, ISOL_value)

def calculate_SOL(i, k, SOL_matrix, SOL_edge_set_matrix):
    if k < 0:
        return ([], np.infty)
    if i >= mc.num_nodes:
        return ([], 0)

    if not np.isnan(SOL_matrix[i][k]):
        return (SOL_edge_set_matrix[i][k], SOL_matrix[i][k])

    objective_values = {}
    candidate_sets = [0 for x in xrange(k+1)]
    for k_i in xrange(k+1):
        ISOL = calculate_ISOL(i, k_i)
        ISOL_value = ISOL[1]
        ISOL_edge_set = ISOL[0]
        SOL = calculate_SOL(i+1, k - k_i, SOL_matrix, SOL_edge_set_matrix)
        SOL_value = SOL[1]
        SOL_edge_set = SOL[0]
        candidate_sets[k_i] = ISOL_edge_set + SOL_edge_set
        objective_values[k_i] = ISOL_value + SOL_value

    min_index = min(objective_values, key=objective_values.get)
    SOL_value = objective_values[min_index]
    SOL_edge_set = candidate_sets[min_index]
    return (SOL_edge_set, SOL_value)

def dp_algorithm(k):
    SOL_matrix = np.full((mc.num_nodes, k+1), np.nan, dtype=float)
    SOL_edge_set_matrix = np.zeros((mc.num_nodes, k+1), dtype=np.ndarray)

    for i in reversed(xrange(mc.num_nodes)):
        for k_prime in xrange(k+1):
            SOL = calculate_SOL(i, k_prime, SOL_matrix, SOL_edge_set_matrix)
            SOL_matrix[i][k_prime] = SOL[1]
            SOL_edge_set_matrix[i][k_prime] = SOL[0]

    return (SOL_edge_set_matrix[0][k], SOL_matrix[0][k])

# Brute force
def brute_force_edges(k):
    pool = Pool(cores)
    candidate_sets = combinations(mc.G.edges(), k)

    objective_values = pool.imap(calculate_F, candidate_sets)
    objective_values = sorted(objective_values, key=lambda x: x[1])

    pool.close()
    pool.join()

    return objective_values[0]

# Other baselines

# Top k edges with highest betweenness centrality
def highest_betweenness_centrality_edges(k):
    betweenness_centrality = nx.edge_betweenness_centrality(mc.G, weight='weight')
    sorted_edges = sorted(betweenness_centrality, key=betweenness_centrality.get)
    sorted_edges = [x for x in reversed(sorted_edges)][:k]
    edges_set = [(x[0],x[1]) for x in sorted_edges]
    return calculate_F(edges_set)

# Top k edges with highest probability
def highest_probability_edges(k):
    sorted_edges = sorted(mc.G.edges(data=True), key=lambda(source, target, data): data)
    sorted_edges = [x for x in reversed(sorted_edges)][:k]
    edges_set = [(x[0],x[1]) for x in sorted_edges]
    return calculate_F(edges_set)

# Top k edges with highest items crossing
def highest_item_edges(k):
    item_nodes = nx.get_node_attributes(mc.G, 'num_items')
    E = mc.G.edges()
    item_edges = {}
    for e in E:
        source = e[0]
        target = e[1]
        item_edges[e] = item_nodes[source]*mc.G[source][target]['weight']
    item_edges = sorted(item_edges, key=item_edges.get)
    sorted_edges = [x for x in reversed(item_edges)][:k]
    edges_set = [(x[0],x[1]) for x in sorted_edges]
    return calculate_F(edges_set)

# Random k edges
def random_edges(k):
    all_edges = mc.G.edges()
    edges_set = [all_edges[x] for x in np.random.choice(len(all_edges), k, replace=False)]
    return calculate_F(edges_set)

# ------------------------------------------------------------
# Evolution with k
# ------------------------------------------------------------

# Get evolution of objective with increasing k
def get_evolution(method, k, edges_per_step=10):
    dataframe = []
    if method == smart_greedy_parallel:
        edges_set = method(k)[0]
        for i in xrange(k):
            row = {}
            row['objective'] = "edges"
            row['k'] = i
            row['objective_value'] = calculate_F(edges_set[:i])[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    elif method == smart_greedy_heuristic:
        edges_set = method(k, edges_per_step)[0]
        for i in range(0, k, edges_per_step):
            row = {}
            row['objective'] = "edges"
            row['k'] = i
            row['objective_value'] = calculate_F(edges_set[:i])[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    else:
        for i in xrange(k):
            row = {}
            row['objective'] = "edges"
            result = method(i)
            row['k'] = i
            row['objective_value'] = result[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    return dataframe
