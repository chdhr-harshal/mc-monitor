#!/usr/local/bin/python

"""
Edge objective functions
"""

from __future__ import division
from markov_chain import MarkovChain
import copy
import random
import numpy as np
import networkx as nx
from multiprocessing import Pool
from itertools import combinations

np.seterr(all="raise")


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

        F += x_dash * F_u

    return (D, F)

def naive_greedy_parallel(k):
    pool = Pool(24)
    # print "Created pool of 24 processes"
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

# ------------------------------------------------------------
# Smart Greedy algorithm
# ------------------------------------------------------------

def calculate_smart_F(args):
    e = args[0]
    rho_dict = args[1]
    B_dict = args[2]

    F = np.float128(0.0)
    rho_dash_dict = {}
    B_dash_dict = {}

    for u in mc.G.nodes():
        x = mc.G.node[u]['num_items']

        if u == e[0]:
            P = mc.G.edge[u][e[1]]['weight']
        else:
            P = 0

        rho_dash_dict[u] = rho_dict[u] + P
        B_dash_dict[u] = B_dict[u] - (2 * P * (1 - rho_dict[u] - P))

        if rho_dash_dict[u] < 1:
            F += x * B_dash_dict[u] / (1 - rho_dash_dict[u])
        else:
            F += 0

    return (e, F, rho_dash_dict, B_dash_dict)

def smart_greedy_parallel(k):
    pool = Pool(24)
    # print "Created pool of 24 processes"
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
        args = [(candidate_edge, rho_dict, B_dict) for candidate_edge in candidate_edges]

        objective_values = pool.imap(calculate_smart_F, args)
        objective_values = sorted(objective_values, key=lambda x: x[1])

        picked_edge = objective_values[0][0]
        rho_dict = objective_values[0][2]
        B_dict = objective_values[0][3]
        picked_set.append(picked_edge)

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
    pool = Pool(24)
    # print "Created pool of 24 processes"
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
def highest_probabibility_edges(k):
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

if __name__ == "__main__":
    n = np.random.randint(10, 20)
    num_items = np.random.randint(n, n*100)
    distribution = np.random.choice(['direct', 'inverse', 'uniform'])
    mc = MarkovChain(n, num_items, distribution)
    k = np.random.randint(1, len(mc.G.edges()))
    print "n={}, k={}, num_items={}, distribution={}".format(n,k,num_items,distribution)

    print "Naive Greedy: {}".format(naive_greedy_parallel(3))
    print "Smart Greedy: {}".format(smart_greedy_parallel(3))
    print "Dynamic programming: {}".format(dp_algorithm(3))
    print "Brute force: {}".format(brute_force_edges(3))
    # print "Betweenness centrality: {}".format(highest_betweenness_centrality_edges(3))
    # print "Probability: {}".format(highest_probabibility_edges(3))
    # print "Items: {}".format(highest_item_edges(3))
    # print "Random: {}".format(random_edges(3))
