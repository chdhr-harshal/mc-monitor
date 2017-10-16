#!/usr/local/bin/python
"""
Node objective functions

Algorithms implemented:
    1. naive_greedy_parallel - In each iteration pick 1 node greedily.
    2. naive_greedy_heuristic - Pick all k nodes at once greedily.
    3. smart_greedy_parallel - In each iteration pick 1 node smart greedily.
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
import argparse

np.seterr(all="raise")
cores = cpu_count()

# ------------------------------------------------------------
# Naive Greedy algorithm
# ------------------------------------------------------------

def calculate_F(S):
    F = np.float128(0.0)
    V_minus_S = set(mc.G.nodes()) - set(S)
    predecessors = []
    for i in V_minus_S:
        predecessors += mc.G.predecessors(i)
    predecessors = set(predecessors)

    for u in predecessors:
        F_u = np.float128(0.0)
        successors = mc.G[u]

        # Calculate rho
        rho = np.float128(0.0)
        for v in successors:
            if v in S:
                rho += successors[v]['weight']

        # Calculate F_u
        x_dash = mc.G.node[u]['num_items'] * ( 1 - rho)

        for v in successors:
            if v not in S:
                P_dash = mc.G.edge[u][v]['weight'] / (1 - rho)
                F_u += P_dash * (1 - P_dash)

        F_u =  x_dash * F_u
        if np.abs(F_u) < 1e-5:
            F += 0
        else:
            F += F_u

    return (S, F)

def naive_greedy_parallel(k):
    pool = Pool(cores)
    picked_set = []
    for i in xrange(k):
        candidate_nodes = set(mc.G.nodes()) - set(picked_set)
        candidate_sets = [picked_set + [v] for v in candidate_nodes]

        objective_values = pool.imap(calculate_F, candidate_sets)

        objective_values = sorted(objective_values, key=lambda x: x[1])
        picked_set = objective_values[0][0]

    pool.close()
    pool.join()

    return calculate_F(picked_set)

def naive_greedy_heuristic(k):
    pool = Pool(cores)
    picked_set = []

    candidate_nodes = [[x] for x in set(mc.G.nodes()) - set(picked_set)]

    objective_values = pool.imap(calculate_F, candidate_nodes)
    objective_values = sorted(objective_values, key=lambda x: x[1])

    picked_set = [x[0][0] for x in objective_values[:k]]

    pool.close()
    pool.join()

    return calculate_F(picked_set)

# ------------------------------------------------------------
# Smart Greedy algorithm
# ------------------------------------------------------------

def calculate_smart_F(args):
    v = args[0]
    rho_dict = args[1]
    B_dict = args[2]
    picked_set = list(args[3]) + list([v])

    F = np.float128(0.0)
    rho_dash_dict = {}
    B_dash_dict = {}
    predecessors = mc.G.predecessors(v)

    for u in mc.G.nodes():
        x = mc.G.node[u]['num_items']

        if u in predecessors:
            P = mc.G.edge[u][v]['weight']
        else:
            P = 0

        successors = mc.G.successors(u)
        if set(successors) - set(picked_set) == set():
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

    return (v, F, rho_dash_dict, B_dash_dict)

def smart_greedy_parallel(k):
    pool = Pool(cores)
    picked_set = []

    rho_dict = {}
    B_dict = {}

    for u in mc.G.nodes():
        rho_dict[u] = 0.0
        B_dict[u] = 0.0

        successors = mc.G[u]
        for v in successors:
            P = mc.G.edge[u][v]['weight']
            B_dict[u] += P * (1 - P)

    while len(picked_set) < k:
        candidate_nodes =  set(mc.G.nodes()) - set(picked_set)
        args = [(candidate_node, rho_dict, B_dict, picked_set) for candidate_node in candidate_nodes]

        objective_values = pool.imap(calculate_smart_F, args)
        objective_values = sorted(objective_values, key=lambda x: x[1])

        picked_node = objective_values[0][0]
        rho_dict = objective_values[0][2]
        B_dict = objective_values[0][3]
        picked_set.append(picked_node)

    pool.close()
    pool.join()

    return calculate_F(picked_set)

# Brute force
def brute_force_nodes(k):
    pool = Pool(cores)
    candidate_sets = combinations(mc.G.nodes(), k)

    objective_values = pool.imap(calculate_F, candidate_sets)
    objective_values = sorted(objective_values, key=lambda x: x[1])

    pool.close()
    pool.join()

    return objective_values[0]

# Other baselines

# Top k nodes with highest betweenness centrality
def highest_betweenness_centrality_nodes(k, betweenness_centrality):
    sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with highest incoming probabibility
def highest_in_probability_nodes(k):
    incoming_probability = mc.G.in_degree(weight='weight')
    sorted_nodes = sorted(incoming_probability, key=incoming_probability.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with incoming edges from highest number of other nodes
def highest_in_degree_centrality_nodes(k, in_deg_centrality):
    sorted_nodes = sorted(in_deg_centrality, key=in_deg_centrality.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with highest closeness centrality
def highest_closeness_centrality_nodes(k, closeness_centrality):
    sorted_nodes = sorted(closeness_centrality, key=closeness_centrality.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with higest items
def highest_item_nodes(k):
    item_nodes = nx.get_node_attributes(mc.G, 'num_items')
    sorted_nodes = sorted(item_nodes, key=item_nodes.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k pagerank nodes
def highest_pagerank_nodes(k, pagerank):
    sorted_nodes = sorted(pagerank, key=pagerank.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Random k nodes
def random_nodes(k):
    random_nodes = np.random.choice(mc.num_nodes, k, replace=False)
    nodes_set = [x for x in random_nodes]
    return calculate_F(nodes_set)

# ------------------------------------------------------------
# Evolution with k
# ------------------------------------------------------------

# Get evolution of objective with increasing k
def get_evolution(method, k):
    dataframe = []
    if method == smart_greedy_parallel or method == naive_greedy_heuristic:
        nodes_set = method(k)[0]
        for i in xrange(k):
            row = {}
            row['objective'] = "nodes"
            row['k'] = i
            row['objective_value'] = calculate_F(nodes_set[:i])[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    elif method == highest_closeness_centrality_nodes:
        closeness_centrality = nx.closeness_centrality(mc.G)
        for i in xrange(k):
            row = {}
            row['objective'] = "nodes"
            row['k'] = i
            row['objective_value'] = method(i, closeness_centrality)[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    elif method == highest_in_degree_centrality_nodes:
        in_deg_centrality = nx.in_degree_centrality(mc.G)
        for i in xrange(k):
            row = {}
            row['objective'] = "nodes"
            row['k'] = i
            row['objective_value'] = method(i, in_deg_centrality)[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    elif method == highest_pagerank_nodes:
        pagerank = nx.pagerank(mc.G, tol=1e-02)
        for i in xrange(k):
            row = {}
            row['objective'] = "nodes"
            row['k'] = i
            row['objective_value'] = method(i, pagerank)[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    elif method == highest_betweenness_centrality_nodes:
        if mc.num_nodes > 1000:
            pivots = 1000
        else:
            pivots = mc.num_nodes
        betweenness_centrality = nx.betweenness_centrality(mc.G, k=pivots)
        for i in xrange(k):
            row = {}
            row['objective'] = "nodes"
            row['k'] = i
            row['objective_value'] = method(i, betweenness_centrality)[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    else:
        for i in xrange(k):
            row = {}
            row['objective'] = "nodes"
            result = method(i)
            row['k'] = i
            row['objective_value'] = result[1]
            row['method_name'] = method.func_name
            row['item_distribution'] = mc.item_distribution
            dataframe.append(row)
    return dataframe
