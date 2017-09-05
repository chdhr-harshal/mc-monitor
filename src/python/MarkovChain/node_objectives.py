#!/usr/local/bin/python
"""
Node objective functions
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

        F += x_dash * F_u

    return (S, F)

def naive_greedy_parallel(k):
    pool = Pool(24)
    # print "Created pool of 24 processes"
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

def calculate_smart_F(args):
    v = args[0]
    rho_dict = args[1]
    B_dict = args[2]

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

        rho_dash_dict[u] = rho_dict[u] + P
        B_dash_dict[u] = B_dict[u] - (2 * P * (1 - rho_dict[u] - P))

        if rho_dash_dict[u] < 1:
            F += x * B_dash_dict[u] / (1 - rho_dash_dict[u])
        else:
            F += 0

    return (v, F, rho_dash_dict, B_dash_dict)

def smart_greedy_parallel(k):
    pool = Pool(24)
    # print "Created pool of 24 processes"
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
        args = [(candidate_node, rho_dict, B_dict) for candidate_node in candidate_nodes]

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
    pool = Pool(24)
    # print "Created pool of 24 processes"
    candidate_sets = combinations(mc.G.nodes(), k)

    objective_values = pool.imap(calculate_F, candidate_sets)
    objective_values = sorted(objective_values, key=lambda x: x[1])

    pool.close()
    pool.join()

    return objective_values[0]

# Other baselines

# Top k nodes with highest betweenness centrality
def highest_betweenness_centrality_nodes(k):
    if mc.num_nodes > 1000:
        pivots = 1000
    else:
        pivots = mc.num_nodes

    betweenness_centrality = nx.betweenness_centrality(mc.G, k=pivots)
    sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with highest incoming probabibility
def highest_in_probabibility_nodes(k):
    incoming_probability = mc.G.in_degree(weight='weight')
    sorted_nodes = sorted(incoming_probability, key=incoming_probability.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with incoming edges from highest number of other nodes
def highest_in_degree_centrality_nodes(k):
    in_deg_centrality = nx.in_degree_centrality(mc.G)
    sorted_nodes = sorted(in_deg_centrality, key=in_deg_centrality.get)
    sorted_nodes = [x for x in reversed(sorted_nodes)]
    nodes_set = sorted_nodes[:k]
    return calculate_F(nodes_set)

# Top k nodes with highest closeness centrality
def highest_closeness_centrality_nodes(k):
    closeness_centrality = nx.closeness_centrality(mc.G)
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

# Random k nodes
def random_nodes(k):
    random_nodes = np.random.choice(mc.num_nodes, k, replace=False)
    nodes_set = [x for x in random_nodes]
    return calculate_F(nodes_set)

if __name__ == "__main__":
    n = np.random.randint(10, 20)
    k = np.random.randint(1, n)
    num_items = np.random.randint(n, n*100)
    distribution = np.random.choice(['direct','inverse','uniform'])
    print "n={}, k={}, num_items={}, distribution={}".format(n,k,num_items, distribution)
    mc = MarkovChain(n,num_items, distribution)


    # Test different baselines
    print "Brute force: {}".format(brute_force_nodes(k))
    print "Naive greedy: {}".format(naive_greedy_parallel(k))
    print "Smart greedy: {}".format(smart_greedy_parallel(k))
    # print "Betweenness centrality: {}".format(highest_betweenness_centrality_nodes(k))
    # print "Incoming probability: {}".format(highest_in_probabibility_nodes(k))
    # print "In-degree: {}".format(highest_in_degree_centrality_nodes(k))
    # print "Closeness centrality: {}".format(highest_closeness_centrality_nodes(k))
    # print "Items: {}".format(highest_item_nodes(k))
    # print "Random: {}".format(random_nodes(k))
