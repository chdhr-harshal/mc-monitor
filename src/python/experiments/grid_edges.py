#!/usr/local/bin/python

"""
Grid graph nodes objective evolution with increasing k
"""

from __future__ import division
import sys
sys.path.insert(1, '..')
import __builtin__

import os
from MarkovChain import *
from MarkovChain.edge_objectives import *
import networkx as nx
import pandas as pd

PLOTS_DATA_DIR = "/home/grad3/harshal/Desktop/MCMonitor/Plots_data/"

dataframe_rows = []

def get_objective_evolution(method, k, iteration):
    rows = get_evolution(method, k)
    for row in rows:
        row['iteration'] = iteration
    global dataframe_rows
    dataframe_rows += rows
    df = pd.DataFrame(dataframe_rows)
    df.to_csv(PLOTS_DATA_DIR + "grid_edges_k_objective_evolution.csv.gz", sep=",",
            header=True, index=False, compression="gzip")

if __name__ == "__main__":
    G = nx.grid_2d_graph(100, 10, create_using=nx.DiGraph())
    G = nx.convert_node_labels_to_integers(G)
    G = nx.stochastic_graph(G, weight='weight')

    num_nodes = len(G)
    num_items = len(G)

    k = 50
    item_distributions = ['uniform', 'direct', 'inverse', 'ego']

    for item_distribution in item_distributions:
        if item_distribution == 'ego':
            iterations = 10
        else:
            iterations = 1
        for iteration in xrange(iterations):
            print "Evaluating item distribution {}".format(item_distribution)

            __builtin__.mc = MarkovChain(num_nodes=num_nodes,
                                         num_items=num_items,
                                         item_distribution=item_distribution,
                                         G=G)

            print "Starting evaluation of methods"
            methods = [random_edges,
                      highest_item_edges,
                      highest_probability_edges,
                      highest_betweenness_centrality_edges,
                      smart_greedy_parallel,
                      smart_greedy_heuristic]

            for method in methods:
                print "Evaluating method {}".format(method.func_name)
                get_objective_evolution(method, k, iteration)
