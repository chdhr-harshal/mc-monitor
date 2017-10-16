import sys
sys.path.insert(1, '..')
import __builtin__

import numpy as np
import pandas as pd
import networkx as nx
from MarkovChain import *
from MarkovChain.node_objectives import *

PLOTS_DATA_DIR = "/home/grad3/harshal/Desktop/MCMonitor/Plots_data/"

num_nodes = 1000
num_items = num_nodes
item_distributions = ['uniform', 'direct', 'inverse', 'ego']
T = 10

# Barabasi Albert graph
ba_G = nx.barabasi_albert_graph(1000, 5)
ba_G = ba_G.to_directed()
ba_G = nx.convert_node_labels_to_integers(ba_G)
ba_G = nx.stochastic_graph(ba_G, weight='weight')

# Geo graph
geo_G = nx.random_geometric_graph(1000, 0.01)
geo_G = geo_G.to_directed()
geo_G = nx.stochastic_graph(geo_G, weight='weight')

# Grid graph
grid_G = nx.grid_2d_graph(100, 10, create_using=nx.DiGraph())
grid_G = nx.convert_node_labels_to_integers(grid_G)
grid_G = nx.stochastic_graph(grid_G, weight='weight')

graphs = ['random', 'ba', 'geo', 'grid']

dataframe_rows = []

for graph in graphs:
    print "Graph {}".format(graph)
    if graph == 'random':
        G = None
    elif graph == 'ba':
        G = ba_G
    elif graph == 'geo':
        G = geo_G
    else:
        G = grid_G
    for item_distribution in item_distributions:
        print "Item distribution {}".format(item_distribution)
        if item_distribution == 'ego':
            iterations = 10
        else:
            iterations = 1
        for iteration in xrange(iterations):
            # Quick hack to make mc global to prevent copying to each process
            __builtin__.mc = MarkovChain(num_nodes=num_nodes,
                                        num_items=num_items,
                                        item_distribution=item_distribution,
                                        G=G)
            for t in xrange(T):
                print "t: {}".format(t)

                tmp_dict = {}
                tmp_dict['graph'] = graph
                tmp_dict['item_distribution'] = item_distribution
                tmp_dict['iteration'] = iteration
                tmp_dict['t'] = t
                tmp_dict['objective_value'] = calculate_F([])[1]

                dataframe_rows.append(tmp_dict)

                __builtin__.mc.simulate_time_step()

df = pd.DataFrame(dataframe_rows)
df.to_csv(PLOTS_DATA_DIR + "time_evolution.csv.gz", sep=",", header=True, index=False, compression="gzip")
