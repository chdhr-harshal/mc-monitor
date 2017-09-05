#!/usr/local/bin/python

import pandas as pd
import numpy as np
import networkx as nx
from MarkovChain import *
import os

DATA_DIR = "/Users/harshal/Projects/markov_traffic/data/ass/"
PLOTS_DATA_DIR = "/Users/harshal/Projects/markov_traffic/Plots_data/"

dataframe_rows = []

def read_txt_file(filename):
    filename = DATA_DIR + filename
    data = pd.read_csv(filename, sep="\t", header=0)
    ass = nx.from_pandas_dataframe(data, 'source', 'target')
    return ass

def get_mc_attributes(G):
    G = G.to_directed()
    G = nx.stochastic_graph(G)
    tm = nx.to_numpy_matrix(G)
    tm = np.squeeze(np.asarray(tm))

    return (G, tm)

def get_objective_evolution(mc, problem, object_distribution, method, k, method_name):
    rows = mc.get_evolution(method, k)
    for row in rows:
        row['problem'] = problem
        row['object_distribution'] = object_distribution
        row['time'] = 0
        row['method'] = method_name

    global dataframe_rows
    dataframe_rows = dataframe_rows + rows


if __name__ == "__main__":
    filename = raw_input("Enter filename: ")
    G = read_txt_file(filename)
    G, tm = get_mc_attributes(G)
    num_nodes = len(G.nodes())
    num_objects = 100000
    total_time = 0
    max_k = 50

    # Uniform object distribution
    print "uniform"
    mc1 = MCNodeObjectives(num_nodes, num_objects, total_time, tm, 'uniform', G)
    get_objective_evolution(mc1, 'node', 'uniform', mc1.smart_greedy, max_k, 'greedy')

    get_objective_evolution(mc1, 'node', 'uniform', mc1.random_nodes, max_k, 'random')

    get_objective_evolution(mc1, 'node', 'uniform', mc1.highest_in_probability_nodes, max_k, 'highest_in_probability')

    get_objective_evolution(mc1, 'node', 'uniform', mc1.highest_in_degree_centrality_nodes, max_k, 'highest_in_degree_centrality')

    get_objective_evolution(mc1, 'node', 'uniform', mc1.highest_closeness_centrality_nodes, max_k, 'highest_closeness_centrality')

    get_objective_evolution(mc1, 'node', 'uniform', mc1.highest_item_nodes, max_k, 'highest_item_nodes')

    # Directly proportional to out-degree object distribution
    print "direct"
    mc1 = MCNodeObjectives(num_nodes, num_objects, total_time, tm, 'direct', G)
    get_objective_evolution(mc1, 'node', 'direct', mc1.smart_greedy, max_k, 'greedy')

    get_objective_evolution(mc1, 'node', 'direct', mc1.random_nodes, max_k, 'random')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_in_probability_nodes, max_k, 'highest_in_probability')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_in_degree_centrality_nodes, max_k, 'highest_in_degree_centrality')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_closeness_centrality_nodes, max_k, 'highest_closeness_centrality')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_item_nodes, max_k, 'highest_item_nodes')

    # Directly proportional to out-degree object distribution
    print "direct"
    mc1 = MCNodeObjectives(num_nodes, num_objects, total_time, tm, 'direct', G)
    get_objective_evolution(mc1, 'node', 'direct', mc1.smart_greedy, max_k, 'greedy')

    get_objective_evolution(mc1, 'node', 'direct', mc1.random_nodes, max_k, 'random')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_in_probability_nodes, max_k, 'highest_in_probability')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_in_degree_centrality_nodes, max_k, 'highest_in_degree_centrality')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_closeness_centrality_nodes, max_k, 'highest_closeness_centrality')

    get_objective_evolution(mc1, 'node', 'direct', mc1.highest_item_nodes, max_k, 'highest_item_nodes')

    # Inversely proportionaly to out-degree object distribution
    print "inverse"
    mc1 = MCNodeObjectives(num_nodes, num_objects, total_time, tm, 'inverse', G)
    get_objective_evolution(mc1, 'node', 'inverse', mc1.smart_greedy, max_k, 'greedy')

    get_objective_evolution(mc1, 'node', 'inverse', mc1.random_nodes, max_k, 'random')

    get_objective_evolution(mc1, 'node', 'inverse', mc1.highest_in_probability_nodes, max_k, 'highest_in_probability')

    get_objective_evolution(mc1, 'node', 'inverse', mc1.highest_in_degree_centrality_nodes, max_k, 'highest_in_degree_centrality')

    get_objective_evolution(mc1, 'node', 'inverse', mc1.highest_closeness_centrality_nodes, max_k, 'highest_closeness_centrality')

    get_objective_evolution(mc1, 'node', 'inverse', mc1.highest_item_nodes, max_k, 'highest_item_nodes')

    # Create a dataframe
    df = pd.DataFrame(dataframe_rows)

    # Export data to a csv in DATA_DIR
    df.to_csv(PLOTS_DATA_DIR + "k_objective_evolution_{}.csv.gz".format(filename), sep=",", header=True, index=False, compression='gzip')

    # Call the R script to make the plots
    # os.system("/Users/harshal/Projects/markov_traffic/src/R/k_variance_plot.r")
