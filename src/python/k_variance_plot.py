#!/usr/local/bin/python

"""
Stores the data for plotting the evolution of the objective function
value over increasing value of k over the time for different settings
"""
from MarkovChain import *
import pandas as pd
import os
from plot_hubway_stations import *

DATA_DIR = "/Users/harshal/Projects/markov_traffic/Plots_data/"

dataframe_rows = []

def get_objective_evolution(mc, problem, object_distribution,
                            method, k, method_name):
    rows = mc.get_evolution(method, k)
    for row in rows:
        row['problem'] = problem
        row['object_distribution'] = object_distribution
        row['time'] = mc.current_time
        row['method'] = method_name

    global dataframe_rows
    dataframe_rows = dataframe_rows + rows

if __name__ == "__main__":
    num_objects, transition_matrix, G, object_distribution = get_mc_attributes(duration=600)
    mc1 = MCNodeObjectives(len(G.nodes()), num_objects, 0, transition_matrix, object_distribution, G)
    max_k = 50
    # Node objective evolution

    # 1. Uniform object distribution
    get_objective_evolution(mc1, 'node', 'actual', mc1.smart_greedy, max_k, 'greedy')

    get_objective_evolution(mc1, 'node', 'actual', mc1.random_nodes, max_k, 'random')

    get_objective_evolution(mc1, 'node', 'actual',
                mc1.highest_in_probability_nodes, max_k, 'highest_in_probability')

    get_objective_evolution(mc1, 'node', 'actual',
                mc1.highest_in_degree_centrality_nodes, max_k, 'highest_in_degree_centrality')

    get_objective_evolution(mc1, 'node', 'actual',
                mc1.highest_closeness_centrality_nodes, max_k, 'highest_closeness_centrality')

    get_objective_evolution(mc1, 'node', 'actual',
                mc1.highest_betweenness_centrality_nodes, max_k, 'highest_betweenness_centrality')

    get_objective_evolution(mc1, 'node', 'actual',
                mc1.highest_closeness_centrality_nodes, max_k, 'highest_item_nodes')

    # Edge objective evolution
    mc1 = MCEdgeObjectives(len(G.nodes()), num_objects, 0, transition_matrix, object_distribution, G)

    # 1. Uniform object distribution
    get_objective_evolution(mc1, 'edge', 'actual', mc1.smart_greedy, max_k, 'greedy')

    get_objective_evolution(mc1, 'edge', 'actual', mc1.random_edges, max_k, 'random')

    get_objective_evolution(mc1, 'edge', 'actual', mc1.highest_probability_edges, max_k, 'high_probability')

    get_objective_evolution(mc1, 'edge', 'actual', mc1.highest_betweenness_centrality_edges, max_k, 'highest_betweenness_centrality')

    get_objective_evolution(mc1, 'edge', 'actual', mc1.dp_algorithm, max_k, 'dynamic_program')

    get_objective_evolution(mc1, 'edge', 'actual', mc1.highest_probability_edges, max_k, 'highest_item_edges')

    # Create a dataframe
    df = pd.DataFrame(dataframe_rows)

    # Export the data to a csv in DATA_DIR
    df.to_csv(DATA_DIR + "k_objective_evolution_hubway.csv.gz", sep=",", header=True, index=False, compression='gzip')

    # Call the R script to make the plots
    # os.system("~/Projects/markov_traffic/src/R/k_variance_plot.r")
