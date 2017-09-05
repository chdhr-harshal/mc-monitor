#!/usr/local/bin/python

"""
Stores the data for plotting the evolution of the baseline objective
function value over the time for different settings
"""
from MarkovChain import *
import pandas as pd
import os
import numpy as np
from hubway import *

DATA_DIR = "/Users/harshal/Projects/markov_traffic/Plots_data/"

dataframe_rows = []

def calculate_baseline_objective(mc, problem, object_distribution):
    while mc.time_left >= 0:
        row = {}
        row['problem'] = problem
        row['object_distribution'] = object_distribution
        row['time'] = mc.current_time
        row['objective_value'] = mc.calculate_f([])

        dataframe_rows.append(row)
        mc.simulate_time_step()

if __name__ == "__main__":
    num_nodes = 145
    num_objects = 1000
    total_time = 20
    transition_matrix = get_transition_matrix("2011-07-28 10:00:00", 600)

    # Create a markov chain with random transition matrix
    # mc = MarkovChain(num_nodes, num_objects, total_time)
    # transition_matrix = mc.transition_matrix

    # transition_matrix = np.array([[0, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ],
    #     [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
    #     [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0],
    #     [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
    #     [0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0],
    #     [0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
    #     [0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
    #     [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    #     [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
    #     [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     ])

    # Node objective evolution

    # 1. Uniform object distribution
    mc1 = MCNodeObjectives(num_nodes, num_objects, total_time, transition_matrix, 'uniform')
    calculate_baseline_objective(mc1, 'node', 'uniform')

    # 2. Directly proportional to out-degree object distribution
    mc2 = MCNodeObjectives(num_nodes, num_objects, total_time, transition_matrix, 'direct')
    calculate_baseline_objective(mc2, 'node', 'direct')

    # 3. Inversely proportional to out-degree object distribution
    mc3 = MCNodeObjectives(num_nodes, num_objects, total_time, transition_matrix, 'inverse')
    calculate_baseline_objective(mc3, 'node', 'inverse')

    # Edge objective evolution

    # 1. Uniform object distribution
    mc1 = MCEdgeObjectives(num_nodes, num_objects, total_time, transition_matrix, 'uniform')
    calculate_baseline_objective(mc1, 'edge', 'uniform')

    # 2. Directly proportional to out-degree object distribution
    mc2 = MCEdgeObjectives(num_nodes, num_objects, total_time, transition_matrix, 'direct')
    calculate_baseline_objective(mc2, 'edge', 'direct')

    # 3. Inversely proportional to out-degree object distribution
    mc3 = MCEdgeObjectives(num_nodes, num_objects, total_time, transition_matrix, 'inverse')
    calculate_baseline_objective(mc3, 'edge', 'inverse')

    # Create a dataframe
    df = pd.DataFrame(dataframe_rows)

    # Export the data to a csv in DATA_DIR
    df.to_csv(DATA_DIR + "baseline_objective_evolution.csv.gz", sep=',', header=True, index=False, compression='gzip')

    # Call the R script to make the plots
    os.system("~/Projects/markov_traffic/src/R/time_variance_plot.r")
