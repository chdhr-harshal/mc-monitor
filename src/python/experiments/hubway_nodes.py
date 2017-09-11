#!/usr/local/bin/python

import sys
sys.path.insert(1, '..')

import pandas as pd
from datetime import timedelta
import __builtin__
import numpy as np
import networkx as nx
from MarkovChain import *
from MarkovChain.node_objectives import *

DATA_DIR = "/home/grad3/harshal/Desktop/markov_traffic/data/"
PLOTS_DATA_DIR = "/home/grad3/harshal/Desktop/markov_traffic/Plots_data/"

dataframe_rows = []

def read_trips_file(filename):
    filename = DATA_DIR + filename
    parser = lambda date: pd.datetime.strptime(date, "%m/%d/%Y %H:%S:%f")
    data = pd.read_csv(filename, sep=",", header=0, index_col=0)
    data['start_date'] = pd.to_datetime(data['start_date'], format="%m/%d/%Y %H:%S:%f")
    data['end_date'] = pd.to_datetime(data['end_date'], format="%m/%d/%Y %H:%S:%f")
    data = data[['start_date', 'strt_statn', 'end_date', 'end_statn']]
    data = data.dropna(axis=0, how='any')
    cols = ['strt_statn', 'end_statn']
    data[cols] = data[cols].applymap(np.int64)
    return data

def read_stations_file(filename):
    filename = DATA_DIR + filename
    data = pd.read_csv(filename, sep=",", header=0)
    data = data[data['status'] == 'Existing']
    return data

def read_status_file(filename):
    filename = DATA_DIR + filename
    data = pd.read_csv(filename, sep=",", header=0, index_col=0)
    data['update'] = pd.to_datetime(data['update'], format="%Y-%m-%d %H:%M:%S")
    return data

def get_mc_attributes(start_time="2012-04-01 10:00:00", duration=120):
    # Create csv read iterator
    data = read_trips_file("hubway_trips_2012.csv")
    start_time = pd.to_datetime(start_time)
    end_time = start_time + timedelta(minutes=duration)

    df = data[(data['start_date'] >= start_time) & (data['end_date'] <= end_time)]
    stations = read_stations_file("hubway_stations.csv")
    status = read_status_file("stationstatus_2012_4.csv")
    status_df = status[status['update'] == start_time]

    # Remove trips starting or ending in the stations which are not present
    # in stations dataframe or stations not present in the status file

    station_ids = set(stations['id'])
    status_df = status_df[status_df['station_id'].isin(station_ids)]

    df = df[(df['strt_statn'].isin(station_ids)) & (df['end_statn'].isin(station_ids))]
    trips_df = pd.DataFrame({'weight' : df.groupby(['strt_statn', 'end_statn']).size()})
    trips_df = trips_df.reset_index()

    print "Creating networkx graph"
    G = nx.from_pandas_dataframe(trips_df, 'strt_statn', 'end_statn', 'weight', create_using=nx.DiGraph())
    G = nx.stochastic_graph(G, weight='weight')

    # Add stations that are present in status_ids but not in trips_df
    status_ids = set(status['station_id'])
    for node in status_ids - set(G.nodes()):
        G.add_node(node)

    return G

def get_objective_evolution(method, k):
    rows = get_evolution(method, k)
    global dataframe_rows
    dataframe_rows += rows
    df = pd.DataFrame(dataframe_rows)
    df.to_csv(PLOTS_DATA_DIR + "hubway_nodes_evolution.csv.gz", sep=",",
            header=True, index=False, compression="gzip")

if __name__ == "__main__":
    k = 10
    G = get_mc_attributes(duration=600)
    num_nodes = len(G)
    num_items = len(G)
    item_distribution = "direct"

    # Make mc global for imported modules
    __builtin__.mc = MarkovChain(num_nodes=num_nodes,
                                 num_items=num_items,
                                 item_distribution=item_distribution,
                                 G=G)

    print "Starting evaluation of methods"
    methods = [random_nodes,
               highest_item_nodes,
               highest_closeness_centrality_nodes,
               highest_in_degree_centrality_nodes,
               highest_in_probability_nodes,
               highest_betweenness_centrality_nodes,
               smart_greedy_parallel]

    for method in methods:
        print "Evaluating method {}".format(method.func_name)
        get_objective_evolution(method, k)
