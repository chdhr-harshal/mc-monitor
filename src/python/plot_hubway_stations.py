#!/usr/local/bin/python

import pandas as pd
from datetime import timedelta
import numpy as np
import networkx as nx
from MarkovChain import *
import os

DATA_DIR = "~/Projects/markov_traffic/data/"
PLOTS_DIR = "~/Projects/markov_traffic/Plots_data/"

def read_trips_file(filename):
    filename = DATA_DIR + filename
    parser = lambda date: pd.datetime.strptime(date, "%m/%d/%Y %H:%S:%f")
    data = pd.read_csv(filename, sep=",", header=0, index_col=0)
    data['start_date'] = pd.to_datetime(data['start_date'], format="%m/%d/%Y %H:%S:%f")
    data['end_date'] = pd.to_datetime(data['end_date'], format="%m/%d/%Y %H:%S:%f")
    data = data[['start_date','strt_statn','end_date','end_statn']]
    data = data.dropna(axis=0, how='any')
    cols=['strt_statn','end_statn']
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
    data['update'] = pd.to_datetime(data['update'], format="%Y-%m-%d %H:%S:%f")
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

    # Remove trips starting or ending in the stations not present in stations dataframe
    # or stations not present in the status file

    station_ids = set(stations['id'])
    status_df = status_df[status_df['station_id'].isin(station_ids)]


    df = df[(df['strt_statn'].isin(station_ids)) & (df['end_statn'].isin(station_ids))]
    trips_df = pd.DataFrame({'weight' : df.groupby(['strt_statn','end_statn']).size()})
    trips_df = trips_df.reset_index()

    print "Creating networkx graph"
    G = nx.from_pandas_dataframe(trips_df, 'strt_statn', 'end_statn', 'weight', create_using=nx.DiGraph())
    G = nx.stochastic_graph(G, weight='weight')

    # Add stations that are present in status_ids but not in trips_df
    status_ids = set(status['station_id'])
    for node in status_ids - set(G.nodes()):
        G.add_node(node)

    print "Creating transition matrix"
    transition_matrix = nx.to_numpy_matrix(G, weight='weight')
    transition_matrix = np.squeeze(np.asarray(transition_matrix))

    print "Creating object assignment and distribution"
    object_assignment = {}
    object_distribution = {}

    for node in G.nodes():
        try:
            object_assignment[node] = status_df[status_df['station_id'] == node].get('nbBikes').item()
        except:
            object_assignment[node] = 0

    num_objects = sum(object_assignment.values())

    for node in G.nodes():
        object_distribution[node] = 1.0 *object_assignment[node]/num_objects

    return (num_objects, transition_matrix, G, object_distribution)

def get_station_coordinate_dataframe(nodes_set):
    stations = read_stations_file("hubway_stations.csv")
    df = stations[stations['id'].isin(nodes_set)]
    return df

def get_paths_coordinate_dataframe(edges_set):
    nodes = []
    for edge in edges_set:
        nodes.append(edge[0])
        nodes.append(edge[1])
    nodes = set(nodes)

    stations = read_stations_file("hubway_stations.csv")
    stations_df = stations[stations['id'].isin(nodes)]

    df = []
    for i in xrange(len(edges_set)):
        edge = edges_set[i]
        source = edge[0]
        target = edge[1]

        temp = {}
        temp['edge_id'] = i
        temp['node'] = source
        temp['node_type'] = 'source'
        temp['lng'] = stations_df[stations_df['id'] == source].get('lng').item()
        temp['lat'] = stations_df[stations_df['id'] == source].get('lat').item()
        df.append(temp)

        temp = {}
        temp['edge_id'] = i
        temp['node'] = target
        temp['node_type'] = 'target'
        temp['lng'] = stations_df[stations_df['id'] == target].get('lng').item()
        temp['lat'] = stations_df[stations_df['id'] == target].get('lat').item()
        df.append(temp)

    df = pd.DataFrame(df)

    return df

if __name__ == "__main__":
    k = 10

    num_objects, transition_matrix, G, object_distribution = get_mc_attributes(duration=600)
    nc = MCNodeObjectives(len(G.nodes()), num_objects, 10, transition_matrix, object_distribution, G)
    ec = MCEdgeObjectives(len(G.nodes()), num_objects, 10, transition_matrix, object_distribution, G)

    nodes_set = nc.smart_greedy(k)[0]
    edges_set = ec.smart_greedy(k)[0]

    nodes_df = get_station_coordinate_dataframe(nodes_set)
    edges_df = get_paths_coordinate_dataframe(edges_set)

    # Export nodes_df
    nodes_df.to_csv(PLOTS_DIR + "hubway_plot_nodes.csv.gz", sep=",", header=True, index=False, compression='gzip')
    edges_df.to_csv(PLOTS_DIR + "hubway_plot_edges.csv.gz", sep=",", header=True, index=False, compression='gzip')

    # Call the R script to make the plots
    os.system("~/Projects/markov_traffic/src/R/plot_hubway_stations.r")
