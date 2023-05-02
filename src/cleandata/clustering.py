import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math

import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore') # just hide warning of Kmean

''' =================================================================================================
funtion to cal proper number ofcluster by using silhouette score
- The silhouette score is a measure of how well each data point in a cluster is separated from other clusters.
  It is a metric that is commonly used to evaluate the quality of clustering results.

- a higher clustering coefficient generally indicates a more highly structured or organized graph, while a lower
  clustering coefficient indicates a more random or disorganized graph.
================================================================================================= '''
def visual_elbow_with_silhouette_score(df, feature_cols, n_cluster_from=2, n_cluster_to=30, random_state=42, plot=True):    
    # retrieve cols name
    cols_name = "".join(feature_cols)
    
    # Get the feature columns
    X = df[feature_cols].values

    # Initialize the KMeans algorithm with a range of cluster numbers
    scores = []
    for k in range(n_cluster_from, n_cluster_to):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        scores.append(silhouette_score(X, kmeans.labels_))

    if plot == True:
      # Plot the elbow curve
      plt.plot(range(n_cluster_from, n_cluster_to), scores, 'bx-')
      plt.xlabel('Number of clusters')
      plt.ylabel('Silhouette Score')
      plt.title(f'Elbow Curve: {cols_name}')
      
      # Add labels to each data point
      for i, score in enumerate(scores):
          plt.annotate(f'{i+n_cluster_from}', xy=(i+n_cluster_from, score), xytext=(5, 5), textcoords='offset points')

      plt.show()

    return np.argmax(scores)+n_cluster_from

''' =================================================================================================
function perform cluster
================================================================================================= '''
def perform_clustering(df, feature_cols, n_clusters=2, random_state=42, assign_by='value'):
    # retrieve cluster name
    prefix = "cluster"
    cluster_name = prefix + "_" + "_".join(feature_cols)
    
    # Get the feature columns
    X = df[feature_cols].values

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    # Add the cluster labels to the original DataFrame
    df[cluster_name] = kmeans.labels_

    # Assign cluster numbers based on number of members or feature values
    if assign_by == 'value':
        centroids = kmeans.cluster_centers_
        sorted_clusters = sorted(range(n_clusters), key=lambda i: centroids[i][0], reverse=True)
        cluster_map = dict(zip(sorted_clusters, range(n_clusters)))
        df[cluster_name] = df[cluster_name].map(cluster_map)
    elif assign_by == 'n_member':
        cluster_counts = df.groupby(cluster_name).size()
        cluster_order = cluster_counts.sort_values().index
        df[cluster_name] = df[cluster_name].replace(dict(zip(cluster_order, range(n_clusters))))

    return df

''' =================================================================================================
function perform cluster each state_num
================================================================================================= '''
def perform_clustering_by_state(df, feature_cols, n_clusters=8, random_state=42, assign_by='value'):
    # retrieve cluster name
    prefix = "cluster_state"
    cluster_name = prefix + "_" + "_".join(feature_cols)
    
    # Get the unique values of the state_num column
    state_values = df['state_num'].unique()

    # Create an empty DataFrame to store the clustered data
    clustered_df = pd.DataFrame()

    # Perform clustering for each unique value of the state_num column
    for state_value in state_values:
        # Get the subset of the DataFrame for the current state value
        state_df = df[df['state_num'] == state_value]

        # Get the feature columns for the current state
        X = state_df[feature_cols].values

        # Perform clustering for the current state
        if len(state_df) < n_clusters:
            n_clusters = len(state_df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X)

        # Add the cluster labels to the original DataFrame
        state_df[cluster_name] = kmeans.labels_

        # Assign cluster numbers based on number of members or feature values
        if assign_by == 'value':
            centroids = kmeans.cluster_centers_
            sorted_clusters = sorted(range(n_clusters), key=lambda i: centroids[i][0], reverse=True)
            cluster_map = dict(zip(sorted_clusters, range(n_clusters)))
            state_df[cluster_name] = state_df[cluster_name].map(cluster_map)
        elif assign_by == 'n_member':
            cluster_counts = state_df.groupby(cluster_name).size()
            cluster_order = cluster_counts.sort_values().index
            state_df[cluster_name] = state_df[cluster_name].replace(dict(zip(cluster_order, range(n_clusters))))

        # Add the clustered data for the current state to the overall DataFrame
        clustered_df = pd.concat([clustered_df, state_df])

    return clustered_df

''' =================================================================================================
function to plot node location color by cluster
================================================================================================= '''

def plot_node_by_cluster(df, high_light_cluster=None, high_light_node=None, base_color = 'Set2', col_name = 'cluster'):
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Load the map of China
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    china = world[world.name == 'China']

    # Create a new figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

    # Plot the map of China
    china.plot(ax=ax, color='white', edgecolor='black')

    # set non highlight
    if high_light_cluster is None and high_light_node is None:
        gray_points = ax.scatter(df['lon'], df['lat'], s=0.5, c=df[col_name], cmap=base_color, alpha=1)
    else:
        gray_points = ax.scatter(df['lon'], df['lat'], s=0.5, c=df[col_name], cmap=base_color, alpha=0.3)

    # Highlight the specified cluster in red, if provided
    if high_light_cluster is not None:
        red_points = ax.scatter(df[df[col_name] == high_light_cluster]['lon'], df[df[col_name] == high_light_cluster]['lat'], s=1, c='red')
    
    # Highlight the specified node in red, if provided
    if high_light_node is not None and high_light_node in df.index:
        ax.scatter(df.loc[high_light_node, 'lon'], df.loc[high_light_node, 'lat'], s=1, c='red')
    
    # Show the plot
    plt.show()

''' =================================================================================================
function to plot map and highlight specify edge
================================================================================================= '''

def plot_edge_by_cluster(df, df_edge, high_light_cluster=None, high_light_node=None, base_color='Set2', col_name=None, edge_feature=None, highlight_value=None, edge_color='red'):
    #sort high value to be on last (plat last = top)
    if edge_feature is not None:
        df_edge = df_edge.sort_values(by=edge_feature, ascending=True)
    
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Load the map of China
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    china = world[world.name == 'China']

    # Create a new figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

    # Plot the map of China
    china.plot(ax=ax, color='white', edgecolor='black')

    # Create a directed graph from the dataframe
    G = nx.from_pandas_edgelist(df_edge, source='st_id', target='st_tg', edge_attr=[edge_feature], create_using=nx.DiGraph())
    
    # Create a dictionary of node attributes from the node dataframe
    node_attrs = df.set_index('node').to_dict('index')

    # Add the node attributes to the graph
    nx.set_node_attributes(G, node_attrs)

    # set non highlight
    if high_light_cluster is None and high_light_node is None:
        if col_name is None:
            gray_points = ax.scatter(df['lon'], df['lat'], s=0.2, color='gray', alpha=1)
        else:
            gray_points = ax.scatter(df['lon'], df['lat'], s=0.2, c=df[col_name], cmap=base_color, alpha=1)
    else:
        gray_points = ax.scatter(df['lon'], df['lat'], s=0.2, c=df[col_name], cmap=base_color, alpha=0.3)


    # Highlight the specified cluster in red, if provided
    if high_light_cluster is not None:
        red_points = ax.scatter(df[df[col_name] == high_light_cluster]['lon'], df[df[col_name] == high_light_cluster]['lat'], s=0.2, c='red')
    
    # Highlight the specified node in red, if provided
    if high_light_node is not None and high_light_node in df.index:
        ax.scatter(df.loc[high_light_node, 'lon'], df.loc[high_light_node, 'lat'], s=0.2, c='red')
        
    # Highlight the specified edge attribute, if provided
    if edge_color is not None and edge_feature is not None:
        if highlight_value is not None:
            edge_color = (1.0, 0.0, 0.0, 1.0)  # red with alpha 1.0 (opaque)
            gray_color = (0.5, 0.5, 0.5, 0.3)  # gray with alpha 0.3 (semi-transparent)
            edge_colors = [edge_color if d[edge_feature] == highlight_value else gray_color for u, v, d in G.edges(data=True)]
        else:
            edge_colors = ['gray' for u, v, d in G.edges(data=True)]
        
        pos = {node: (G.nodes[node]['lon'], G.nodes[node]['lat']) for node in G.nodes()}
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.5, arrows=False)
    
    # Show the plot
    plt.show()


''' =================================================================================================
function to plot histogram
================================================================================================= '''
def plot_hist(array, bins=50):
    plt.hist(array, bins=bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of numpy array')
    plt.show()

''' =================================================================================================
function compute all node properties by specigy weight
- degree, Degree_Centrality, Clustering_Coefficients, Closeness_Centrality, Betweenness_Centrality, Eigenvector_Centrality
================================================================================================= '''
def compute_node_properties(df_data, df_pos, weight = 'distance'):
    Gc = nx.from_pandas_edgelist(df_data, source='st_id', target='st_tg', edge_attr=['train_max_speed', 'stay_time', 'distance', 'travel_time'], create_using=nx.DiGraph())

    # create a dictionary of node attributes from your df
    node_attrs = df_pos.set_index('node').to_dict('index')

    # add the node attributes to G2
    nx.set_node_attributes(Gc, node_attrs)

    # calculate centrality measures
    centrality_measures = pd.DataFrame({'Degree_Centrality': nx.degree_centrality(Gc),
                                        'Clustering_Coefficients': nx.clustering(Gc, weight=weight),
                                        'Closeness_Centrality': nx.closeness_centrality(Gc, distance=weight),
                                        'Betweenness_Centrality': nx.betweenness_centrality(Gc, weight=weight),
                                        'Eigenvector_Centrality': nx.eigenvector_centrality(Gc, weight=weight)})
    
    # get degee each node
    df_pos_degree = pd.DataFrame(Gc.degree(Gc.nodes()), columns = ['node', 'degree']).set_index('node')

    # merge pos and centra value
    df_pos_comb = df_pos.merge(df_pos_degree, left_on='node', right_index=True)
    df_pos_comb = df_pos_comb.merge(centrality_measures, left_on='node', right_index=True)

    return df_pos_comb

''' =================================================================================================
function compute label cluster all specify cols
================================================================================================= '''
def compute_all_cluster_with_best_n(df_node_properties, col_names = ['degree', 'Degree_Centrality', 'Clustering_Coefficients', 'Closeness_Centrality', 'Betweenness_Centrality', 'Eigenvector_Centrality'], plot=True):
    n_max_scores = []
    n_max = df_node_properties['state_num'].nunique()+1 # define max cluster number = num of province
    for col_name in col_names:
        n_max_score = visual_elbow_with_silhouette_score(df_node_properties, feature_cols = [col_name], n_cluster_from=2, n_cluster_to=n_max, random_state=42, plot=plot)
        df_node_properties = perform_clustering(df_node_properties, feature_cols = [col_name], n_clusters=n_max_score, random_state=42, assign_by='value')
        df_node_properties = perform_clustering_by_state(df_node_properties, feature_cols = [col_name], n_clusters=n_max_score, random_state=42, assign_by='value')
        n_max_scores.append(n_max_score)

    return df_node_properties, n_max_scores



''' =================================================================================================
function encode uniqie value inspecify col
================================================================================================= '''
def encode_col_value(df_data, col, new_col_name):
    df_data[new_col_name], uniques = pd.factorize(df_data[col])
    
    return df_data, uniques

''' =================================================================================================
function map pos df value (pos) on each row of edge df (data)
================================================================================================= '''
def map_pos_on_data(df_data, df_pos):
    df_data['st_id_lat'] = df_data['st_id'].map(df_pos.set_index('node')['lat'])
    df_data['st_id_lon'] = df_data['st_id'].map(df_pos.set_index('node')['lon'])
    df_data['st_tg_lat'] = df_data['st_tg'].map(df_pos.set_index('node')['lat'])
    df_data['st_tg_lon'] = df_data['st_tg'].map(df_pos.set_index('node')['lon'])
    df_data['st_id_state_num'] = df_data['st_id'].map(df_pos.set_index('node')['state_num'])
    df_data['st_tg_state_num'] = df_data['st_tg'].map(df_pos.set_index('node')['state_num'])
    
    return df_data

''' =================================================================================================
function check train which are travel across country or not
================================================================================================= '''
def calculate_across(row):
    if row['st_id_state_num'] == row['st_tg_state_num']:
        return 0
    else:
        return 1

''' =================================================================================================
function check overnight train
================================================================================================= '''  
def calculate_over_night(row):
    if math.floor(row['dep_time']) != math.floor(row['next_arr_time']):
        return 1
    else:
        return 0
''' =================================================================================================
function main for clustering from datasset
================================================================================================= '''
def main_clustering(df_edge, df_node):
    # load file from following 2 clean data
    # - clean_data_no_latlon.csv
    # - df_pos_with_state_clean2.csv
    df_data = df_edge.copy()
    df_pos = df_node.copy()

    # keep only col needed
    df_data = df_data[['st_id', 'st_tg', 'train_max_speed', 'distance', 'travel_time', 'stay_time']] # keep onlt needed(want) cols
    df_pos = df_pos[['node', 'lat', 'lon', 'state_num']] # keep only needed(want) cols

    # calculate node properties
    df_node_properties = compute_node_properties(df_data, df_pos, 
                                                weight = 'distance' # we can specify weight of node property for calulcation (refer to value on node feature)
                                                )

    # get df with cluster labels and list of n_max_scores of each properties
    df_pos_cluster, n_max_scores = compute_all_cluster_with_best_n(df_node_properties, 
                                                            plot=False # if Yes, it mean it will show all plot of ellbow plot each node properties
                                                            )

    print('finish clustering node')

    # load data w/ keep all
    df_data2 = df_edge.copy()

    # encode (assign unique) number to train name
    df_data_edge, _ = encode_col_value(df_data2, col='train', new_col_name='train_no')

    # map all pos information if each station into each row of edge data
    df_data_edge = map_pos_on_data(df_data_edge, df_pos)

    # classify by train speed
    # encode (assign unique) number to train_max_speed
    df_data_edge = df_data_edge.sort_values(by='train_max_speed', ascending=True) # for easily by sort --> then lower cluater number = lower speed
    df_edge_cluster, _ = encode_col_value(df_data_edge, col='train_max_speed', new_col_name='cluster_speed')

    
    # classify by across
    # assign 1 to across province train, others are 0 which is intervene country
    df_edge_cluster['cluster_across'] = df_edge_cluster.apply(calculate_across, axis=1)

    # classify by overnight
    # assign 1 to overnight train, other = 0
    df_edge_cluster['cluster_overnight'] = df_edge_cluster.apply(calculate_over_night, axis=1)

    # classify by distance
    df_edge_cluster = df_edge_cluster.sort_values(by='distance', ascending=True) # for easily by sort --> then lower cluater number = shorter distance
    df_edge_cluster = perform_clustering(df_edge_cluster, feature_cols = ['distance'], 
                                            n_clusters=3, # i use 3 cluster to indentify short. medium, long distance 
                                            random_state=42, 
                                            assign_by='value' # mode of function: this mode will order number of cluster by value in specify col (lower number = higher value of col value)
                                            )
    
    print('finish edge clustering')

    return df_pos_cluster, df_edge_cluster

if __name__ =="__main__":
    print('please import this module by "from clusternig import *"')