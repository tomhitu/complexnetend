import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

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
function to plot node location color by cluster
================================================================================================= '''
def plot_node_by_cluster(df, high_light_cluster=None, high_light_node=None, base_color='Set2', col_name='cluster'):
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Load the map of France
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    paris = world[world['name'] == 'Paris']

    # Create a new figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

    # Plot the map of Paris
    paris.plot(ax=ax, color='white', edgecolor='black')

    # set non highlight
    if high_light_cluster is None and high_light_node is None:
        gray_points = ax.scatter(df['lat'], df['lon'], s=0.5, c=df[col_name], cmap=base_color, alpha=1)
    else:
        gray_points = ax.scatter(df['lat'], df['lon'], s=0.5, c=df[col_name], cmap=base_color, alpha=0.3)

    # Highlight the specified cluster in red, if provided
    if high_light_cluster is not None:
        red_points = ax.scatter(df[df[col_name] == high_light_cluster]['lat'], df[df[col_name] == high_light_cluster]['lon'], s=1, c='red')

    # Highlight the specified node in red, if provided
    if high_light_node is not None and high_light_node in df.index:
        ax.scatter(df.loc[high_light_node, 'lat'], df.loc[high_light_node, 'lon'], s=1, c='red')

    # Show the plot
    plt.show()

''' =================================================================================================
function to plot map and highlight specify edge
================================================================================================= '''

def plot_edge_by_cluster(df_node, df_edge, high_light_cluster=None, high_light_node=None, base_color='Set2', col_name=None, edge_feature=None, highlight_value=None, edge_color='red'):
    # Drop duplicate columns
    df_node = df_node.loc[:, ~df_node.columns.duplicated()]

    # Load the map of France
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    paris = world[world['name'] == 'Paris']

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    # Plot the map of Paris
    paris.plot(ax=ax, color='white', edgecolor='black')

    # set non highlight
    if high_light_cluster is None and high_light_node is None:
        if col_name is None:
            gray_points = ax.scatter(df_node['lat'], df_node['lon'], s=0.2, color='gray', alpha=1)
        else:
            gray_points = ax.scatter(df_node['lat'], df_node['lon'], s=0.5, c=df_node[col_name], cmap=base_color, alpha=1)
    else:
        gray_points = ax.scatter(df_node['lat'], df_node['lon'], s=0.5, c=df_node[col_name], cmap=base_color, alpha=0.3)

    # Highlight the specified cluster in red, if provided
    if high_light_cluster is not None:
        red_points = ax.scatter(df_node[df_node[col_name] == high_light_cluster]['lat'], df_node[df_node[col_name] == high_light_cluster]['lon'], s=1, c='red')

    # Highlight the specified node in red, if provided
    if high_light_node is not None and high_light_node in df_node.index:
        ax.scatter(df_node.loc[high_light_node, 'lat'], df_node.loc[high_light_node, 'lon'], s=1, c='red')
    
    # Create a directed graph from the dataframe
    G = nx.from_pandas_edgelist(df_edge, source='source', target='target', edge_attr=[edge_feature], create_using=nx.DiGraph())
    
    # Create a dictionary of node attributes from the node dataframe
    node_attrs = df_node.set_index('node_id').to_dict('index')

    # Add the node attributes to the graph
    nx.set_node_attributes(G, node_attrs)
    
    # Highlight the specified edge attribute, if provided
    if edge_color is not None and edge_feature is not None:
        if highlight_value is not None:
            edge_color = (1.0, 0.0, 0.0, 1.0)  # red with alpha 1.0 (opaque)
            gray_color = (0.5, 0.5, 0.5, 0.3)  # gray with alpha 0.3 (semi-transparent)
            edge_colors = [edge_color if d[edge_feature] == highlight_value else gray_color for u, v, d in G.edges(data=True)]
        else:
            edge_colors = ['gray' for u, v, d in G.edges(data=True)]
        
        # pos = {node: (df_node.loc[node, 'lat'], df_node.loc[node, 'lon']) for node in G.nodes()}
        pos = {node: (G.nodes[node]['lat'], G.nodes[node]['lon']) for node in G.nodes()}
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.5, arrows=False)
    
    # Show the plot
    plt.show()


''' =================================================================================================
function compute all node properties by specigy weight
- degree, Degree_Centrality, Clustering_Coefficients, Closeness_Centrality, Betweenness_Centrality, Eigenvector_Centrality
================================================================================================= '''
def compute_node_properties(df_data, df_pos, weight = 'distance'):
    Gc = nx.from_pandas_edgelist(df_data, source='source', target='target', edge_attr=['distance'], create_using=nx.DiGraph())

    # create a dictionary of node attributes from your df
    node_attrs = df_pos.set_index('node_id').to_dict('index')

    # add the node attributes to G2
    nx.set_node_attributes(Gc, node_attrs)

    # calculate centrality measures
    centrality_measures = pd.DataFrame({'Degree_Centrality': nx.degree_centrality(Gc),
                                        'Clustering_Coefficients': nx.clustering(Gc, weight=weight),
                                        'Closeness_Centrality': nx.closeness_centrality(Gc, distance=weight),
                                        'Betweenness_Centrality': nx.betweenness_centrality(Gc, weight=weight),
                                        'Eigenvector_Centrality': nx.eigenvector_centrality(Gc, weight=weight)})
    
    # get degee each node
    df_pos_degree = pd.DataFrame(Gc.degree(Gc.nodes()), columns = ['node_id', 'degree']).set_index('node_id')

    # merge pos and centra value
    df_pos_comb = df_pos.merge(df_pos_degree, left_on='node_id', right_index=True)
    df_pos_comb = df_pos_comb.merge(centrality_measures, left_on='node_id', right_index=True)

    return df_pos_comb

''' =================================================================================================
function compute label cluster all specify cols
- update: edit if value of col are all same --> make it no cluster and assign value cluster to 0
================================================================================================= '''
def compute_all_cluster_with_best_n(df_node_properties, col_names = ['degree', 'cluster_lat_lon','Degree_Centrality', 'Clustering_Coefficients', 'Closeness_Centrality', 'Betweenness_Centrality', 'Eigenvector_Centrality'], plot=True):
    n_max_scores = []
    for col_name in col_names:
        if df_node_properties[col_name].nunique()==1:
            # retrieve cluster name
            prefix = "cluster"
            cluster_name = prefix + "_" + "_".join([col_name])
            df_node_properties[cluster_name] = 0
            n_max_scores.append(1)
        else:
            n_max_score = visual_elbow_with_silhouette_score(df_node_properties, feature_cols = [col_name], n_cluster_from=2, n_cluster_to=10, random_state=42, plot=plot)
            df_node_properties = perform_clustering(df_node_properties, feature_cols = [col_name], n_clusters=n_max_score, random_state=42, assign_by='value')
            n_max_scores.append(n_max_score)

    return df_node_properties, n_max_scores




if __name__ =="__main__":
    print('please import this module by "from dataset2_clustering import *"')