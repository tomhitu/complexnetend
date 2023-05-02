import numpy as np
import pandas as pd
from dataset2_clustering import *

def main():
    # load file from following 2 clean data
    # - dataset2_df_nodes_v3.csv
    # - dataset2_df_edges_v3.csv
    df_node = pd.read_csv('dataset2_df_nodes_v3.csv')
    df_edge = pd.read_csv('dataset2_df_edges_v3.csv')
    df_node_cluster, df_edge_cluster = main_clustering(df_edge = df_edge, df_node = df_node)

    # save into CSV file
    df_node_cluster.to_csv('dataset2_df_node_cluster.csv', index=False)

    # save into CSV file
    df_edge_cluster.to_csv('dataset2_df_edge_cluster.csv', index=False)

if __name__ == "__main__":
    main()