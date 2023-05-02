import numpy as np
import pandas as pd
from clustering import *

def main():
    # load file from following 2 clean data
    # - clean_data_no_latlon.csv
    # - df_pos_with_state_clean2.csv
    df_data = pd.read_csv('clean_data_no_latlon.csv')
    df_pos = pd.read_csv('df_pos_with_state_clean2.csv')
    df_pos_cluster, df_edge_cluster = main_clustering(df_edge = df_data, df_node = df_pos)

    # save into CSV file for node clsustering
    df_pos_cluster.to_csv('df_pos_cluster.csv', index=False)

    # save into CSV file for edge clustering
    df_edge_cluster.to_csv('df_edge_cluster.csv', index=False)

if __name__ == "__main__":
    main()