import numpy as np
import pandas as pd
from dataset2_pre_proc import *

def main():
    # load raw data and clean it
    # during processing, data will same into temp file if the program are shut down when mapping province
    # mapping province may excess limitation of Geopy, yiu can run this file again and it will continue from leaving
    # raw_path_folder = folder of raw data
    # temp_filename = filename & path of temp file
    # note: after finish all clean temp file will be delete
    # you may check, the code can be run by looking update of "temp_filename" when processing
    df_nodes, df_edges = main_prep_data(raw_path_folder="MultilayerParis-master", temp_filename = "dataset2_df_nodes_v3_notcomplet.csv")
    
    # save in to CSV file
    df_nodes.to_csv('dataset2_df_nodes_v3.csv', index=False)
    df_edges.to_csv('dataset2_df_edges_v3.csv', index=False)
    print('finish save all clean dataset')

if __name__ == "__main__":
    main()