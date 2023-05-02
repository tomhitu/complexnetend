import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dataset2_pred_edge import *
from dataset2_pred_degree import *

def main():
    # load data
    df_edge = pd.read_csv('dataset2_df_edges_v3.csv')
    df_node = pd.read_csv('dataset2_df_nodes_v3.csv')

    df_edge = df_edge[['source', 'target', 'distance', 'type_num']] # keep onlt needed(want) cols
    df_node = df_node[['node_id', 'lat', 'lon', 'type_num']] # keep only needed(want) cols

    # norm labels because it is in differnebt range scale
    scaler_labels = MinMaxScaler()
    df_edge.iloc[:, 2:4] = scaler_labels.fit_transform(df_edge.iloc[:, 2:4])

    # define new node
    new_node_lat = 2.388096
    new_node_lon = 48.870187

    ''' [IMPRTANCE to CONSIDER SWITCHING MODE between RUN MODEL AGAIN or USE EXITING TRAINED MODEL] '''
    # mode = 0 = train again
    # mode = 1 = use previous model
    degree_mode = 1
    edge_mode = 1

    if degree_mode == 0:
        ''' Degree train again ''' # it will take sometime, however you can use trined model following part
        n_degree = main_pred_degree(df_data = df_edge, df_pos = df_node,
                                new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                train_new = True, folder_name = 'dataset2_degree_pred_conf',
                                num_epochs=100, lr=0.1, batch_size=64, 
                                scheduler_step_size=10, scheduler_gamma=0.8, 
                                print_epoch=True, step_print=10
                                )

    elif degree_mode == 1:
        ''' Degree use previous model '''
        n_degree = main_pred_degree(df_data = df_edge, df_pos = df_node,
                                new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                train_new = False, folder_name = 'dataset2_degree_pred_conf')

    print(f'pred n_degree: {n_degree}')

    if edge_mode == 0:
        ''' Edge train again ''' # it will take sometime, however you can use trined model following part
        df_new_edge, df_new_node, new_node, neighbor_node = main_edge_feature_pred(df_data = df_edge, df_pos = df_node,
                                                                                   new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                                                                   n_degree = n_degree, train_new = True, folder_name = 'dataset2_edge_pred_conf',
                                                                                   split_portion = 0.1, hidden_channel=128, num_epochs=80, lr=0.01, batch_size=1024,
                                                                                   scheduler_step_size=5, scheduler_gamma=0.8, print_epoch=True, step_print=10,
                                                                                  )
    elif edge_mode == 1:
        ''' Edge use previous model '''
        df_new_edge, df_new_node, new_node, neighbor_node = main_edge_feature_pred(df_data = df_edge, df_pos = df_node,
                                                                                    new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                                                                    n_degree = n_degree, folder_name = 'dataset2_edge_pred_conf')
    
    # denorm labels
    cols_to_denormalize = ['distance', 'type_num']
    df_new_edge[cols_to_denormalize] = scaler_labels.inverse_transform(df_new_edge[cols_to_denormalize])

    print(f'pred edge: \n{df_new_edge.loc[df_new_edge["target"] == new_node]}')

    '''
    plot graph
    - jsut for chekcing, no need if perform on real UI (use Yuqin dispplay instead)
    '''
    # create new G
    G2 = nx.from_pandas_edgelist(df_new_edge, source='source', target='target', edge_attr=['distance', 'type_num'])

    # create a dictionary of node attributes from your df
    node_attrs = df_new_node.set_index('node_id').to_dict('index')

    # add the node attributes to G2
    nx.set_node_attributes(G2, node_attrs)

    # plot all nodes and edges on the map
    plot_all_on_map(G = G2, high_light_node1=[new_node], high_light_node2=[neighbor_node[0]])

if __name__ == "__main__":
    main()