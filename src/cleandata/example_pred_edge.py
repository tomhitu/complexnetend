import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pred_edge import *
from pred_degree import *

def main():
    # load data
    df_data = pd.read_csv('clean_data_no_latlon.csv')
    df_data = df_data[['st_id', 'st_tg', 'train_max_speed', 'distance', 'travel_time']] # keep onlt needed(want) cols

    df_pos = pd.read_csv('df_pos_with_state_clean2.csv')
    df_pos = df_pos[['node', 'lat', 'lon', 'state_num']] # keep only needed(want) cols

    # norm labels because it is in differnebt range scale
    scaler_labels = MinMaxScaler()
    df_data.iloc[:, 2:5] = scaler_labels.fit_transform(df_data.iloc[:, 2:5])

    # define new node
    new_node_lat = 37.063816
    new_node_lon = 92.803324

    ''' [IMPRTANCE to CONSIDER SWITCHING MODE between RUN MODEL AGAIN or USE EXITING TRAINED MODEL] '''
    # mode = 0 = train again
    # mode = 1 = use previous model
    degree_mode = 1
    edge_mode = 1

    # pred degree
    if degree_mode == 0:    
        ''' Degree train again ''' # it will take sometime, however you can use trined model following part
        n_degree = main_pred_degree(df_data = df_data, df_pos = df_pos,
                                new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                train_new = True, folder_name = 'degree_pred_conf',
                                num_epochs=100, lr=0.1, batch_size=128, 
                                scheduler_step_size=10, scheduler_gamma=0.7, 
                                print_epoch=True, step_print=10
                                )
    elif degree_mode == 1:
        ''' Degree use previous model '''
        n_degree = main_pred_degree(df_data = df_data, df_pos = df_pos,
                                new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                train_new = False, folder_name = 'degree_pred_conf')
    
    print(f'pred n_degree: {n_degree}')

    # predict edge
    if edge_mode == 0:
        ''' Edge train again ''' # it will take sometime, however you can use trined model following part
        df_new_data, df_new_pos, new_st_id, neighbor_node = main_edge_feature_pred(df_data = df_data, df_pos = df_pos,
                                                                                new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                                                                n_degree = n_degree, train_new = True, folder_name = 'edge_pred_conf',
                                                                                col_inputs = ['lat_id', 'lon_id', 'state_num_id', 'lat_tg', 'lon_tg', 'state_num_tg'],
                                                                                split_portion = 0.1, hidden_channel=128, num_epochs=50, lr=0.01, batch_size=1024,
                                                                                scheduler_step_size=5, scheduler_gamma=0.6, print_epoch=True, step_print=10,
                                                                                )
    elif edge_mode == 1:
        ''' Edge use previous model '''
        df_new_data, df_new_pos, new_st_id, neighbor_node = main_edge_feature_pred(df_data = df_data, df_pos = df_pos,
                                                                                    new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                                                                    n_degree = n_degree)
    print(f'new_state_num: {df_new_pos[df_new_pos["node"]==new_st_id]["state_num"].iloc[0]}')
    
    # denorm labels
    cols_to_denormalize = ['train_max_speed', 'distance', 'travel_time']
    df_new_data[cols_to_denormalize] = scaler_labels.inverse_transform(df_new_data[cols_to_denormalize])

    # apply train speed mapping
    df_new_data['train_max_speed'] = df_new_data['train_max_speed'].apply(reverse_train_speed).apply(lambda x: x[1])

    print(f'pred edge: \n{df_new_data.loc[df_new_data["st_id"] == new_st_id]}')

    '''
    plot graph
    - jsut for chekcing, no need if perform on real UI (use Yuqin dispplay instead)
    '''
    # create new G
    G2 = nx.from_pandas_edgelist(df_new_data, source='st_id', target='st_tg', edge_attr=['distance', 'travel_time', 'train_max_speed'])

    # create a dictionary of node attributes from your df
    node_attrs = df_new_pos.set_index('node').to_dict('index')

    # add the node attributes to G2
    nx.set_node_attributes(G2, node_attrs)

    # plot all nodes and edges on the map
    plot_all_on_map(G = G2, node_attrs = node_attrs, high_light_node1=[new_st_id], high_light_node2=neighbor_node)

if __name__ == "__main__":
    main()