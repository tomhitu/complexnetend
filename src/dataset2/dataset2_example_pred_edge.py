from sklearn.preprocessing import MinMaxScaler

from .dataset2_pred_edge import *
from .dataset2_pred_degree import *


def main(new_node_lat, new_node_lon):
    new_node_lat = float(new_node_lat)
    new_node_lon = float(new_node_lon)

    """
    get the current path
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    """
    get the data path
    """
    df_path = os.path.abspath(os.path.join(current_path, 'dataset2_df_nodes_v3.csv'))
    pos_path = os.path.abspath(os.path.join(current_path, 'dataset2_df_edges_v3.csv'))
    # load data
    df_edge = pd.read_csv(pos_path)
    df_node = pd.read_csv(df_path)

    df_edge = df_edge[['source', 'target', 'distance', 'type_num']] # keep onlt needed(want) cols
    df_node = df_node[['node_id', 'lat', 'lon', 'type_num']] # keep only needed(want) cols

    # norm labels because it is in differnebt range scale
    scaler_labels = MinMaxScaler()
    df_edge.iloc[:, 2:4] = scaler_labels.fit_transform(df_edge.iloc[:, 2:4])

    # define new node
    # new_node_lat = 2.23
    # new_node_lon = 48.41

    ''' Degree train again ''' # it will take sometime, however you can use trined model following part
    # n_degree = main_pred_degree(df_data = df_edge, df_pos = df_node,
    #                         new_node_lat = new_node_lat, new_node_lon = new_node_lon,
    #                         train_new = True, folder_name = 'dataset2_degree_pred_conf')

    ''' Degree use previous model '''
    n_degree = main_pred_degree(df_data = df_edge, df_pos = df_node,
                            new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                            train_new = False, folder_name = 'dataset2_degree_pred_conf')

    print(f'pred n_degree: {n_degree}')

    ''' Edge train again ''' # it will take sometime, however you can use trined model following part
    # df_new_data, df_new_pos, new_st_id, neighbor_node = main_edge_feature_pred(df_data = df_edge, df_pos = df_node,
    #                                                                            new_node_lat = new_node_lat, new_node_lon = new_node_lon,
    #                                                                            n_degree = n_degree, train_new = True, folder_name = 'dataset2_edge_pred_conf',
    #                                                                            split_portion = 0.1, hidden_channel=128, num_epochs=80, lr=0.01, batch_size=1024,
    #                                                                            scheduler_step_size=5, scheduler_gamma=0.8, print_epoch=True, step_print=10,
    #                                                                           )
    ''' Edge use previous model '''
    df_new_edge, df_new_node, new_node, neighbor_node = main_edge_feature_pred(df_data = df_edge, df_pos = df_node,
                                                                                new_node_lat = new_node_lat, new_node_lon = new_node_lon,
                                                                                n_degree = n_degree, folder_name = 'dataset2_edge_pred_conf')
    
    # denorm labels
    cols_to_denormalize = ['distance', 'type_num']
    df_new_edge[cols_to_denormalize] = scaler_labels.inverse_transform(df_new_edge[cols_to_denormalize])

    print(f'pred edge: \n{df_new_edge.loc[df_new_edge["target"] == new_node]}')

    df_new_edges = df_new_edge.loc[df_new_edge["target"] == new_node]

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
    # plot_all_on_map(G=G2, high_light_node1=[new_node], high_light_node2=[neighbor_node[0]])

    new_neighbor_node = []
    new_edges_distance = []
    new_edges_type_num = []

    if df_new_edges.shape[0] > 0:
        for index, row in df_new_edges.iterrows():
            new_neighbor_node.append(row['source'])
            new_edges_distance.append(row['distance'])
            new_edges_type_num.append(row['type_num'])

    return n_degree, new_neighbor_node, new_edges_distance, new_edges_type_num

if __name__ == "__main__":
    main(2, 48)