from sklearn.preprocessing import MinMaxScaler
import os.path

"""
.pred_edge for application.py
pred_edge for local
"""
from .pred_edge import *
from .pred_degree import *
import os

df_data, df_pos, scaler_labels = None, None, None


def init():
    """
    get the current path
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    """
    get the data path
    """
    df_path = os.path.abspath(os.path.join(current_path, '../../data/clean_data_no_latlon.csv'))
    pos_path = os.path.abspath(os.path.join(current_path, '../../data/df_pos_with_state_clean.csv'))
    # load data
    global df_data, df_pos, scaler_labels
    df_data = pd.read_csv(df_path)
    df_data = df_data[['st_id', 'st_tg', 'train_max_speed', 'distance', 'travel_time']]  # keep only needed(want) cols
    df_pos = pd.read_csv(pos_path)
    df_pos = df_pos[['node', 'lat', 'lon', 'state_num']]  # keep only needed(want) cols

    # norm labels because it is in different range scale
    scaler_labels = MinMaxScaler()
    df_data.iloc[:, 2:5] = scaler_labels.fit_transform(df_data.iloc[:, 2:5])


def predEdges(new_node_lon, new_node_lat):

    new_node_lat = float(new_node_lat)
    new_node_lon = float(new_node_lon)

    # define new node
    # new_node_lat = 37.063816
    # new_node_lon = 92.803324

    """
    :param new_node_lat:
    :param new_node_lon:
    :return:
    """

    """ Edge train again """
    # it will take sometime, however you can use trained model following part
    # n_degree = main_pred_degree(df_data = df_data, df_pos = df_pos,
    #                         new_node_lat = new_node_lat, new_node_lon = new_node_lon,
    #                         train_new = True, folder_name = 'degree_pred_conf')
    ''' Degree use previous model '''
    n_degree = main_pred_degree(df_data=df_data, df_pos=df_pos, new_node_lat=new_node_lat, new_node_lon=new_node_lon,
                                train_new=False, folder_name='degree_pred_conf')
    # print(f'pred n_degree: {n_degree}')

    ''' Edge train again ''' # it will take sometime, however you can use trined model following part

    ''' Edge use previous model '''
    df_new_data, df_new_pos, new_st_id, neighbor_node = main_edge_feature_pred(df_data=df_data, df_pos=df_pos,
                                                                               new_node_lat=new_node_lat, new_node_lon=new_node_lon,
                                                                               n_degree=n_degree)
    
    # denorm labels
    cols_to_denormalize = ['train_max_speed', 'distance', 'travel_time']
    df_new_data[cols_to_denormalize] = scaler_labels.inverse_transform(df_new_data[cols_to_denormalize])

    # apply train speed mapping
    df_new_data['train_max_speed'] = df_new_data['train_max_speed'].apply(reverse_train_speed).apply(lambda x: x[1])

    # print(f'pred edge: \n{df_new_data.loc[df_new_data["st_id"] == new_st_id]}')

    df_new_edges = df_new_data.loc[df_new_data["st_id"] == new_st_id]

    new_neighbor_node = []
    new_edges_distance = []
    new_edges_travel_time = []
    new_edges_train_speed = []
    new_node_id = 0

    if df_new_edges.shape[0] > 0:
        for index, row in df_new_edges.iterrows():
            new_neighbor_node.append(row['st_tg'])
            new_edges_distance.append(row['distance'])
            new_edges_travel_time.append(row['travel_time'])
            new_edges_train_speed.append(row['train_max_speed'])
            new_node_id = row['st_id']

    print(f'new_neighbor_node: {new_node_id}')

    return n_degree, new_neighbor_node, new_edges_distance, new_edges_travel_time, new_edges_train_speed, new_node_id


if __name__ == "__main__":
    init()
    lon = 92.803324
    lat = 37.063816
    predEdges(lon, lat)