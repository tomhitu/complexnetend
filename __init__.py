"""
Toolbox for local computation of pre-analysis and real-time actions.
"""

import pandas as pd

from src.assets import mapdraw as mdraw
from src.assets import minpath as mpath
from src.assets import example_pred_edge as epe

from src.cleandata import pre_proc_data as cdata
from src.cleandata import pos_map_province as posmap

from src.cleandata import pred_edge as pedge
from src.cleandata import pred_degree as pdeg

from src.assets import NetworkResilience as nres


"""
Clean your data
"""
def cleandata(data):
    required_columns = ['train', 'st_no', 'mileage', 'st_id', 'date', 'arr_time', 'dep_time', 'stay_time', 'lat', 'lon']

    # check dataframe if similar to chinese railway data
    if all(column in data.columns for column in required_columns):
        df, pos = cdata.chinese_railway_prep(data)
        print("Your data has been cleaned successfully.")
        print(f'len_df: {len(df)}, len_pos: {len(pos)}')
        return df, pos
    else:
        print("Your data has different columns. Please check your data.")
        print(f'Your data columns: {data.columns}')
        print(f'Required columns: {required_columns}')
        return None, None


"""
clean data with lat lon
"""
def clustercoord(_, pos):
    print("Please make sure that you have run cleandata() to get pos dict.")
    # convert pos dict to df
    df_pos = pd.DataFrame(pos.items(), columns=['node', 'pos'])
    df_pos[['lat', 'lon']] = df_pos['pos'].apply(lambda x: pd.Series({'lat': x[0], 'lon': x[1]}))

    # map province(state) and city into each lat-lon
    df_pos[['city', 'state']] = df_pos.apply(lambda x: pd.Series(posmap.extract_city_state(str(x['lat']), str(x['lon']))), axis=1)

    return df_pos


"""
map province and city into each lat-lon
"""
def mapprovince(data):
    df_pos_with_state_clean = posmap.clean_blank_province(data)

    required_columns = ['node', 'lat', 'lon', 'geometry', 'city', 'state']
    if all(column in df_pos_with_state_clean.columns for column in required_columns):
        ''' -- clean city data -- '''
        # load non clean data from file and clean it
        merged_df = pd.merge(df_pos_with_state_clean, data[['node', 'city']], on='node')
        df_pos_with_state_clean2 = posmap.clean_blank_city(merged_df)

        return df_pos_with_state_clean2
    else:
        print("Your data has different columns. Please check your data.")
        print(f'Your data columns: {data.columns}')
        print(f'Required columns: {required_columns}')
        print(f'You can run clustercoord() to get required columns.')
        return None


"""
prediction of edges how to add new node into the graph with lon, lat and pre-trained model
"""
def predictedges(new_node_lat, new_node_lon):
    epe.init()
    n_degree, new_neighbor_node, new_edges_distance, new_edges_travel_time, new_edges_train_speed, new_node_id = epe.predEdges(new_node_lon, new_node_lat)
    return n_degree, new_neighbor_node, new_edges_distance, new_edges_travel_time, new_edges_train_speed, new_node_id


"""
prediction with new model
"""
def trainnewmodel(df_data, df_pos, folder_degree, folder_edge, attr_dim, new_node_lat=37.063816, new_node_lon=92.803324):
    print("Please make sure that you can get df_data and df_pos.")
    required_columns = ['st_id', 'st_tg', 'train_max_speed', 'distance', 'travel_time']
    required_pos_columns = ['node', 'lat', 'lon', 'state_num']
    if all(column in df_data.columns for column in required_columns) and \
            all(column in df_pos.columns for column in required_pos_columns):
        num_epochs = attr_dim['num_epochs']
        lr = attr_dim['lr']
        batch_size = attr_dim['batch_size']
        scheduler_step_size = attr_dim['scheduler_step_size']
        scheduler_gamma = attr_dim['scheduler_gamma']
        print_epoch = attr_dim['print_epoch']
        step_print = attr_dim['step_print']
        n_degree = pdeg.main_pred_degree(df_data=df_data, df_pos=df_pos,
                                         new_node_lat=new_node_lat, new_node_lon=new_node_lon,
                                         train_new=True, folder_name=folder_degree,
                                         num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                                         scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma,
                                         print_epoch=print_epoch, step_print=step_print)
        if n_degree is None:
            print("Please check your data.")
        else:
            print("format both are correct.")
            df_new_data, df_new_pos, new_st_id, neighbor_node = pedge.main_edge_feature_pred(df_data=df_data, df_pos=df_pos,
                                                                                             new_node_lat=new_node_lat, new_node_lon=new_node_lon,
                                                                                             n_degree=n_degree, train_new=True,
                                                                                             folder_name=folder_edge,
                                                                                             col_inputs=['lat_id', 'lon_id', 'state_num_id', 'lat_tg', 'lon_tg', 'state_num_tg'],
                                                                                             split_portion=0.1, hidden_channel=128, num_epochs=50, lr=0.01, batch_size=1024,
                                                                                             scheduler_step_size=5, scheduler_gamma=0.6, print_epoch=True, step_print=10)
            return n_degree, df_new_data, df_new_pos, new_st_id, neighbor_node
    else:
        print("Your data has different columns. Please check your data.")
        print(f'Your df_data columns: {df_data.columns}')
        print(f'Required df_data columns: {required_columns}')
        print(f'Your df_pos columns: {df_pos.columns}')
        print(f'Required df_pos columns: {required_pos_columns}')
        print(f'You can run testclusterpos() and testmapprovince() to get required columns.')

    return None, None, None, None, None


"""
transfer data into front-end
"""
def setjson(nodes_cluster, edges_cluster):
    nodes, edges, maxclusternum, maxedgenum = mdraw.setdatajson(nodes_cluster, edges_cluster)
    return nodes, edges, maxclusternum, maxedgenum


"""
delete node and show change
"""
def deletenode(node_id, G):
    before, after = nres.delete_node_any(node_id, G)
    return before, after


"""
Dijkstra algorithm with shortest path
"""
def shortestway(start_node, end_node, G):
    path, distance = mpath.getG_any(start_node, end_node, G)
    return path, distance


if __name__ == "__main__":
    print("Please run test.py")