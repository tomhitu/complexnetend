import pandas as pd
import math
import networkx as nx
import pickle

import json
import numpy

import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class initmap():
    def __init__(self, path='railway'):
        if path == 'railway':
            self.path = '../../data/Railway Data_JL.xlsx'
        else:
            self.path = path

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        # convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # calculate differences between latitudes and longitudes
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # apply Haversine formula to calculate distance
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_in_km = 6371 * c  # approximate radius of Earth in kilometers
        distance_in_miles = distance_in_km * 0.621371  # convert kilometers to miles

        return distance_in_miles

    def get_train_speed(self, train_id):
        KM_TO_MI_FACTOR = 0.6213711922
        if train_id.startswith('G') or train_id.startswith('C'):
            return 350 * KM_TO_MI_FACTOR  # max speed of 350 km/h = 217.48 mph
        elif train_id.startswith('D'):
            return 260 * KM_TO_MI_FACTOR  # max speed of 260 km/h = 161.56 mph
        elif train_id.startswith('Z') or train_id.startswith('T'):
            return 160 * KM_TO_MI_FACTOR  # max speed of 160 km/h = 99.42 mph
        elif train_id.startswith('K'):
            return 120 * KM_TO_MI_FACTOR  # max speed of 120 km/h = 74.56 mph
        else:
            return 100 * KM_TO_MI_FACTOR  # default speed, assuming max. speed less than 120km/h = 62.14 mph

    def get_train_speed(self, train_id):
        KM_TO_MI_FACTOR = 0.6213711922
        if str(train_id).startswith('G') or str(train_id).startswith('C'):
            return 350 * KM_TO_MI_FACTOR  # max speed of 350 km/h = 217.48 mph
        elif str(train_id).startswith('D'):
            return 260 * KM_TO_MI_FACTOR  # max speed of 260 km/h = 161.56 mph
        elif str(train_id).startswith('Z') or str(train_id).startswith('T'):
            return 160 * KM_TO_MI_FACTOR  # max speed of 160 km/h = 99.42 mph
        elif str(train_id).startswith('K'):
            return 120 * KM_TO_MI_FACTOR  # max speed of 120 km/h = 74.56 mph
        else:
            return 100 * KM_TO_MI_FACTOR  # default speed, assuming max. speed less than 120km/h = 62.14 mph

    def get_map(self):
        print(self.path)
        data = pd.read_excel(self.path)

        print(data.head(3))

        # clean data

        df = data.copy()

        df0 = df.copy()
        df0['adj_mileage'] = pd.Series([float('nan')] * len(df0))

        for i in range(len(df0)):
            mil = str(df0['mileage'][i])
            if not mil.isnumeric():
                if df0['st_no'][i] == 1:
                    df0['adj_mileage'][i] = 0
                else:
                    dist_ll = self.calculate_distance(df0['lat'][i - 1], df0['lon'][i - 1], df0['lat'][i], df0['lon'][i])
                    df0['adj_mileage'][i] = dist_ll + df0['adj_mileage'][i - 1]
            else:
                df0['adj_mileage'][i] = df0['mileage'][i]

        df2 = df0.copy()

        df2['st_tg'] = pd.Series([float('nan')] * len(df2))
        df2['dist_to_tg'] = pd.Series([float('nan')] * len(df2))

        for i in range(len(df2) - 1):
            if df2['st_no'][i] + 1 == df2['st_no'][i + 1]:
                df2.at[i, 'st_tg'] = int(df2['st_id'][i + 1])
                df2.at[i, 'dist_to_tg'] = df2['adj_mileage'][i + 1] - df2['adj_mileage'][i]

        # df2['st_tg'].fillna(-1, inplace=True)
        # df2['st_tg'] = df2['st_tg'].astype(int)

        df3 = df2.copy()
        df3 = df3.dropna(subset=['st_tg'])

        # from above i found that there are some self travelling at the same time too, i think we should clean it
        # i sure that is an error because time is the same with both line

        df4 = df3.copy()
        df4 = df4[~(df4['st_id'] == df4['st_tg'])]

        df5 = df4[['train', 'st_id', 'st_tg', 'arr_time', 'dep_time', 'stay_time', 'dist_to_tg']].copy()

        # location of station
        pos = {station: (longitude, latitude) for station, latitude, longitude in
               zip(df2['st_id'], df2['lat'], df2['lon'])}

        df6 = pd.merge(df5, pd.DataFrame(pos).T.rename(columns={0: 'lat', 1: 'lon'}),
                       how='left', left_on='st_id', right_index=True)
        df6 = pd.merge(df6, pd.DataFrame(pos).T.rename(columns={0: 'tg_lat', 1: 'tg_lon'}),
                       how='left', left_on='st_tg', right_index=True)

        df7 = df6
        df7['dist_to_tg'][4092] = 1
        df7['dist_to_tg'][8389] = 1

        df8 = df7.copy()  # to avoid modifying the original dataframe
        df8['train'] = df8['train'].astype(str)  # convert 'train' column to string type
        df8['train_speed'] = df8['train'].apply(self.get_train_speed)  # apply the function to create 'train_speed' column

        df9 = df8.copy()  # make a copy of df8 to avoid modifying it
        df9['travel_time'] = df9['dist_to_tg'] / df9[
            'train_speed']  # calculate travel time using 'dist_to_tg' and 'train_speed' columns

        # Create graph # direct graph # cannot return in plot route
        G = nx.from_pandas_edgelist(df9, source='st_id', target='st_tg',
                                    edge_attr=['dist_to_tg', 'travel_time', 'dep_time'])

        # Add node positions to node attributes
        nx.set_node_attributes(G, pos, 'pos')


        # store graph data
        with open("graph.pickle", "wb") as f:
            pickle.dump(G, f)

        # with open("graph.pickle", "rb") as f:
        #     G_loaded = pickle.load(f)
        #
        # assert nx.is_isomorphic(G, G_loaded)

        # Add node positions to node attributes
        # nx.set_node_attributes(G, pos, 'pos')

        # format = {
        #     "nodes": [
        #         {
        #             "color": "#4f19c7",
        #             "label": "jquery",
        #             "attributes": {},
        #             "y": -404.26147,
        #             "x": -739.36383,
        #             "id": "jquery",
        #             "size": 4.7252817
        #         }
        #     ],
        #     "edges": [
        #         {
        #             "sourceID": "jquery",
        #             "attributes": {},
        #             "targetID": "jsdom",
        #             "size": 1
        #         }
        #     ]
        # }

        # nodesList = df9['st_id'].unique()
        # nodes = []
        #
        # for item in nodesList:
        #     node_temp = {}
        #     node_temp['name'] = item
        #     node_temp['value'] = [pos[item][0], pos[item][1]]
        #     nodes.append(node_temp)
        #
        # edges = [
        #     {
        #         "source": st_id,
        #         "target": st_tg,
        #         "train": train
        #     } for st_id, st_tg, train in zip(df9['st_id'], df9['st_tg'], df9['train']) if st_tg != -1
        # ]
        #
        # a = {'nodes': nodes, 'edges': edges}
        # b = json.dump(a, open('railway_data.json', 'w'), cls=NumpyEncoder)


def getData():
    with open('railway_data.json', 'r') as fcc_file:
      fcc_data = json.load(fcc_file)
      return fcc_data


'''
function to find shorest xxx from node to node

which xxx can be any edge attribute
'''


def find_shortest_path_with_attr(G, source, target, edge_attr):
    # Find the shortest path between the source and target nodes
    shortest_path = nx.shortest_path(G, source=source, target=target, weight=edge_attr)

    # Compute the total travel distance and time along the shortest path
    travel_distance = sum(G[u][v][edge_attr] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    travel_time = sum(G[u][v]['travel_time'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))

    # Return the shortest path, travel distance, and travel time
    return shortest_path, travel_distance, travel_time


def getG(snode, tnode):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'graph.pickle')
    with open(image_path, "rb") as f:
        G_loaded = pickle.load(f)

    source_node = int(snode)
    target_node = int(tnode)
    edge_attr = 'dist_to_tg'
    shortest_path, travel_distance, travel_time = find_shortest_path_with_attr(G_loaded, source_node, target_node,
                                                                               edge_attr=edge_attr)
    print("Shortest ", edge_attr, " from node", source_node, "to node", target_node, ":", shortest_path)
    print("Total travel distance:", travel_distance)
    return shortest_path, travel_distance, travel_time


def setdatajson():
    nodes_cluster = pd.read_csv('../../data/df_pos_cluster.csv')
    edges_cluster = pd.read_csv('../../data/df_edge_cluster_light.csv')
    nodes = []
    # state_num
    # degree
    # Degree_Centrality
    # Clustering_Coefficients
    # Closeness_Centrality
    # Betweenness_Centrality
    # Eigenvector_Centrality
    # cluster_degree
    # cluster_state_degree
    # cluster_Degree_Centrality
    # cluster_state_Degree_Centrality
    # cluster_Clustering_Coefficients
    # cluster_state_Clustering_Coefficients
    # cluster_Closeness_Centrality
    # cluster_state_Closeness_Centrality
    # cluster_Betweenness_Centrality
    # cluster_state_Betweenness_Centrality
    # cluster_Eigenvector_Centrality
    # cluster_state_Eigenvector_Centrality
    for item, row in nodes_cluster.iterrows():
        node_temp = {}
        node_temp['name'] = row['node']
        node_temp['value'] = [row['lon'], row['lat']]
        node_temp['state_num'] = row['state_num']
        node_temp['degree'] = row['degree']
        node_temp['Degree_Centrality'] = row['Degree_Centrality']
        node_temp['Clustering_Coefficients'] = row['Clustering_Coefficients']
        node_temp['Closeness_Centrality'] = row['Closeness_Centrality']
        node_temp['Betweenness_Centrality'] = row['Betweenness_Centrality']
        node_temp['Eigenvector_Centrality'] = row['Eigenvector_Centrality']
        node_temp['cluster_degree'] = row['cluster_degree']
        node_temp['cluster_state_degree'] = row['cluster_state_degree']
        node_temp['cluster_Degree_Centrality'] = row['cluster_Degree_Centrality']
        node_temp['cluster_state_Degree_Centrality'] = row['cluster_state_Degree_Centrality']
        node_temp['cluster_Clustering_Coefficients'] = row['cluster_Clustering_Coefficients']
        node_temp['cluster_state_Clustering_Coefficients'] = row['cluster_state_Clustering_Coefficients']
        node_temp['cluster_Closeness_Centrality'] = row['cluster_Closeness_Centrality']
        node_temp['cluster_state_Closeness_Centrality'] = row['cluster_state_Closeness_Centrality']
        node_temp['cluster_Betweenness_Centrality'] = row['cluster_Betweenness_Centrality']
        node_temp['cluster_state_Betweenness_Centrality'] = row['cluster_state_Betweenness_Centrality']
        node_temp['cluster_Eigenvector_Centrality'] = row['cluster_Eigenvector_Centrality']
        node_temp['cluster_state_Eigenvector_Centrality'] = row['cluster_state_Eigenvector_Centrality']

        nodes.append(node_temp)

    edges = []
    # st_id
    # st_tg
    # train
    # train_max_speed
    # st_id_state_num
    # st_tg_state_num
    # cluster_speed
    # cluster_across
    # cluster_overnight
    # cluster_distance

    for item, row in edges_cluster.iterrows():
        edge_temp = {}
        edge_temp['source'] = row['st_id']
        edge_temp['target'] = row['st_tg']
        edge_temp['train'] = row['train']
        edge_temp['train_max_speed'] = row['train_max_speed']
        edge_temp['st_id_state_num'] = row['st_id_state_num']
        edge_temp['st_tg_state_num'] = row['st_tg_state_num']
        edge_temp['cluster_speed'] = row['cluster_speed']
        edge_temp['cluster_across'] = row['cluster_across']
        edge_temp['cluster_overnight'] = row['cluster_overnight']
        edge_temp['cluster_distance'] = row['cluster_distance']

        edges.append(edge_temp)

    a = {'nodes': nodes, 'edges': edges}
    b = json.dump(a, open('railway_data.json', 'w'), cls=NumpyEncoder)


if __name__ == '__main__':
    # app = initmap('railway')
    # app.get_map()
    # getG()
    setdatajson()
