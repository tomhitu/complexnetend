"""
@ author: Yuqin Xia
@ date: 2023-4-11
@ description: cluster data and store the data to json file
@ raw dataset: df_pos_with_state_clean.csv, clean_data_no_latlon.csv
@ data size: row_12240, size_898KB, row_40740, size_3.87MB
"""

import pandas as pd
import os
import json
import networkx as nx
import pickle

'''
read data from df_pos_with_state.xlsx
'''
current_dir = os.path.dirname(__file__)
'''
node, lat, lon, state, state_num
'''
nodes_path = os.path.abspath(os.path.join(current_dir, 'dataset2_df_nodes_v3.csv'))
nodes = pd.read_csv(nodes_path)
'''
read data from clean_data_no_latlon.xlsx
st_id, st_tg, train, train_max_speed
arr_time, dep_time, next_arr_time, stay_time, distance
travel_time, arr_time_t, dep_time_t, next_arr_time_t, travel_time_t
'''
edges_path = os.path.abspath(os.path.join(current_dir, 'dataset2_df_edges_v3.csv'))
edges = pd.read_csv(edges_path)

'''
get data from edges_path, create a new column 'index' to store the index of each edge
'''
edges['index'] = edges.index

'''
get all neighbors of each node
'''
node_dict = {}
for index, row in edges.iterrows():
    node1 = int(row['source'])
    node2 = int(row['target'])
    distance = row['distance']

    # if current node is not in the dict, add it
    if node1 not in node_dict:
        node_dict[node1] = {'neighbors': {}}

    # add the neighbor of current node
    node_dict[node1]['neighbors'][node2] = distance

    # do the same thing for node2
    if node2 not in node_dict:
        node_dict[node2] = {'neighbors': {}}

    node_dict[node2]['neighbors'][node1] = distance

neighbors_path = os.path.abspath(os.path.join(current_dir, 'paris_node_neighbors.json'))
# store the dict to a json file
with open(neighbors_path, 'w') as f:
    json.dump(node_dict, f)


# Create graph # direct graph # cannot return in plot route
G = nx.from_pandas_edgelist(edges, source='source', target='target',
                            edge_attr=['distance'])

# location of station
pos = {int(station): (longitude, latitude) for station, latitude, longitude in
       zip(nodes['node_id'], nodes['lat'], nodes['lon'])}

# Add node positions to node attributes
nx.set_node_attributes(G, pos, 'pos')


# store graph data
with open("parisgraph.pickle", "wb") as f:
    pickle.dump(G, f)