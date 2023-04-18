"""
@ author: Yuqin Xia
@ date: 2023-4-11
@ description: cluster data and store the data to json file
@ raw dataset: df_pos_with_state_clean.csv, clean_data_no_latlon.csv
@ data size: row_2720, size_110KB, row_64150, size_6298KB
"""

import pandas as pd
import os
import networkx as nx
import pickle
import json

'''
read data from df_pos_with_state.xlsx
'''
current_dir = os.path.dirname(__file__)
'''
node, lat, lon, state, state_num
'''
nodes_path = os.path.abspath(os.path.join(current_dir, '../../data/df_pos_with_state_clean.csv'))
nodes = pd.read_csv(nodes_path)
'''
read data from clean_data_no_latlon.xlsx
st_id, st_tg, train, train_max_speed
arr_time, dep_time, next_arr_time, stay_time, distance
travel_time, arr_time_t, dep_time_t, next_arr_time_t, travel_time_t
'''
edges_path = os.path.abspath(os.path.join(current_dir, '../../data/clean_data_no_latlon.csv'))
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
    node1 = row['st_id']
    node2 = row['st_tg']
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

neighbors_path = os.path.abspath(os.path.join(current_dir, '../../data/node_neighbors.json'))
# store the dict to a json file
with open(neighbors_path, 'w') as f:
    json.dump(node_dict, f)

'''
filter the edges by node,
node will have two new subscribe 'before' and 'after'
which store all edges source is itself and target is itself respectively
'''
#  nodes['before'] = nodes['node'].apply(lambda x: edges[edges['st_tg'] == x]['index'].tolist())
nodes['after'] = nodes['node'].apply(lambda x: edges[edges['st_id'] == x]['index'].tolist())

'''
use edges value to calculate the degree of each node
'''
nodes['degree'] = nodes['node'].apply(lambda x: len(edges[edges['st_id'] == x]) + len(edges[edges['st_tg'] == x]))

'''
get the biggest degree node in each state
'''
nodes['max_degree_node'] = nodes['state_num'].apply(lambda x: nodes[(nodes['state_num'] == x) & (nodes['degree'] == nodes[nodes['state_num'] == x]['degree'].max())]['node'].tolist())
nodes['max_degree_node'] = nodes['max_degree_node'].apply(lambda x: x[0] if len(x) > 0 else -1)


'''
get betweenness centrality of each node
'''


with open('graph.pickle', 'rb') as f:
    G = pickle.load(f)

betweenness_centrality = nx.betweenness_centrality(G)

'''
add new column 'betweenness_centrality' to nodes
'''
nodes['betweenness_centrality'] = nodes['node'].apply(lambda x: betweenness_centrality[x])

# print(betweenness_centrality)

'''
get closeness centrality of each node
'''
closeness_centrality = nx.closeness_centrality(G)
'''
add new column 'closeness_centrality' to nodes
'''
nodes['closeness_centrality'] = nodes['node'].apply(lambda x: closeness_centrality[x])

# print(closeness_centrality)

# print the first 3 rows of edges
# print(edges.head(3))

'''
sort nodes by state_num with decreasing order
'''
nodes_sorted = nodes.sort_values(by=["state_num", "degree"], ascending=[False, False])

'''
save nodes_sorted to nodes_sorted.csv
'''
nodes_sorted_path = os.path.abspath(os.path.join(current_dir, '../../data/nodes_sorted.csv'))
edges_sorted_path = os.path.abspath(os.path.join(current_dir, '../../data/edges_sorted.csv'))
nodes_sorted.to_csv(nodes_sorted_path, index=False, mode='w')
edges.to_csv(edges_sorted_path, index=False, mode='w')
