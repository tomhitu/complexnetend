"""
@ author: Yuqin Xia
@ date: 2023-4-11
@ description: for the railway map data find the shortest path
@ raw dataset: Paris data.xlsx
@ data size: nodes
"""

import pandas as pd
import os
import json
import networkx as nx
import pickle


nodes, neighbors = None, None


def init():
    print('init')
    global nodes, neighbors

    '''
    get the current path
    '''
    current_path = os.path.abspath(os.path.dirname(__file__))
    '''
    get the data path
    '''
    nodes_path = os.path.abspath(os.path.join(current_path, 'dataset2_df_nodes_v3.csv'))
    neighbors_path = os.path.abspath(os.path.join(current_path, 'paris_node_neighbors.json'))
    '''
    read the data
    '''
    nodes = pd.read_csv(nodes_path)
    # print(nodes.head())

    with open(neighbors_path, 'r') as f:
        neighbors = json.load(f)


def find_shortest_path_with_attr(G, source, target, edge_attr):
    # Find the shortest path between the source and target nodes
    shortest_path = nx.shortest_path(G, source=source, target=target, weight=edge_attr)

    # Compute the total travel distance and time along the shortest path
    travel_distance = sum(G[u][v][edge_attr] for u, v in zip(shortest_path[:-1], shortest_path[1:]))

    # Return the shortest path, travel distance, and travel time
    return shortest_path, travel_distance


def getG(snode, tnode):
    current_path = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.abspath(os.path.join(current_path, 'parisgraph.pickle'))
    with open(image_path, "rb") as f:
        G_loaded = pickle.load(f)

    source_node = int(snode)
    target_node = int(tnode)
    edge_attr = 'distance'
    shortest_path, travel_distance = find_shortest_path_with_attr(G_loaded, source_node, target_node,
                                                                               edge_attr=edge_attr)
    # print("Shortest ", edge_attr, " from node", source_node, "to node", target_node, ":", shortest_path)
    # print("Total travel distance:", travel_distance)
    return shortest_path, travel_distance


# check if the start node and end node are in the nodes
def is_in_nodes(start, end):
    # print(start, end)
    # print(type(start), type(end))
    # print(nodes['node_id'][0])
    start = int(start)
    end = int(end)
    if start in nodes['node_id'].values.astype(int) and end in nodes['node_id'].values.astype(int):
        return True
    else:
        return False


# get the distance between two nodes
def get_distance(start, end):
    start_lat = nodes[nodes['node_id'] == start]['lat'].values[0]
    start_lon = nodes[nodes['node_id'] == start]['lon'].values[0]
    end_lat = nodes[nodes['node_id'] == end]['lat'].values[0]
    end_lon = nodes[nodes['node_id'] == end]['lon'].values[0]
    distance = ((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5
    return distance


def list_insert(shortest_list, snode, enode):
    ntx_shortest_path, travel_distance = getG(snode, enode)
    ntx_shortest_path = [int(x) for x in ntx_shortest_path]
    list2 = []
    for step in shortest_list:
        list2.append(step['node'])

    dict1 = {elem: index for index, elem in enumerate(ntx_shortest_path)}
    dict2 = {elem: index for index, elem in enumerate(list2)}

    result_dict = {}

    for elem in ntx_shortest_path:
        if elem in list2:
            index_sum = dict1[elem] + dict2[elem]
            result_dict[index_sum] = elem

    result_list = [result_dict[key] for key in sorted(result_dict)]
    # print(result_list)

    # convert set to list
    # result_list = [x for x in ntx_shortest_path if x in list2]
    #
    # print('ntx_shortest_path', ntx_shortest_path)
    # print('shortest_list', list2)
    # print('result_list', result_list)

    if len(result_list) == 2:
        return shortest_list
    else:
        final_path = []
        shorti = 0
        for i in range(len(result_list)):
            # print('result_list[i]', result_list[i])
            if shortest_list[shorti]['node'] == result_list[i]:
                # print('same node', shortest_list[shorti]['node'])
                final_path.append({
                    'node': shortest_list[shorti]['node'],
                    'distance': shortest_list[shorti]['distance']
                })
            else:
                temp_distance = 0
                node1 = result_list[i - 1]
                temp_start_index = shorti
                # print('find different node', shortest_list[shorti]['node'])
                while shortest_list[shorti]['node'] != result_list[i]:
                    temp_distance += shortest_list[shorti]['distance']
                    shorti += 1
                # print('back to same node', shortest_list[shorti]['node'])
                node2 = shortest_list[shorti]['node']
                ntx_interval_path, travel_interval_distance = getG(node1, node2)
                ntx_interval_path = [int(x) for x in ntx_interval_path]
                if travel_interval_distance < temp_distance:
                    # print('shortest interval better')
                    for interval_step in ntx_interval_path:
                        final_path.append({
                            'node': interval_step,
                            'distance': 0
                        })
                    interval_temp_len = len(final_path)
                    final_path[interval_temp_len - 1]['distance'] = travel_interval_distance
                else:
                    # print('catch the next node')
                    # print('temp_start_index', temp_start_index)
                    # print('shorti', shorti)
                    while temp_start_index != shorti + 1:
                        final_path.append({
                            'node': shortest_list[temp_start_index]['node'],
                            'distance': shortest_list[temp_start_index]['distance']
                        })
                        temp_start_index += 1
                    # print('now node', shortest_list[shorti]['node'])
            shorti += 1
        return final_path


# if node has been visited
def is_visited(node, path):
    time = 0
    for p in path:
        if p['node'] == node:
            return True, time
        time += 1
    return False, 0


# get the shortest distance between two nodes
def get_shortest_distance(end, path, forbbiden_nodes):
    """
    :param end: target
    :param path: current path
    :return: path: find the shortest distance between current node and target
    """
    length = len(path)
    # find all the edges cross the current node
    # print('current_node', path[length - 1]['node'])
    # print('forbbiden_nodes', forbbiden_nodes)

    current_neighbors = neighbors[str(path[length - 1]['node'])]['neighbors']

    if len(current_neighbors) == 0:
        if length == 1:
            return None
        else:
            path[length - 2]['neighbors'].remove(path[length - 1]['node'])
            path.pop()
            return get_shortest_distance(end, path, forbbiden_nodes)

    path[length - 1]['neighbors'] = [int(key) for key in current_neighbors.keys()]

    current_closet_neighbor = -1
    current_next_distance = -1
    ccdistance = -1
    # get the key and value of dict
    for key, value in current_neighbors.items():
        key = int(key)
        if key == end:
            path.append({
                'state': 2,  # 2 means node has been the target
                'node': key,
                'distance': int(value)
            })
            return path
        value = float(value)
        cc_now_distance = get_distance(key, end)
        ifvisited, visited_index = is_visited(key, path)
        if key in forbbiden_nodes:
            continue
        if ifvisited:
            forbbiden_nodes.append(key)
            continue
        if current_closet_neighbor == -1:
            current_closet_neighbor = key
            current_next_distance = value
        else:
            if ccdistance == -1:
                ccdistance = cc_now_distance
            if ccdistance > cc_now_distance:
                current_closet_neighbor = key
                current_next_distance = value
                ccdistance = cc_now_distance

    if current_closet_neighbor != -1:
        ifvisited, visited_index = is_visited(current_closet_neighbor, path)

        if ifvisited:
            forbbiden_nodes.append(current_closet_neighbor)
            path[visited_index]['neighbors'].remove(path[visited_index+1]['node'])
            for step in range(0, length - visited_index):
                path.pop()
            # print('has visited ', current_closet_neighbor, 'add to forbbiden')
            # print('now path')
            # for step in path:
            #     print(step['node'])
            return get_shortest_distance(end, path, forbbiden_nodes)

        # print('this node is new ', current_closet_neighbor)
        path.append({
            'state': 1,  # 1 means node will continue to search
            'node': current_closet_neighbor,
            'distance': current_next_distance
        })

        return get_shortest_distance(end, path, forbbiden_nodes)

    else:
        # print('no neighbor can use')
        path[length - 3]['neighbors'].remove(path[length - 2]['node'])
        path.pop()
        path.pop()
        # print('last path is')
        # for step in path:
        #     print(step['node'])
        return get_shortest_distance(end, path, forbbiden_nodes)


# filter the path
def filter_path(dist_path):
    """
    :param path: current path
    :return: path: filter the path
    """
    shortest_route = []
    shortest_distance = 0
    if dist_path is None:
        shortest_distance = -1
    else:
        for step in dist_path:
            shortest_route.append(str(step['node']))
            shortest_distance += step['distance']
    shortest_distance = float(shortest_distance)
    print('shortest route: ', shortest_route)
    print('shortest distance: ', shortest_distance)
    return shortest_route, shortest_distance


# get the path between two nodes
def get_path(start, end):
    """
    :param start: current node
    :param end: end node
    :param arr_time: user arrival this station time
    :param gap_time: if user need change train but transfer time is less than gap_time, then transfer will be fail
    :description: get the path between two nodes
    :return: path
    """
    if is_in_nodes(start, end):
        start = int(start)
        end = int(end)
        beginnode = nodes[nodes['node_id'] == start]
        # len_after = len(ast.literal_eval(beginnode['after'].values[0]))
        len_after = 1
        # find no need to set distribution system
        if len_after == 0:
            print('no alter next step')
            return [{
                'state': -1
            }]
        else:
            forbbiden = [-1]
            shortest_distance = []
            pass2 = {
                'state': 0,  # 0 means the node is the first
                'node': start,
                'distance': 0,
            }
            shortest_distance.append(pass2)
            shortest_distance = get_shortest_distance(end, shortest_distance, forbbiden)
            compare_shortest_distance = list_insert(shortest_distance, start, end)
            return filter_path(compare_shortest_distance)
    else:
        print('The two nodes are not in the nodes')
        return None, None


if __name__ == '__main__':
    '''
    get the start node and end node
    '''
    init()
    get_path(9311, 4364)
