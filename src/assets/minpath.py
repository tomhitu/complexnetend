"""
@ author: Yuqin Xia
@ date: 2023-4-11
@ description: for the railway map data find the shortest path
@ raw dataset: Railway Data_JL.xlsx
@ data size: row_69638, size_4470KB
"""

import pandas as pd
import os
import time
import ast
import json
import networkx as nx
import pickle


nodes, edges, neighbors = None, None, None


def init():
    print('init')
    global nodes, edges, neighbors

    '''
    get the current path
    '''
    current_path = os.path.abspath(os.path.dirname(__file__))
    '''
    get the data path
    '''
    nodes_path = os.path.abspath(os.path.join(current_path, '../../data/nodes_sorted.csv'))
    edges_path = os.path.abspath(os.path.join(current_path, '../../data/edges_sorted.csv'))
    neighbors_path = os.path.abspath(os.path.join(current_path, '../../data/node_neighbors.json'))
    '''
    read the data
    '''
    nodes = pd.read_csv(nodes_path)
    print(nodes.head())
    edges = pd.read_csv(edges_path)

    with open(neighbors_path, 'r') as f:
        neighbors = json.load(f)


def find_shortest_path_with_attr(G, source, target, edge_attr):
    # Find the shortest path between the source and target nodes
    shortest_path = nx.shortest_path(G, source=source, target=target, weight=edge_attr)

    # Compute the total travel distance and time along the shortest path
    travel_distance = sum(G[u][v][edge_attr] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    travel_time = sum(G[u][v]['travel_time'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))

    # Return the shortest path, travel distance, and travel time
    return shortest_path, travel_distance, travel_time


def getG(snode, tnode):
    current_path = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.abspath(os.path.join(current_path, 'graph.pickle'))
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


# check if the start node and end node are in the nodes
def is_in_nodes(start, end):
    # print(start, end)
    # print(type(start), type(end))
    # print(nodes['node'][0])
    if str(start) in nodes['node'].values.astype(str) and str(end) in nodes['node'].values.astype(str):
        return True
    else:
        return False


# check if the start node and end node are in the same state
def is_same_state(start, end):
    start_node = nodes[nodes['node'] == start]
    start_index = start_node['node'].values[0]
    start_state = start_node['state_num'].values[0]
    start_max = start_node['max_degree_node'].values[0]
    end_node = nodes[nodes['node'] == end]
    end_index = end_node['node'].values[0]
    end_state = end_node['state_num'].values[0]
    end_max = end_node['max_degree_node'].values[0]
    if start_state == end_state:
        return True, start_max, end_max
    elif start_index == start_max and end_index == end_max:
        return True, start_max, end_max
    else:
        return False, start_max, end_max


# get the distance between two nodes
def get_distance(start, end):
    start_lat = nodes[nodes['node'] == start]['lat'].values[0]
    start_lon = nodes[nodes['node'] == start]['lon'].values[0]
    end_lat = nodes[nodes['node'] == end]['lat'].values[0]
    end_lon = nodes[nodes['node'] == end]['lon'].values[0]
    distance = ((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5
    return distance


def list_insert(shortest_list, snode, enode):
    ntx_shortest_path, travel_distance, travel_time = getG(snode, enode)
    ntx_shortest_path = [int(x) for x in ntx_shortest_path]
    list2 = []
    for step in shortest_list:
        list2.append(step['node'])

    # convert set to list
    result_list = [x for x in ntx_shortest_path if x in list2]

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
                ntx_interval_path, travel_interval_distance, travel_interval_time = getG(node1, node2)
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



# if add day, then the time will be 24:00:00
def add_day(day, dep_time, arr_time):
    if arr_time - dep_time > 1:
        day += 1
    dep_time = dep_time % 1
    arr_time = arr_time % 1
    if arr_time < dep_time:
        day += 1
    return day, arr_time


# if node has been visited
def is_visited(node, path):
    time = 0
    for p in path:
        if p['node'] == node:
            return True, time
        time += 1
    return False, 0


# get the fewer transfer stations between two nodes
def get_fewer_transfer(end, gap_time, path):
    """
    :param end: target
    :param gap_time: if user need change train but transfer time is less than gap_time, then transfer will be fail
    :param path: current path
    :description: get the fewer transfer stations between two nodes
    :return: path
    """
    length = len(path)
    allow_time = gap_time

    if length == 1:
        allow_time = 0

    alter_nodes = []

    if len(path[length - 1]['after']) == 0:
        if length == 1:
            return None
        else:
            # delete the path[length - 2]['after'] which is the current node
            path[length - 2]['after'].remove(path[length - 1]['edge_index'])
            # delete the last after node
            path.pop()
            return get_fewer_transfer(end, gap_time, path)

    after_nodes = path[length - 1]['after']
    for edge_index in after_nodes:
        current_index = edge_index
        edge = edges[edges['index'] == edge_index]
        dep_time = edge['dep_time'].values[0]

        day = path[length - 1]['day']

        # user has to be at the station before the train departs with allow_time
        if path[length - 1]['arr_time'] > dep_time % 1 - allow_time:
            day += 1

        if edge['st_tg'].values[0] == end:
            next_arr_time = edge['next_arr_time'].values[0]
            day, next_arr_time = add_day(day, dep_time, next_arr_time)
            alter_nodes.append({
                'edge_index': edge_index,
                'arr_index': edge_index,
                'node': end,
                'day': day,
                'arr_time': next_arr_time
            })
        else:
            current_index += 1
            while edges['train'][current_index] == edges['train'][current_index - 1]:
                if edges['st_tg'][current_index] == end:
                    next_arr_time = edges['next_arr_time'][current_index]
                    day, next_arr_time = add_day(day, dep_time, next_arr_time)
                    alter_nodes.append({
                        'edge_index': edge_index,
                        'arr_index': current_index,
                        'node': end,
                        'day': day,
                        'arr_time': next_arr_time,
                    })
                    break
                current_index += 1

    if len(alter_nodes) == 0:
        alter_closer_nodes = []
        for edge_index in path[length - 1]['after']:
            current_index = edge_index
            current_closet_index = current_index
            closet_distance = get_distance(end, edges['st_tg'][current_closet_index])
            current_index += 1
            while edges['train'][current_index] == edges['train'][current_index - 1]:
                temp_distance = get_distance(end, edges['st_tg'][current_index])
                if temp_distance < closet_distance:
                    closet_distance = temp_distance
                    current_closet_index = current_index
                current_index += 1
            alter_closer_nodes.append({
                'edge_index': edge_index,
                'arr_index': current_closet_index,
                'node': edges['st_tg'][current_closet_index],
                'distance': edges['distance'][current_closet_index],
            })

        closet_current_distance = get_distance(end, alter_closer_nodes[0]['node'])

        closet_current_index = 0
        if len(alter_closer_nodes) > 1:
            for step in alter_closer_nodes:
                temp_distance = get_distance(end, step['node'])
                if temp_distance < closet_current_distance:
                    closet_current_distance = temp_distance
                    closet_current_index = alter_closer_nodes.index(step)
        closet_current = alter_closer_nodes[closet_current_index]
        for step in range(closet_current['edge_index'], closet_current['arr_index'] + 1):
            length = len(path)
            day = path[length - 1]['day']
            dep_time = edges['dep_time'][step]
            if path[length - 1]['arr_time'] > dep_time % 1 - allow_time:
                day += 1
            next_arr_time = edges['next_arr_time'][step]
            day, next_arr_time = add_day(day, dep_time, next_arr_time)
            path.append({
                'state': 1,  # 1 means node will continue to search
                'node': edges['st_tg'][step],
                'day': day,
                'arr_time': next_arr_time,
                'train': edges['train'][step],
                'edge_index': step,
                'distance': edges['distance'][step],
                'after': ast.literal_eval(nodes[nodes['node'] == edges['st_tg'][step]]['after'].values[0]),
            })

        return get_fewer_transfer(end, gap_time, path)

    else:
        closet = alter_nodes[0]['arr_time'] + alter_nodes[0]['day']
        closet_index = 0
        closet_arr_index = 0
        for step in alter_nodes:
            if step['arr_time'] + step['day'] < closet:
                closet = step['arr_time'] + step['day']
                closet_index = step['edge_index']
                closet_arr_index = step['arr_index']
        for step in range(closet_index, closet_arr_index + 1):
            length = len(path)
            day = path[length - 1]['day']
            dep_time = edges['dep_time'][step]
            if path[length - 1]['arr_time'] > dep_time % 1 - allow_time:
                day += 1
            next_arr_time = edges['next_arr_time'][step]
            day, next_arr_time = add_day(day, dep_time, next_arr_time)
            path.append({
                'state': 2,  # 2 means node has been the target
                'node': edges['st_tg'][step],
                'day': day,
                'arr_time': next_arr_time,
                'train': edges['train'][step],
                'edge_index': step,
                'distance': edges['distance'][step],
                'after': None,
            })
        return path


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
def filter_path(less_path, dist_path):
    """
    :param path: current path
    :return: path: filter the path
    """
    less_route = []
    less_time = []
    less_distance = 0
    shortest_route = []
    shortest_distance = 0
    if less_path is None:
        less_distance = -1
    else:
        for step in less_path:
            less_route.append(str(step['node']))
            less_time = step['arr_time']
            less_distance += step['distance']
    if dist_path is None:
        shortest_distance = -1
    else:
        for step in dist_path:
            shortest_route.append(str(step['node']))
            shortest_distance += step['distance']
    less_distance = float(less_distance)
    shortest_distance = float(shortest_distance)
    print('less route: ', less_route)
    print('less time: ', less_time)
    print('less distance: ', less_distance)
    print('shortest route: ', shortest_route)
    print('shortest distance: ', shortest_distance)
    return less_route, less_time, less_distance, shortest_route, shortest_distance


# get the path between two nodes
def get_path(start, end, gap_time):
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
        beginnode = nodes[nodes['node'] == start]
        len_after = len(ast.literal_eval(beginnode['after'].values[0]))
        check_state, max_start, max_end = is_same_state(start, end)
        # find no need to set distribution system
        check_state = True
        if len_after == 0:
            print('no alter next step')
            return [{
                'state': -1,  # -1 means the node has none alter next step
                # 'node': start,
                # 'day': 0,
                # 'arr_time': arr_time,
                # 'train': None,
                # 'edge_index': None,
                # 'after': None,
                # 'distance': 0
            }]
        elif check_state:
            current_date = time.strftime('%H:%M:%S', time.localtime(time.time()))
            current_time = float(current_date[0:2])/24 + float(current_date[3:5])/60/24 + float(current_date[6:8])/60/60/24
            less_transfer = []
            forbbiden = [-1]
            pass1 = {
                'state': 0,  # 0 means the node is the first
                'node': start,
                'day': 0,
                'arr_time': current_time,
                'train': None,
                'edge_index': None,
                'after': ast.literal_eval(beginnode['after'].values[0]),
                'distance': 0
            }
            less_transfer.append(pass1)
            shortest_distance = []
            pass2 = {
                'state': 0,  # 0 means the node is the first
                'node': start,
                'distance': 0,
            }
            shortest_distance.append(pass2)
            less_transfer = get_fewer_transfer(end, gap_time, less_transfer)
            shortest_distance = get_shortest_distance(end, shortest_distance, forbbiden)
            # if less_transfer is None:
            #     print('There is no path between the two nodes')
            # else:
            #     print(less_transfer)
            #     for edge in less_transfer:
            #         print(edge['node'], edge['arr_time'], edge['train'], edge['distance'])
            # if shortest_distance is None:
            #     print('There is no path between the two nodes')
            # else:
            #     final_distance = 0
                # print(shortest_distance)
                # for edge in shortest_distance:
                #     final_distance += edge['distance']
                #     print(edge['node'], edge['distance'])
                # print('final distance', final_distance)
            # return shortest_distance
            compare_shortest_distance = list_insert(shortest_distance, start, end)
            return filter_path(less_transfer, compare_shortest_distance)
        else:
            first_path = get_path(start, max_start, gap_time)
            # print(get_path(start, max_start, gap_time))
            second_path = get_path(max_start, max_end, gap_time)
            # print(get_path(max_start, max_end, gap_time))
            third_path = get_path(max_end, end, gap_time)
            # print(get_path(max_end, end, gap_time))
            if first_path is None or second_path is None or third_path is None:
                print('There is no path between the two nodes')
                return None, None
            else:
                second_path.pop(0)
                third_path.pop(0)
                final_path = first_path + second_path + third_path
                print(final_path)
                final_dist = 0
                for step in final_path:
                    final_dist += step['distance']

                print('final distance', final_dist)
            # print('The two nodes are not in the same state')
            # return None, None
    else:
        print('The two nodes are not in the nodes')
        return None, None


if __name__ == '__main__':
    '''
    get the start node and end node
    '''
    # start_node = input('Please input the start node: ')
    # start_node = int(start_node)
    # end_node = input('Please input the end node: ')
    # end_node = int(end_node)
    # get current Beijing time by system
    # 7/24h + 30/60/24min = 0.3125
    transfer_time = 30/60/24
    init()
    get_path(1684, 2197, transfer_time)
