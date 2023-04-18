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

nodes, edges, neighbors = None, None, None


def init():
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
    edges = pd.read_csv(edges_path)

    with open(neighbors_path, 'r') as f:
        neighbors = json.load(f)


# check if the start node and end node are in the nodes
def is_in_nodes(start, end):
    if str(start) in nodes['node'].values.astype(str) and str(end) in nodes['node'].values.astype(str):
        return True
    else:
        return False


# check if the start node and end node are in the same state
def is_same_state(start, end):
    start_state = nodes[nodes['node'] == start]['state_num'].values[0]
    end_state = nodes[nodes['node'] == end]['state_num'].values[0]
    if start_state == end_state:
        return True
    else:
        return True


# get the distance between two nodes
def get_distance(start, end):
    start_lat = nodes[nodes['node'] == start]['lat'].values[0]
    start_lon = nodes[nodes['node'] == start]['lon'].values[0]
    end_lat = nodes[nodes['node'] == end]['lat'].values[0]
    end_lon = nodes[nodes['node'] == end]['lon'].values[0]
    distance = ((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5
    return distance


# if add day, then the time will be 24:00:00
def add_day(day, dep_time, arr_time):
    if arr_time - dep_time > 1:
        day += 1
    dep_time = dep_time % 1
    arr_time = arr_time % 1
    if arr_time < dep_time:
        day += 1
    return day, arr_time


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
def get_shortest_distance(end, path):
    """
    :param end: target
    :param path: current path
    :return: path: find the shortest distance between current node and target
    """
    length = len(path)
    # find all the edges cross the current node

    current_neighbors = neighbors[str(path[length - 1]['node'])]['neighbors']

    if len(current_neighbors) == 0:
        if length == 1:
            return None
        else:
            path[length - 2]['forbidden'].append(path[length - 1]['node'])
            path.pop()
            return get_shortest_distance(end, path)

    current_closet_neighbor = -1
    current_next_distance = -1
    # get the key and value of dict
    for key, value in current_neighbors.items():
        key = int(key)
        if key == end:
            path.append({
                'state': 2,  # 2 means node has been the target
                'node': key,
                'distance': int(value),
                'forbidden': [],
            })
            return path
        value = float(value)
        if current_closet_neighbor == -1:
            current_closet_neighbor = key
            current_next_distance = value
        elif key not in path[length - 1]['forbidden']:
            if get_distance(current_closet_neighbor, end) > get_distance(key, end):
                current_closet_neighbor = key
                current_next_distance = value

    path.append({
        'state': 1,  # 1 means node will continue to search
        'node': current_closet_neighbor,
        'distance': current_next_distance,
        'forbidden': [],
    })

    return get_shortest_distance(end, path)


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
        elif is_same_state(start, end):
            current_date = time.strftime('%H:%M:%S', time.localtime(time.time()))
            current_time = float(current_date[0:2])/24 + float(current_date[3:5])/60/24 + float(current_date[6:8])/60/60/24
            less_transfer = []
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
                'forbidden': []
            }
            shortest_distance.append(pass2)
            less_transfer = get_fewer_transfer(end, gap_time, less_transfer)
            shortest_distance = get_shortest_distance(end, shortest_distance)
            # if less_transfer is None:
            #     print('There is no path between the two nodes')
            # else:
            #     print(less_transfer)
                # for edge in less_transfer:
                #     print(edge['node'], edge['arr_time'], edge['train'], edge['distance'])
            # if shortest_distance is None:
            #     print('There is no path between the two nodes')
            # else:
            #     print(shortest_distance)
                # for edge in shortest_distance:
                #     print(edge['node'], edge['distance'])
            return filter_path(less_transfer, shortest_distance)
        else:
            print('The two nodes are not in the same state')
            return None, None
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
    get_path(947, 1948, transfer_time)
