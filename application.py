import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
import json

from src.assets import minpath
from src.assets import example_pred_edge
from src.dataset2 import dataset2_example_pred_edge

from src import dataresilience
from src.assets import NetworkResilience

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

CORS(app, resources={r'/*': {'origins': '*'}})

base_url = 'data/'


# get data from json
def get_data_by_json(keyword):
    if keyword == 'map':
        with open('./src/assets/railway_data.json', 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
            return fcc_data
    elif keyword == 'paris':
        with open('./src/dataset2/Paris_data_heavy.json', 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
            return fcc_data
    else:
        return {}


@app.route('/')
def hello_world():
    return 'Hello from Flask!'


@app.route('/hidden_edges', methods=['GET'])
def hidden_edges():
    global hidden_datas

    if request.method == 'GET':
        try:
            keyword = request.args.get('keywords')
            hiddennodes, hiddenedges = NetworkResilience.gethidden(0)

            hiddennodes = [int(d) for d in hiddennodes]
            hiddenedges = [(int(d[0]), int(d[1])) for d in hiddenedges]

            print(hiddennodes)
            print(hiddenedges)

            hidden_datas = {
                'status': 0,
                'hiddennodes': hiddennodes,
                'hiddenedges': hiddenedges
            }

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return jsonify(hidden_datas)


@app.route('/attack_network', methods=['GET'])
def attack_network():
    global attack_datas
    if request.method == 'GET':
        try:
            attack_china, attack_paris = dataresilience.get_attack_json()
            attack_datas = {
                'status': 0,
                'attack_china': attack_china,
                'attack_paris': attack_paris
            }
        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 1})

        return jsonify(attack_datas)


@app.route('/pred_edges', methods=['POST'])
def predict_edges():
    global pred_edge

    if request.method == 'POST':

        try:
            post_data = request.get_json()
            lon = post_data.get('longitude')
            lat = post_data.get('latitude')

            n_degree, new_neighbor_node, new_edges_distance, new_edges_travel_time, new_edges_train_speed, new_node_id = example_pred_edge.predEdges(lon, lat)

            pred_edge = {'status': 0,
                         'n_degree': n_degree,
                         'new_neighbor_node': new_neighbor_node,
                         'new_edges_distance': new_edges_distance,
                         'new_edges_travel_time': new_edges_travel_time,
                         'new_edges_train_speed': new_edges_train_speed,
                         'new_node_id': new_node_id}

        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 1})

        else:
            return jsonify(pred_edge)

@app.route('/pred_paris_edges', methods=['POST'])
def predict_paris_edges():
    global pred_paris_edge

    if request.method == 'POST':

        try:
            post_data = request.get_json()
            lon = post_data.get('longitude')
            lat = post_data.get('latitude')

            n_degree, new_neighbor_node, new_edges_distance, new_edges_type_num = dataset2_example_pred_edge.main(lat, lon)

            pred_paris_edge = {'status': 0,
                         'n_degree': n_degree,
                         'new_neighbor_node': new_neighbor_node,
                         'new_edges_distance': new_edges_distance,
                         'new_edges_type_num': new_edges_type_num}
        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 1})

        else:
            return jsonify(pred_paris_edge)


@app.route('/delete_nodes', methods=['POST'])
def delete_nodes():
    global delete_data

    if request.method == 'POST':

        try:
            post_data = request.get_json()
            node_id = post_data.get('nodeid')
            delete_map_type = post_data.get('type')
            before, after = NetworkResilience.delete_node(node_id, delete_map_type)
            delete_data = {'status': 0,
                           'beforedel': before,
                           'afterdel': after}

            print(delete_data)

        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 1})

        else:
            return jsonify(delete_data)


@app.route('/shortest_path', methods=['POST'])
def shortest_path_generate():
    global data

    if request.method == 'POST':

        try:
            post_data = request.get_json()
            snode = post_data.get('startnode')
            enode = post_data.get('endnode')
            transfer_time = 30/60/24
            less_route, less_time, less_distance, shortest_route, shortest_distance = minpath.get_path(snode, enode, transfer_time)
            print(type(less_route))
            print(type(less_time))

            less_h = int(less_time * 7)
            less_time = less_time * 7 - less_h
            less_m = int(less_time * 60)
            less_time = less_time * 60 - less_m
            less_s = 0 + int(less_time * 60)
            less_h_s = ''
            less_m_s = ''
            less_s_s = ''
            if less_h < 10:
                less_h_s = '0' + str(less_h)
            if less_m < 10:
                less_m_s = '0' + str(less_m)
            if less_s < 10:
                less_s_s = '0' + str(less_s)
            arr_time = less_h_s + ':' + less_m_s + ':' + less_s_s
            print(type(less_distance))
            print(type(shortest_route))
            print(type(shortest_distance))
            data = {'status': 0, 'less_route': less_route, 'less_time': arr_time, 'less_distance': less_distance, 'shortest_route': shortest_route, 'shortest_distance': shortest_distance}

        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 1})

        else:
            return jsonify(data)


@app.route('/map_generate', methods=['GET'])
def map_generate():
    global node_all_data

    if request.method == 'GET':
        try:
            keyword = request.args.get('type')
            node_data = get_data_by_json(keyword)
            minpath.init()
            example_pred_edge.init()
            NetworkResilience.init()

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return jsonify(node_data)


@app.route('/paris_map_generate', methods=['GET'])
def paris_map_generate():
    global paris_node_data

    if request.method == 'GET':
        try:
            keyword = request.args.get('type')
            paris_node_data = get_data_by_json(keyword)

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return jsonify(paris_node_data)


@app.route('/get_data', methods=['GET'])
def get_data():
    global data
    if request.method == 'GET':
        try:
            arg = request.args.get('name')
            response_data = data.get(arg)

            return jsonify(response_data)

        except Exception as e:
            traceback.print_exc()

            return None


if __name__ == '__main__':
    app.run(port=8000)
