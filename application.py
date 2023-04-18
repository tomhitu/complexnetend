import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
import json

from src.assets import mapdraw
from src.assets import minpath
from src.assets import example_pred_edge

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

CORS(app, resources={r'/*': {'origins': '*'}})

base_url = 'data/'


# get data and then output
def get_data_by_keyword(keyword):
    df = pd.read_csv(base_url + 'animal-crossing-fish-info.csv')
    df['price'] = df['price'].astype(str)
    price = df[df['name'] == keyword]['price'].iloc[0]
    image = df[df['name'] == keyword]['image'].iloc[0].split('\t')[0] + '>'
    fish_info = {'price': str(price), 'image': image}

    return {'fish_info': fish_info}


# get data from json
def get_data_by_json(keyword):
    if keyword == 'all':
        with open('./data/npmdepgraph.min10.json', 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
            return fcc_data
    elif keyword == 'map':
        with open('./src/assets/railway_data.json', 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
            return fcc_data
    else:
        return {}


@app.route('/hidden_edges', methods=['GET'])
def hidden_edges():
    global hidden_datas

    if request.method == 'GET':
        try:
            keyword = request.args.get('keywords')

            hidden_datas = {
                'status': 0,
                'hiddennodes':[243, 1124, 394, 164, 2231, 2244, 144, 344, 2336, 2230],
                'hiddenedges':[
                {'source': 243, 'target': 1124},
                {'source': 394, 'target': 164},
                {'source': 2231, 'target': 2244},
                {'source': 144, 'target': 344},
                {'source': 2336, 'target': 2230},

            ]}

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return jsonify(hidden_datas)


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


@app.route('/shortest_path', methods=['POST'])
def shortest_path_generate():
    global data

    if request.method == 'POST':

        try:
            post_data = request.get_json()
            snode = post_data.get('startnode')
            enode = post_data.get('endnode')
            shortest_path, travel_distance, travel_time = mapdraw.getG(snode, enode)
            travel_distance = float(travel_distance)
            print(type(shortest_path))
            print(type(travel_distance))
            print(type(travel_time))
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
            data = {'status': 0, 'route': shortest_path, 'distance': travel_distance, 'time': travel_time, 'less_route': less_route, 'less_time': arr_time, 'less_distance': less_distance, 'shortest_route': shortest_route, 'shortest_distance': shortest_distance}

        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 1})

        else:
            return jsonify(data)


@app.route('/map_generate', methods=['GET'])
def map_generate():
    global node_data

    if request.method == 'GET':
        try:
            keyword = request.args.get('type')
            node_data = get_data_by_json(keyword)
            minpath.init()
            example_pred_edge.init()

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return jsonify(node_data)


@app.route('/data_generate', methods=['POST'])
def data_generate():
    global data

    if request.method == 'POST':

        try:
            post_data = request.get_json()
            keyword = post_data.get('search')
            data = get_data_by_keyword(keyword)
            message = {'status': 'success'}

        except Exception as e:
            traceback.print_exc()
            return jsonify({'status': 'fail'})

        else:
            return jsonify(message)


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
