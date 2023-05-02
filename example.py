import __init__ as tbox
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import numpy
import pickle
import matplotlib.pyplot as plt


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


"""
Get your current file path.
"""
current_path = os.path.abspath(os.path.dirname(__file__))


def getdata():
    """
    Read your data
    """
    china_path = os.path.abspath(os.path.join(current_path, 'data/Railway Data_JL.xlsx'))
    # paris_path = os.path.abspath(os.path.join(current_path, 'data/Paris_data_heavy.json'))
    data = pd.read_excel(china_path)  # read
    return data


def testcleandata(save_path, ifsave=True):
    data = getdata()  # read
    df, pos = tbox.cleandata(data)  # get clean data

    # print
    print(f'len_df: {len(data)}, len_pos: {len(pos)}')
    print(f'df: \n{data.head(2)}')
    print(f'pos:')
    count = 2
    for key, value in pos.items():
        if count == 0:
            break
        print(key, value)
        count -= 1
    if ifsave:
        # save into csv
        df.to_csv(save_path, index=False)


def testclusterpos(save_path, ifsave=True):
    data = getdata()  # read
    _, pos = tbox.cleandata(data)  # get clean data
    df_pos = tbox.clustercoord(pos, False)  # cluster pos
    print(df_pos.head(2))

    if ifsave:
        # save into csv
        df_pos.to_csv(save_path, index=False)


def testmapprovince(read_path, save_path, ifsave=True):
    """
    Read your data
    """
    df_pos_with_state_path = os.path.abspath(os.path.join(current_path, read_path))
    df_pos_with_state = pd.read_csv(df_pos_with_state_path)
    df_pos_with_state_clean2 = tbox.mapprovince(df_pos_with_state)  # clean pos
    print(df_pos_with_state_clean2.head(2))

    if ifsave:
        # save into csv
        df_pos_with_state_clean2.to_csv(save_path, index=False)


def testpredictedges(lon, lat):
    n_degree, new_neighbor_node, new_edges_distance, new_edges_travel_time, new_edges_train_speed, new_node_id = tbox.predictedges(lon, lat)
    print(f'n_degree: {n_degree}')
    print(f'new_neighbor_node: {new_neighbor_node}')
    print(f'new_edges_distance: {new_edges_distance}')
    print(f'new_edges_travel_time: {new_edges_travel_time}')
    print(f'new_edges_train_speed: {new_edges_train_speed}')
    print(f'new_node_id: {new_node_id}')


def testtrainnewmodel(read_df_path, read_pos_path, save_degree_folder_name, save_edge_folder_name, attr_dim):
    """
    Read your data
    """
    df_path = os.path.abspath(os.path.join(current_path, read_df_path))
    pos_path = os.path.abspath(os.path.join(current_path, read_pos_path))
    df_data = pd.read_csv(df_path)
    required_cols = ['st_id', 'st_tg', 'train_max_speed', 'distance', 'travel_time']
    if not set(required_cols).issubset(df_data.columns):
        raise Exception(f'Please make sure that your data has all required cols: {required_cols}')
    print("df_data format is correct")
    df_data = df_data[['st_id', 'st_tg', 'train_max_speed', 'distance', 'travel_time']]  # keep only needed(want) cols
    df_pos = pd.read_csv(pos_path)
    required_cols = ['node', 'lat', 'lon', 'state_num']
    if not set(required_cols).issubset(df_pos.columns):
        raise Exception(f'Please make sure that your data has all required cols: {required_cols}')
    df_pos = df_pos[['node', 'lat', 'lon', 'state_num']]  # keep only needed(want) cols
    print("df_pos format is correct")

    # norm labels because it is in different range scale
    scaler_labels = MinMaxScaler()
    df_data.iloc[:, 2:5] = scaler_labels.fit_transform(df_data.iloc[:, 2:5])
    n_degree, df_new_data, df_new_pos, new_st_id, neighbor_node = tbox.trainnewmodel(df_data, df_pos, save_degree_folder_name, save_edge_folder_name, attr_dim)  # train new model
    print(f'n_degree: {n_degree}')
    print(f'df_new_data: \n{df_new_data.head(2)}')
    print(f'df_new_pos: \n{df_new_pos.head(2)}')
    print(f'new_st_id: {new_st_id}')
    print(f'neighbor_node: {neighbor_node}')



def testsetjson(node_path, edge_path, savepatha, savepathb, savepathc, ifsave=True):
    nodes_cluster = pd.read_csv(node_path, encoding='ISO-8859-1')
    edges_cluster = pd.read_csv(edge_path)
    nodes, edges, maxclusternum, maxedgenum = tbox.setjson(nodes_cluster, edges_cluster)

    if ifsave:
        a = {'nodes': nodes, 'edges': edges}
        b = json.dump(a, open(savepatha, 'w'), cls=NumpyEncoder)
        c = json.dump(maxclusternum, open(savepathb, 'w'), cls=NumpyEncoder)
        d = json.dump(maxedgenum, open(savepathc, 'w'), cls=NumpyEncoder)


def testdeletenode(node_id, map_path):
    image_path = os.path.abspath(os.path.join(current_path, map_path))
    with open(image_path, "rb") as f:
        G_loaded = pickle.load(f)
    before, after = tbox.deletenode(node_id, G_loaded)
    print(f'before: {before}')
    print(f'after: {after}')


def testshortestway(source, target, map_path):
    image_path = os.path.abspath(os.path.join(current_path, map_path))
    with open(image_path, "rb") as f:
        G_loaded = pickle.load(f)
    path, distance = tbox.shortestway(source, target, G_loaded)
    print(f'path: {path}')
    print(f'distance: {distance}')


def testresiliencedata(path, graphic_path, node_path, edge_path, ifsave=True):
    data = pd.read_excel(path)  # read
    G, df_node, df_edge = tbox.resiliencedata(data)
    print(df_node.head(2))
    print(df_edge.head(2))

# --------------------------get the node and edge for node properties and GCN model ------------------------------------
    #get the dataset1_RL_ data for CGN model
    df_node = tbox.resilienceinfo(G, df_node)
    if ifsave:
        with open(graphic_path, "wb") as f:
            pickle.dump(G, f)
        df_node.to_csv(node_path, index=False)
        df_edge.to_csv(edge_path, index=False)


def testnetproperties(G_path):
    with(open(G_path, "rb")) as f:
        G = pickle.load(f)
    num_nodes, num_edges, num_connected_components, Max_components_node, Max_components_edges, k_cores, density, diameter, avg_distance, efficiency = tbox.netproperties(
        G)
    print('node number：', num_nodes)
    print('edge number：', num_edges)
    print('The number of connected graphs:', num_connected_components)
    print('The number of nodes of the largest connected component：', Max_components_node)
    print('The number of edges of the largest connected component：', Max_components_edges)
    print('The k-core of the largest connected component：', k_cores)
    print('The network density of the largest connected component：', density)
    print('Diameter of the largest connected component：', diameter)
    print('Average distance of the largest connected component：', avg_distance)
    print('The efficiency of the largest connected component：', efficiency)


def testgetattackrate(G_path, attackcsv, attackjson, ifsave=True, ifshow=False):
    with(open(G_path, "rb")) as f:
        G = pickle.load(f)
    Attack_Ratio, relative_size, relative_size_deg, relative_size_betw, relative_size_kshell, relative_size_ci = tbox.getattackrate(
        G)

    if ifsave:
        # save csv
        save = pd.DataFrame(
            {'Attack_Ratio': Attack_Ratio, 'relative_size': relative_size, 'relative_size_deg': relative_size_deg,
             'relative_size_betw': relative_size_betw, 'relative_size_kshell': relative_size_kshell,
             'relative_size_ci': relative_size_ci})
        save.to_csv(attackcsv, index=False)

        Attack_Ratio = Attack_Ratio.tolist()

        data = {"Attack_Ratio": Attack_Ratio, "relative_size": relative_size, "relative_size_deg": relative_size_deg,
                "relative_size_betw": relative_size_betw, "relative_size_kshell": relative_size_kshell,
                "relative_size_ci": relative_size_ci}

        with open(attackjson, 'w') as f:
            json.dump(data, f)

    if ifshow:
        plt.figure(figsize=(10, 8))
        plt.plot(Attack_Ratio, relative_size, 'bo-', label='Random Attack')
        plt.plot(Attack_Ratio, relative_size_deg, 'ro-', label='Degree Attack')
        plt.plot(Attack_Ratio, relative_size_betw, 'go-', label='Betweenness Attack')
        plt.plot(Attack_Ratio, relative_size_kshell, 'yo-', label='Kshell Attack')
        plt.plot(Attack_Ratio, relative_size_ci, 'ko-', label='Collective Influence Attack')
        plt.title("Relative Size of Largest Connected Component")
        plt.ylabel("Relative Size")
        plt.xlabel("Attack Ratio")
        plt.legend()
        plt.show()


def testdegreecount(G_path, degreejson, ifsave=True, ifshow=False):
    with(open(G_path, "rb")) as f:
        G = pickle.load(f)
    degree_list, count_list = tbox.degreecount(G)
    print(f'degree_list: {degree_list}')
    print(f'count_list: {count_list}')
    if ifsave:
        data = {"degree_list": degree_list, "count_list": count_list}
        with open(degreejson, 'w') as f:
            json.dump(data, f)

    if ifshow:
        plt.plot(degree_list, count_list, 'bo-')
        plt.title("Degree-Count Relationship")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        plt.show()


def testdegreedistribution(G_path, degreejson, ifsave=True, ifshow=False):
    with(open(G_path, "rb")) as f:
        G = pickle.load(f)
    degree_count, degree_distribution = tbox.degreedistribution(G)
    print(f'degree_count: {degree_count}')
    print(f'degree_distribution: {degree_distribution}')

    if ifsave:
        degree_count = list(degree_count)
        data = {"degree_count": degree_count, "degree_distribution": degree_distribution}
        with open(degreejson, 'w') as f:
            json.dump(data, f)

    if ifshow:
        plt.bar(degree_count, degree_distribution)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Fraction of Nodes")
        plt.show()

        plt.scatter(degree_count, degree_distribution)
        # 将横坐标和纵坐标设置为对数比例
        plt.xscale('log')
        plt.yscale('log')


def testgethidden(df_node_path, df_edge_path, lr, iftrain, foldername):
    df_node = pd.read_csv(df_node_path)
    df_edge = pd.read_csv(df_edge_path)
    hidden_edge = tbox.gethidden(df_node, df_edge, lr, iftrain, foldername)
    print(hidden_edge)


