"""
Created on 2023/04/18 20:20
"""
import pandas as pd
import json
import numpy


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


def getinfo():
    nodes_cluster = pd.read_csv('dataset2_df_node_cluster.csv')
    edges_cluster = pd.read_csv('dataset2_df_edge_cluster_light.csv')
    nodes = []
    # node_id
    # lon
    # lat
    # type
    # type_num
    # cluster_lat_lon
    # degree
    # Degree_Centrality
    # Clustering_Coefficients
    # Closeness_Centrality
    # Betweenness_Centrality
    # Eigenvector_Centrality
    # cluster_degree
    # cluster_Degree_Centrality
    # cluster_cluster_lat_lon
    # cluster_Clustering_Coefficients
    # cluster_Closeness_Centrality
    # cluster_Betweenness_Centrality
    # cluster_Eigenvector_Centrality
    for item, row in nodes_cluster.iterrows():
        node_temp = {}
        node_temp['name'] = row['node_id']
        node_temp['value'] = [row['lat'], row['lon']]
        node_temp['type'] = row['type']
        node_temp['type_num'] = row['type_num']
        node_temp['cluster_lat_lon'] = row['cluster_lat_lon']
        node_temp['degree'] = row['degree']
        node_temp['Degree_Centrality'] = row['Degree_Centrality']
        node_temp['Clustering_Coefficients'] = row['Clustering_Coefficients']
        node_temp['Closeness_Centrality'] = row['Closeness_Centrality']
        node_temp['Betweenness_Centrality'] = row['Betweenness_Centrality']
        node_temp['Eigenvector_Centrality'] = row['Eigenvector_Centrality']
        node_temp['cluster_degree'] = row['cluster_degree']
        node_temp['cluster_Degree_Centrality'] = row['cluster_Degree_Centrality']
        node_temp['cluster_Clustering_Coefficients'] = row['cluster_Clustering_Coefficients']
        node_temp['cluster_Closeness_Centrality'] = row['cluster_Closeness_Centrality']
        node_temp['cluster_Betweenness_Centrality'] = row['cluster_Betweenness_Centrality']
        node_temp['cluster_Eigenvector_Centrality'] = row['cluster_Eigenvector_Centrality']
        nodes.append(node_temp)

    edges = []
    # source
    # target
    # name
    # distance
    # type_num
    # node_type_num
    # node_tg_type_num
    # cluster_distance
    # cluster_type
    # cluster_across

    for item, row in edges_cluster.iterrows():
        edge_temp = {}
        edge_temp['source'] = row['source']
        edge_temp['target'] = row['target']
        coors = [nodes_cluster[nodes_cluster['node_id'] == row['source']]['lat'].values[0],
                 nodes_cluster[nodes_cluster['node_id'] == row['source']]['lon'].values[0]]
        coort = [nodes_cluster[nodes_cluster['node_id'] == row['target']]['lat'].values[0],
                 nodes_cluster[nodes_cluster['node_id'] == row['target']]['lon'].values[0]]
        edge_temp['value'] = [coors, coort]
        edge_temp['name'] = row['name']
        edge_temp['distance'] = row['distance']
        edge_temp['type_num'] = row['type_num']
        edge_temp['node_type_num'] = row['node_type_num']
        edge_temp['node_tg_type_num'] = row['node_tg_type_num']
        edge_temp['cluster_distance'] = row['cluster_distance']
        edge_temp['cluster_type'] = row['cluster_type']
        edge_temp['cluster_across'] = row['cluster_across']

        edges.append(edge_temp)

    a = {'nodes': nodes, 'edges': edges}
    b = json.dump(a, open('Paris_data_heavy.json', 'w'), cls=NumpyEncoder)


if __name__ == '__main__':
    getinfo()