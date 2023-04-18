from pre_proc_data import chinese_railway_prep
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np


# get the edge list
def get_edge(df_org):
    edge_df = df_org[['st_id','st_tg']]
    df_sort = pd.concat([edge_df['st_id'], edge_df['st_tg']], axis=0, ignore_index=True)
    df_sort = pd.DataFrame(df_sort, columns=['data'])
    df_sort = df_sort.drop_duplicates(subset='data')
    df_sort = df_sort.sort_values(by='data')
    df_sort['st_num'] = range(0, len(df_sort))
    df_sort = df_sort.set_index(df_sort.columns[0], drop=True)
    df_sort.index.name = None
    df_sort1 = df_sort.copy()
    df_sort1.columns = ['st_id_num']
    df_sort2 = df_sort.copy()
    df_sort2.columns = ['st_tg_num']
    df_merged = edge_df.join(df_sort1, on='st_id')
    df_merged= df_merged.join(df_sort2, on='st_tg')
    df_edge = df_merged[['st_id_num', 'st_tg_num']].copy()
    return df_edge


# get the node list
def get_node(p):
    df_node = pd.DataFrame([(k, v[0], v[1]) for k, v in p.items()], columns=['node', 'lat', 'lon'])
    df_node = (df_node.sort_values(by=df_node.columns[0], ascending=True)).reset_index(drop=True)
    df_node['node'] = range(0, len(df_node))
    return df_node


# get the 2nd dataset get the df_node and df_edge from 2nd dataset clear
def get_2nd_dataset(df_node, df_edge):
    df_edge = df_edge[['source','target']]
    df_edge = df_edge.rename(columns={'source': 'st_id', 'target': 'st_tg'})
    df_edge = get_edge(df_edge)
    df_node = df_node.rename(columns={'node_id': 'node'})
    df_node = df_node.drop(['type','type_num'], axis=1)
    df_node = (df_node.sort_values(by=df_node.columns[0], ascending=True)).reset_index(drop=True)
    df_node['node'] = range(0, len(df_node))
    return df_node, df_edge


# get the data
def get_data(df_node, df_edge):
    df_test_node =  df_node
    df_test_edge = df_edge.rename(columns={'st_id_num': 'start', 'st_tg_num': 'end'})
    x = torch.tensor(df_test_node.values, dtype=torch.float)
    edge_index = torch.tensor([df_test_edge['start'].values,df_test_edge['end'].values], dtype=torch.long)
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index)
    # 创建NetworkX图对象
    G = nx.Graph()
    # 添加节点和标签
    for i in range(data.num_nodes):
        G.add_node(i)
    # 添加边
    for j in range(data.edge_index.shape[1]):
        src = data.edge_index[0, j]
        dst = data.edge_index[1, j]
        G.add_edge(src.item(), dst.item())
    return data,G


def get_graph(path):
    data = pd.read_excel(path)  # read
    df_org, pos = chinese_railway_prep(data)  # get clean data
    df_edge = get_edge(df_org)
    df_node = get_node(pos)
    data,G = get_data(df_node, df_edge)
    return G,df_node,df_edge


def get_graph_2nd(df_node,df_edge):
    data,G = get_data(df_node, df_edge)
    return G, df_node, df_edge

#---------------------------------------------------node property------------------------------------------------------------
#Calculate the degree, degree centrality, betweenness centrality, closeness centrality, eigenvector centrality, clustering coefficient


def get_degree(G):
    degree_dict = dict(G.degree(G.nodes()))
    df_degree = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['degree'])
    return df_degree


def get_clustering(G):
    clustering_dict = nx.clustering(G)
    df_clustering = pd.DataFrame.from_dict(clustering_dict, orient='index', columns=['clustering'])
    return df_clustering


def get_core_number(G):
    core_number_dict = nx.core_number(G)
    df_core_number = pd.DataFrame.from_dict(core_number_dict, orient='index', columns=['core_number'])
    return df_core_number


def get_degree_centrality(G):
    degree_centrality_dict = nx.degree_centrality(G)
    df_degree_centrality = pd.DataFrame.from_dict(degree_centrality_dict, orient='index', columns=['degree_centrality'])
    return df_degree_centrality


def get_betweenness_centrality(G):
    betweenness_centrality_dict = nx.betweenness_centrality(G)
    df_betweenness_centrality = pd.DataFrame.from_dict(betweenness_centrality_dict, orient='index', columns=['betweenness_centrality'])
    return df_betweenness_centrality


def get_closeness_centrality(G):
    closeness_centrality_dict = nx.closeness_centrality(G)
    df_closeness_centrality = pd.DataFrame.from_dict(closeness_centrality_dict, orient='index', columns=['closeness_centrality'])
    return df_closeness_centrality


def get_eigenvector_centrality(G):
    eigenvector_centrality_dict = nx.eigenvector_centrality(G)
    df_eigenvector_centrality = pd.DataFrame.from_dict(eigenvector_centrality_dict, orient='index', columns=['eigenvector_centrality'])
    return df_eigenvector_centrality


#---------------------------------------------------get the node property as dataframe------------------------------------------------------------
def get_node_properties(G,df_feature):

    df_degree = get_degree(G)
    df_clustering = get_clustering(G)
    df_degree_centrality = get_degree_centrality(G)
    df_betweenness_centrality = get_betweenness_centrality(G)
    df_closeness_centrality = get_closeness_centrality(G)
    df_eigenvector_centrality = get_eigenvector_centrality(G)
    df_node_feature = pd.concat([df_degree, df_clustering, df_degree_centrality, df_betweenness_centrality, df_closeness_centrality, df_eigenvector_centrality], axis=1)
    df_feature = df_feature.join(df_node_feature, on='node')
    return df_feature


#---------------------------------------------------network property------------------------------------------------------------
#Calculate the number of nodes, number of edges, number of connected components, number of nodes in the largest connected component, density, diameter, average distance, efficiency
def get_network_properties(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_connected_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    Max_components_G = nx.Graph(G.subgraph(largest_cc))
    Max_components_node = Max_components_G.number_of_nodes()
    Max_components_edges = Max_components_G.number_of_edges()
    k_cores = nx.k_core(Max_components_G)
    density = nx.density(Max_components_G)
    diameter = nx.diameter(Max_components_G)
    avg_distance = nx.average_shortest_path_length(Max_components_G)
    efficiency = nx.global_efficiency(Max_components_G)
    # print('节点数：', num_nodes)
    # print('边数：', num_edges)
    # print('连通图的数量：', num_connected_components)
    # print('最大连通分量的节点数：', Max_components_node)
    # print('最大连通分量的边数：', Max_components_node)
    # print('最大连通分量的k核：', k_cores)
    # print('最大连通分量的网络密度：', density)
    # print('最大连通分量的直径：', diameter)
    # print('最大连通分量的平均距离：', avg_distance)
    # print('最大连通分量的效率：', efficiency)
    return num_nodes, num_edges, num_connected_components, Max_components_node, Max_components_edges, k_cores, density, diameter, avg_distance, efficiency


#---------------------------------------------------detele node ------------------------------------------------------------
def delete_node(G,node_id):
    G_delete = G.copy()
    G_delete.remove_node(node_id)
    num_nodes, num_edges, num_connected_components, Max_components_node, Max_components_edges, k_cores, density, diameter, avg_distance, efficiency = get_network_properties(G)
    df_G = pd.DataFrame({'G_before': [num_nodes, num_edges, num_connected_components, Max_components_node, Max_components_edges, k_cores, density, diameter, avg_distance, efficiency]}, index=['Number of Nodes', 'Number of Edges', 'Number of Connected Components', 'Size of Largest Connected Component', 'Number of Edges in Largest Connected Component', 'K-Cores', 'Density', 'Diameter', 'Average Distance', 'Efficiency'])
    num_nodes, num_edges, num_connected_components, Max_components_node, Max_components_edges, k_cores, density, diameter, avg_distance, efficiency = get_network_properties(G_delete)
    df_G['G_after'] = [num_nodes, num_edges, num_connected_components, Max_components_node, Max_components_edges, k_cores, density, diameter, avg_distance, efficiency]
    return df_G


def plot_degrees(G):
    degrees = dict(G.degree())
    # 统计每个度值的节点数量
    degree_count = {}
    for node, degree in degrees.items():
        if degree in degree_count:
            degree_count[degree] += 1
        else:
            degree_count[degree] = 1
    # 将字典转换为两个列表，用于绘图
    degree_list = sorted(degree_count.keys())
    count_list = [degree_count[k] for k in degree_list]
    return degree_list, count_list


def plot_Degree_Distribution(G):
    degree_count = nx.degree_histogram(G)
    degree_distribution = [(float(i)/sum(degree_count)) for i in degree_count]
    return range(len(degree_count)),degree_distribution


#---------------------------------------------------attack------------------------------------------------------------
# attack by random degree and betweenness k-core and closeness centrality
#Define functions for random attacks and deliberate attacks with degree values
def random_attack(G, p):
    largest_cc = len(max(nx.connected_components(G), key=len))
    nodes_to_remove = np.random.choice(list(G.nodes()), size=int(p * len(G)), replace=False)
    G.remove_nodes_from(nodes_to_remove)
    # Calculate the maximum connected subgraph size
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    return largest_cc_size / largest_cc


#Define functions for random attacks and deliberate attacks with degree values
def degree_attack(G, p):
    largest_cc = len(max(nx.connected_components(G), key=len))
    degree_dict = dict(G.degree())
    sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    nodes_to_remove = [sorted_degree[i][0] for i in range(int(p * len(G)))]
    G.remove_nodes_from(nodes_to_remove)
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    return largest_cc_size / largest_cc


#Define functions for attacks with betweenness values
def betweenness_attack(G, p):
    largest_cc = len(max(nx.connected_components(G), key=len))
    betweenness_dict = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)
    nodes_to_remove = [sorted_betweenness[i][0] for i in range(int(p * len(G)))]
    G.remove_nodes_from(nodes_to_remove)
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    return largest_cc_size / largest_cc


#Define functions for attacks with closeness values
def kshell_attack(G, p):
    largest_cc = len(max(nx.connected_components(G), key=len))
    kshell_dict = nx.core_number(G)
    sorted_kshell = sorted(kshell_dict.items(), key=lambda x: x[1], reverse=True)
    nodes_to_remove = [sorted_kshell[i][0] for i in range(int(p * len(G)))]
    G.remove_nodes_from(nodes_to_remove)
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    return largest_cc_size / largest_cc


#Define functions for attacks with closeness values
def collective_influence(G):
    ci_dict = {}
    l = 3
    for node in G.nodes():
        ci = (G.degree(node) - 1) * sum([(G.degree(j) - 1) for j in nx.single_source_shortest_path_length(G, node, cutoff=l)])
        ci_dict[node] = ci
    return ci_dict


#Define functions for attacks with collective_influence(CI) values
def ci_attack(G, p):
    largest_cc = len(max(nx.connected_components(G), key=len))
    ci_dict = collective_influence(G)
    sorted_ci = sorted(ci_dict.items(), key=lambda x: x[1], reverse=True)
    nodes_to_remove = [sorted_ci[i][0] for i in range(int(p * len(G)))]
    G.remove_nodes_from(nodes_to_remove)
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    return largest_cc_size / largest_cc


#---------------------------------------------------plot------------------------------------------------------------
def plot_attack(G):
    relative_size = [1]
    for p in np.arange(0, 1, 0.1):
        relative_size.append(random_attack(G.copy(), p))
    relative_size_deg = [1]
    for p in np.arange(0, 1, 0.1):
        relative_size_deg.append(degree_attack(G.copy(), p))
    relative_size_betw = [1]
    for p in np.arange(0, 1, 0.1):
        relative_size_betw.append(betweenness_attack(G.copy(), p))
    relative_size_kshell = [1]
    for p in np.arange(0, 1, 0.1):
        relative_size_kshell.append(kshell_attack(G.copy(), p))
    relative_size_ci = [1]
    for p in np.arange(0, 1, 0.1):
        relative_size_ci.append(ci_attack(G.copy(), p))

    Attack_Ratio = np.arange(0, 1.1, 0.1)
    # plt.figure(figsize=(10, 8))
    # plt.plot(Attack_Ratio, relative_size, 'bo-', label='Random Attack')
    # plt.plot(Attack_Ratio, relative_size_deg, 'ro-', label='Degree Attack')
    # plt.plot(Attack_Ratio, relative_size_betw, 'go-', label='Betweenness Attack')
    # plt.plot(Attack_Ratio, relative_size_kshell, 'yo-', label='Kshell Attack')
    # plt.plot(Attack_Ratio, relative_size_ci, 'ko-', label='Collective Influence Attack')
    # plt.title("Relative Size of Largest Connected Component")
    # plt.ylabel("Relative Size")
    # plt.xlabel("Attack Ratio")
    # plt.legend()
    # plt.show()
    return Attack_Ratio, relative_size, relative_size_deg, relative_size_betw, relative_size_kshell, relative_size_ci



