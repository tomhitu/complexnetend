import pandas as pd
import networkx as nx

import torch
from torch_geometric.data import Data

def get_data(df_node,df_edge):
    # creat the graph
    G = nx.from_pandas_edgelist(df_edge, source='source', target='target')
    # define the value of node and edge
    # df_node = df_node
    df_edge = nx.to_pandas_edgelist(G)#[['source', 'target']]
    x = torch.tensor(df_node.values, dtype=torch.float)
    edge_index = torch.tensor([df_edge['source'].values,df_edge['target'].values], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data

import torch_geometric.transforms as T
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Split_F = T.Compose([
    T.NormalizeFeatures(),
    # T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1,
                      num_test=0.1,
                      # the graph is undirected
                      is_undirected=False,
                      # don't add the negative samples to the training set
                      add_negative_train_samples=False),
])

from torch_geometric.utils import  negative_sampling
def negative_sample(data):
    # 从训练集中采样与正边相同数量的负边
    neg_edge_label_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1), method='sparse')
    # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
    ALLedge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_label_index],
        dim=-1,
    )
    ALLedge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_label_index.size(1))
    ], dim=0)
    return ALLedge_label, ALLedge_label_index



from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, feature_num, hidden_channels, out_num):
        super().__init__()
        self.conv1 = GCNConv(feature_num, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_num)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量，一行为所有特征
        start_node = z[edge_label_index[0]]#调用起始列所有点的特征值，所有起始点的特征值所以行是起始点数量列是特征数量
        end_node = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        result = (start_node * end_node).sum(dim=-1)#按所有位置相乘一个一个然后按点的特征值相加即一行中的所有列相加
        # print(r.size())   (7284)所有点连边的特征值，即行数为节点对，即连边的最后特征值
        return result

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        pre_out = self.decode(z, edge_label_index)
        return pre_out


from sklearn.metrics import roc_auc_score
def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()#将out作为输入将其转化成0-1之间的概率
        model.train()
    auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    return auc


def train(data, lr=0.01):
    model = GCN(data.num_features, 128, 64)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        ALLedge_label, ALLedge_label_index = negative_sample(data)
        out = model(data.x, data.edge_index, ALLedge_label_index).view(-1)
        loss = criterion(out, ALLedge_label)
        loss.backward()
        optimizer.step()
    return model

def get_out(model,data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        # print(z.size())
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return out

def get_preAcc(data,out):
    df_pre = pd.DataFrame(out.cpu().numpy(), columns=['score'])
    df_edges = pd.DataFrame(data.edge_label_index.T.cpu().numpy(), columns=['source', 'target'])
    df_preorig = pd.DataFrame(data.edge_label_index.T.cpu().numpy(), columns=['source', 'target'])
    df_pre['label'] = df_pre['score'].apply(lambda x: 1 if x > 0.71 else 0)
    df_pre = pd.concat([df_edges, df_pre], axis=1)
    # get the hidden edges label = 1
    pre_data = df_pre[df_pre['label'] == 1]
    # how many the hidden edges label = 1 in original data
    pre_data = pd.merge(df_preorig, pre_data, on=['source', 'target'])
    acc = len(pre_data)/len(df_preorig)
    return acc


# get the all possible edges of the undirected graph
def get_all_edges(num_nodes,train_data):
    # Create a complete graph
    graph = nx.complete_graph(num_nodes)
    # Convert edges to DataFrame
    # df_alledges= pd.DataFrame(list(graph.edges), columns=['source', 'target'])
    alledges_label_index = torch.tensor(list(graph.edges)).t().contiguous()
    data = Data(x=train_data.x,edge_index=train_data.edge_index,edge_label_index = alledges_label_index)
    return data


# 遍历所有连边 input the train_data and Threshold,choose the type 1 is get the hidden edges csv file ,0 is get the hidden list as dataframe
def get_hidden_edges(dataname,Threshold,model,type):
    node_num = 2708
    train_data = torch.load(dataname + '.pth')
    data = get_all_edges(node_num,train_data)
    out = get_out(model,data)
    df_pre = pd.DataFrame(out.cpu().numpy(), columns=['score'])
    df_edges = pd.DataFrame(data.edge_label_index.T.cpu().numpy(), columns=['source', 'target'])
    df_pre['label'] = df_pre['score'].apply(lambda x: 1 if x > Threshold else 0)
    df_pre = pd.concat([df_edges, df_pre], axis=1)
    if type == 1:
        train_data, val_data, test_data = Split_F(train_data)
        df_pre = df_pre[df_pre['label'] == 1]
        dfr_orig1 =  pd.DataFrame(train_data.edge_label_index.T.cpu().numpy(), columns=['source', 'target'])#
        dfr_orig2 = pd.DataFrame(dfr_orig1[['target','source']].values, columns=['source', 'target'])
        hidden_edge1 = pd.merge(dfr_orig1, df_pre, on=['source', 'target'])
        hidden_edge2 = pd.merge(dfr_orig2, df_pre, on=['source', 'target'])
        hidden_edge = pd.concat([hidden_edge1, hidden_edge2])
        hidden_edge = df_pre[~df_pre.index.isin(hidden_edge.index)]
        hidden_edge = hidden_edge.drop(['score','label'], axis=1)
        hidden_edge.to_csv('hidden_edges.csv', index=False)
    else:
        return df_pre


#根据点和边的csv文件，返回模型
def train_save_model(node,edges,lrdata):
    df_node = pd.read_csv(node+'.csv')
    df_edge = pd.read_csv(edges + '.csv')
    data = get_data(df_node,df_edge)
    train_data, val_data, test_data = Split_F(data)
    model = train(train_data, lrdata)
    print("Training is completed and save the model as pre_hidden_model.pth ! ")
    torch.save(data,'data.pth')
    torch.save(model.state_dict(), 'pre_hidden_model.pth')



#给定一个数据的点和边还有要预测的边，返回预测的边的得分
def get_pre_edges(df_node,df_edge,pre_edges,type, lrdata):
    data = get_data(df_node,df_edge)
    train_data, val_data, test_data = Split_F(data)
    if type == 1:
        model = train(train_data, lrdata)
    else:
        model = GCN(data.num_features, 128, 64)
        model.load_state_dict(torch.load('pre_hidden_model.pth'))
    edge_label_index  = torch.tensor([pre_edges['source'].values,pre_edges['target'].values], dtype=torch.long)
    data_pre = Data(x=train_data.x,edge_index=train_data.edge_index,edge_label_index = edge_label_index)
    out = get_out(model,data_pre)
    df_pre = pd.DataFrame(out.cpu().numpy(), columns=['score'])
    df_edges = pd.DataFrame(data_pre.edge_label_index.T.cpu().numpy(), columns=['source', 'target'])
    df_pre['label'] = df_pre['score'].apply(lambda x: 1 if x > 0.71 else 0)
    df_pre = pd.concat([df_edges, df_pre], axis=1)
    return df_pre


#Pick a random number of points to predict
def hidden_edges(df_node, df_edge, lrdata, iftrain=False, folderpath='pre_hidden_model.pth'):
    data = get_data(df_node, df_edge)
    train_data, val_data, test_data = Split_F(data)
    if iftrain == True:
        print('start training new model')
        model = train(train_data, lrdata)
        torch.save(model.state_dict(), folderpath)
    else:
        print('load model')
        model = GCN(data.num_features, 128, 64)
        model.load_state_dict(torch.load(folderpath))
    train_data.edge_label, train_data.edge_label_index = negative_sample(train_data)
    out = get_out(model, train_data)
    acc = roc_auc_score(train_data.edge_label.cpu().numpy(), out.cpu().numpy())
    df_pre = pd.DataFrame(out.cpu().numpy(), columns=['score'])
    df_edges = pd.DataFrame(train_data.edge_label_index.T.cpu().numpy(), columns=['source', 'target'])
    df_pre['label'] = df_pre['score'].apply(lambda x: 1 if x > 0.71 else 0)
    df_pre = pd.concat([df_edges, df_pre], axis=1)
    # df_pre = df_pre[df_pre['label'] == 1]
    dfr_orig1 = pd.DataFrame(data.edge_index.T.cpu().numpy(), columns=['source', 'target'])  #
    dfr_orig2 = pd.DataFrame(dfr_orig1[['target', 'source']].values, columns=['source', 'target'])
    hidden_edge1 = pd.merge(dfr_orig1, df_pre, on=['source', 'target'])
    hidden_edge2 = pd.merge(dfr_orig2, df_pre, on=['source', 'target'])
    hidden_edge = pd.concat([hidden_edge1, hidden_edge2])
    hidden_edge = df_pre[~df_pre.index.isin(hidden_edge.index)]
    # hidden_edge = hidden_edge.drop(['score','label'], axis=1)
    print("The accuracy of the model is :%.2f%%" % (acc * 100))
    return hidden_edge
