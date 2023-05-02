import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt

import networkx as nx

from sklearn.neighbors import NearestNeighbors

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree

import joblib
import json

''' =================================================================================================
Define class to get nearest neighbor node
================================================================================================= '''

class NodeNeighborPredictor:
    def __init__(self, node_df):
        self.nodes_df = node_df[['lat', 'lon', 'node_id']]
        self.tree = self._fit_tree(self.nodes_df)

    def predict_neighbors(self, lat, lon, n_neighbors):
        new_node_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        dist, idx = self.tree.query(new_node_df[['lat', 'lon']].values, k=n_neighbors)
        neighbor_nodes = self.nodes_df.iloc[idx[0]]['node_id'].tolist()
        return neighbor_nodes

    def _fit_tree(self, nodes_df):
        tree = KDTree(nodes_df[['lat', 'lon']].values)
        return tree
    
# class NodeNeighborPredictor:
#     def __init__(self, node_df):
#         self.nodes_df = node_df[['lat', 'lon', 'node']]
#         self.tree = self._fit_tree(self.nodes_df)

#     def predict_neighbors(self, lat, lon, n_neighbors):
#         new_node_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
#         dist, idx = self.tree.kneighbors(new_node_df[['lat', 'lon']].values, n_neighbors=n_neighbors)
#         neighbor_nodes = self.nodes_df.iloc[idx[0]]['node'].tolist()
#         return neighbor_nodes

#     def _fit_tree(self, nodes_df):
#         tree = NearestNeighbors(metric='haversine', radius=10)
#         tree.fit(nodes_df[['lat', 'lon']].values)
#         return tree

''' =================================================================================================
Define the NN Feed forward model (fully connected)
================================================================================================= '''
class ModelFFW_new_edge_pred(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channel):
        super(ModelFFW_new_edge_pred, self).__init__()
        self.linear1 = nn.Linear(input_channel, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, int(hidden_channel/2))
        self.linear3 = nn.Linear(int(hidden_channel/2), output_channel)
        self.relu = nn.ReLU()   

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

''' =================================================================================================
function to add cluster number into 'df_new_latlon'
    
function input:
- df: 'df_pos_with_state_clean'
- df_new_latlon: df of new desired lat-lon

function return:
- df_new_latlon with col 'state_num'
    
RF model inputs:
- 'df_pos_with_state_clean' col 'lat' and 'lon'
RF labels:
- 'df_pos_with_state_clean' col 'state_num'
================================================================================================= '''
def prep_classify_data(df_pos, df_new_node, split_portion=0.1):
    '''
    function to add cluster number into 'df_new_latlon'
    
    function input:
    - df: 'df_pos_with_state_clean'
    - df_new_latlon: df of new desired lat-lon
    function return:
    - df_new_latlon with col 'state_num'
    
    RF model inputs:
    - 'df_pos_with_state_clean' col 'lat' and 'lon'
    RF labels:
    - 'df_pos_with_state_clean' col 'state_num'
    
    '''
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_pos[['lat', 'lon']], df_pos['type_num'], test_size=split_portion, random_state=42)

    # fit RandomForestClassifier model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # predict encoded state numbers for test set
    y_pred = rf.predict(X_test)

    # calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'classification accuracy score: {accuracy}')

    # predict encoded state numbers for all data points
    df_new_node.loc[:, 'type_num'] = rf.predict(df_new_node[['lat', 'lon']])
    
    return df_new_node

''' =================================================================================================
function pred neighbor node
================================================================================================= '''
def train_and_get_node_neighbor(node_df, df_new_node, n_neighbors=1):
    '''
    function pred neighbor node
    '''
    pred_neighbor = NodeNeighborPredictor(node_df)
    neighbor_nodes = pred_neighbor.predict_neighbors(lat = df_new_node['lat'], 
                                                     lon = df_new_node['lon'],
                                                     # state_num = df_new_node['state_num'],
                                                     n_neighbors = n_neighbors)
    return neighbor_nodes

''' =================================================================================================
function to train new edge feature pred model
    
function input:
- df: df for train and test model (only have needed)
- split_portion: split test-train portion
- hidden_channel: number of hidden nurual
- model hyperparameter: e.g. num_epochs, lr etc.
- model config: e.g. print_epoch state
    
function return:
- model: model variable that can use for pred or save
- model channel: input, output, hidden
- norm_scalar: norm scalar using for norm new input
================================================================================================= '''
def train_function_new_edge_pred_FFW(df, col_inputs = ['lat_id', 'lon_id', 'state_num_id', 'lat_tg', 'lon_tg', 'state_num_tg'],
                                     split_portion = 0.1, hidden_channel=128, num_epochs=100, lr=0.01, batch_size=128,
                                     scheduler_step_size=10, scheduler_gamma=0.8, print_epoch=True, step_print=10):
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'running on: {device}')
    
    ''' split train-test dataset '''
    # split 80-20 w/ shuffle
    df_train, df_test = train_test_split(df, test_size=split_portion, shuffle=True)

    # normalize inputs (col 6 to end)
    scaler = StandardScaler()
    
    # scaler = MinMaxScaler()
    # df_train.iloc[:, :4] = scaler.fit_transform(train_df.iloc[:, :4])
    # df_test.iloc[:, :4] = scaler.transform(test_df.iloc[:, :4])
    df_train[col_inputs] = scaler.fit_transform(df_train[col_inputs])
    df_test[col_inputs] = scaler.transform(df_test[col_inputs])    
    
    ''' split inputs and labels '''
    # col_inputs = ['lat_id', 'lon_id', 'state_num_id', 'lat_tg', 'lon_tg', 'state_num_tg']
    X_train = df_train[col_inputs] # split inputs for train
    y_train = df_train.drop(col_inputs, axis=1) # split output for train
    
    X_test = df_test[col_inputs]  # split inputs for test
    y_test = df_test.drop(col_inputs, axis=1) # split output for test

    ''' make data into tensor and dataLoader '''
    # convert your Pandas DataFrame to a PyTorch tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)

    # create TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # create DataLoader: batch try: 32, 64, 128, 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    ''' define and config model'''
    input_channel = X_train.shape[1]
    output_channel = y_train.shape[1]
    hidden_channel = hidden_channel
    model = ModelFFW_new_edge_pred(input_channel=input_channel, output_channel=output_channel, hidden_channel=hidden_channel)
    model.to(device) # move to device if possible
    
    ''' loss function '''
    # loss_fn = mpe_loss_xy
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss()
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()

    ''' optimizer & scheduler '''
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

#     ''' check '''
#     print(X_train.head())
    
#     print(y_train.head())
    
    
    ''' train model '''
    for epoch in range(num_epochs):
        # model.to(device) # move to device if possible
        model.train()
        # iterate over the batches
        for i, (inputs, outputs) in enumerate(train_loader):
            # inputs = inputs.to(device)
            # outputs = outputs.to(device)
            y_pred = model(inputs)
            loss = loss_fn(y_pred, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update the learning rate at the end of each epoch
        scheduler.step()

        # calculate the MPE for each epoch
        model.eval()
        y_test_pred = model(X_test_tensor)
        test_loss = loss_fn(y_test_tensor, y_test_pred).item()

        # print the MSE every 100 epochs
        if print_epoch == True:
            if epoch % step_print == 0:
                print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test loss: {test_loss:.4f}, , Lr: {scheduler.get_last_lr()[0]:.6f}")
                
    # print the loss for the last epoch
    # if print_epoch == True:
    print(f"Epoch {num_epochs}, Train: {loss.item():.4f}, Test loss: {test_loss:.4f}, , Lr: {scheduler.get_last_lr()[0]:.6f}")          
    
    model.to('cpu')
    return model, input_channel, output_channel, hidden_channel, scaler

''' =================================================================================================
function to use model to pred edge feature
function inputs:
- model: trained model
- df_input_pred (same format of input when train model)
================================================================================================= '''
def pred_new_edge_feature(model, df_input_pred, norm_scaler):
    model.eval() # set model to eval (to grad)
    # norm all input
    df_input_pred.iloc[:, :] = norm_scaler.transform(df_input_pred.iloc[:, :])

    # convert to tensor
    X_test_tensor = torch.tensor(df_input_pred.values, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_test_tensor)

    return pred.detach().numpy()

''' =================================================================================================
export trained model and config
================================================================================================= '''
def edge_pred_export_model(model, input_channel, output_channel, hidden_channel, scaler, folder_name='edge_pred_conf'):
    # create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save the PyTorch model to disk
    torch.save(model.state_dict(), f'{folder_name}/model.pt')

    # save the Scaler object to disk
    joblib.dump(scaler, f'{folder_name}/scaler.pkl')

    # save the channel values to a JSON file
    data = {'input_channel': input_channel, 'output_channel': output_channel, 'hidden_channel': hidden_channel}
    with open(f'{folder_name}/channels.json', 'w') as f:
        json.dump(data, f)

''' =================================================================================================
load saved model and config
================================================================================================= '''
def edge_pred_import_model(folder_name='edge_pred_conf'):
    # load the saved PyTorch model state dictionary
    model_state_dict = torch.load(f'{folder_name}/model.pt')

    # load the channel values from the JSON file
    with open(f'{folder_name}/channels.json', 'r') as f:
        data = json.load(f)
        input_channel = data['input_channel']
        output_channel = data['output_channel']
        hidden_channel = data['hidden_channel']

    # create a new instance of the model using the saved architecture
    model = ModelFFW_new_edge_pred(input_channel, output_channel, hidden_channel)
    model.load_state_dict(model_state_dict)

    # load the saved Scaler object
    scaler = joblib.load(f'{folder_name}/scaler.pkl')

    return model, scaler, input_channel, output_channel, hidden_channel

''' =================================================================================================
function map pos df into df (map lat and lon refer to node_id (st_id))
================================================================================================= '''
def map_pos_into_data(df_data, df_pos):
    # merge the dataframes on 'node'
    df_data = df_data.rename(columns={'type_num': 'type_num_edge'})
    df_data = pd.merge(df_data, df_pos, left_on='source', right_on='node_id', how='left')
    df_data = pd.merge(df_data, df_pos, left_on='target', right_on='node_id', how='left', suffixes=('_id', '_tg'))

    # drop the 'node' columns
    df_data.drop(['node_id_id', 'node_id_tg'], axis=1, inplace=True)
    df_data = df_data.rename(columns={'type_num_edge': 'type_num'})
    return df_data
    
''' =================================================================================================
function plot all node and edge on map
- if specify node 1 --> make it red
- if specify node 1 --> make it blue
================================================================================================= '''
def plot_all_on_map(G, high_light_node1=None, high_light_node2=None):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    paris = world[world['name'] == 'Paris']

    # Reproject the map to the Lambert-93 projection
    paris = paris.to_crs(epsg=2154)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 10), dpi=200)

    # Plot the map of Paris
    paris.plot(ax=ax, color='white', edgecolor='black')

    # Create a default pos dictionary
    pos = {node: (G.nodes[node]['lon'], G.nodes[node]['lat']) for node in G.nodes()}

    # Highlight the specified nodes in red and blue, if provided
    if high_light_node1 is not None:
        highlighted_nodes1 = [node_id for node_id in high_light_node1 if node_id in pos]
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes1, node_size=5, node_color='red', ax=ax)

    if high_light_node2 is not None:
        highlighted_nodes2 = [node_id for node_id in high_light_node2 if node_id in pos]
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes2, node_size=2, node_color='blue', ax=ax)

    if high_light_node1 is None and high_light_node2 is None:
        # Plot all nodes and edges in gray
        nx.draw_networkx_nodes(G, pos, node_size=1, node_color='gray', ax=ax)

    # Plot all edges in gray
    nx.draw_networkx_edges(G, pos, width=0.2, edge_color='gray', ax=ax, arrows=False)

    # Color the edges between the highlighted nodes red
    if high_light_node1 is not None and high_light_node2 is not None:
        red_edges = [(u, v) for u, v in G.edges if (u in high_light_node1 and v in high_light_node2) or (v in high_light_node1 and u in high_light_node2)]
        nx.draw_networkx_edges(G, pos, edgelist=red_edges, width=0.2, edge_color='red', ax=ax, arrows=False)
        
    # Show the plot
    plt.show()


    
''' =================================================================================================
main function to perform new edge prediction
================================================================================================= '''
def main_edge_feature_pred(df_data, df_pos, new_node_lat, new_node_lon, n_degree, train_new = False,
                           col_inputs = ['lat_id', 'lon_id', 'type_num_id', 'lat_tg', 'lon_tg', 'type_num_tg'],
                           split_portion = 0.1, hidden_channel=128, num_epochs=30, lr=0.1, batch_size=128,
                           scheduler_step_size=10, scheduler_gamma=0.8, print_epoch=True, step_print=10,
                           folder_name = 'dataset2_edge_pred_conf'
                          ):
    # map lat-lon to df
    # print(df_data.head())
    df_data2 = map_pos_into_data(df_data, df_pos)
    df_data2 = df_data2.drop(['source', 'target'], axis=1) 
    # print(df_data2.head())
    
    # make new lat-lon to df
    df_new_node = pd.DataFrame({'lat': [new_node_lat], 'lon': [new_node_lon]}, index=[0]) # make it to df
    
    # prep classify new node
    df_new_node = prep_classify_data(df_pos = df_pos, df_new_node = df_new_node, split_portion = 0.1)
    
    # if train_new == True, it will train model again and replace previous saved model config
    if train_new == True:
        ''' auto save result after train '''
        model, input_channel, output_channel, hidden_channel, scaler = train_function_new_edge_pred_FFW(df = df_data2, col_inputs = col_inputs,
                                                                                                        split_portion = split_portion, hidden_channel=hidden_channel, num_epochs=num_epochs,
                                                                                                        lr=lr, batch_size=batch_size,
                                                                                                        scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma,
                                                                                                        print_epoch=print_epoch, step_print=step_print)
        # save model
        edge_pred_export_model(model = model,
                               input_channel = input_channel, output_channel = output_channel,
                               hidden_channel = hidden_channel, scaler = scaler,
                               folder_name = folder_name)
        
    # import model
    model_edge_pred, scaler_edge_pred, input_channel_edge_pred, output_channel_edge_pred, hidden_channel_edge_pred = edge_pred_import_model(folder_name=folder_name)
    
    # pred neighbor_node
    neighbor_node = train_and_get_node_neighbor(node_df = df_pos, df_new_node = df_new_node, n_neighbors = n_degree)
    
    # map pair of new connection
    new_pair = df_pos[df_pos['node_id'].isin(neighbor_node)].copy()
    new_pair.loc[:, 'lat_id'] = df_new_node.loc[0, 'lat']
    new_pair.loc[:, 'lon_id'] = df_new_node.loc[0, 'lon']
    new_pair.loc[:, 'type_num_id'] = df_new_node.loc[0, 'type_num']
    new_pair.loc[:, 'lat_tg'] = new_pair.loc[:, 'lat']
    new_pair.loc[:, 'lon_tg'] = new_pair.loc[:, 'lon']
    new_pair.loc[:, 'type_num_tg'] = new_pair.loc[:, 'type_num']
    new_pair_node = new_pair['node_id']
    new_pair = new_pair.drop(['node_id', 'lat', 'lon', 'type_num'], axis=1)
    
    # pred new edge
    edge_pred = pred_new_edge_feature(model = model_edge_pred, df_input_pred = new_pair, norm_scaler = scaler_edge_pred)
    
    # define new station number
    # max_st_id = max(df_pos['node_id'])
    new_st_id = 'new_node'

    # get new df edge feature
    df_edge_pred = pd.DataFrame(edge_pred, columns=['distance', 'type_num'])
    df_edge_pred['source'] = neighbor_node
    df_edge_pred['target'] = new_st_id
    # df_edge_pred['train'] = df_edge_pred['train_max_speed'].apply(reverse_train_speed).apply(lambda x: x[0])
    # df_edge_pred['train_max_speed'] = df_edge_pred['train_max_speed'].apply(reverse_train_speed).apply(lambda x: x[1])

    # get new df pos (df_pos2)
    df_new_node2 = df_new_node.copy()
    df_new_node2['node_id'] =new_st_id
    df_pos2 = pd.concat([df_pos, df_new_node2], ignore_index=True)
    df_pos2[df_pos2['node_id']==new_st_id]

    # get new df_data (df_data3)
    # df_data2 = pd.concat([df_edge_pred.drop('train', axis=1), df_data], ignore_index=True)
    df_data2 = pd.concat([df_edge_pred, df_data], ignore_index=True)

    
    return df_data2, df_pos2, new_st_id, neighbor_node
