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
Class of Model to pred degree of node
================================================================================================= '''
class ModelFFW_pred_degree(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channel):
        super(ModelFFW_pred_degree, self).__init__()
        self.linear1 = nn.Linear(input_channel, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, output_channel)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
''' =================================================================================================
function to pred node degree
================================================================================================= '''

def train_model_predict_node_degree(df_pos, num_epochs=100, lr=0.01, batch_size=128, scheduler_step_size=30, scheduler_gamma=0.9, print_epoch=True, step_print=10):
    '''define training function'''
    def train_function_degree_pred_FFW(df_train, df_test, drop_col, num_epochs=100, lr=0.01, batch_size=128, scheduler_step_size=10, scheduler_gamma=0.8, print_epoch=True, step_print=10):
    
        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'running on: {device}')
        # prepare data
        df_train = df_train.drop(drop_col, axis=1) # drop col (meaning = drop feature)
        df_test = df_test.drop(drop_col, axis=1) # drop col (meaning = drop feature)

        X_train = df_train[['lat', 'lon']] # split inputs for train
        y_train = df_train.drop(['lat', 'lon'], axis=1) # split output for train

        X_test = df_test[['lat', 'lon']]  # split inputs for test
        y_test = df_test.drop(['lat', 'lon'], axis=1) # split output for test

        # convert your Pandas DataFrame to a PyTorch tensor
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)

        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)

        # create TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        # create DataLoader: batch try: 32, 64, 128, 256
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        input_channel = X_train.shape[1]
        output_channel = y_train.shape[1]
        hidden_channel = 256
        model = ModelFFW_pred_degree(input_channel=input_channel, output_channel=output_channel, hidden_channel=hidden_channel)
        model.to(device) # move to device if possible
        ''' loss function '''
        # loss_fn = nn.MSELoss()
        loss_fn = nn.L1Loss()
        # loss_fn = nn.SmoothL1Loss()
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCEWithLogitsLoss()

        ''' optimizer '''
        # optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.Adagrad(model.parameters(), lr=lr)
        # optimizer = optim.Adadelta(model.parameters(), lr=lr)
        # optimizer = optim.RMSprop(model.parameters(), lr=lr)

        # Initialize the learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        epochs = num_epochs


        for epoch in range(epochs):
            model.train()
            # iterate over the batches
            for i, (inputs, outputs) in enumerate(train_loader):
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
        #     print(f"Epoch {epochs}, Test Loss: {loss.item():.4f}, Test MPE: {test_loss:.4f}, , Lr: {scheduler.get_last_lr()[0]:.6f}") 
        print(f"Epoch {epochs}, Train: {loss.item():.4f}, Test loss: {test_loss:.4f}, , Lr: {scheduler.get_last_lr()[0]:.6f}")          
        return model.cpu(), test_loss, y_test_pred.cpu().detach().numpy(), input_channel, output_channel, hidden_channel

    # prepare data
    df_model = df_pos.copy()
    
    # split 80-20 w/ shuffle
    train_df, test_df = train_test_split(df_model, test_size=0.1, shuffle=True)
    
    # normalize inputs (col 6 to end)
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    train_df.iloc[:, :2] = scaler.fit_transform(train_df.iloc[:, :2])
    test_df.iloc[:, :2] = scaler.transform(test_df.iloc[:, :2])
    
    # train model
    model, _, _, input_channel, output_channel, hidden_channel = train_function_degree_pred_FFW(df_train = train_df,
                                                                   df_test = test_df, 
                                                                   drop_col = [], 
                                                                   num_epochs=num_epochs, lr=lr, batch_size=batch_size, 
                                                                   scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma, 
                                                                   print_epoch=print_epoch, step_print = step_print)
    
    return model, scaler, input_channel, output_channel, hidden_channel

''' =================================================================================================
export trained model and config
================================================================================================= '''
def degree_pred_export_model(model, input_channel, output_channel, hidden_channel, scaler, folder_name='degree_pred_conf'):
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
def degree_pred_import_model(folder_name='degree_pred_conf'):
    # load the saved PyTorch model state dictionary
    model_state_dict = torch.load(f'{folder_name}/model.pt')

    # load the channel values from the JSON file
    with open(f'{folder_name}/channels.json', 'r') as f:
        data = json.load(f)
        input_channel = data['input_channel']
        output_channel = data['output_channel']
        hidden_channel = data['hidden_channel']

    # create a new instance of the model using the saved architecture
    model = ModelFFW_pred_degree(input_channel, output_channel, hidden_channel)
    model.load_state_dict(model_state_dict)

    # load the saved Scaler object
    scaler = joblib.load(f'{folder_name}/scaler.pkl')

    return model, scaler, input_channel, output_channel, hidden_channel

''' =================================================================================================
function pred degree
================================================================================================= '''
def pred_degree(lat, lon, model_degree, scaler_degree, input_channel, output_channel, hidden_channel):
    data = {'lat': [lat], 'lon': [lon]}
    df = pd.DataFrame(data, index=[0])
    df[['lat', 'lon']] = scaler_degree.transform(df[['lat', 'lon']])
    X_tensor = torch.tensor(df.values, dtype=torch.float32)
    
    model_degree.eval()
    with torch.no_grad():
        pred = model_degree(X_tensor)
    
    return round(pred.detach().numpy()[0][0])
    # return pred.detach().numpy()[0]
    
''' =================================================================================================
function main pred degree
================================================================================================= '''
def main_pred_degree(df_data, df_pos, new_node_lat, new_node_lon, train_new = False,
                    folder_name = 'dataset2_degree_pred_conf',
                    num_epochs=100, lr=0.01, batch_size=128, 
                    scheduler_step_size=30, scheduler_gamma=0.9, 
                    print_epoch=True, step_print=10
                    ):
    # Create graph # direct graph
    # G = nx.from_pandas_edgelist(df_data, source='source', target='target', edge_attr=['distance', 'type_num'], create_using=nx.DiGraph())
    G = nx.from_pandas_edgelist(df_data, source='source', target='target', edge_attr=['distance', 'type_num'])

    # add node attributes from DataFrame to graph
    for node, data in df_pos.set_index('node_id').iterrows():
        G.nodes[node].update(data.to_dict())

    # get node degree into df with node as an index
    G_degree = pd.DataFrame(G.degree(G.nodes()), columns = ['node_id', 'degree']).set_index('node_id')

    # obtain df_pos2 which have information of state_num and degree
    df_pos2 = df_pos.set_index('node_id')
    df_pos2 = pd.merge(df_pos2, G_degree, left_index=True, right_index=True)
    
    # training model
    if train_new == True:
        model_degree, scaler_degree, input_channel, output_channel, hidden_channel = train_model_predict_node_degree(df_pos2[['lat', 'lon', 'degree']],
                                                                                                                     num_epochs=num_epochs, lr=lr, batch_size=batch_size, 
                                                                                                                     scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma, 
                                                                                                                     print_epoch=print_epoch, step_print=step_print)
        '''save model'''
        degree_pred_export_model(model=model_degree,
                             input_channel=input_channel, output_channel=output_channel, hidden_channel=hidden_channel,
                             scaler=scaler_degree, folder_name=folder_name)
    
    '''load model'''
    model_degree, scaler_degree, input_channel, output_channel, hidden_channel = degree_pred_import_model(folder_name=folder_name)
    
    # pred degree
    n_neighbor = pred_degree(new_node_lat, new_node_lon, model_degree, scaler_degree, input_channel, output_channel, hidden_channel)
    
    return n_neighbor