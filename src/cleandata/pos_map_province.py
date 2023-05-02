import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim # pip install geopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

''' =================================================================================================
function to extract city and state information from lat-lon data
================================================================================================= '''
def extract_city_state(lat, lon):
    # Initialize the geolocator object
    geolocator = Nominatim(user_agent="geoapiExercises")
    
    # Reverse geocode the coordinates and extract the address components
    location = geolocator.reverse(lat+","+lon, language='en')
    address = location.raw['address']
        
    # Extract the city and state information from the address
    city = address.get('city', '')
    state = address.get('state', '')

    # Return the city and state information as a tuple
    return (city, state)

''' =================================================================================================
function to clean blank province
================================================================================================= '''
def clean_blank_province(df_pos_with_state):
    # keep only necessary columns
    df_pos_with_state = df_pos_with_state[['node', 'lat', 'lon', 'state']]

    # split data into labeled and unlabeled datasets
    df_labeled = df_pos_with_state[df_pos_with_state['state'].notnull()].copy()
    df_unlabeled = df_pos_with_state[df_pos_with_state['state'].isnull()].copy()

    # encode state column into numbers
    states = df_labeled['state'].unique()
    state_map = {state: i for i, state in enumerate(states)}
    df_labeled.loc[:, 'state_num'] = df_labeled['state'].map(state_map)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_labeled[['lat', 'lon']], df_labeled['state_num'], test_size=0.1, random_state=42)

    # fit RandomForestClassifier model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # predict encoded state numbers for test set
    y_pred = rf.predict(X_test)

    # calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'classification accuracy score: {accuracy}')

    # predict encoded state numbers for all data points
    df_unlabeled.loc[:, 'state_num'] = rf.predict(df_unlabeled[['lat', 'lon']])

    # map predicted state numbers back to state names
    df_unlabeled.loc[:, 'state'] = df_unlabeled['state_num'].map({i: state for state, i in state_map.items()})

    # combine labeled and unlabeled dataframes
    df_pos_with_state_clean = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    return df_pos_with_state_clean

''' =================================================================================================
function to clean blank city
================================================================================================= '''
def clean_blank_city(df_pos_with_state):
    # keep only necessary columns
    df_pos_with_state = df_pos_with_state[['node', 'lat', 'lon', 'state_num','city', 'state']]
    df_pos_with_state['city_state'] = df_pos_with_state['city']+"_"+df_pos_with_state['state']
    
    # split data into labeled and unlabeled datasets
    df_labeled = df_pos_with_state[df_pos_with_state['city_state'].notnull()].copy()
    df_unlabeled = df_pos_with_state[df_pos_with_state['city_state'].isnull()].copy()

    # encode state column into numbers
    states = df_labeled['city_state'].unique()
    state_map = {state: i for i, state in enumerate(states)}
    df_labeled.loc[:, 'city_num'] = df_labeled['city_state'].map(state_map)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_labeled[['lat', 'lon', 'state_num']], df_labeled['city_num'], test_size=0.05, random_state=42)

    # fit RandomForestClassifier model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # predict encoded state numbers for test set
    y_pred = rf.predict(X_test)

    # calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'classification accuracy score: {accuracy}')

    # predict encoded state numbers for all data points
    df_unlabeled.loc[:, 'city_num'] = rf.predict(df_unlabeled[['lat', 'lon', 'state_num']])

    # map predicted state numbers back to state names
    df_unlabeled.loc[:, 'city_state'] = df_unlabeled['city_num'].map({i: state for state, i in state_map.items()})

    # combine labeled and unlabeled dataframes
    df_pos_with_state_clean = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    # delete "_"
    df_pos_with_state_clean['city'] = df_pos_with_state_clean['city_state'].str.split('_').str[0]

    # drop city_state
    df_pos_with_state_clean = df_pos_with_state_clean.drop(columns=['city_state'])

    # order col
    df_pos_with_state_clean = df_pos_with_state_clean[['node', 'lat', 'lon', 'city', 'city_num', 'state', 'state_num']]

    return df_pos_with_state_clean

