import pandas as pd
import numpy as np
import datetime
import re
import os

''' =================================================================================================
function change time format into numeric time
- 24hr = 1
- e.g. 7hr and 30min = 7/24 + 30/60/24
================================================================================================= '''
def apply_numeric_time(val):
    if type(val) == float:
        return round(val, 6)
    else:
        hour = val.hour
        minute = val.minute
        second = val.second
        time_num = hour/24+minute/(24*60)+second/(24*60*60)
        return round(time_num, 6)   # prevent diff # at 6 digit uncertainty ~+-0.17s
    
''' =================================================================================================
function to add train (max) speed into df
================================================================================================= '''
def get_train_speed(train_id):
    if str(train_id).startswith('G') or str(train_id).startswith('C'):
        return 350
    elif str(train_id).startswith('D'):
        return 260
    elif str(train_id).startswith('Z') or str(train_id).startswith('T'):
        return 160
    elif str(train_id).startswith('K'):
        return 120
    else:
        return 100


''' =================================================================================================
function convert floating time to time format.
================================================================================================= '''
def convert_time_format(val):
    val = val % 1
    hr = val * 24 // 1
    mins = (((val * 24) % 1) * 60 )// 1
    hrs_mins = hr * 60
    full_mins = hrs_mins + mins
    zero_time = datetime.datetime(1900, 1, 1, 0, 0, 0)
    desired_time = zero_time + datetime.timedelta(minutes=full_mins)
    
    
    return pd.to_datetime(desired_time.strftime("%H:%M:%S")).time()

''' =================================================================================================
main function for clean chinese_railway dataset.
================================================================================================= '''

def chinese_railway_prep(df):
  
  # manually fix reverse error on row 4091-4096
  # but miledge still correct in raw data --> i will not reverse it
  df = df[['train', 'st_no', 'mileage', 'st_id', 'date', 'arr_time', 'dep_time', 'stay_time', 'lat', 'lon']]
  df.iloc[4091:4096, 3:] = df.iloc[4091:4096, 3:].iloc[::-1].values
  df = df[['train', 'st_no', 'st_id', 'date', 'arr_time', 'dep_time', 'stay_time', 'mileage', 'lat', 'lon']]
  
  # manually swap valus in col 'mileage' of row 8389 and 8390
  # because it present decresase mileage in raw data
  temp8389 = df.loc[8389, 'mileage']
  temp8390 = df.loc[8390, 'mileage']
  df.loc[8390, 'mileage'] = temp8389
  df.loc[8389, 'mileage'] = temp8390
  
  # apply numeric time format into df (24hr = 1)
  df['arr_time'] = df['arr_time'].apply(apply_numeric_time)
  df['dep_time'] = df['dep_time'].apply(apply_numeric_time)

  # change cols 'date' into number
  df['date'] = df['date'].apply(lambda s: int(re.findall('\d+', s)[0]))

  # adj timing with date information (e.g. date 2, time will start with 1.xx)
  df['arr1'] = df.apply(lambda row: round(row['arr_time']+row['date']-1,6)  if row['arr_time'] <= row['dep_time'] else (round(row['arr_time']+row['date']-2,6) if row['date'] > 1 else round(row['arr_time'],6)), axis=1) 
  df['dep1'] = round(df['dep_time']+df['date']-1,6)
  
  # add next arrvie time into each row
  # add next st order into each row too (in order to check whether nex station is connected or not)
  # if next station is not connected --> next_arr_time = 'NA'
  df['next_arr_time'] = df['arr1'].shift(-1)
  df['next_st_no'] = df['st_no'].shift(-1)
  df['next_arr_time'] = df.apply(lambda row: 'NA'  if row['next_st_no'] < row['st_no'] else row['next_arr_time'], axis=1)

  # adjust departure time (dep1) if values less than arrive time (arr1)
  # in current solution, i make dep1=arr1+stay_time if stay_time is available
  #   - it make sense because the train cannot departure before arrive
  #   - however if arrive time is incorrect --> it make dep incorrcet too.
  df['stay_time'] = df['stay_time'].replace(['-', ' '], 0) # replace emtry string with 0
  df['stay_time'] = df['stay_time'].astype(float) # make sure it is a number
  df['dep1'] = df.apply(lambda row: row['arr1']+row['stay_time']/24/60 if (row['dep1'] < row['arr1'] and row['stay_time'] != 0) else (row['arr1'] if row['dep1'] < row['arr1']  else row['dep1']), axis=1)
  
  # calcuate stay time
  # however, it still already have stay time come from raw data
  # i dont sure all if raw stay time is correct or not --> use new calculation!
  df['adj_stay_time'] = df.apply(lambda row: (row['dep1'] - row['arr1']) , axis=1)

  # add next station (st_tg) into each row
  # if next row are not connected to current row --> make st_tg to np.nan
  # checking new connected by looking at value 'next_arr_time' because current if 'next_arr_time'=='NA'
  df['st_tg'] = df['st_id'].shift(-1)
  df['st_tg'] = df.apply(lambda row: np.nan if row['next_arr_time'] == 'NA' else row['st_tg'], axis=1)

  # add 'next_mileage' into each row
  # make 'next_mileage' to 0. it no next station
  df['mileage'] = df['mileage'].replace(' ', 0) # replace emtry string with 0
  df['mileage'] = df['mileage'].astype(int) # make col to int
  df['next_mileage'] = df['mileage'].shift(-1)
  df['next_mileage'] = df.apply(lambda row: np.nan if row['next_arr_time'] == 'NA' else row['next_mileage'], axis=1)

  # calcuate travel_time by 'next_arr_time' - 'dep1'
  # if not next_arr_tme --> travel_time = 'NA'
  df['travel_time'] = df.apply(lambda row:  row['next_arr_time']-row['dep1'] if row['next_arr_time'] != 'NA' else 'NA', axis=1)

  # get all position (po: lat-lon) information before perform drop ununccessary row
  # perform pos get at thsi step because all st_id still in a
  pos = {station: (latitude, longitude) for station, latitude, longitude in zip(df['st_id'], df['lat'], df['lon'])}

  # drop the row that no data about st_tg
  # because the program already move next station infomation into previous station
  df = df.dropna(subset=['st_tg'])

  # drop data that contain same station as a st_id and st_tg
  # e.g. row 10617 (after previous process)
  # train	st_no	st_id	date	arr_time	dep_time	stay_time	mileage	lat	    lon	        arr1	    dep1	    next_arr_time	next_st_no	adj_stay_time	st_tg	  travel_time	next_mileage
  # D2102	14	  2106	1	    0.870139	0.871528	2	        941	29. 735707	113.900578	0.870139	0.871528	0.870139	    15.0	      0.001389	    2106.0	-0.001389	  941.0
  df = df[~(df['st_id'] == df['st_tg'])]

  # keep only col we want
  df = df[['train', 'st_id', 'st_tg', 'mileage', 'next_mileage', 'arr1', 'dep1','next_arr_time',  'adj_stay_time', 'travel_time']].copy()
  df['next_arr_time'] = df['next_arr_time'].astype(float)

  # add train speed into df
  df['train'] = df['train'].astype(str) # convert 'train' column to string type
  df['train_max_speed'] = df['train'].apply(get_train_speed)  # apply the function to create 'train_speed' column

  # calculate distance
  # 1st try to cal from next_mileage - mileage if 'next_mileage' != 0
  # 2nd using ditance = travel_time * train_max_speed if travel_time > 0 (exist data error)
  # 3rd make it 0
  df['distance'] = df.apply(lambda row: row['next_mileage']-row['mileage'] if row['next_mileage'] != 0 else (row['travel_time']*24*row['train_max_speed'] if row['travel_time']>0 else 0), axis=1)

  # re calculate travel time from distance if previous calulated travel time is negative
  # 1st travel_time = distanec/train_speed if travel_time < 0 (just correct for negative one) and distance != 0 (from previous some distance = 0 in 3rd try)
  # 2nd make it same
  df['adj_travel_time'] = df.apply(lambda row: row['distance']/row['train_max_speed']/24 if (row['travel_time'] < 0 and row['distance']!=0) else row['travel_time'], axis=1)

  ''' ** after above step i didnot found any no value of travel time with negative values or 0 ** '''

  # only keep the col we want and change the name of col
  df = df[['st_id', 'st_tg', 'train', 'train_max_speed', 'arr1', 'dep1', 'next_arr_time', 'adj_stay_time', 'distance', 'adj_travel_time']]
  df = df.rename(columns={'arr1': 'arr_time', 
                              'dep1': 'dep_time',
                              'next_arr_time': 'next_arr_time',
                              'adj_stay_time': 'stay_time',
                              'adj_travel_time': 'travel_time'
                              })
  
  # add col of time of time format
  df['arr_time_t'] = df['arr_time'].apply(convert_time_format)
  df['dep_time_t'] = df['dep_time'].apply(convert_time_format)
  df['next_arr_time_t'] = df['next_arr_time'].apply(convert_time_format)
  df['travel_time_t'] = df['travel_time'].apply(convert_time_format)
  
  # change type some cols
  df['st_id'] = df['st_id'].astype(int)
  df['st_tg'] = df['st_tg'].astype(int)
  df['train'] = df['train'].astype(str)
  df['train_max_speed'] = df['train_max_speed'].astype(int)
  df['distance'] = df['distance'].astype(float)

  return df, pos


if __name__ == '__main__':
    raw_data = pd.read_excel('Railway_Data_JL.xlsx')
    df, pos = chinese_railway_prep(raw_data)
    print(len(df), len(pos))
    print(df.head())