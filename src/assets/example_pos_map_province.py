import pandas as pd
import numpy as np
from pre_proc_data import chinese_railway_prep
from pos_map_province import extract_city_state, clean_blank_province

if __name__ == '__main__':
    
    '''
    - this step will take a long time because it have to call geopy API to get province mapping
    - good to try, but you can skip and load df_pos_with_state.csv that already mapped
    '''
    # # load pos
    # path = 'Railway_Data_JL.xlsx'  #path
    # data = pd.read_excel(path)  # read
    # _, pos = chinese_railway_prep(data) # get clean data
    
    # # convert pos dict to df
    # df_pos = pd.DataFrame(pos.items(), columns=['node', 'pos'])
    # df_pos[['lat', 'lon']] = df_pos['pos'].apply(lambda x: pd.Series({'lat': x[0], 'lon': x[1]}))

    # # map province(state) and city into each lat-lon
    # df_pos[['city', 'state']] = df_pos.apply(lambda x: pd.Series(extract_city_state(str(x['lat']), str(x['lon']))), axis=1)
    
    # # save into csv
    # df_pos.to_csv('df_pos_with_state.csv', index=False)

    '''
    - after this will load data from df_pos_with_state.csv
    - after this want to clean blank province by using RandomForestClassifier to specify suitable cluster
    - and also assign each province into seperate number
    '''
    # load non clean data from file and clean it
    df_pos_with_state = pd.read_csv('df_pos_with_state.csv')
    df_pos_with_state_clean  = clean_blank_province(df_pos_with_state)
    
    print(f'df_pos_with_state_clean \n {df_pos_with_state_clean.head()}')

    # save into csv
    df_pos_with_state_clean.to_csv('df_pos_with_state_clean.csv', index=False)

