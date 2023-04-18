import pandas as pd
import numpy as np
from pre_proc_data import chinese_railway_prep


if __name__ == "__main__":
    path = 'Railway_Data_JL.xlsx'  #path
    data = pd.read_excel(path)  # read
    df, pos = chinese_railway_prep(data) # get clean data
    

    # print
    print(f'len_df: {len(df)}, len_pos: {len(pos)}')
    print(f'df: \n{df.head(2)}')
    print(f'pos:')
    count = 2
    for key, value in pos.items():
        if count == 0:
            break
        print(key, value)
        count -= 1
        
    df.to_csv('clean_data_no_latlon.csv', index=False)
