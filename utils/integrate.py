import os
import re
import numpy as np
import pandas as pd


vrdata_path = 'data/vrdata/preprocessed/'
vrdata_filenames = os.listdir(vrdata_path)
vrdata_filenames.sort()
n_ped = 51

# combine all VR log files 
dfs = pd.DataFrame()
for f in vrdata_filenames:
    df_i = pd.read_csv(f'{vrdata_path}/{f}')
    df_i['pid'] = int(re.findall(r'\d+', f)[0])
    if df_i.shape[0] == 0:
        continue
    print(f'Read {f}, num={df_i.shape[0]}')
    dfs = pd.concat([dfs, df_i], axis=0)

# get dummy of object
dfs = pd.concat([dfs, pd.get_dummies(dfs['HitObject'], prefix='HitObject')], axis=1)


dfs.drop(columns=['Unnamed: 0', 'Scenario', 'FamCar_Location_x', 'FamCar_Location_y', 
       'FamCar_Velocity_x', 'FamCar_Distance', 'FamCar_Ped_CenterDistance_x',
       'LeftController_Location_x', 'LeftController_Location_y', 'LeftController_Location_z',
       'RightController_Location_x', 'RightController_Location_y', 'RightController_Location_z',
       'direction', 'DiffPedLocation_y', 'Ped_Velocity', 'Ped_Velocity_abs', 'Ped_Velocity_abs_smoothed', 
       'body_rotation', 'body_rotation_unwrap', 'body_rotation_int', 
       'head_vel_relative', 'eye_vel_relative', 'eye_head_relative'], inplace=True)

dfs.to_csv('data/dfs_combined_new.csv', index=False)
print(f'Done!, num={dfs.shape[0]}')