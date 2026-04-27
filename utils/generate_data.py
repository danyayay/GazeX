import os
import math
import random 
import argparse
import numpy as np
import pandas as pd
import torch 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.util_eye import extract_fix_sac_sp_df, extract_att_sac_sp_df_using_hit


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_sampling_together(xs, ys, psids, train_val_test_ratio=[0.7, 0.1, 0.2]):
    n_data = xs.shape[0]
    n_train = int(n_data * train_val_test_ratio[0])
    n_val = int(n_data * train_val_test_ratio[1])

    # generate random indices
    indices = np.random.permutation(n_data) # 0-indexed
    indices_train = indices[:n_train]
    indices_val = indices[n_train:n_train+n_val]
    indices_test = indices[n_train+n_val:]

    # get train and test data
    x_train, y_train, psid_train = xs[indices_train], ys[indices_train], psids[indices_train]
    x_val, y_val, psid_val = xs[indices_val], ys[indices_val], psids[indices_val]
    x_test, y_test, psid_test = xs[indices_test], ys[indices_test], psids[indices_test]

    return x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test


def fit_time_series_clusters(features_train, K=5):
    # ---- Step 4: Normalize + Cluster ---- 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_train)

    kmeans = KMeans(n_clusters=K, random_state=42)
    print('Fitting kmeans with K =', K)
    kmeans.fit(X_scaled)
    return kmeans, scaler


def individual_based_sampling_together(
        xs, ys, psids, train_val_test_ratio=[0.7, 0.1, 0.2], verbose=False):
    pids, unique_pids = get_unique_pids(psids)
    n_indiv = len(unique_pids)
    n_train = int(n_indiv * train_val_test_ratio[0])
    n_val = int(n_indiv * train_val_test_ratio[1])

    # generate random indices
    random.shuffle(unique_pids)
    pid_train = unique_pids[:n_train]
    pid_val = unique_pids[n_train:n_train+n_val]
    pid_test = unique_pids[n_train+n_val:]

    # get train and test data
    x_train = xs[[pid in pid_train for pid in pids]] # (data_point, len_obs, input_dim)
    y_train = ys[[pid in pid_train for pid in pids]]
    psid_train = psids[[pid in pid_train for pid in pids]]

    x_val = xs[[pid in pid_val for pid in pids]]
    y_val = ys[[pid in pid_val for pid in pids]]
    psid_val = psids[[pid in pid_val for pid in pids]]

    x_test = xs[[pid in pid_test for pid in pids]]
    y_test = ys[[pid in pid_test for pid in pids]]
    psid_test = psids[[pid in pid_test for pid in pids]]
    
    if verbose:
        return x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test, pids, pid_train, pid_val, pid_test
    else:
        return x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test


def individual_based_sampling_together_with_clusters(
        xs, ys, psids, xs_tsc, ys_tsc, features, 
        window_accumulated_dict, K=5, train_val_test_ratio=[0.7, 0.1, 0.2]):
    x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test, pids, pid_train, pid_val, pid_test = \
        individual_based_sampling_together(xs, ys, psids, train_val_test_ratio=train_val_test_ratio, verbose=True)

    # extend xs, ys with tsc features and clusters
    features_train = np.array([])
    for p in pid_train: 
        start = get_first_item_from_p_curr(window_accumulated_dict, int(p))
        end = get_last_item_from_p_curr(window_accumulated_dict, int(p))
        features_train = np.concatenate((features_train, features[start:end, :]), axis=0) if features_train.size else features[start:end, :]
    kmeans, scaler = fit_time_series_clusters(features_train, K=K)
    n_features = scaler.mean_.shape[0]

    x_train_tsc = xs_tsc[[pid in pid_train for pid in pids]] # (data_point, pad+1, len_obs, input_dim)
    y_train_tsc = ys_tsc[[pid in pid_train for pid in pids]]
    
    x_val_tsc = xs_tsc[[pid in pid_val for pid in pids]]
    y_val_tsc = ys_tsc[[pid in pid_val for pid in pids]]
    
    x_test_tsc = xs_tsc[[pid in pid_test for pid in pids]]
    y_test_tsc = ys_tsc[[pid in pid_test for pid in pids]]

    # extend with tsc features
    n_tsc_dim = 3 
    x_train_tsc_cluster = kmeans.predict(scaler.transform(
        x_train_tsc[..., -n_features:].reshape(-1, n_features))).reshape(x_train_tsc.shape[:n_tsc_dim])
    y_train_tsc_cluster = kmeans.predict(scaler.transform(
        y_train_tsc[..., -n_features:].reshape(-1, n_features))).reshape(y_train_tsc.shape[:n_tsc_dim])

    x_val_tsc_cluster = kmeans.predict(scaler.transform(
        x_val_tsc[..., -n_features:].reshape(-1, n_features))).reshape(x_val_tsc.shape[:n_tsc_dim])
    y_val_tsc_cluster = kmeans.predict(scaler.transform(
        y_val_tsc[..., -n_features:].reshape(-1, n_features))).reshape(y_val_tsc.shape[:n_tsc_dim])

    x_test_tsc_cluster = kmeans.predict(scaler.transform(
        x_test_tsc[..., -n_features:].reshape(-1, n_features))).reshape(x_test_tsc.shape[:n_tsc_dim])
    y_test_tsc_cluster = kmeans.predict(scaler.transform(
        y_test_tsc[..., -n_features:].reshape(-1, n_features))).reshape(y_test_tsc.shape[:n_tsc_dim])

    x_train_tsc = np.concatenate((x_train_tsc, x_train_tsc_cluster[..., np.newaxis]), axis=-1) # (data_point, pad+1, len_obs, input_dim + 1)
    y_train_tsc = np.concatenate((y_train_tsc, y_train_tsc_cluster[..., np.newaxis]), axis=-1) # (data_point, pad+1, len_pred, output_dim + 1)
    x_val_tsc = np.concatenate((x_val_tsc, x_val_tsc_cluster[..., np.newaxis]), axis=-1)
    y_val_tsc = np.concatenate((y_val_tsc, y_val_tsc_cluster[..., np.newaxis]), axis=-1)
    x_test_tsc = np.concatenate((x_test_tsc, x_test_tsc_cluster[..., np.newaxis]), axis=-1)
    y_test_tsc = np.concatenate((y_test_tsc, y_test_tsc_cluster[..., np.newaxis]), axis=-1)

    return x_train, y_train, psid_train, x_train_tsc, y_train_tsc, \
        x_val, y_val, psid_val, x_val_tsc, y_val_tsc,\
        x_test, y_test, psid_test, x_test_tsc, y_test_tsc


def scenario_based_sampling_together(xs, ys, psids, train_val_test_ratio=[0.7, 0.1, 0.2]):
    unique_psids = get_unique_psids(psids)
    n_psid = len(unique_psids)
    n_train = int(n_psid * train_val_test_ratio[0])
    n_val = int(n_psid * train_val_test_ratio[1])

    # generate random indices
    random.shuffle(unique_psids)
    psid_train = unique_psids[:n_train]
    psid_val = unique_psids[n_train:n_train+n_val]
    psid_test = unique_psids[n_train+n_val:]

    # get train and test data
    x_train = xs[[psid in psid_train for psid in psids]]
    y_train = ys[[psid in psid_train for psid in psids]]
    psid_train = psids[[psid in psid_train for psid in psids]]

    x_val = xs[[psid in psid_val for psid in psids]]
    y_val = ys[[psid in psid_val for psid in psids]]
    psid_val = psids[[psid in psid_val for psid in psids]]

    x_test = xs[[psid in psid_test for psid in psids]]
    y_test = ys[[psid in psid_test for psid in psids]]
    psid_test = psids[[psid in psid_test for psid in psids]]

    return x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test


def individual_based_sampling(dfs, train_val_test_ratio=[0.7, 0.1, 0.2]):
    n_indiv = dfs['pid'].unique().shape[0]
    n_train = int(n_indiv * train_val_test_ratio[0])
    n_val = int(n_indiv * train_val_test_ratio[1])

    # generate random indices
    indices = np.random.permutation(n_indiv) + 1 # 1-indexed
    pid_train = indices[:n_train]
    pid_val = indices[n_train:n_train+n_val]
    pid_test = indices[n_train+n_val:]

    # get train and test data
    dfs_train = dfs[dfs['pid'].isin(pid_train)]
    dfs_val = dfs[dfs['pid'].isin(pid_val)]
    dfs_test = dfs[dfs['pid'].isin(pid_test)]

    return dfs_train, dfs_val, dfs_test


def scenario_based_sampling(dfs, train_val_test_ratio=[0.7, 0.1, 0.2]):
    # index unique ids
    df_unique_ids = dfs[['pid', 'sid']].drop_duplicates()
    df_unique_ids.reset_index(drop=True, inplace=True)
    df_unique_ids.reset_index(drop=False, inplace=True)
    df_unique_ids.rename(columns={'index': 'id'}, inplace=True)
    
    # merge 
    dfs = pd.merge(dfs, df_unique_ids, on=['pid', 'sid'], how='left')
    n_total = df_unique_ids.shape[0]
    n_train = int(n_total * train_val_test_ratio[0])
    n_val = int(n_total * train_val_test_ratio[1])

    # generate random indices
    indices = np.random.permutation(n_total)
    pid_train = indices[:n_train]
    pid_val = indices[n_train:n_train+n_val]
    pid_test = indices[n_train+n_val:]

    # get train and test data
    dfs_train = dfs[dfs['id'].isin(pid_train)]
    dfs_val = dfs[dfs['id'].isin(pid_val)]
    dfs_test = dfs[dfs['id'].isin(pid_test)]

    return dfs_train, dfs_val, dfs_test 


def get_unique_pids(psids, return_pids=True):
    pids = [psid.split('_')[0] for psid in psids]
    unique_pids = list(set(pids))
    unique_pids.sort()
    if return_pids:
        return pids, unique_pids 
    else:
        return unique_pids


def get_unique_psids(psids):
    unique_psids = list(set(psids))
    unique_psids.sort()
    return unique_psids


def time_based_split(
        df, x_offsets, y_offsets, stride, pad=2, with_clusters=False):
    # get the shape 
    df.reset_index(drop=True, inplace=True)
    n_samples = df.shape[0]
    x, y, psid = [], [], []
    x_tsc, y_tsc = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(n_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t, stride):
        x_t = df.loc[(t + x_offsets).tolist(), ...]
        y_t = df.loc[(t + y_offsets).tolist(), ...]
        psid_t = df.loc[t, 'psid']
        x.append(x_t)
        y.append(y_t)
        psid.append(psid_t)
        if with_clusters:
            x_tsc_t, y_tsc_t = [], []
            padded = pd.concat([df.loc[[0]]] * pad, ignore_index=True).rename(
                index=dict(zip(np.arange(pad).tolist(), (np.arange(pad)-pad).tolist())))
            df_padded = pd.concat([padded, df])
            for k in range(pad+1):
                x_tsc_k = df_padded.loc[(t + x_offsets - k).tolist(), ...]
                y_tsc_k = df_padded.loc[(t + y_offsets - k).tolist(), ...]
                x_tsc_t.append(x_tsc_k)
                y_tsc_t.append(y_tsc_k)
            x_tsc_t = np.stack(x_tsc_t, axis=0)  # [pad+1, len_pred, n_features]
            y_tsc_t = np.stack(y_tsc_t, axis=0)  # [pad+1, len_pred, n_features]
            x_tsc.append(x_tsc_t)
            y_tsc.append(y_tsc_t)
    x = np.stack(x, axis=0) # [n_samples, len_pred, n_features]
    y = np.stack(y, axis=0) # [n_samples, len_pred, n_features]
    psid = np.stack(psid, axis=0) # [n_samples]
    if with_clusters:
        x_tsc = np.stack(x_tsc, axis=0) # [n_samples, pad+1, len_pred, n_features]
        y_tsc = np.stack(y_tsc, axis=0) # [n_samples, pad+1, len_pred, n_features]
    return x, y, psid, x_tsc, y_tsc


def preprocess(dfs):
    # get the needed columns
    cols = ['psid', 'Ped_Location_x_smoothed', 'Ped_Location_y_smoothed', 
            'Ped_Velocity_x_smoothed', 'Ped_Velocity_y_smoothed', 
            'Ped_Velocity_smoothed','Ped_Velocity_Rotation_smoothed', 
            'body_rotation_smoothed', 'Ped_Rotation_z_smoothed', 'EyeGaze_Rotation_z_smoothed', 
            'head_vel_relative_smoothed', 'eye_vel_relative_smoothed', 'eye_head_relative_smoothed',
            'PodLeader_Location_x', 'PodLeader_Location_y',
            'PodFollower_Location_x', 'PodFollower_Location_y',
            'PodLeader_Ped_CenterDistance_x', 'PodLeader_Ped_CenterDistance_y',
            'PodFollower_Ped_CenterDistance_x', 'PodFollower_Ped_CenterDistance_y',
            'PodLeader_EhmiStatus', 'PodFollower_EhmiStatus', 
            'HitObject_env', 'HitObject_goal', 'HitObject_neighbor', 'HitObject_pod_leader', 'HitObject_pod_follower', 'HitObject_widget',
            'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 'TimeElapsedTrial', 'ConfidenceValue']
    dfs = dfs[cols]
    dfs.rename(columns={
        'Ped_Location_x_smoothed': 'loc_x', 'Ped_Location_y_smoothed': 'loc_y', 
        'Ped_Velocity_x_smoothed': 'vel_x', 'Ped_Velocity_y_smoothed': 'vel_y', 
        'Ped_Velocity_smoothed': 'vel_r', 'Ped_Velocity_Rotation_smoothed': 'vel_yaw', 
        'body_rotation_smoothed': 'body_yaw', 'Ped_Rotation_z_smoothed': 'head_yaw', 'EyeGaze_Rotation_z_smoothed': 'eye_yaw', 
        'head_vel_relative_smoothed': 'head_vel_relyaw', 'eye_vel_relative_smoothed': 'eye_vel_relyaw', 'eye_head_relative_smoothed': 'eye_head_relyaw',
        'PodLeader_Location_x': 'leader_x', 'PodLeader_Location_y': 'leader_y',
        'PodFollower_Location_x': 'follower_x', 'PodFollower_Location_y': 'follower_y',
        'PodLeader_Ped_CenterDistance_x': 'dist_pedleader_x', 'PodLeader_Ped_CenterDistance_y': 'dist_pedleader_y',
        'PodFollower_Ped_CenterDistance_x': 'dist_pedfollower_x', 'PodFollower_Ped_CenterDistance_y': 'dist_pedfollower_y',
        'PodLeader_EhmiStatus': 'ehmileader', 'PodFollower_EhmiStatus': 'ehmifollower',
        'HitObject_goal': 'hit_goal', 'HitObject_pod_leader': 'hit_leader', 'HitObject_pod_follower': 'hit_follower'
        }, inplace=True)
    # transform yaw into (cos, sin) pairs
    dfs.loc[:, ['vel_sinyaw']] = np.sin(np.deg2rad(dfs['vel_yaw'])).fillna(-999)
    dfs.loc[:, ['vel_cosyaw']] = np.cos(np.deg2rad(dfs['vel_yaw'])).fillna(-999)
    dfs.loc[:, ['body_sinyaw']] = np.sin(np.deg2rad(dfs['body_yaw'])).fillna(-999)
    dfs.loc[:, ['body_cosyaw']] = np.cos(np.deg2rad(dfs['body_yaw'])).fillna(-999)
    dfs.loc[:, ['head_sinyaw']] = np.sin(np.deg2rad(dfs['head_yaw'])).fillna(-999)
    dfs.loc[:, ['head_cosyaw']] = np.cos(np.deg2rad(dfs['head_yaw'])).fillna(-999)
    dfs.loc[:, ['eye_sinyaw']] = np.sin(np.deg2rad(dfs['eye_yaw'])).fillna(-999)
    dfs.loc[:, ['eye_cosyaw']] = np.cos(np.deg2rad(dfs['eye_yaw'])).fillna(-999)
    # pod angle
    dfs['leader_yaw'] = np.rad2deg(np.atan2((dfs['leader_y'] - dfs['loc_y']).values, (dfs['leader_x'] - dfs['loc_x']).values))
    dfs['follower_yaw'] = np.rad2deg(np.atan2((dfs['follower_y'] - dfs['loc_y']).values, (dfs['follower_x'] - dfs['loc_x']).values))

    # make sure the indicator columns are in good format
    dfs.loc[:, ['ehmileader', 'ehmifollower']] = dfs[['ehmileader', 'ehmifollower']].replace(
        {None: '0', 'green': '1', 'red': '-1'}).astype(int)
    dfs.loc[:, ['hit_goal', 'hit_leader', 'hit_follower', 'HitObject_env', 'HitObject_neighbor', 'HitObject_widget']] = dfs[
        ['hit_goal', 'hit_leader', 'hit_follower', 'HitObject_env', 'HitObject_neighbor', 'HitObject_widget']].astype(int)
    dfs.loc[:, ['hit_others']] = dfs.HitObject_env | dfs.HitObject_neighbor | dfs.HitObject_widget
    # add perceived emhi status
    dfs.loc[:, ['perceived_ehmileader']] = dfs["ehmileader"].where(dfs["hit_leader"] == 1).ffill().fillna(0).astype(int)
    dfs.loc[:, ['perceived_ehmifollower']] = dfs["ehmifollower"].where(dfs["hit_follower"] == 1).ffill().fillna(0).astype(int)
    # make sure the distances are not NaN
    dfs.loc[:, ['dist_pedfollower_x']] = dfs['dist_pedfollower_x'].fillna(-9999)
    dfs.loc[:, ['dist_pedfollower_y']] = dfs['dist_pedfollower_y'].fillna(-9999)

    # extract events from eye tracking
    dfs = dfs.groupby("psid", group_keys=False).apply(extract_att_sac_sp_df_using_hit)
    
    dfs.to_csv('data/dfs_ready.csv', index=False)

    # remove columns
    dfs.drop(columns=['GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 
                      'TimeElapsedTrial', 'ConfidenceValue',
                      'HitObject_env', 'HitObject_neighbor', 'HitObject_widget',
                      'mask_keep', 'eye_yaw_vel'], inplace=True)
    print('Columns=', dfs.columns.tolist())
    return dfs


def generate_seq2seq_io_data(
        dfs, len_obs, len_pred, past_ratio, stride, split_mode, with_clusters=False):
    dfs = preprocess(dfs)
    
    # get the split
    x, y, psid, x_tsc, y_tsc = [], [], [], [], []
    if split_mode == 'time_based':
        # 0 is the latest observed sample, observe the len_obs, and predict the len_pred 
        x_offsets = np.arange(-len_obs+1, 1, 1)
        y_offsets = np.arange(1, len_pred+1, 1)
        for _, group in dfs.groupby('psid'):
            if group.shape[0] < len_obs + len_pred:
                print(f"Group {group['psid'].values[0]} is too short, skipping")
                continue
            _x, _y, _psid, _x_tsc, _y_tsc = time_based_split(
                group, x_offsets, y_offsets, stride, with_clusters=with_clusters)
            x.append(_x)
            y.append(_y)
            psid.append(_psid)
            if with_clusters:
                x_tsc.append(_x_tsc)
                y_tsc.append(_y_tsc)
    else:
        raise ValueError(f"Split mode {split_mode} not supported")

    #       try to pad or find another way to store the such data
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    psid = np.concatenate(psid, axis=0)
    if with_clusters:
        x_tsc = np.concatenate(x_tsc, axis=0)
        y_tsc = np.concatenate(y_tsc, axis=0)
    return x, y, psid, x_tsc, y_tsc, dfs.columns


def get_dict_item(d, key1, key2):
    if d.get((key1, key2)) is not None:
        return d[(key1, key2)]
    else:
        return get_dict_item(d, key1, key2-1)
    

def get_last_item_from_p_prev(d, p_curr):
    s_curr = 12
    if d.get((p_curr-1, s_curr)) is not None:
        return d[(p_curr-1, s_curr)]
    else:
        return get_dict_item(d, p_curr-1, s_curr-1)


def get_first_item_from_p_curr(d, p_curr):
    s_curr = 1
    if d.get((p_curr, s_curr)) is not None:
        return d[(p_curr, s_curr)]
    else:
        return get_dict_item(d, p_curr, s_curr+1)
    

def get_last_item_from_p_curr(d, p_curr):
    s_curr = 12
    if d.get((p_curr, s_curr)) is not None:
        return d[(p_curr, s_curr)]
    else:
        return get_dict_item(d, p_curr, s_curr-1)
    

def generate_sliding_windows(dfs, window_size=5, step_size=3):
    data = dfs
    pids = data.pid.unique()
    sids = data.sid.unique()
    pids.sort()
    sids.sort()

    half_window = window_size // 2
    features = []
    positions = []
    window_dict = dict()
    window_accumulated_dict = dict()

    for k, p in enumerate(pids):
        window_dict[(p.item(), 0)] = 0
        if k == 0:
            window_accumulated_dict[(p.item(), 0)] = 0
        else:
            window_accumulated_dict[(p.item(), 0)] = get_last_item_from_p_prev(window_accumulated_dict, p.item())
        for j, s in enumerate(sids):
            # Prepare data for each scenario
            # Prepare data for each scenario
            d = data[(data.pid == p) & (data.sid == s)]
            if d.shape[0] != 0:
                d.loc[:, ['window_speed', 'window_acc', 'window_angle']] = np.nan
                # ---- Step 2: Compute features ----
                speed = d['Ped_Velocity_smoothed'].values
                acc = np.gradient(speed)
                angle_change = np.gradient(d['Ped_Velocity_Rotation_smoothed'].values)

                # Pad the time series with edge values
                pad_speed = np.pad(speed, (half_window, half_window), mode='edge')
                pad_acc = np.pad(acc, (half_window, half_window), mode='edge')
                pad_angle = np.pad(angle_change, (half_window, half_window), mode='edge')

                # ---- Step 3: Sliding window feature extraction ----
                n_slides = 0
                for i in range(half_window, len(speed) - half_window, step_size):
                    window_speed = pad_speed[i:i+window_size]
                    window_acc = pad_acc[i:i+window_size]
                    window_angle = pad_angle[i:i+window_size]
                    
                    f = [
                        np.mean(window_speed),
                        # np.std(window_speed),
                        np.mean(window_acc),
                        # np.std(window_acc),
                        np.mean(window_angle),
                        # np.std(window_angle),
                    ]
                    features.append(f)
                    positions.append(i)  # center of window
                    n_slides += 1
                    d.iloc[i, -3:] = f
                print(f"Processing pid={p} sid={s}: {len(speed)} points, {n_slides} windows extracted.")
                # record the start and end of each scenario
                if s == 1:
                    window_dict[(p.item(), s.item())] = n_slides
                else:
                    # Update the end of the previous scenario
                    window_dict[(p.item(), s.item())] = n_slides + get_dict_item(window_dict, p.item(), s.item()-1)
                window_accumulated_dict[(p.item(), s.item())] = n_slides + get_dict_item(window_accumulated_dict, p.item(), s.item()-1)
            d.loc[:, ['window_speed']] = d['window_speed'].interpolate(method='nearest').ffill().bfill()
            d.loc[:, ['window_acc']] = d['window_acc'].interpolate(method='nearest').ffill().bfill()
            d.loc[:, ['window_angle']] = d['window_angle'].interpolate(method='nearest').ffill().bfill()
            dfs.loc[(dfs.pid == p) & (dfs.sid == s), ['window_speed', 'window_acc', 'window_angle']] = d[['window_speed', 'window_acc', 'window_angle']]
    features = np.array(features)
    # window_accumulated_dict = {f"{k[0]}_{k[1]}": v for k, v in window_accumulated_dict.items()}
    print(f"Total features extracted: {features.shape[0]} windows, {features.shape[1]} features each.")

    return dfs, features, positions, window_dict, window_accumulated_dict


def generate_train_val_test_together(args):
    # read data 
    dfs = pd.read_csv(os.path.join(args.data_dir, 'dfs_combined.csv'), low_memory=False)
    dfs.loc[:, 'psid'] = dfs['pid'].astype(str) + '_' + dfs['sid'].astype(str)
    assert np.sum(args.train_val_test_ratio) == 1, "Train, validation, test ratio should sum to 1"
    
    out_dir = os.path.join(args.data_dir, 
        f'{args.sampling_mode[:5]}_{args.split_mode.split("_")[0]}_o{args.len_obs}_p{args.len_pred}_s{args.stride}')
    if args.sample: out_dir = out_dir + '_sample'
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # extend with sliding window feature sum
    if args.with_clusters:
        dfs, features, _, _, window_accumulated_dict = generate_sliding_windows(dfs, window_size=5, step_size=3)

    # x: (n_samples, len_obs, input_dim), y: (n_samples, len_pred, output_dim)
    xs, ys, psids, xs_tsc, ys_tsc, columns = generate_seq2seq_io_data(
        dfs, len_obs=args.len_obs, len_pred=args.len_pred, 
        past_ratio=args.past_ratio, stride=args.stride, 
        split_mode=args.split_mode, with_clusters=args.with_clusters)
    
    if args.sampling_mode == 'individual':
        if args.with_clusters:
            x_train, y_train, psid_train, x_tsc_train, y_tsc_train, x_val, y_val, psid_val, x_tsc_val, y_tsc_val, x_test, y_test, psid_test, x_tsc_test, y_tsc_test = \
                individual_based_sampling_together_with_clusters(xs, ys, psids, xs_tsc, ys_tsc, features, window_accumulated_dict, args.K, args.train_val_test_ratio)
        else: 
            x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test = \
                individual_based_sampling_together(xs, ys, psids, args.train_val_test_ratio, verbose=False)
    elif args.sampling_mode == 'scenario':
        x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test = \
            scenario_based_sampling_together(xs, ys, psids, args.train_val_test_ratio)
    elif args.sampling_mode == 'random':
        x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test = \
            random_sampling_together(xs, ys, psids, args.train_val_test_ratio)
    
    if args.sample: 
        if args.with_clusters:
            x_train, y_train, psid_train, x_tsc_train, y_tsc_train, \
                x_val, y_val, psid_val, x_tsc_val, y_tsc_val, \
                x_test, y_test, psid_test, x_tsc_test, y_tsc_test = \
                    get_sampled_data_with_clusters(x_train, y_train, psid_train, x_tsc_train, y_tsc_train, 
                        x_val, y_val, psid_val, x_tsc_val, y_tsc_val, x_test, y_test, psid_test, x_tsc_test, y_tsc_test,
                        args.n_sample, args.train_val_test_ratio)
        else:
            x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test = \
                get_sampled_data(x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test, args.n_sample, args.train_val_test_ratio)
    
    print("x_train shape: ", x_train.shape, ", y_train shape: ", y_train.shape, ", pid len: ", len(get_unique_pids(psid_train, return_pids=False)))
    if args.with_clusters: print("    x_tsc_train.shape ", x_tsc_train.shape, ", y_tsc_train.shape ", y_tsc_train.shape)
    if args.sampling_mode != 'random': print("    pid: ", get_unique_pids(psid_train, return_pids=False))
    print("x_val shape: ", x_val.shape, ", y_val shape: ", y_val.shape, ", pid len: ", len(get_unique_pids(psid_val, return_pids=False)))
    if args.with_clusters: print("    x_tsc_val.shape ", x_tsc_val.shape, ", y_tsc_val.shape ", y_tsc_val.shape)
    if args.sampling_mode != 'random': print("    pid: ", get_unique_pids(psid_val, return_pids=False))
    print("x_test shape: ", x_test.shape, ", y_test shape: ", y_test.shape, ", pid len: ", len(get_unique_pids(psid_test, return_pids=False)))
    if args.with_clusters: print("    x_tsc_test.shape ", x_tsc_test.shape, ", y_tsc_test.shape ", y_tsc_test.shape)
    if args.sampling_mode != 'random': print("    pid: ", get_unique_pids(psid_test, return_pids=False))

    if args.with_clusters:
        np.savez_compressed(os.path.join(out_dir, 'train'), x=x_train, y=y_train, x_tsc=x_tsc_train, y_tsc=y_tsc_train, columns=columns)
        np.savez_compressed(os.path.join(out_dir, 'val'), x=x_val, y=y_val, x_tsc=x_tsc_val, y_tsc=y_tsc_val, columns=columns)
        np.savez_compressed(os.path.join(out_dir, 'test'), x=x_test, y=y_test, x_tsc=x_tsc_test, y_tsc=y_tsc_test, columns=columns)
    else:
        np.savez_compressed(os.path.join(out_dir, 'train'), x=x_train, y=y_train, columns=columns)
        np.savez_compressed(os.path.join(out_dir, 'val'), x=x_val, y=y_val, columns=columns)
        np.savez_compressed(os.path.join(out_dir, 'test'), x=x_test, y=y_test, columns=columns)


def get_sampled_data(
        x_train, y_train, psid_train,
        x_val, y_val, psid_val, 
        x_test, y_test, psid_test,
        n_sample, train_val_test_ratio
    ):
    n_train = int(n_sample * train_val_test_ratio[0])
    n_val = int(n_sample * train_val_test_ratio[1])
    n_test = int(n_sample * train_val_test_ratio[2])
    print(f'Sampling the first {n_train} train, {n_val} val, {n_test} test examples for fast validation')
    x_train, y_train, psid_train = x_train[:n_train], y_train[:n_train], psid_train[:n_train]
    x_val, y_val, psid_val = x_val[:n_val], y_val[:n_val], psid_val[:n_val]
    x_test, y_test, psid_test = x_test[:n_test], y_test[:n_test], psid_test[:n_test]
    return x_train, y_train, psid_train, x_val, y_val, psid_val, x_test, y_test, psid_test


def get_sampled_data_with_clusters(
        x_train, y_train, psid_train, x_train_tsc, y_train_tsc,
        x_val, y_val, psid_val, x_val_tsc, y_val_tsc,
        x_test, y_test, psid_test, x_test_tsc, y_test_tsc,
        n_sample, train_val_test_ratio
    ):
    n_train = int(n_sample * train_val_test_ratio[0])
    n_val = int(n_sample * train_val_test_ratio[1])
    n_test = int(n_sample * train_val_test_ratio[2])
    print(f'Sampling the first {n_train} train, {n_val} val, {n_test} test examples for fast validation')
    x_train, y_train, psid_train, x_train_tsc, y_train_tsc = \
        x_train[:n_train], y_train[:n_train], psid_train[:n_train], x_train_tsc[:n_train], y_train_tsc[:n_train]
    x_val, y_val, psid_val, x_val_tsc, y_val_tsc = \
        x_val[:n_val], y_val[:n_val], psid_val[:n_val], x_val_tsc[:n_train], y_val_tsc[:n_train]
    x_test, y_test, psid_test, x_test_tsc, y_test_tsc = \
        x_test[:n_test], y_test[:n_test], psid_test[:n_test], x_test_tsc[:n_test], y_test_tsc[:n_test]
    return x_train, y_train, psid_train, x_train_tsc, y_train_tsc, \
        x_val, y_val, psid_val, x_val_tsc, y_val_tsc, \
        x_test, y_test, psid_test, x_test_tsc, y_test_tsc


def main(args):
    seed_everything(args.random_seed)
    print(f"Generating train, val, and test data... \n{args}")
    generate_train_val_test_together(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/", help="Output directory")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--n_sample", type=int, default=1000, help="Number of samples to sample")
    parser.add_argument("--sample", default=False, action='store_true',
                        help="whether firstly sample few data points for testing")
    parser.add_argument("--sampling_mode", type=str, default='individual',
                        choices=['individual', 'scenario', 'random'],
                        help="The way to sample train and test data: [individual, scenario, random]")
    parser.add_argument("--split_mode", type=str, default="time_based", 
                        choices=['time_based'],
                        help="Splitting based in which method: [time_based]")
    parser.add_argument("--train_val_test_ratio", default=[0.7, 0.1, 0.2], nargs='+', type=float, help="Train, validation, test ratio")

    # time based split mode (data frequency = 0.05s)
    parser.add_argument("--len_obs", type=int, default=40, help="Observation length for time based split")
    parser.add_argument("--len_pred", type=int, default=40, help="Prediction length for time based split")
    parser.add_argument("--stride", type=int, default=4)
    
    # distance based split mode
    parser.add_argument("--past_ratio", type=float, default=0.3, help="Past ratio for distance based split")
    
    # time series clustering
    parser.add_argument("--with_clusters", default=False, action='store_true', help="Whether to include time series clustering features")
    parser.add_argument("--K", type=int, default=5, help="Number of clusters for time series clustering")
    
    args = parser.parse_args()
    main(args)


# run 
# python -m utils.generate_data 