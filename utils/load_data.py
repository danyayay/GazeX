"""Data loading and preprocessing utilities for trajectory prediction.

This module provides classes and functions for loading trajectory data,
preprocessing it into train/val/test splits, and creating PyTorch DataLoaders
for model training and evaluation.

Classes:
    TrajectoryDataset: PyTorch Dataset for trajectory data.
    StandardScaler: Normalizes time series data.
    StandardScalerAux: Normalizes auxiliary (contextual) information.

Functions:
    load_dataset: Loads and preprocesses trajectory data.
    get_dataloaders: Creates PyTorch DataLoaders for training/evaluation.
    nanstd: Computes standard deviation while handling NaN values.
"""

import os
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
        

class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory prediction.
    
    Handles loading and preprocessing of trajectory sequences with support
    for multiple trajectory representations (absolute, relative delta, etc.)
    and auxiliary contextual information.
    
    Attributes:
        in_trajs: Input trajectory coordinates.
        in_feats: Input features (non-trajectory data).
        out_trajs: Output trajectory coordinates.
        out_feats: Output features (non-trajectory data).
        auxs: Auxiliary contextual information (e.g., experiment setup).
        traj_format: Trajectory representation format.
    """
    
    def __init__(self, inputs, outputs, auxs, traj_format='rel_delta'):
        """Initialize the trajectory dataset.
        
        Args:
            inputs: Input array of shape (n_samples, seq_len, n_features).
            outputs: Output array of shape (n_samples, pred_len, n_features).
            auxs: Auxiliary information array.
            traj_format: Trajectory coordinate format.
                'rel_delta': Relative differences (default).
                'rel_to_origin': Relative to first frame.
                'rel_to_t': Relative to last observation frame.
                'absolute': Use absolute coordinates.
        """
        traj_len = 2
        self.in_trajs = torch.tensor(inputs[:, :, :traj_len], dtype=torch.float32)
        self.in_feats = torch.tensor(inputs[:, :, traj_len:], dtype=torch.float32)
        self.out_trajs = torch.tensor(outputs[:, :, :traj_len], dtype=torch.float32)
        self.out_feats = torch.tensor(outputs[:, :, traj_len:], dtype=torch.float32)
        self.auxs = torch.tensor(auxs).long()
        self.is_return_aux = True if len(auxs) else False

        self.traj_format = traj_format
        if traj_format == 'rel_delta':
            self.in_trajs_rel = torch.zeros(self.in_trajs.shape, dtype=torch.float32)
            self.in_trajs_rel[:, 1:, :] = self.in_trajs[:, 1:] - self.in_trajs[:, :-1]
            self.out_trajs_rel = torch.zeros(self.out_trajs.shape, dtype=torch.float32)
            self.out_trajs_rel[:, 0, :] = self.out_trajs[:, 0, :] - self.in_trajs[:, -1, :]
            self.out_trajs_rel[:, 1:, :] = self.out_trajs[:, 1:] - self.out_trajs[:, :-1]
            assert (torch.cumsum(self.in_trajs_rel[0], dim=0) + torch.unsqueeze(self.in_trajs[0], dim=0) == self.in_trajs[0]).any()
            assert (torch.cumsum(self.out_trajs_rel[0], dim=0) + torch.unsqueeze(self.in_trajs[0][-1], dim=0) == self.out_trajs[0]).any()
        elif traj_format == 'rel_to_origin':
            self.in_trajs_rel = self.in_trajs - self.in_trajs[:, 0].unsqueeze(1)
            self.out_trajs_rel = self.out_trajs - self.in_trajs[:, 0].unsqueeze(1)
        elif traj_format == 'rel_to_t':
            self.in_trajs_rel = self.in_trajs - self.in_trajs[:, -1].unsqueeze(1)
            self.out_trajs_rel = self.out_trajs - self.in_trajs[:, -1].unsqueeze(1)
        
        # print('------')
        # # print statistics 
        # print('Data statistics:')
        # if self.in_feats.shape[2]:
        #     print(f"in_feats: \n  mean={np.round(self.in_feats.mean(axis=(0,1)).numpy(), 2)}, \tstd={np.round(self.in_feats.std(axis=(0,1)).numpy(), 2)},", \
        #         f"\n  min ={np.round(self.in_feats.min(axis=1).values.min(axis=0).values.numpy(), 2)}, \tmax={np.round(self.in_feats.max(axis=1).values.max(axis=0).values.numpy(), 2)}")
        # print(f"in_trajs_rel : mean={np.round(self.in_trajs_rel.mean(axis=(0,1)).numpy(), 2)}, \tstd={np.round(self.in_trajs_rel.std(axis=(0,1)).numpy(), 2)}")
        # print(f"out_trajs_rel: mean={np.round(self.out_trajs_rel.mean(axis=(0,1)).numpy(), 2)}, \tstd={np.round(self.out_trajs_rel.std(axis=(0,1)).numpy(), 2)}")
        # if auxs.shape[1]: print(f"auxs: mean={auxs.mean(axis=0)}, std:{auxs.std(axis=0)}")

    def __len__(self):
        return len(self.in_trajs)
    
    def __getitem__(self, idx):
        auxs = self.auxs[idx] if self.is_return_aux else torch.tensor([])
        if self.traj_format != 'absolute':
            in_trajs_rel = self.in_trajs_rel[idx]
            out_trajs_rel = self.out_trajs_rel[idx]
        else:
            in_trajs_rel = self.in_trajs[idx]
            out_trajs_rel = self.out_trajs[idx]
        inputs = torch.cat([self.in_trajs[idx], self.in_feats[idx]], dim=-1)
        inputs_rel = torch.cat([in_trajs_rel, self.in_feats[idx]], dim=-1)
        # outputs = self.out_trajs[idx]
        outputs = torch.cat([self.out_trajs[idx], self.out_feats[idx]], dim=-1)
        outputs_rel = torch.cat([out_trajs_rel, self.out_feats[idx]], dim=-1)
        return inputs, outputs, inputs_rel, outputs_rel, auxs
    

def load_dataset(
        data_dir, base_motion=['traj', 'vel'], target=['traj'],
        use_headeye=None, use_pod=None,
        use_expt=None, use_person=None, aux_format='raw'):
    """Load and preprocess trajectory data from NPZ files.

    The NPZ files store data with shape (n_samples, seq_len, n_features).
    Column 0 is the string PSID identifier; columns 1+ are float features.
    Both x (observation) and y (prediction) arrays share the same column layout.

    Feature selection maps option names to column names in the NPZ:
      base_motion: 'traj' -> [loc_x, loc_y], 'vel' -> [vel_x, vel_y]
      use_headeye: head/eye direction and gaze event columns
      use_pod:     automated shuttle position, distance, and eHMI columns
      use_expt:    experimental setup columns stored in dts_qn.csv (aux)
      use_person:  participant demographic columns stored in dts_qn.csv (aux)

    The returned dict contains x/y/aux arrays for train/val/test, plus
    'ts_names' (selected column names) and 'target_indices' (positions of
    prediction targets within ts_names, used by Supervisor to slice y).
    """

    # Maps each feature option to the NPZ column names it selects.
    # Mutually exclusive groups (head direction, eye direction, gaze event)
    # are separated so the elif logic below stays readable.
    HEADEYE_COLS = {
        # head direction (mutually exclusive)
        'head_in_space':        ['head_yaw'],
        'head_in_walking':      ['head_vel_relyaw'],
        'head_vislet':          ['head_sinyaw', 'head_cosyaw'],
        # eye direction (mutually exclusive)
        'eye_in_space':         ['eye_yaw'],
        'eye_in_walking':       ['eye_vel_relyaw'],
        'eye_vislet':           ['eye_sinyaw', 'eye_cosyaw'],
        'eye_n_head':     ['eye_head_relyaw', 'head_vel_relyaw'],
        'eye_n_head_abs': ['eye_head_relyaw', 'head_yaw'],
        # gaze event (mutually exclusive)
        'hit_all':        ['hit_env', 'hit_goal', 'hit_leader', 'hit_follower'],
        'hit_pod':        ['hit_leader', 'hit_follower'],
        # 'eye_pehmi':      ['perceived_ehmileader', 'perceived_ehmifollower'],
        'gaze_events':          ['ind_att', 'ind_sac', 'ind_noise'],
        'presence_of_attn':     ['ind_att'],
        'attn_on_traffic':      ['ind_att_pod', 'ind_att_nonpod'],
        'attn_distribution':    ['ind_att_leader', 'ind_att_follower', 'ind_att_goal', 'ind_att_others'],
    }
    POD_COLS = {
        'pod_loc':        ['leader_x', 'follower_x'],
        'podleader_loc':  ['leader_x'],
        'pod_dist':       ['dist_pedleader_x', 'dist_pedleader_y', 'dist_pedfollower_x', 'dist_pedfollower_y'],
        'podleader_dist': ['dist_pedleader_x', 'dist_pedleader_y'],
        'pod_ehmi':       ['ehmileader', 'ehmifollower'],
    }
    TARGET_COLS = {
        'traj':       ['loc_x', 'loc_y'],
        'vel':        ['vel_x', 'vel_y'],
        'eye_degree': ['eye_yaw'],
        'eye_vislet': ['eye_sinyaw', 'eye_cosyaw'],
    }
    PERSON_COLS = {
        'age':       ['age'],
        'gender':    ['gender'],
        'education': ['education'],
        'hand':      ['dominant_hand'],
        'trust':     ['trust_score'],
        'pb':        ['v_score', 'l_score', 'p_score'],
        'cluster':   ['cluster'],
        'as':        ['as_familiarity', 'as_experience'],
    }
    # assert check
    if use_headeye is not None:
        assert np.array([i in HEADEYE_COLS.keys() for i in use_headeye]).all(), f"use_headeye must be one of {list(HEADEYE_COLS.keys())} or None, got: {use_headeye}"
    if use_pod is not None:
        assert np.array([i in POD_COLS.keys() for i in use_pod]).all(), f"use_pod must be one of {list(POD_COLS.keys())} or None, got: {use_pod}"
    if use_person is not None:
        assert np.array([i in PERSON_COLS.keys() for i in use_person]).all(), f"use_person must be one of {list(PERSON_COLS.keys())} or None, got: {use_person}"
    if 'traj' not in base_motion and 'vel' not in base_motion:
        raise ValueError(f"base_motion must include 'traj' or 'vel', got: {base_motion}")
    
    # Load column names from the train split (same layout across all splits).
    columns = np.load(os.path.join(data_dir, 'train.npz'), allow_pickle=True)['columns']

    def col_indices(names):
        return np.where(np.isin(columns, names))[0].tolist()

    # --- Build input feature index list ---
    # Column 0 is the PSID string; numeric features start at index 1.
    cols_input_indices = []
    if 'traj' in base_motion:
        cols_input_indices.extend(col_indices(['loc_x', 'loc_y']))
    if 'vel' in base_motion:
        cols_input_indices.extend(col_indices(['vel_x', 'vel_y']))

    # head/eye direction and gaze events (within each sub-group, options are mutually exclusive)
    if use_headeye is not None:
        _head_opts  = ['head_in_space', 'head_vislet', 'head_in_walking']
        _eye_opts   = ['eye_in_space', 'eye_vislet', 'eye_in_walking', 'eye_n_head', 'eye_n_head_abs']
        _gaze_opts  = ['hit_all', 'hit_pod', 'eye_pehmi', 'event_overall',
                       'attn_overall', 'attn_traffic', 'attn_detail']
        for group in [_head_opts, _eye_opts, _gaze_opts]:
            for opt in group:
                if opt in use_headeye:
                    cols_input_indices.extend(col_indices(HEADEYE_COLS[opt]))
                    break  # only one option per sub-group

    # pod features (options are additive, not mutually exclusive)
    if use_pod is not None:
        for opt, names in POD_COLS.items():
            if opt in use_pod:
                cols_input_indices.extend(col_indices(names))

    print(f'Using features: {columns[cols_input_indices]}')

    # --- Build prediction target column names ---
    cols_target = []
    for opt, names in TARGET_COLS.items():
        if opt in target:
            cols_target.extend(names)

    # --- Build auxiliary feature names (from dts_qn.csv) ---
    if aux_format == 'onehot':
        enc = OneHotEncoder(handle_unknown='ignore', drop='first')
    aux_names = []
    if use_expt:
        aux_names.extend(['eHMI', 'yielding', 'angle', 'traffic_flow'])
    if use_person is not None:
        for opt, names in PERSON_COLS.items():
            if opt in use_person:
                aux_names.extend(names)

    print(f'Using auxiliary features: {aux_names}')

    # Load participant/questionnaire table for aux features.
    dts_path = os.path.join(data_dir, '..', 'dts_qn.csv')
    print(f'Loading datatable and questionnaire at {dts_path}...')
    dts = pd.read_csv(dts_path)

    # --- Load NPZ splits ---
    # x and y share the same column layout; both are sliced with cols_input_indices.
    print(f'Loading dataset...')
    data = dict()
    for cat in ['train', 'val', 'test']:
        print(f'Loading {data_dir}/{cat}.npz')
        data_cat = np.load(os.path.join(data_dir, f'{cat}.npz'), allow_pickle=True)
        data_cat_x = data_cat['x'].copy()
        data_cat_x[..., 1:] = data_cat['x'][..., 1:].astype(np.float32)

        data['x_' + cat] = np.nan_to_num(data_cat_x[..., cols_input_indices].astype(np.float32), nan=-999)
        data['y_' + cat] = np.nan_to_num(data_cat['y'][..., cols_input_indices].astype(np.float32), nan=-999)
        data['psid_' + cat] = data_cat_x[:, 0, 0]

    for cat in ['train', 'val', 'test']:
        dts_filtered = dts[dts['psid'].isin(data['psid_' + cat])]
        dts_filtered = dts_filtered.set_index('psid').loc[data['psid_' + cat]].reset_index()
        data['aux_' + cat] = dts_filtered[aux_names].values
        print(f"x_{cat}.shape: {data['x_' + cat].shape}, y_{cat}.shape: {data['y_' + cat].shape}, aux_{cat}.shape: {data['aux_' + cat].shape}")

    # --- Transform aux data if needed ---
    if len(aux_names) != 0:
        aux_data = np.concatenate([data['aux_train'], data['aux_val'], data['aux_test']], axis=0)
        if aux_format == 'onehot':
            print('Transforming aux data to onehot encoding...')
            enc.fit(aux_data)
            for cat in ['train', 'val', 'test']:
                data['aux_' + cat] = enc.transform(data['aux_' + cat]).toarray()
                print(f"aux_onehot_{cat}.shape: {data['aux_' + cat].shape}")

    data['ts_names'] = columns[cols_input_indices]
    data['aux_names'] = aux_names
    # target_indices: positions of prediction targets within ts_names.
    # Used by Supervisor to slice y_ -> y via y_[..., target_indices].
    data['target_indices'] = np.where(np.isin(data['ts_names'], cols_target))[0].tolist()
    return data


def get_dataloaders(data, batch_size, traj_format='rel_delta', is_pgm=False):
    if not is_pgm:
        dataset_train = TrajectoryDataset(
            data['x_train'], data['y_train'], data['aux_train'], traj_format=traj_format)
        dataset_val = TrajectoryDataset(
            data['x_val'], data['y_val'], data['aux_val'], traj_format=traj_format)
        dataset_test = TrajectoryDataset(
            data['x_test'], data['y_test'], data['aux_test'], traj_format=traj_format) 
    else:
        raise NotImplementedError

    data_loader = dict()
    data_loader['train_loader'] = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader['val_loader'] = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    data_loader['test_loader'] = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    return data_loader


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, data_train, group_indices_dict, target_indices=None, device='cpu', ts_names=None):
        '''
        data_train: numpy array, (n_samples, seq_len, n_features)
        target_indices: list of int, which indices to inverse transform
        ts_names: list of str, column names (used to exclude sin/cos features from normalization)
        '''
        self.target_indices = target_indices
        self.mean = torch.zeros(data_train.shape[-1], dtype=torch.float32).to(device)
        self.std = torch.ones(data_train.shape[-1], dtype=torch.float32).to(device)
        data_train = torch.from_numpy(data_train).to(device)

        # motion and eye
        cols = [*group_indices_dict['motion'][2:], *group_indices_dict['headeye']]
        binary_cols = [col for col in range(data_train.shape[-1]) if len(data_train[..., col].unique()) <= 2]
        # sin/cos features are already bounded in [-1, 1]; z-score normalization would break sin²+cos²=1
        sincos_cols = [i for i, name in enumerate(ts_names)
                       if name.endswith('sinyaw') or name.endswith('cosyaw')] if ts_names is not None else []
        cols = [col for col in cols if col not in binary_cols and col not in sincos_cols]
        # pod
        if 'pod' in group_indices_dict.keys():
            cols_pod = group_indices_dict['pod']
            pod_dim = int(len(cols_pod) / 2) 
        print('\nNormalizing all columns except trajs and binary columns...')
        
        # motion and headeye
        self.mean[cols] = data_train[:, :, cols].mean(dim=(0,1))
        self.std[cols] = data_train[:, :, cols].std(dim=(0,1))
        
        # pod
        if 'pod' in group_indices_dict.keys():
            # repeat mean and std for leader and follower
            data_pod = torch.concat(data_train[:, :, cols_pod].split(pod_dim, dim=-1), dim=0)
            self.mean[cols_pod] = data_pod.nanmean(dim=(0,1)).repeat(1, 1, 2)
            self.std[cols_pod] = nanstd(data_pod, dim=(0,1)).repeat(1, 1, 2)

            # all 
            cols = [*cols, *cols_pod]

        print('------ Time series -------')
        print('columns indices: ', cols)
        print('standard scaler mean:', np.round(self.mean.cpu().numpy()[cols], 2))
        print('standard scaler std: ', np.round(self.std.cpu().numpy()[cols], 2))

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std[self.target_indices]) + self.mean[self.target_indices]


def nanstd(x, dim=None, keepdim=False, unbiased=True):
    # mask out nans
    mask = ~torch.isnan(x)
    count = mask.sum(dim=dim, keepdim=True)
    
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    diff = x - mean
    diff = torch.where(mask, diff, torch.zeros_like(diff))
    
    sq_diff = diff ** 2
    var = sq_diff.sum(dim=dim, keepdim=True) / (count - (1 if unbiased else 0))
    
    std = torch.sqrt(var)
    if not keepdim and dim is not None:
        std = std.squeeze(dim)
    return std


class StandardScalerAux:
    """
    Standard the input
    """

    def __init__(self, data_train, device, eps=1e-6):
        '''
        data_train: numpy array, (n_samples, n_features)
        '''
        data_train = torch.from_numpy(data_train).float().to(device)
        self.mean = data_train.mean(dim=0)
        self.std = data_train.std(dim=0) + eps
        print('------ Aux info ------')
        print('standard scaler mean:', np.round(self.mean.cpu().numpy(), 2))
        print('standard scaler std: ', np.round(self.std.cpu().numpy(), 2))

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
        
    
if __name__ == '__main__':

    data_dir = 'data/indiv_time_o40_p40_s4'
    batch_size = 32
    traj_format = 'rel_delta'

    data = load_dataset(
        data_dir,
        base_motion=['traj', 'vel'],
        target=['traj'],
        use_headeye=None,
        use_pod=None,
        use_expt=True,
        use_person=None,
        aux_format='onehot',
    )
    dataloaders = get_dataloaders(data, batch_size, traj_format=traj_format)