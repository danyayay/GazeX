import os
import re
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import load_runtime_args, to_hierarchical_config
from utils.load_data import load_dataset


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


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


def get_consecutive_trues(bool_array):
    # Identify consecutive True groups
    diff = np.diff(np.concatenate(([False], bool_array, [False])).astype(int))
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]

    # Length of consecutive True sequences
    lengths = end_indices - start_indices
    # print("Consecutive True lengths:", lengths)  # Output: [2, 1, 3]
    return start_indices, lengths


def split_continuous_and_categorical_data(data):
    continuous_data = []
    categorical_data = []
    for i in range(data.shape[1]):
        if data[:, i].dtype == np.float32 or data[:, i].dtype == np.float64:
            continuous_data.append(data[:, i])
        else:
            categorical_data.append(data[:, i])
    return np.array(continuous_data), np.array(categorical_data)


def update_input_dim(arg_val, arg_name):
    # add mask column if needed 
    if 'eye' in arg_name.lower() or 'follow' in arg_name.lower():
        base_n = 1 
    else: 
        base_n = 0
    # add other columns 
    if arg_val == 'degree':
        return base_n + 1
    elif arg_val == 'sincos':
        return base_n + 2
    elif arg_val == 'xyrel':
        return base_n + 2
    elif arg_val == 'polar':
        return base_n + 1
    elif arg_val == 'polar_degree':
        return base_n + 2
    elif arg_val == 'polar_sincos':
        return base_n + 3
    elif arg_val == 'no':
        return 0
    else:
        raise ValueError(f"Invalid argument: {arg_val}")
    

def reverse_delta_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (batch, seq_len, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (batch, seq_len, 2)
    """
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj


def reverse_delta_to_abs_mx(rel_traj_mx, start_pos):
    """
    Inputs:
    - rel_traj_mx: pytorch tensor of shape (batch, seq_len, 2, n_type)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj_mx: pytorch tensor of shape (batch, seq_len, 2, n_type)
    """
    displacement = torch.cumsum(rel_traj_mx, dim=1)
    start_pos = torch.unsqueeze(torch.unsqueeze(start_pos, dim=1), dim=3)
    abs_traj_mx = displacement + start_pos
    return abs_traj_mx


def reverse_offset_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (batch, seq_len, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (batch, seq_len, 2)
    """
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = rel_traj + start_pos
    return abs_traj


def reverse_offset_to_abs_mx(rel_traj_mx, start_pos):
    """
    Inputs:
    - rel_traj_mx: pytorch tensor of shape (batch, seq_len, 2, n_type)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (batch, seq_len, 2, n_type)
    """
    start_pos = torch.unsqueeze(torch.unsqueeze(start_pos, dim=1), dim=3)
    abs_traj_mx = rel_traj_mx + start_pos
    return abs_traj_mx


class EntropyLossLogits(nn.Module):
    def __init__(self):
        super(EntropyLossLogits, self).__init__()

    def forward(self, x, eps=1e-7):
        if type(x) is torch.Tensor:
            b = x * torch.log(torch.clamp(x, min=eps))
        elif type(x) is np.ndarray:
            b = x * np.log(np.clip(x, eps, None))
        b = -1.0 * b.sum(-1)
        return b.mean()
    

class BalanceLoss(nn.Module):
    def __init__(self):
        super(BalanceLoss, self).__init__()

    def forward(self, x, eps=1e-7):
        # x.shape (batch_size*len_pred, n_type)
        if type(x) is torch.Tensor:
            # loss = torch.mean(torch.sum(x**2, dim=-1))
            loss = torch.mean(x**2)
        elif type(x) is np.ndarray:
            # loss = np.mean(np.sum(x**2, axis=-1))
            loss = np.mean(x**2)
        return loss
    

def entropy_loss_mrnn(layer_qz_T, eps=1e-20):
    if type(layer_qz_T) is torch.Tensor:
        entropy = - layer_qz_T * torch.log(layer_qz_T + eps)
        entropy = torch.mean(torch.sum(entropy, (1, 2)))
    elif type(layer_qz_T) is np.ndarray:
        entropy = - layer_qz_T * np.log(layer_qz_T + eps)
        entropy = np.mean(np.sum(entropy, axis=(1, 2)))
    return entropy


def get_config_file(config_filename, args):
    return load_runtime_args(config_filename, cli_args=args)


def save_config_file(config_filedir, args):
    config_filename = os.path.join(config_filedir, 'config.yaml')
    with open(config_filename, 'w') as f:
        yaml.safe_dump(to_hierarchical_config(args), f, sort_keys=False)
    print(f"Configuration saved to {config_filename}")


def get_input_group_indices(ts_names):
    group_indices_dict = {}
    group_dim_dict = {}
    # motion group
    group_indices_dict['motion'] = np.where(np.isin(
        ts_names, ['loc_x', 'loc_y', 'vel_x', 'vel_y']))[0].tolist()
    if len(group_indices_dict['motion']) > 0: 
        group_dim_dict['motion'] = len(group_indices_dict['motion'])
    # headeye group 
    group_indices_dict['headeye'] = np.where(np.isin(
        ts_names, ['head_yaw', 'head_sinyaw', 'head_cosyaw', 'head_vel_relyaw',
                    'eye_yaw', 'eye_sinyaw', 'eye_cosyaw', 'eye_vel_relyaw', 'eye_head_relyaw',
                    'hit_env', 'hit_goal', 'hit_leader', 'hit_follower',
                    'perceived_ehmileader', 'perceived_ehmifollower', 
                    'ind_sac', 'ind_noise', 'ind_att', 'ind_att_pod', 'ind_att_nonpod', 
                    'ind_att_leader', 'ind_att_follower', 'ind_att_goal', 'ind_att_others']))[0].tolist()
    if len(group_indices_dict['headeye']) > 0:
        group_dim_dict['headeye'] = len(group_indices_dict['headeye'])
    # pod group
    group_indices_dict['pod'] = np.where(np.isin(
        ts_names, ['dist_pedleader_x', 'dist_pedleader_y', 'ehmileader',
                        'dist_pedfollower_x', 'dist_pedfollower_y', 'ehmifollower', 
                        'leader_x', 'leader_y', 'follower_x', 'follower_y']))[0].tolist()
    if len(group_indices_dict['pod']) > 0:
        group_dim_dict['pod'] = len(group_indices_dict['pod'])
    
    return group_indices_dict, group_dim_dict


def setattr_seq_len(args):
    # extract sequence length from data directory
    match = re.search(r"_o(\d+)_p(\d+)", args.data_dir)
    assert match is not None, "Invalid data directory"
    setattr(args, 'len_obs', int(match.group(1)))
    setattr(args, 'len_pred', int(match.group(2)))
    return args


def setattr_input_dim(args):
    # base input dim 
    input_dim_base = 0 
    if 'traj' in args.base_motion: # basics: x, y
        input_dim_base += 2
    if 'vel' in args.base_motion: # basics: vx, vy
        input_dim_base += 2 
    
    # input dim 
    ats_dim = 0
    # use head/eye/hit features
    if args.use_headeye is not None:
        # head direction
        if 'head_in_space' in args.use_headeye: 
            ats_dim += 1
        elif 'head_vislet' in args.use_headeye: 
            ats_dim += 2
        elif 'head_in_walking' in args.use_headeye:
            ats_dim += 1
        # eye direction 
        if 'eye_in_space' in args.use_headeye: 
            ats_dim += 1
        elif 'eye_vislet' in args.use_headeye: 
            ats_dim += 2
        elif 'eye_in_walking' in args.use_headeye:
            ats_dim += 1
        elif 'eye_n_head' in args.use_headeye:
            ats_dim += 1
        # eye gaze
        if 'hit_all' in args.use_headeye: 
            ats_dim += 4
        elif 'hit_pod' in args.use_headeye:
            ats_dim += 2
        elif 'eye_pehmi' in args.use_headeye:
            ats_dim += 2
        elif 'event_overall' in args.use_headeye:
            ats_dim += 3
        elif 'attn_overall' in args.use_headeye:
            ats_dim += 1
        elif 'attn_traffic' in args.use_headeye:
            ats_dim += 2
        elif 'attn_detail' in args.use_headeye:
            ats_dim += 4
    
    # use pod features
    if args.use_pod is not None:
        if 'pod_loc' in args.use_pod:
            ats_dim += 2
        elif 'pod_dist' in args.use_pod:
            ats_dim += 4
        elif 'pod_ehmi' in args.use_pod:
            ats_dim += 2
        
    input_dim = ats_dim + input_dim_base
    setattr(args, 'input_dim', input_dim)
    setattr(args, 'ats_dim', ats_dim)

    return args


def setattr_aux_info(args):
    data = load_dataset(
        args.data_dir, base_motion=args.base_motion, target=args.target,
        use_headeye=args.use_headeye, use_pod=args.use_pod,
        use_expt=args.use_expt, use_person=args.use_person, aux_format=args.aux_format)
    aux_dim = data['aux_val'].shape[-1]
    ts_names = data['ts_names'].tolist()
    target_dim = len(data['target_indices'])
    setattr(args, 'aux_dim', aux_dim)
    setattr(args, 'ts_names', ts_names)
    setattr(args, 'target_dim', target_dim)
    return args


def setattr_output_dim(args):
    # base output dim
    output_dim_base = 0
    if 'traj' in args.target:
        output_dim_base += 2
    if 'vel' in args.target:
        output_dim_base += 2
    if 'eye_degree' in args.target:
        output_dim_base += 1
    if 'eye_vislet' in args.target:
        output_dim_base += 2

    # output dim 
    if args.model in ['lstm', 'lstm_eye', 'lstm_eye_mx', 'aux_lstm', 'lstm_att', 'multimodallstm']:
        if not args.is_deter: # stochastic
            if args.diagonal_cov:
                setattr(args, 'output_dim', output_dim_base*2)
            else:
                setattr(args, 'output_dim', int(output_dim_base+(output_dim_base*(output_dim_base+1))/2))
        else: # deterministic 
            setattr(args, 'output_dim', output_dim_base)
    else:
        setattr(args, 'output_dim', output_dim_base)
    return args


def setattrs(args):
    args = setattr_seq_len(args)
    args = setattr_input_dim(args)
    args = setattr_output_dim(args)
    args = setattr_aux_info(args)
    return args




def get_device() -> torch.device:
    """Detect and return the best available compute device.
    
    Automatically selects the best available device in the following order:
    1. CUDA (NVIDIA GPU) - if available
    2. MPS (Apple Metal Performance Shaders) - if on macOS
    3. CPU - fallback default
    
    Returns:
        torch.device: The best available device for computation
        
    Examples:
        >>> device = get_device()
        >>> tensor = torch.randn(100, 100, device=device)
        >>> print(f"Using device: {device}")
        Using device: cuda
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed information about available compute devices.
    
    Returns:
        dict: Dictionary containing device availability and info
        
    Examples:
        >>> info = get_device_info()
        >>> print(f"CUDA available: {info['cuda_available']}")
        >>> print(f"MPS available: {info['mps_available']}")
    """
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": str(get_device()),
    }
