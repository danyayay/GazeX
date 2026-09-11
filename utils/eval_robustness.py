"""Evaluate robustness of trajectory prediction to gaze/head measurement noise.

Injects synthetic noise into the gaze and head-orientation channels at
inference time and reports FDE (deterministic / mu mode).

Noise types:
  1. Additive Gaussian on the yaw angle:  θ̃_t = θ_t + ε_t,  ε_t ~ N(0, σ²)
  2. Intermittent dropout: hold at last valid value for a random fraction p
  3. Eye→Head substitution: replace eye channel with head channel

Usage:
    conda run -n work2 python scripts/eval_noise_robustness.py
"""

import os
import sys
import copy
import argparse
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.util import seed_everything, get_config_file, setattrs, get_input_group_indices
from utils.init_model import init_model
from utils.device_utils import get_device
from utils.load_data import load_dataset, get_dataloaders, StandardScaler, StandardScalerAux
from utils.metric import compute_ade_fde_np, compute_rmse_over_horizon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODELS = {
    'head': {
        'config': 'logs/indiv_time_o40_p40_s4/20260416_105420_MultiModalLSTM/config.yaml',
        'headeye_type': 'head_vel_relyaw',   # 1D angle in degrees
        'headeye_cols': ['head_vel_relyaw'],
        'sub_source_cols': None,  # no eye→head substitution for head-only model
    },
    'eye': {
        'config': 'logs/indiv_time_o40_p40_s4/sqlite_final/20250918_113122_LSTMse/config.yaml',
        'headeye_type': 'eye_vislet',         # sin/cos representation
        'headeye_cols': ['eye_sinyaw', 'eye_cosyaw'],
        'sub_source_cols': ['head_sinyaw', 'head_cosyaw'],  # for eye→head substitution
    },
    'eye_context': {
        'config': 'logs/indiv_time_o40_p40_s4/sqlite_final/20250919_053434_LSTMse/config.yaml',
        'headeye_type': 'eye_in_walking',     # 1D angle in degrees
        'headeye_cols': ['eye_vel_relyaw'],
        'sub_source_cols': ['head_vel_relyaw'],  # for eye→head substitution
    },
}

NOISE_SIGMAS = [1, 2, 5, 10, 15, 20, 30]        # degrees
DROPOUT_FRACS = [0.10, 0.25, 0.50]

# ---------------------------------------------------------------------------
# Noise injection functions (operate on raw NPZ arrays before normalization)
# ---------------------------------------------------------------------------

def inject_gaussian_noise_angle(x_data, col_indices, sigma_deg, columns, rng):
    """Add Gaussian noise to angle-valued columns (in degrees).

    For sin/cos columns, converts to angle, adds noise, converts back.
    For direct angle columns, adds noise directly.
    """
    x = x_data.copy()
    col_names = columns[col_indices]

    if len(col_indices) == 2 and 'sinyaw' in col_names[0] and 'cosyaw' in col_names[1]:
        # sin/cos representation: convert to angle, add noise, convert back
        sin_vals = x[:, :, col_indices[0]].astype(np.float64)
        cos_vals = x[:, :, col_indices[1]].astype(np.float64)
        angles = np.degrees(np.arctan2(sin_vals, cos_vals))  # in degrees
        noise = rng.normal(0, sigma_deg, size=angles.shape)
        noisy_angles = angles + noise
        x[:, :, col_indices[0]] = np.sin(np.radians(noisy_angles)).astype(np.float32)
        x[:, :, col_indices[1]] = np.cos(np.radians(noisy_angles)).astype(np.float32)
    else:
        # Direct angle columns (degrees): add noise directly
        for idx in col_indices:
            noise = rng.normal(0, sigma_deg, size=x[:, :, idx].shape)
            x[:, :, idx] = x[:, :, idx].astype(np.float64) + noise
    return x


def inject_dropout(x_data, col_indices, dropout_frac, rng):
    """Intermittent dropout: for a random fraction p of the observation window,
    hold the gaze channel at its last valid value (sample-forward fill)."""
    x = x_data.copy()
    n_samples, seq_len, _ = x.shape

    for i in range(n_samples):
        # Select random timesteps to drop
        n_drop = max(1, int(seq_len * dropout_frac))
        drop_indices = rng.choice(seq_len, size=n_drop, replace=False)
        drop_mask = np.zeros(seq_len, dtype=bool)
        drop_mask[drop_indices] = True

        for t in range(seq_len):
            if drop_mask[t]:
                # Hold at last valid value (forward fill)
                if t > 0:
                    x[i, t, col_indices] = x[i, t - 1, col_indices]
                # If t==0 and dropped, keep original (no previous value)
    return x


def inject_eye_head_substitution(x_data, eye_col_indices, head_col_indices, columns):
    """Replace the eye channel with the head channel."""
    x = x_data.copy()
    eye_names = columns[eye_col_indices]
    head_names = columns[head_col_indices]

    if len(eye_col_indices) != len(head_col_indices):
        raise ValueError(
            f"Eye ({eye_names}) and head ({head_names}) column counts don't match")

    for ei, hi in zip(eye_col_indices, head_col_indices):
        x[:, :, ei] = x_data[:, :, hi]
    return x


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_with_noise(args, model, device, x_test_noisy, data_orig, seed=42):
    """Run evaluation on noisy test data. Returns (ade, fde)."""
    seed_everything(seed)

    # Rebuild datasets with noisy test x
    data = copy.deepcopy(data_orig)
    data['x_test'] = x_test_noisy

    # Build dataloaders
    data_loader = get_dataloaders(data, args.batch_size, args.traj_format, is_pgm=False)

    # Build scaler from TRAINING data (unchanged)
    ts_names = data['ts_names']
    group_indices_dict, _ = get_input_group_indices(ts_names)
    target_indices = data['target_indices']

    scaler_torch = StandardScaler(
        data['x_train'], group_indices_dict=group_indices_dict,
        target_indices=target_indices, device=device, ts_names=ts_names)

    scaler_aux = None
    if args.is_normalize_aux and args.aux_format == 'raw' and data['aux_train'].shape[-1] > 0:
        scaler_aux = StandardScalerAux(data['aux_train'], device=device)

    # Run inference
    model.eval()
    x_list, y_list, pred_list = [], [], []

    with torch.no_grad():
        for batch in data_loader['test_loader']:
            x, y_full_, x_rel, y_rel_full_, aux = batch
            x = x.to(device)
            y_full_ = y_full_.to(device)
            x_rel = x_rel.to(device)
            aux = aux.to(device)

            if args.is_normalize_ts:
                x_rel = scaler_torch.transform(x_rel)
            if scaler_aux is not None:
                aux = scaler_aux.transform(aux)

            y = y_full_[..., target_indices]

            # mu inference (deterministic mean)
            raw_out = model(x_rel, aux, num_samples=1, is_infer_mu=True)

            # Convert to absolute coordinates
            from utils.util import reverse_delta_to_abs, reverse_offset_to_abs
            if args.traj_format == 'rel_delta':
                pred = reverse_delta_to_abs(raw_out[..., :2], x[:, -1, :2])
            elif args.traj_format == 'rel_to_origin':
                pred = reverse_offset_to_abs(raw_out[..., :2], x[:, -1, :2])
            elif args.traj_format == 'rel_to_t':
                pred = reverse_offset_to_abs(raw_out[..., :2], x[:, -1, :2])
            else:
                pred = raw_out[..., :2]

            x_list.append(x.cpu().numpy())
            y_list.append(y.cpu().numpy())
            pred_list.append(pred.cpu().numpy())

    y_all = np.concatenate(y_list, axis=0)
    preds = np.concatenate(pred_list, axis=0)

    y_traj = y_all[..., :2]
    p_traj = preds[..., :2]

    ade, fde = compute_ade_fde_np(p_traj, y_traj)
    return float(ade), float(fde)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model_and_data(model_key, device):
    """Load model, checkpoint, and data for a given model configuration."""
    cfg = MODELS[model_key]
    config_path = cfg['config']

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args = get_config_file(config_path, args)

    # Force mu inference mode
    args.is_infer_mu = True
    args.train = False

    args = setattrs(args)
    model = init_model(args, device=device)

    # Load checkpoint
    ckpt_path = args.ckpt_path
    model = model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # Load raw data
    data = load_dataset(
        args.data_dir, base_motion=args.base_motion, target=args.target,
        use_headeye=args.use_headeye, use_pod=args.use_pod,
        use_expt=args.use_expt, use_person=args.use_person,
        aux_format=args.aux_format)

    # Also load full raw NPZ to get columns for noise injection
    raw_npz = np.load(os.path.join(args.data_dir, 'test.npz'), allow_pickle=True)
    all_columns = raw_npz['columns']
    raw_x_test = raw_npz['x'].copy()
    raw_x_test[..., 1:] = raw_x_test[..., 1:].astype(np.float32)

    return args, model, data, all_columns, raw_x_test


def get_col_indices_in_raw(all_columns, col_names):
    """Get column indices in the raw NPZ data."""
    indices = []
    for name in col_names:
        idx = np.where(all_columns == name)[0]
        if len(idx) == 0:
            raise ValueError(f"Column '{name}' not found in raw data")
        indices.append(idx[0])
    return indices


def get_col_indices_in_selected(ts_names, col_names):
    """Get column indices in the selected (model input) feature set."""
    indices = []
    for name in col_names:
        idx = np.where(np.array(ts_names) == name)[0]
        if len(idx) == 0:
            raise ValueError(f"Column '{name}' not found in ts_names: {ts_names}")
        indices.append(idx[0])
    return indices


def apply_noise_to_selected_data(x_test_clean, ts_names, all_columns, raw_x_test,
                                  model_cfg, noise_type, noise_param, rng):
    """Apply noise to the selected test features.

    Strategy: apply noise in the RAW domain (before column selection),
    then re-select the relevant columns to get noisy model input.
    """
    headeye_cols = model_cfg['headeye_cols']
    raw_col_indices = get_col_indices_in_raw(all_columns, headeye_cols)

    # Get the column selection indices used by load_dataset
    selected_col_indices = get_col_indices_in_selected(ts_names, headeye_cols)

    if noise_type == 'gaussian':
        sigma_deg = noise_param
        # Apply noise in the raw domain
        raw_noisy = inject_gaussian_noise_angle(
            raw_x_test, np.array(raw_col_indices), sigma_deg, all_columns, rng)
        # Copy clean data and replace only the noisy columns
        x_noisy = x_test_clean.copy()
        for sel_idx, raw_idx in zip(selected_col_indices, raw_col_indices):
            x_noisy[:, :, sel_idx] = np.nan_to_num(
                raw_noisy[:, :, raw_idx].astype(np.float32), nan=-999)

    elif noise_type == 'dropout':
        dropout_frac = noise_param
        x_noisy = x_test_clean.copy()
        # Apply dropout directly on the selected features
        x_noisy = inject_dropout(x_noisy, selected_col_indices, dropout_frac, rng)

    elif noise_type == 'substitution':
        sub_source_cols = model_cfg['sub_source_cols']
        if sub_source_cols is None:
            return None  # not applicable for head-only model

        raw_sub_indices = get_col_indices_in_raw(all_columns, sub_source_cols)
        raw_noisy = inject_eye_head_substitution(
            raw_x_test, np.array(raw_col_indices), np.array(raw_sub_indices), all_columns)

        x_noisy = x_test_clean.copy()
        for sel_idx, raw_idx in zip(selected_col_indices, raw_col_indices):
            x_noisy[:, :, sel_idx] = np.nan_to_num(
                raw_noisy[:, :, raw_idx].astype(np.float32), nan=-999)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return x_noisy


def run_experiments():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    device = get_device()
    rng = np.random.RandomState(42)

    results = {}  # {condition: {model_key: fde}}

    for model_key in ['head', 'eye', 'eye_context']:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_key}")
        print(f"{'='*60}")

        seed_everything(42)
        args, model, data, all_columns, raw_x_test = load_model_and_data(model_key, device)
        ts_names = data['ts_names']
        x_test_clean = data['x_test'].copy()
        model_cfg = MODELS[model_key]

        # --- Clean baseline ---
        print(f"\n--- Clean (σ=0°) ---")
        ade, fde = evaluate_with_noise(args, model, device, x_test_clean, data)
        condition = 'clean'
        results.setdefault(condition, {})[model_key] = fde
        print(f"  {model_key}: ADE={ade:.4f}, FDE={fde:.4f}")

        # --- Gaussian noise sweep ---
        for sigma in NOISE_SIGMAS:
            print(f"\n--- Gaussian σ={sigma}° ---")
            x_noisy = apply_noise_to_selected_data(
                x_test_clean, ts_names, all_columns, raw_x_test,
                model_cfg, 'gaussian', sigma, rng)
            ade, fde = evaluate_with_noise(args, model, device, x_noisy, data)
            condition = f'gaussian_{sigma}'
            results.setdefault(condition, {})[model_key] = fde
            print(f"  {model_key}: ADE={ade:.4f}, FDE={fde:.4f}")

        # --- Dropout ---
        for p in DROPOUT_FRACS:
            print(f"\n--- Dropout p={p:.2f} ---")
            x_noisy = apply_noise_to_selected_data(
                x_test_clean, ts_names, all_columns, raw_x_test,
                model_cfg, 'dropout', p, rng)
            ade, fde = evaluate_with_noise(args, model, device, x_noisy, data)
            condition = f'dropout_{p}'
            results.setdefault(condition, {})[model_key] = fde
            print(f"  {model_key}: ADE={ade:.4f}, FDE={fde:.4f}")

        # --- Eye→Head substitution ---
        print(f"\n--- Eye→Head substitution ---")
        x_noisy = apply_noise_to_selected_data(
            x_test_clean, ts_names, all_columns, raw_x_test,
            model_cfg, 'substitution', None, rng)
        if x_noisy is not None:
            ade, fde = evaluate_with_noise(args, model, device, x_noisy, data)
            condition = 'substitution'
            results.setdefault(condition, {})[model_key] = fde
            print(f"  {model_key}: ADE={ade:.4f}, FDE={fde:.4f}")
        else:
            condition = 'substitution'
            results.setdefault(condition, {})[model_key] = 'N/A'
            print(f"  {model_key}: N/A (head-only model)")

    # --- Print results table ---
    print(f"\n\n{'='*70}")
    print("RESULTS TABLE: FDE (cm), deterministic mode")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'+ Head':>10} {'+ Eye':>10} {'+ Eye+ctx':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    conditions = [
        ('clean', 'Clean (σ=0°)'),
    ] + [
        (f'gaussian_{s}', f'σ={s}°') for s in NOISE_SIGMAS
    ] + [
        (f'dropout_{p}', f'Dropout p={p:.2f}') for p in DROPOUT_FRACS
    ] + [
        ('substitution', 'Eye→Head'),
    ]

    for cond_key, cond_label in conditions:
        vals = results.get(cond_key, {})
        head_val = vals.get('head', '---')
        eye_val = vals.get('eye', '---')
        ctx_val = vals.get('eye_context', '---')

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        print(f"{cond_label:<25} {fmt(head_val):>10} {fmt(eye_val):>10} {fmt(ctx_val):>10}")

    # Save results to file
    import json
    results_serializable = {}
    for k, v in results.items():
        results_serializable[k] = {mk: (float(mv) if isinstance(mv, float) else mv) for mk, mv in v.items()}

    out_path = 'scripts/noise_robustness_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_experiments()