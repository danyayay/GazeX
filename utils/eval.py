import re
import argparse
import numpy as np
import pandas as pd
import os
from model.historic import predict_historic_model
from utils.metric import compute_ade_fde_np, compute_rmse_over_horizon
from model.supervisor import Supervisor


def test_stochastic_preds(args, model, callback, device, best_ckpt_path):
    """Run test at mu, min-1, min-5, and min-20 using the best checkpoint.

    Returns a DataFrame with RMSE-per-horizon columns for each inference mode.
    """
    supervisor = Supervisor(
        model=model, callback=callback,
        device=device, ckpt_path=best_ckpt_path, args=args
    )

    df = pd.DataFrame(columns=['mu', 'min1', 'min5', 'min20'])

    print('\n##### Testing with the best checkpoint #####')

    supervisor.n_samples = 1
    print('#### mu (deterministic mean) ####')
    _, _, rmse = supervisor.test(0, False, is_infer_mu=True)
    df['mu'] = rmse

    for k, col in [(1, 'min1'), (5, 'min5'), (20, 'min20')]:
        supervisor.n_samples = k
        print(f'#### min-{k} ####')
        _, _, rmse = supervisor.test(k, False, is_infer_mu=False)
        df[col] = rmse

    return df


def evaluate_by_angle(args, test_data_dir=None, preds_path=None):
    """Evaluate predictions separately for each approach angle (45/90/135 degrees)."""

    test_data_dir = test_data_dir or args.test_data_dir
    preds_path = preds_path or args.preds_path

    match = re.search(r'_o(\d+)_p(\d+)', test_data_dir)
    assert match is not None, 'Invalid data directory'
    len_pred = int(match.group(2))

    model_name = 'historic' if preds_path == 'historic' else preds_path.split('/')[-2].split('_')[-1]

    print(f'Loading data at {test_data_dir}...')
    data_test = np.load(test_data_dir, allow_pickle=True)
    psid_list = data_test['x'][:, 0, 0]
    x = data_test['x'][..., 1:3].astype(np.float32)
    y_labels = data_test['y'][..., 1:3].astype(np.float32)

    if model_name == 'historic':
        y_preds = predict_historic_model(x, len_pred)
    else:
        print(f'Loading predictions at {preds_path}')
        preds = np.load(preds_path, allow_pickle=True)
        x_labels = preds['x_labels'].astype(np.float32)[..., :2]
        y_labels = preds['y_labels'].astype(np.float32)
        y_preds = preds['y_preds'].astype(np.float32)
        assert (x_labels - x).sum() < 1, 'x_labels and x do not match'

    dts_path = os.path.join(os.path.dirname(test_data_dir), '..', 'dts_combined.csv')
    if not os.path.exists(dts_path):
        print(f'Warning: dts_combined.csv not found at {dts_path}, falling back to overall evaluation')
        ade, fde = compute_ade_fde_np(y_preds, y_labels)
        print(f'Model: {model_name}, ADE: {ade:.4f}, FDE: {fde:.4f}')
        return {}

    dts = pd.read_csv(dts_path)
    psid_to_angle = dict(zip(dts['psid'].astype(str), dts['angle']))
    angles = np.array([psid_to_angle.get(str(p), np.nan) for p in psid_list])

    overall_ade, overall_fde = compute_ade_fde_np(y_preds, y_labels)
    print(f'\nModel: {model_name}')
    print(f'Overall  ADE={overall_ade:.4f}, FDE={overall_fde:.4f}')
    print('=' * 60)

    results = {}
    unique_angles = sorted([a for a in np.unique(angles) if not np.isnan(a)])
    for angle in unique_angles:
        mask = angles == angle
        ade, fde = compute_ade_fde_np(y_preds[mask], y_labels[mask])
        rmse = compute_rmse_over_horizon(y_preds[mask], y_labels[mask])
        results[angle] = {'count': int(mask.sum()), 'ade': ade, 'fde': fde, 'rmse': rmse}

        print(f'Angle {angle} (n={mask.sum()}): ADE={ade:.4f}, FDE={fde:.4f}')
        for i in range(0, rmse.shape[0], 5):
            print(f'  Horizon {i+1}: {rmse[i]:.4f}')
        
    print("\n" + "=" * 60)
    print("Summary by Angle:")
    print("=" * 60)
    for angle in unique_angles:
        if angle in results:
            result = results[angle]
            print(f"Angle {angle}: ADE={result['ade']:.4f}, FDE={result['fde']:.4f}, n={result['count']}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', default='data/indiv_time_o40_p40_s4/test.npz', type=str)
    parser.add_argument('--preds_path', default='logs/indiv_time_o40_p40_s4/sqlite_final/20250919_053434_LSTMse/20260412_182802_MultiModalLSTM/test_last_1_True.npz', type=str)
    parser.add_argument('--eval_by_angle', action='store_true', default=True)
    args = parser.parse_args()

    evaluate_by_angle(args)
