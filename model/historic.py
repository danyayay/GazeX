import re
import argparse
import numpy as np
from utils.metric import compute_ade_fde_np, compute_rmse_over_horizon


def predict_historic_model(x, len_pred=20):
    # x.shape = (batch_size, len_past, input_dim)
    # y.shape = (batch_size, len_future, output_dim)
     
    diff = np.diff(x, axis=1)[:, -1:, :]
    y_preds = x[:, -1:, :] + np.cumsum(np.repeat(diff, len_pred, axis=1), axis=1)

    return y_preds


def main(args):
    # get horizons 
    match = re.search(r"_o(\d+)_p(\d+)", args.test_data_dir)
    assert match is not None, "Invalid data directory"
    setattr(args, 'len_obs', int(match.group(1)))
    setattr(args, 'len_pred', int(match.group(2)))

    # load data 
    print(f'Loading data at {args.test_data_dir}...')
    data_test = np.load(args.test_data_dir, allow_pickle=True)
    x = data_test['x'][..., 1:3].astype(np.float32)
    y_labels = data_test['y'][..., 1:3].astype(np.float32)
    y_preds = predict_historic_model(x, len_pred=args.len_pred)

    # compute loss 
    ade, fde = compute_ade_fde_np(y_preds, y_labels)
    print("Historic model:")
    print(f"Overall ADE: {ade:.4f} \nOverall FDE: {fde:.4f}")

    rmse = compute_rmse_over_horizon(y_preds, y_labels)
    print('Horizon, RMSE:')
    for i in range(0, rmse.shape[0]):
        print(f'{i},\t{rmse[i]:.4f}') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", default='data/ind_time_sample/test_p20_f20.npz', type=str)
    args = parser.parse_args()

    main(args)