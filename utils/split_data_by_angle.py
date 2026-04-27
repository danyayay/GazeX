import os 
import numpy as np
import pandas as pd


def main():

    dts_merged = pd.read_csv('data/dts_qn.csv')

    train = np.load('data/indiv_time_o40_p40_s4/train.npz', allow_pickle=True)
    val = np.load('data/indiv_time_o40_p40_s4/val.npz', allow_pickle=True)
    test = np.load('data/indiv_time_o40_p40_s4/test.npz', allow_pickle=True)

    for angle_idx in [0, 1, 2]:
        angle = angle_idx*45+45
        out_dir = f'data/indiv_time_angle{angle}_o40_p40_s4'
        psid_angle = dts_merged[dts_merged.angle == angle_idx].psid.unique()
        columns = train['columns']

        train_psid = train['x'][..., 0, 0]
        val_psid = val['x'][..., 0, 0]
        test_psid = test['x'][..., 0, 0]

        train_psid_in_angle = [psid for psid in list(set(train_psid)) if psid in psid_angle]
        val_psid_in_angle = [psid for psid in list(set(val_psid)) if psid in psid_angle]
        test_psid_in_angle = [psid for psid in list(set(test_psid)) if psid in psid_angle]

        print(f'Angle {angle}:')
        print(f'    Train: {len(train_psid_in_angle)}')
        print(f'    Validation: {len(val_psid_in_angle)}')
        print(f'    Test: {len(test_psid_in_angle)}')

        x_train = train['x'][np.isin(train_psid, train_psid_in_angle)]
        y_train = train['y'][np.isin(train_psid, train_psid_in_angle)]
        x_val = val['x'][np.isin(val_psid, val_psid_in_angle)]
        y_val = val['y'][np.isin(val_psid, val_psid_in_angle)]
        x_test = test['x'][np.isin(test_psid, test_psid_in_angle)]
        y_test = test['y'][np.isin(test_psid, test_psid_in_angle)]

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        np.savez_compressed(os.path.join(out_dir, 'train'), x=x_train, y=y_train, columns=columns)
        np.savez_compressed(os.path.join(out_dir, 'val'), x=x_val, y=y_val, columns=columns)
        np.savez_compressed(os.path.join(out_dir, 'test'), x=x_test, y=y_test, columns=columns)


if __name__ == '__main__':
    main()