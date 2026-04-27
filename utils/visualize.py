import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.historic import predict_historic_model
from utils.util import seed_everything
from utils.metric import compute_rmse_over_sample, compute_mape_over_sample, compute_rmse_over_sample_minK, compute_mape_over_sample_minK


def main(args):
    print(args)
    seed_everything(args.random_seed)

    dot_size = 2
    dot_size = 2
    colors_dict = {
        'obs': 'black', 'gt': 'red', 'gt_eyearr': 'rosybrown', 'pred_eyearr': 'purple', 'podleader_obs': 'black', 'podleader_fut': 'pink',
        'baseline': 'orange', 'eyevislet': 'green', 'eyereldeg': 'cyan', 'eyereldegexpt': 'blue'
        # 'historic': 'blue', 'lstm': 'cyan', 'lstm+base': 'cyan', 'lstm+eyevislet': 'slateblue', 'lstm+eyedegree': 'deepskyblue', 'lstm+headvislet': 'aquamarine', 'lstmeye': 'teel',
        # 'historic': 'blue', 'lstm': 'cyan', 'lstm+base': 'cyan', 'lstm+eyevislet': 'springgreen', 'lstm+eyedegree': 'orange', 'lstm+headvislet': 'hotpink', 'lstmeye': 'blue',
        # 'auxlstm': 'green', 'auxlstmbest': 'olive', 'auxlstmnorm': 'brown', 'lstmeyemx': 'blue', 'auxlstm+eyedegree': 'brown', 'auxlstm+eyevislet': 'green',
        # 'seq2seq': 'purple', 'cvae': 'magenta', 'cvae2': 'magenta', 'socialcvae': 'magenta',
        # 'socialvae': 'blue', 'cvaecpard': 'orange',
        # 'nsdedecoderonly': 'orange', 'mlstm': 'pink', 'mlstmt': 'magenta', 'ds3m': 'magenta', 'mxlstm': 'magenta'
    }
    model_name_mapper = {
        'baseline': 'baseline', 'eyevislet': 'GazeX-LSTM (eye vislet)', 'eyereldeg': 'GazeX-LSTM (eye-in-walking)', 'eyereldegexpt': 'GazeX-LSTM (eye-in-walking + expt)'
        # 'lstm+base': 'lstm(base)', 'lstm+eyedegree': 'lstm(eyedegree)', 'lstm+headvislet': 'lstm(headvislet)', 'lstm+eyevislet': 'lstm(eyevislet)',
        # 'lstmeye': 'LSTMeye', 'lstmeyemx': 'LSTMeyeMX', 'auxlstm+eyedegree': 'AuxLSTM(eyedegree)', 'auxlstm+eyevislet': 'AuxLSTM(eyevislet)',
        # 'lstmeye_eyevislet': 'LSTMeye(eyevislet)',
        # 'lstm': 'LSTM', 'auxlstm': 'Aux-LSTM', 'nsdedecoder': 'NSDE', 'mlstm': 'MLSTM', 'mlstmt': 'MLSTMT', 'ds3m': 'DS3M', 'mxlstm': 'MXLSTM', 
        # 'cvae': 'cVAE', 'socialcvae': 'SocialCVAE', 'cvaecpard': 'cVAEcpard', 'socialvae': 'SocialVAE'
    }
    models_dict = {}
    rmse_df = pd.DataFrame()
    mape_df = pd.DataFrame()
    n_plot_per_row = 4
    plots_index_list = list('abcdefghijklmnopqrstuvwxyz')

    for preds_path in args.preds_paths:
        file_name = preds_path.split('/')[-2]
        model_name = preds_path.split('/')[-2].split('_')[-1].lower()
        # load data 
        print(f'Loading data at {preds_path}')
        preds = np.load(preds_path, allow_pickle=True)
        x_labels, y_labels, y_preds_model = preds['x_labels'].astype(np.float32), \
            preds['y_labels'].astype(np.float32), preds['y_preds'].astype(np.float32)
        print(f'x_labels: {x_labels.shape}, y_labels: {y_labels.shape}, y_preds: {y_preds_model.shape}')
        models_dict[model_name] = {'x_labels': x_labels, 'y_labels': y_labels, 'y_preds': y_preds_model}
        is_infer_mu = len(y_preds_model.shape) == 3
        if is_infer_mu: # when only having deterministic predictions
            rmse_model = compute_rmse_over_sample(models_dict[model_name]['y_preds'], models_dict[model_name]['y_labels'])
            mape_model = compute_mape_over_sample(models_dict[model_name]['y_preds'], models_dict[model_name]['y_labels'])
            models_dict[model_name]['rmse'] = rmse_model
            models_dict[model_name]['mape'] = mape_model
        else: # when multiple predictions exist
            rmse_model, rmse_minK_idx = compute_rmse_over_sample_minK(y_preds_model, y_labels)
            mape_model, mape_minK_idx = compute_mape_over_sample_minK(y_preds_model, y_labels)
            num_samples = y_preds_model.shape[0]
            models_dict[model_name]['rmse'] = rmse_model
            models_dict[model_name]['mape'] = mape_model
            models_dict[model_name]['rmse_minK_idx'] = rmse_minK_idx
            models_dict[model_name]['mape_minK_idx'] = mape_minK_idx
        rmse_df[file_name] = rmse_model
        mape_df[file_name] = mape_model
    
    # load historic model
    # y_preds_historic = predict_historic_model(x_labels)
    
    last_model_name = model_name
    loss = models_dict[model_name][args.metric]
    l = np.arange(x_labels.shape[0]).tolist()
    rmse_df['id'] = l
    mape_df['id'] = l
    if args.order in ['best', 'worst']:
        l_sorted = [x for _, x in sorted(zip(loss, l))]
    elif args.order == 'list':
        l_sorted = args.order_list.copy()
        setattr(args, 'n_fig', len(l_sorted))
        
    len_fut = y_labels.shape[1]
    alpha_list = np.linspace(0.2, 1, len_fut)[::-1]

    # visualize
    n_row, n_col = math.ceil(args.n_fig/n_plot_per_row), n_plot_per_row
    _, axes = plt.subplots(n_row, n_col, figsize=(n_col*2.8, n_row*2.6))
    if args.order == 'random': l_sorted = []
    axes = axes.flatten()
    for i in range(args.n_fig):
        # find the index
        # find the index
        if args.order == 'random':
            j = np.random.choice(l)
            l_sorted.append(j)
        elif args.order == 'best' or args.order =='list':
            j = l_sorted[0]
            l_sorted.remove(j)
        elif args.order == 'worst':
            j = l_sorted[-1]
            l_sorted.remove(j)
        axes[i].scatter(x_labels[j,:,0], x_labels[j,:,1], label='observation', color=colors_dict['obs'], s=dot_size*0.25)
        axes[i].scatter(x_labels[j,-1,0], x_labels[j,-1,1], color=colors_dict['obs'], s=dot_size)
        # axes[i].scatter(y_preds_historic[j,:,0], y_preds_historic[j,:,1], label='historic', color='blue', s=dot_size, alpha=alpha_list)

        ## show all 20 predictions if needed
        if args.show_min20_preds:
            for model_name in models_dict.keys():
                x_labels, y_labels, y_preds_model = models_dict[model_name]['x_labels'], \
                    models_dict[model_name]['y_labels'], models_dict[model_name]['y_preds']
                is_infer_mu = len(y_preds_model.shape) == 3
                if is_infer_mu:
                    axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
                else:
                    for p in range(num_samples):
                        axes[i].scatter(y_preds_model[p,j,:,0], y_preds_model[p,j,:,1], color=colors_dict[model_name], s=dot_size*0.5, alpha=0.04)
                    min_idx = models_dict[model_name][f'{args.metric}_minK_idx'][j]
                    axes[i].scatter(y_preds_model[min_idx,j,:,0], y_preds_model[min_idx,j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
        
        ## show the best prediction
        axes[i].scatter(x_labels[j,:,0], x_labels[j,:,1], label='observation', color=colors_dict['obs'], s=dot_size*0.25)
        axes[i].scatter(x_labels[j,-1,0], x_labels[j,-1,1], color=colors_dict['obs'], s=dot_size)
        # axes[i].scatter(y_preds_historic[j,:,0], y_preds_historic[j,:,1], label='historic', color='blue', s=dot_size, alpha=alpha_list)

        ## show all 20 predictions if needed
        if args.show_min20_preds:
            for model_name in models_dict.keys():
                x_labels, y_labels, y_preds_model = models_dict[model_name]['x_labels'], \
                    models_dict[model_name]['y_labels'], models_dict[model_name]['y_preds']
                is_infer_mu = len(y_preds_model.shape) == 3
                if is_infer_mu:
                    axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
                else:
                    for p in range(num_samples):
                        axes[i].scatter(y_preds_model[p,j,:,0], y_preds_model[p,j,:,1], color=colors_dict[model_name], s=dot_size*0.5, alpha=0.04)
                    min_idx = models_dict[model_name][f'{args.metric}_minK_idx'][j]
                    axes[i].scatter(y_preds_model[min_idx,j,:,0], y_preds_model[min_idx,j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
        
        ## show the best prediction
        for model_name in models_dict.keys():
            x_labels, y_labels, y_preds_model = models_dict[model_name]['x_labels'], \
                models_dict[model_name]['y_labels'], models_dict[model_name]['y_preds']
            if is_infer_mu:
                axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], label=model_name_mapper.get(model_name), color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
            else:
                min_idx = models_dict[model_name][f'{args.metric}_minK_idx'][j]
                axes[i].scatter(y_preds_model[min_idx,j,:,0], y_preds_model[min_idx,j,:,1], label=model_name_mapper.get(model_name), color=colors_dict[model_name], s=dot_size*0.5, alpha=alpha_list)

        axes[i].scatter(y_labels[j,:,0], y_labels[j,:,1], label='groundtruth', color='red', s=dot_size*0.5, alpha=alpha_list)
        axes[i].set_xlabel('x (cm)')
        axes[i].set_ylabel('y (cm)')
        if not args.zoom_in:
            axes[i].set_ylim(-300, 420)
            axes[i].set_xlim(-400, 320)
        axes[i].set_title(f'Ex{j}: {model_name}={loss[j]:.2f}')
    plt.tight_layout()

    # plot legend
    handles, labels = axes[i].get_legend_handles_labels()
    for i in range(args.n_fig, n_row*n_col):
        axes[i].axis('off')
    axes[n_row*n_col - 1].legend(handles, labels)


    # plot legend
    handles, labels = axes[i].get_legend_handles_labels()
    for i in range(args.n_fig, n_row*n_col):
        axes[i].axis('off')
    axes[n_row*n_col - 1].legend(handles, labels)

    if not os.path.exists(args.figdir):
        os.makedirs(args.figdir)

    if len(args.preds_paths) > 1:
        name = ['-'.join(i.rstrip('.npz').split('/')[2:]) for i in args.preds_paths]
        filename = str(args.random_seed)+f'-comparison_{args.order}__'+f"{'__'.join(name)}"
        print(len(filename))
        if len(filename) > 50:
            filename = filename[:50]
        fig_path = args.figdir+filename+'.png'
    else:
        filename = str(args.random_seed)+f'-{args.order}'+f"{'-'.join(args.preds_paths[0].rstrip('.npz').split('/')[2:])}"
        if len(filename) > 50:
            filename = filename[:50]
        fig_path = args.figdir+filename+".png"
    plt.savefig(fig_path)
    print(f'Saved figure at {fig_path}')
    

    # to load eye tracking data from test.npz
    if args.show_eyearr:
        # eye 
        test = np.load('data/indiv_time_o40_p40_s4/test.npz', allow_pickle=True)
        eye_idx = np.where(test['columns'] == 'eye_yaw')[0][0]
        eye_arr_obs = test['x'][..., eye_idx].astype(float)
        eye_arr_fut = test['y'][..., eye_idx].astype(float)
        eye_arr = np.concatenate((eye_arr_obs, eye_arr_fut), axis=-1)
        # loc
        loc_x = np.concatenate((test['x'][..., 1], test['y'][..., 1]), axis=-1).astype(float)
        loc_y = np.concatenate((test['x'][..., 2], test['y'][..., 2]), axis=-1).astype(float)
        # compute eye direction
        gaze_rad = np.deg2rad(eye_arr)
        dx = np.cos(gaze_rad)
        dy = np.sin(gaze_rad)
        # eye_sin_idx = np.where(test['columns'] == 'eye_sinyaw')[0][0]
        # eye_cos_idx = np.where(test['columns'] == 'eye_cosyaw')[0][0]
        # eye_arr_obs = test['x'][..., [eye_sin_idx, eye_cos_idx]].astype(float)
        # eye_arr_fut = test['y'][..., [eye_sin_idx, eye_cos_idx]].astype(float)
        # eye_arr = np.concatenate((eye_arr_obs, eye_arr_fut), axis=-2)
        # dx = eye_arr[..., 1]
        # dy = eye_arr[..., 0]
        # # loc
        # loc_x = np.concatenate((test['x'][..., 1], test['y'][..., 1]), axis=-1).astype(float)
        # loc_y = np.concatenate((test['x'][..., 2], test['y'][..., 2]), axis=-1).astype(float)

        # show pod if needed
        if args.show_pod:
            leader_idx_x = np.where(test['columns'] == 'dist_pedleader_x')[0][0]
            leader_obs_x = test['x'][..., leader_idx_x].astype(float)
            leader_fut_x = test['y'][..., leader_idx_x].astype(float)
            leader_x = np.concatenate((leader_obs_x, leader_fut_x), axis=-1)
            leader_x += loc_x

            leader_idx_y = np.where(test['columns'] == 'dist_pedleader_y')[0][0]
            leader_obs_y = test['x'][..., leader_idx_y].astype(float)
            leader_fut_y = test['y'][..., leader_idx_y].astype(float)
            leader_y = np.concatenate((leader_obs_y, leader_fut_y), axis=-1)
            leader_y += loc_y
        
        # reset l 
        if args.order in ['best', 'worst']:
            l_sorted = [x for _, x in sorted(zip(loss, l))]
        elif args.order == 'list':
            l_sorted = args.order_list.copy()

        # visualize 
        n_row, n_col = math.ceil(args.n_fig/n_plot_per_row), n_plot_per_row
        _, axes = plt.subplots(n_row, n_col, figsize=(n_col*2.8, n_row*2.6))
        axes = axes.flatten()
        for i in range(args.n_fig):
            if args.order == 'random':
                j = l_sorted[0]
                l_sorted.remove(j)
            elif args.order == 'best' or args.order =='list':
                j = l_sorted[0]
                l_sorted.remove(j)
            elif args.order == 'worst':
                j = l_sorted[-1]
                l_sorted.remove(j)
            axes[i].scatter(x_labels[j,:,0], x_labels[j,:,1], label='observation', color=colors_dict['obs'], s=dot_size*0.25)
            axes[i].scatter(x_labels[j,-1,0], x_labels[j,-1,1], color=colors_dict['obs'], s=dot_size)
            # axes[i].scatter(y_preds_historic[j,:,0], y_preds_historic[j,:,1], label='historic', color='blue', s=dot_size, alpha=alpha_list)

            ## show all 20 predictions if needed
            if args.show_min20_preds:
                for model_name in models_dict.keys():
                    x_labels, y_labels, y_preds_model = models_dict[model_name]['x_labels'], \
                        models_dict[model_name]['y_labels'], models_dict[model_name]['y_preds']
                    is_infer_mu = len(y_preds_model.shape) == 3
                    if is_infer_mu:
                        axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
                    else:
                        for p in range(num_samples):
                            axes[i].scatter(y_preds_model[p,j,:,0], y_preds_model[p,j,:,1], color=colors_dict[model_name], s=dot_size*0.5, alpha=0.04)
                        min_idx = models_dict[model_name][f'{args.metric}_minK_idx'][j]
                        axes[i].scatter(y_preds_model[min_idx,j,:,0], y_preds_model[min_idx,j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)

            ## show the best prediction
            axes[i].scatter(x_labels[j,:,0], x_labels[j,:,1], label='observation', color=colors_dict['obs'], s=dot_size*0.25)
            axes[i].scatter(x_labels[j,-1,0], x_labels[j,-1,1], color=colors_dict['obs'], s=dot_size)
            # axes[i].scatter(y_preds_historic[j,:,0], y_preds_historic[j,:,1], label='historic', color='blue', s=dot_size, alpha=alpha_list)

            ## show all 20 predictions if needed
            if args.show_min20_preds:
                for model_name in models_dict.keys():
                    x_labels, y_labels, y_preds_model = models_dict[model_name]['x_labels'], \
                        models_dict[model_name]['y_labels'], models_dict[model_name]['y_preds']
                    is_infer_mu = len(y_preds_model.shape) == 3
                    if is_infer_mu:
                        axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
                    else:
                        for p in range(num_samples):
                            axes[i].scatter(y_preds_model[p,j,:,0], y_preds_model[p,j,:,1], color=colors_dict[model_name], s=dot_size*0.5, alpha=0.04)
                        min_idx = models_dict[model_name][f'{args.metric}_minK_idx'][j]
                        axes[i].scatter(y_preds_model[min_idx,j,:,0], y_preds_model[min_idx,j,:,1], color=colors_dict[model_name], s=dot_size, alpha=alpha_list)

            ## show the best prediction
            for model_name in models_dict.keys():
                if is_infer_mu:
                    axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], label=model_name_mapper.get(model_name), color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
                    axes[i].scatter(y_preds_model[j,:,0], y_preds_model[j,:,1], label=model_name_mapper.get(model_name), color=colors_dict[model_name], s=dot_size, alpha=alpha_list)
                else:
                    x_labels, y_labels, y_preds_model = models_dict[model_name]['x_labels'], \
                        models_dict[model_name]['y_labels'], models_dict[model_name]['y_preds']
                    min_idx = models_dict[model_name][f'{args.metric}_minK_idx'][j]
                    axes[i].scatter(y_preds_model[min_idx,j,:,0], y_preds_model[min_idx,j,:,1], label=model_name_mapper.get(model_name), color=colors_dict[model_name], s=dot_size*0.5, alpha=alpha_list)
            N = 4

            # show the true eye direction
            axes[i].quiver(loc_x[j, ::N], loc_y[j, ::N], dx[j, ::N], dy[j, ::N],
                angles='xy', scale_units='xy', scale=None, color=colors_dict['gt_eyearr'], width=0.005, headlength=5, alpha=0.8,
                label="Eye gaze direction")
            
            # show the predicted eye direction
            if args.show_predicted_eyearr:
                y_preds_model = models_dict[last_model_name]['y_preds']
                if y_preds_model.shape[-1] > 2:
                    min_idx = models_dict[last_model_name][f'{args.metric}_minK_idx'][j]

                    loc_x_pred = y_preds_model[min_idx, j, :, 0]
                    loc_y_pred = y_preds_model[min_idx, j, :, 1]

                    if y_preds_model.shape[-1] == 3:
                        gaze_rad_pred = np.deg2rad(y_preds_model[min_idx, j, :, -1])
                        dx_pred = np.cos(gaze_rad_pred)
                        dy_pred = np.sin(gaze_rad_pred)
                    elif y_preds_model.shape[-1] == 4:
                        dx_pred = y_preds_model[min_idx, j, :, -1]
                        dy_pred = y_preds_model[min_idx, j, :, -2]                    

                    axes[i].quiver(loc_x_pred[::N], loc_y_pred[::N], dx_pred[::N], dy_pred[::N],
                        angles='xy', scale_units='xy', scale=None, color=colors_dict['pred_eyearr'], width=0.005, headlength=5,
                        angles='xy', scale_units='xy', scale=None, color=colors_dict['pred_eyearr'], width=0.005, headlength=5,
                        label="Predicted eye gaze direction")

            axes[i].scatter(y_labels[j,:,0], y_labels[j,:,1], label='groundtruth', color=colors_dict['gt'], s=dot_size*0.5, alpha=alpha_list)
            axes[i].scatter(y_labels[j,:,0], y_labels[j,:,1], label='groundtruth', color=colors_dict['gt'], s=dot_size*0.5, alpha=alpha_list)
            axes[i].set_xlabel('x (cm)')
            axes[i].set_ylabel('y (cm)')
            axes[i].set_title(f'Ex{j}: {model_name}={loss[j]:.2f}')
        
        # plot pod if needed
        if args.show_pod:
            axes[i].scatter(leader_x[j,:], leader_y[j,:], label='pod leader', color='pink', s=1)
        
        # plot legend
        handles, labels = axes[i].get_legend_handles_labels()
        for i in range(args.n_fig, n_row*n_col):
            axes[i].axis('off') 
        axes[n_row*n_col - 1].legend(handles, labels)

        plt.tight_layout()
        
        # plot legend
        handles, labels = axes[i].get_legend_handles_labels()
        for i in range(args.n_fig, n_row*n_col):
            axes[i].axis('off') 
        axes[n_row*n_col - 1].legend(handles, labels)

        if not os.path.exists(args.figdir):
            os.makedirs(args.figdir)

        if len(args.preds_paths) > 1:
            name = ['-'.join(i.rstrip('.npz').split('/')[2:]) for i in args.preds_paths]
            filename = str(args.random_seed)+f'-comparison_{args.order}__'+f"{'__'.join(name)}"
            print(len(filename))
            if len(filename) > 50:
                filename = filename[:50]
            fig_path = args.figdir + filename + '_eyearr.png'
        else:
            filename = str(args.random_seed)+f'-{args.order}'+f"{'-'.join(args.preds_paths[0].rstrip('.npz').split('/')[2:])}"
            if len(filename) > 50:
                filename = filename[:50]
            fig_path = args.figdir + filename + "_eyearr.png"
        plt.savefig(fig_path)
        print(f'Saved figure at {fig_path}')

    if os.path.exists('rmse.csv'):
        rmse_df_ = pd.read_csv('rmse.csv')
        for col in rmse_df.columns.tolist():
            if col not in rmse_df_.columns:
                rmse_df_[col] = rmse_df[col]
    if os.path.exists('mape.csv'):
        mape_df_ = pd.read_csv('mape.csv')
        for col in mape_df.columns.tolist():
            if col not in mape_df_.columns:
                mape_df_[col] = mape_df[col]
    else:
        rmse_df_ = rmse_df.copy()
        mape_df_ = mape_df.copy()
    rmse_df_.to_csv('rmse.csv', index=False)
    mape_df_.to_csv('mape.csv', index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=22, type=int)
    parser.add_argument("--preds_paths", default=[
        # 'logs/indi_time_o40_p40_s4/20250623_055623_MLSTM/test_0_best.npz',
        # 'logs/indi_time_o40_p40_s4/20250621_165447_MLSTM/train_0_best.npz',
        # 'logs/indi_time_o40_p40_s4/20250621_181328_MLSTM/test_0_best.npz',
        # 'logs/indi_time_o40_p40_s4/20250501_042127_LSTM/test_0_best.npz',
        # 'logs/indi_time_o40_p40_s4/20250502_110030_AuxLSTM/test_0_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250820_101056_AuxLSTM/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250820_150254_CVAE/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250821_025042_CVAEcpard/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250820_185550_SocialVAE/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/best_models/20250825_230126_LSTMbase/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/best_models/20250825_133240_LSTMeyedegree/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/best_models/20250826_222825_LSTMheaddegree/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250829_132241_LSTMeyeMX/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250918_093737_baseline/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250918_113122_eyevislet/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250918_135144_eyereldeg/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250919_053434_eyereldegExpt/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250825_230126_LSTM+base/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250825_133240_LSTM+eyedegree/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250902_221914_AuxLSTM+eyedegree/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250828_233503_LSTM+eyevislet/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250902_034540_AuxLSTM+eyevislet/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250918_093737_baseline/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250918_113122_eyevislet/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250918_135144_eyereldeg/test_best.npz', 
        'logs/indiv_time_o40_p40_s4/best/20250919_053434_eyereldegExpt/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250825_230126_LSTM+base/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250825_133240_LSTM+eyedegree/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250902_221914_AuxLSTM+eyedegree/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250828_233503_LSTM+eyevislet/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best_aux/20250902_034540_AuxLSTM+eyevislet/test_best.npz', 
        # 'logs/indiv_time_o40_p40_s4/best/20250828_233503_LSTM+eyevislet/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/best/20250825_133240_LSTM+eyedegree/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/best/20250829_025224_LSTM+headvislet/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/best/20250828_210934_LSTMeye/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/hp_eye_sqlite93/20250827_214358_LSTMeye/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250816_150704_LSTM/test_best.npz',
        # 'logs/indiv_time_o40_p40_s4/20250812_235059_SocialCVAE/test_0_last.npz',
        ], nargs='+', type=str)
    
    # parser.add_argument("--preds_paths", default=[
    #     'logs/rand_time_o40_p40_s4/20250429_104209_LSTM/train_0_best.npz',
    #     'logs/rand_time_o40_p40_s4/20250429_104411_Seq2Seq/train_0_best.npz',
    #     'logs/rand_time_o40_p40_s4/20250429_103945_AuxLSTM/train_0_best.npz'], nargs='+', type=str)
    # parser.add_argument("--preds_paths", default=[
    #     'logs/ind_time_o40_p40_s4/20250428_164831_LSTM/test_0_best.npz',
    #     'logs/ind_time_o40_p40_s4/20250428_165446_AuxLSTM/test_0_best.npz',
    #     'logs/ind_time_o40_p40_s4/20250428_170106_Seq2Seq/test_0_best.npz'], nargs='+', type=str)
    # parser.add_argument("--preds_paths", default=[
    #     'logs/ind_time_o20_p60/20250424_095133_best_LSTM/test_0_best.npz',
    #     'logs/ind_time_o20_p60/20250423_185720_best_AuxLSTM/test_0_best.npz',
    #     'logs/ind_time_o20_p60/20250425_130119_best_Seq2Seq/test_0_best.npz'], nargs='+', type=str)
        # 'logs/ind_time_o20_p60/20250310_183624_Seq2Seq/test_0_best.npz', 
    parser.add_argument("--figdir", default='figures_model/', type=str)
    parser.add_argument("--n_fig", default=20, type=int)
    parser.add_argument("--order", default='list', choices=['random', 'best', 'worst', 'list'], type=str)
    parser.add_argument("--order_list", default=
    # [1635, 2017, 2938, 2018, 1810, 1636, 2939, 1366, 1102, 2220, 1986, 2219, 2937, 1441,  657, 1365, 1985, 1634,  594, 2936], 
    # [656, 3002, 2936, 684, 1793, 749, 2015, 1815, 563, 2820, 2939, 3851, 2940, 827, 113, 594, 351, 638, 3179, 2675, 2937],
    [2123, 1504, 1370, 3782, 487, 602, 1367, 1102, 1833, 1813],
       nargs='+', type=int, help='list of indices to order the figures')
    parser.add_argument("--metric", default='rmse', choices=['rmse', 'mape'], type=str)
    parser.add_argument("--show_eyearr", default=True)
    parser.add_argument("--show_eyearr", default=True)
    parser.add_argument("--show_predicted_eyearr", default=False)
    parser.add_argument("--show_min20_preds", default=False)
    parser.add_argument("--show_pod", default=True)
    parser.add_argument("--zoom_in", default=True, action='store_true')

    args = parser.parse_args()
    main(args)