import os
import math
import scipy
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class Interaction(object):
    def __init__(self, log_filename, is_recenter=True, is_save_prep=True, is_savefig=True, is_plot_intermediate=True, vis_fam=False, suffix=''):
        self.log_path = os.path.join('data/vrdata', f'combined/{log_filename}.csv')
        self.pid = log_filename.split('_')[0]
        self.dt_fam_path = os.path.join('data/dtdata', 'DT_Fam.csv')
        self.dt_path = os.path.join('data/dtdata', f'DT_{self.pid}.csv')
        self.fig_dir = os.path.join('figures', f'{log_filename}{suffix}')
        
        self.is_recenter = is_recenter
        self.is_save_prep = is_save_prep
        self.is_savefig = is_savefig
        self.is_plot_intermediate = is_plot_intermediate
        self.vis_fam = vis_fam
        self.suffix = suffix
        
        self.load_data()
        
        if self.is_save_prep:
            print(f'Saving {log_filename}_prep.csv')
            if self.suffix == '_noremoval':
                if not os.path.exists('data/vrdata/noremoval'):
                    os.makedirs('data/vrdata/noremoval')
                self.df.to_csv(os.path.join('data/vrdata/noremoval', f'{log_filename}_prep.csv'))
            else:
                if not os.path.exists('data/vrdata/preprocessed'):
                    os.makedirs('data/vrdata/preprocessed')
                self.df.to_csv(os.path.join('data/vrdata/preprocessed', f'{log_filename}_prep.csv'))
    
    def load_data(self):
        print('\n#############################')
        print(f'Reading {self.log_path}')
        self.df_all = pd.read_csv(self.log_path)
        self.df_all['sid'] = self.df_all.Scenario.apply(lambda x: int(x.lstrip('F'))-1 if x.startswith('F') else int(x))
        self.df_all['pid'] = self.pid

        # formal dt 
        self.dt = pd.read_csv(self.dt_path)
        self.dt['sid'] = self.dt.index + 1
        self.dt['pid'] = self.pid
        self.dt.drop(columns=['---'], inplace=True)
        self.dt = self.dt[['pid', 'sid', 'eHMI', 'yielding', 'angle', 'traffic flow']]

        # fam dt 
        self.dt_fam = pd.read_csv(self.dt_fam_path)
        self.dt_fam['sid'] = self.dt_fam.index
        self.dt_fam['pid'] = self.pid
        self.dt_fam.drop(columns=['---'], inplace=True)
        self.dt_fam = self.dt_fam[['pid', 'sid', 'eHMI', 'yielding', 'direction']]

        self.preprocess_data()
        self.split_data_by_tasks()

        # check how many times fam tasks are shown
        n_fam = self.df_fam.Scenario.nunique()
        if n_fam > 7: self.dt_fam = pd.concat([self.dt_fam]*math.ceil(n_fam/7), ignore_index=True)

        print(f'\nData table: shape = {self.dt.shape}\n')

        print(f'Logged familiarization data')
        print(f'    shape   = {self.df_fam.shape}')
        print(f'    n_scenarios = {self.df_fam.Scenario.unique()}\n')

        print(f'Logged formal data')
        print(f'    shape   = {self.df.shape}')
        print(f'    n_scenarios = {self.df.Scenario.unique()}\n')

    def preprocess_data(self):
        self.df_all = self.df_all.drop_duplicates()
        if self.is_recenter: self.recenter_locations()
        self.preprocess_eyegaze_data()
        self.calculate_distance_with_interaction_point()
        self.calculate_distance_between_agents()
        self.match_columns_for_all_data()
        self.get_pod_trigger_points()
        if self.suffix != '_noremoval':
            self.remove_pod_already_left()
            self.remove_mistakenly_triggered_scenarios()
        self.remove_no_eye_tracking()

        if self.df_all.shape[0] > 0:
            self.smooth_trajectories_gf()
            self.preprocess_speed()
            if self.suffix != '_noremoval':
                self.trim_white_to_green()
                self.remove_trials_with_bad_speed()
                self.remove_collision()
            self.get_time_elapsed_for_each_trial()
        else:
            print('No data left after preprocessing for pid:', self.pid)

    def remove_no_eye_tracking(self):
        # remove those without eye gazing data
        no_eye_tracking_mask = self.df_all.EyeGaze_Rotation_z_smoothed.isna()
        print(f'Removing scenarios {self.df_all.loc[no_eye_tracking_mask].Scenario.unique()} without eye tracking data')
        self.df_all = self.df_all[~no_eye_tracking_mask]

    def split_data_by_tasks(self):
        self.df_fam = self.df_all[self.df_all.Scenario.str.startswith('F') == True]
        self.df = self.df_all[self.df_all.Scenario.str.startswith('F') != True]

    def check_log_freq(self):
        def get_mean_std_each_group(group):
            group['TimeDiff'] = group.TimeElapsed - group.TimeElapsed.shift(1)
            return round(group['TimeDiff'].mean(), 4), round(group['TimeDiff'].std(), 4)
        print(self.df[['TimeElapsed', 'Scenario']].groupby('Scenario', group_keys=False).apply(
            get_mean_std_each_group, include_groups=False))

    ################# Visualization #####################
    def visualize(self):
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        # if there are fam tasks in data
        if self.vis_fam:
            if self.df_all.Scenario.str.startswith('F').any():
                self.visualize_familiarization_expt()
        self.visualize_formal_expt()
    
    def visualize_familiarization_expt(self):
        self.plot_expt_process_over_time(self.df_fam, is_fam=True)
        self.plot_interaction_trajectories_together(self.df_fam, is_fam=True)
        self.plot_gazehit_together(self.df_fam, is_fam=True)

        self.plot_when_passed_collision_point_per_trial(self.df_fam, is_fam=True)
        self.plot_ped_rotation_z_per_trial(self.df_fam, is_fam=True)
        self.plot_ped_rotation_z_rel_per_trial(self.df_fam, is_fam=True)

        self.plot_ped_velocity_per_trial(self.df_fam, is_fam=True)
        self.plot_pod_trajectory_velocity_per_trial(self.df_fam, 'pod_fam', is_fam=True)

        # intermediate plots 
        if self.is_plot_intermediate:
            self.plot_ped_rotation_diff_per_trial(self.df_fam, is_fam=True)
            self.plot_ped_y_diff_per_trial(self.df_fam, is_fam=True)
            self.plot_trajectory_per_trial(self.df_fam, is_fam=True)
    
    def visualize_formal_expt(self):
        self.plot_expt_process_over_time(self.df)
        self.plot_interaction_trajectories_together(self.df)
        self.plot_gazehit_together(self.df)

        self.plot_when_passed_collision_point_per_trial(self.df)
        self.plot_ped_rotation_z_per_trial(self.df)
        self.plot_ped_rotation_z_rel_per_trial(self.df)

        self.plot_ped_velocity_per_trial(self.df)
        self.plot_pod_trajectory_velocity_per_trial(self.df, 'pod_leader')
        self.plot_pod_trajectory_velocity_per_trial(self.df, 'pod_follower')

        # intermediate plots 
        if self.is_plot_intermediate:
            self.plot_ped_rotation_diff_per_trial(self.df)
            self.plot_ped_y_diff_per_trial(self.df)
            self.plot_trajectory_per_trial(self.df)
    
    def plot_expt_process_over_time(self, df, is_fam=False):
        fig = plt.figure(figsize=(5, 3))
        plt.plot(df.TimeElapsed)
        plt.xlabel('Log index')
        plt.ylabel('Time')
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_expt_process_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_expt_process_formal.png', bbox_inches='tight')
        plt.close()

    def plot_interaction_trajectories_together(self, df, is_plot_pod=False, is_smooth=True, is_fam=False):
        # define the boundary
        max_x, min_x = 400, -400
        max_y, min_y = -250, 450
        print('Interaction area:')
        print(f'x ({min_x}, {max_x})')
        print(f'y ({min_y}, {max_y})')

        # plot
        fig = plt.figure(figsize=(7, 5))
        if is_smooth:
            plt.scatter(df.Ped_Location_x, df.Ped_Location_y, c=df['sid'], s=1, cmap='grey')
            plt.scatter(df.Ped_Location_x_smoothed, df.Ped_Location_y_smoothed, c=df['sid'], s=1, cmap='Paired')
        else: 
            plt.scatter(df.Ped_Location_x, df.Ped_Location_y, c=df['sid'], s=1, cmap='Paired')

        # if plot pod trajectory
        if is_plot_pod: plt.plot(df.PodLeader_Location_x, df.PodLeader_Location_y, label='pod')

        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.title('Interaction trajectories')
        plt.colorbar()
        fig.tight_layout()
        
        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_ped_traj_agg_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_ped_traj_agg_formal.png', bbox_inches='tight')
        plt.close()

    def plot_when_passed_collision_point_per_trial(self, df, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1

        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    # pedestrian 
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Distance, label='Pedestrian')
                    axes[k, l].scatter(df_j.TimeElapsedTrial.iloc[df_j.Ped_Distance.argmin()], df_j.Ped_Distance.min())
                    if not is_fam:
                        # pod leader
                        axes[k, l].plot(df_j.TimeElapsedTrial, df_j.PodLeader_Distance, label='Pod Leader')
                        axes[k, l].scatter(df_j.TimeElapsedTrial.iloc[df_j.PodLeader_Distance.argmin()], df_j.PodLeader_Distance.min())
                        # plot the start moving point 
                        diff_dist = df_j.PodLeader_Distance.diff()
                        axes[k, l].axvline(x=df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], color='gray')
                        # pod follower
                        if df_j.PodFollower_Velocity_x.isnull().sum() == 0:
                            axes[k, l].plot(df_j.TimeElapsedTrial, df_j.PodFollower_Distance, label='Pod Follower')
                            axes[k, l].scatter(df_j.TimeElapsedTrial.iloc[df_j.PodFollower_Distance.argmin()], df_j.PodFollower_Distance.min())
                    else:
                        # pod fam
                        axes[k, l].plot(df_j.TimeElapsedTrial, df_j.FamCar_Distance, label='Pod Fam')
                        axes[k, l].scatter(df_j.TimeElapsedTrial.iloc[df_j.FamCar_Distance.argmin()], df_j.FamCar_Distance.min())
                        # plot the start moving point 
                        diff_dist = df_j.FamCar_Distance.diff()
                        axes[k, l].axvline(x=df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], color='gray')
                        
                    axes[k, l].annotate(
                        int(df_j.Ped_Distance.loc[diff_dist[diff_dist != 0].index[1]]), 
                        xy=(df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], 0), 
                        xytext=(df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], 0.5), color='gray')
                    # legend 
                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                        axes[k, l].set_ylim(0, 2000)
                        # plot ehmi status 
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                        axes[k, l].set_ylim(0, 4400)
                        # plot ehmi status 
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    axes[k, l].legend()
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_collision_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_collision_formal.png', bbox_inches='tight')       
        plt.close()
    
    def plot_ped_rotation_z_per_trial(self, df, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1
        min_y = df[['Ped_Rotation_z', 'Ped_Rotation_z_smoothed', 'EyeGaze_Rotation_z', 'EyeGaze_Rotation_z_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].min().min()
        max_y = df[['Ped_Rotation_z', 'Ped_Rotation_z_smoothed', 'EyeGaze_Rotation_z', 'EyeGaze_Rotation_z_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].max().max()

        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    p = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Rotation_z, alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Rotation_z_smoothed, c=p[0].get_color(), label='head orientation')

                    p = axes[k, l].plot(df_j.TimeElapsedTrial, np.ma.masked_where(df_j.ConfidenceValue == 0.0, df_j.EyeGaze_Rotation_z), alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.EyeGaze_Rotation_z_smoothed, c=p[0].get_color(), label='eye gaze')

                    p, = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_Rotation, alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_Rotation_smoothed, c=p.get_color(), label='velocity rotation')

                    p, = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_unwrap, alpha=0.3)
                    # axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_unwrap, c=p.get_color(), linestyle='--')
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_smoothed, c=p.get_color(), label='body rotation')
                    # axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation, c='pink')

                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    axes[k, l].set_xlim(0, df.TimeElapsedTrial.max())
                    axes[k, l].set_ylim(min_y, max_y)
                    axes[k, l].legend(loc='upper right')
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(f'{self.fig_dir}/{self.pid}_headeye_fam.png', bbox_inches='tight')
            else:
                plt.savefig(f'{self.fig_dir}/{self.pid}_headeye_formal.png', bbox_inches='tight')
        plt.close()

    def plot_ped_rotation_z_rel_per_trial(self, df, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1
        min_y = df[['head_vel_relative', 'head_vel_relative_smoothed', 'eye_vel_relative', 'eye_vel_relative_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].min().min()
        max_y = df[['head_vel_relative', 'head_vel_relative_smoothed', 'eye_vel_relative', 'eye_vel_relative_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].max().max()

        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    p, = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.head_vel_relative, alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.head_vel_relative_smoothed, label='relative head orientation', c=p.get_color())

                    p, = axes[k, l].plot(df_j.TimeElapsedTrial, np.ma.masked_where(df_j.ConfidenceValue == 0.0, df_j.eye_vel_relative), alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.eye_vel_relative_smoothed, c=p.get_color(), label='relative eye gaze')

                    p, = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_Rotation, alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_Rotation_smoothed, label='velocity rotation', c=p.get_color())

                    # p, = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.eye_head_relative, alpha=0.3)
                    # axes[k, l].plot(df_j.TimeElapsedTrial, df_j.eye_head_relative_smoothed, c=p.get_color(), label='eye head relative')
                    p, = axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_unwrap, alpha=0.3)
                    # axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_unwrap, c=p.get_color(), linestyle='--')
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_smoothed, c=p.get_color(), label='body rotation')
                    # axes[k, l].plot(df_j.TimeElapsedTrial, df_j.body_rotation_int, c='pink')

                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    axes[k, l].set_xlim(0, df.TimeElapsedTrial.max())
                    axes[k, l].set_ylim(min_y, max_y)
                    axes[k, l].legend(loc='upper right')
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(f'{self.fig_dir}/{self.pid}_headeye_rel_fam.png', bbox_inches='tight')
            else:
                plt.savefig(f'{self.fig_dir}/{self.pid}_headeye_rel_formal.png', bbox_inches='tight')
        plt.close()

    def plot_ped_rotation_diff_per_trial(self, df, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1

        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.head_vel_relative, label='head vel diff')
                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    axes[k, l].legend(loc='upper right')
                    axes[k, l].set_xlim(0, df.TimeElapsedTrial.max())
                    axes[k, l].set_ylim(-180, 180)
                    axes[k, l].axhline(y=-self.threshold, color='grey', linestyle='--')
                    axes[k, l].axhline(y=self.threshold, color='grey', linestyle='--')
                    axes[k, l].annotate(
                        self.threshold, 
                        xy=(df.TimeElapsedTrial.max()-2, self.threshold), 
                        xytext=(df.TimeElapsedTrial.max()-2, self.threshold), color='gray')                
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(f'{self.fig_dir}/{self.pid}_head_vel_diff_fam.png', bbox_inches='tight')
            else:
                plt.savefig(f'{self.fig_dir}/{self.pid}_head_vel_diff_formal.png', bbox_inches='tight')
        plt.close()

    def plot_ped_y_diff_per_trial(self, df, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1

        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.DiffPedLocation_y, label='y diff')
                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                        # plot ehmi status 
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    axes[k, l].legend(loc='upper right')
                    axes[k, l].set_xlim(0, df.TimeElapsedTrial.max())
                    axes[k, l].set_ylim(df.DiffPedLocation_y.min(), df.DiffPedLocation_y.max())
                    axes[k, l].axhline(y=0, color='grey', linestyle='--')                   
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(f'{self.fig_dir}/{self.pid}_ped_y_diff_fam.png', bbox_inches='tight')
            else:
                plt.savefig(f'{self.fig_dir}/{self.pid}_ped_y_diff_formal.png', bbox_inches='tight')
        plt.close()

    def plot_gazehit_together(self, df, is_fam=False):
        df_eye = df[df.ConfidenceValue != 0.0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.histplot(data=df_eye, y="HitObject", ax=axes[0])
        axes[0].set_xlabel('Duration')
        axes[0].set_title('Detailed')
        # sns.histplot(data=df_eye, y="HitActor_matched", ax=axes[1])
        # axes[1].set_title('Rough')
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_eye_agg_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_eye_agg_formal.png', bbox_inches='tight')
        plt.close()

    def plot_ped_velocity_per_trial(self, df, is_smooth=True, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1

        COLOR_VEL_x = 'red'
        COLOR_VEL_y = 'green'
        COLOR_VEL = 'blue'
        SPEED_MAX = max(df.Ped_Velocity_smoothed.abs().max(), 160)

        # per trial
        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    # original velocity
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_x, c=COLOR_VEL_x, alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_y, c=COLOR_VEL_y, alpha=0.3)
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity, c=COLOR_VEL, alpha=0.3)
                    # smoothed velocity 
                    if is_smooth:
                        axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_x_smoothed, c=COLOR_VEL_x, label='x')
                        axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_y_smoothed, c=COLOR_VEL_y, label='y')
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Velocity_smoothed, c=COLOR_VEL)
                    axes[k, l].set_ylabel('Velocity')
                    axes[k, l].set_xlabel('Time (second)')
                    axes[k, l].set_ylim(-SPEED_MAX, SPEED_MAX)
                    axes[k, l].set_xlim(0, df.TimeElapsedTrial.max())
                    axes[k, l].legend()
                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                        diff_dist = df_j.FamCar_Distance.diff()
                        # plot ehmi status 
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                        #     axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                        diff_dist = df_j.PodLeader_Distance.diff()
                        # plot ehmi status 
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                            axes[k, l].fill_between(df_j.TimeElapsedTrial, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=axes[k, l].get_xaxis_transform())
                    axes[k, l].axhline(y=0, color='gray', linestyle='--')
                    # plot start point
                    axes[k, l].axvline(x=df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], color='gray')
                    axes[k, l].annotate(
                        int(df_j.Ped_Distance.loc[diff_dist[diff_dist != 0].index[1]]), 
                        xy=(df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], -150), 
                        xytext=(df_j.TimeElapsedTrial.loc[diff_dist[diff_dist != 0].index[1]], -150), color='gray')
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_ped_speed_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_ped_speed_formal.png', bbox_inches='tight')
        plt.close()

    def plot_pod_trajectory_velocity_per_trial(self, df, agent, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1

        if agent == 'pod_leader':
            col_name = 'PodLeader_Location_x'
        elif agent == 'pod_follower':
            col_name = 'PodFollower_Location_x'
        elif agent == 'pod_fam':
            col_name = 'FamCar_Location_x'
        else:
            raise ValueError(f'No such {agent}')
        print(f'Drawing for {agent}...')

        COLOR_TRAJ = "#69b3a2"
        COLOR_VEL = "#3399e6"

        # per trial
        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    # VR trajectory
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j[col_name], label='Trajectory', c=COLOR_TRAJ)
                    axes[k, l].set_ylabel('Trajectory (cm)', c=COLOR_TRAJ)
                    axes[k, l].tick_params(axis="y", labelcolor=COLOR_TRAJ, length=0)

                    ax2 = axes[k, l].twinx()
                    # VR velocity
                    if agent == 'pod_leader':
                        ax2.plot(df_j.TimeElapsedTrial, df_j.PodLeader_Velocity_x, label='Velocity', c=COLOR_VEL)
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            ax2.fill_between(df_j.TimeElapsedTrial, -150, 150, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=ax2.get_xaxis_transform())
                            ax2.fill_between(df_j.TimeElapsedTrial, -150, 150, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=ax2.get_xaxis_transform())
                    elif agent == 'pod_follower':
                        ax2.plot(df_j.TimeElapsedTrial, df_j.PodFollower_Velocity_x, label='Velocity', c=COLOR_VEL)
                        if len([s for s in df_j.columns if "EhmiStatus" in s]):
                            ax2.fill_between(df_j.TimeElapsedTrial, -150, 150, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=ax2.get_xaxis_transform())
                            ax2.fill_between(df_j.TimeElapsedTrial, -150, 150, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=ax2.get_xaxis_transform())
                    elif agent == 'pod_fam':
                        ax2.plot(df_j.TimeElapsedTrial, df_j.FamCar_Velocity_x, label='Velocity', c=COLOR_VEL)
                        # if len([s for s in df_j.columns if "EhmiStatus" in s]):
                        #     ax2.fill_between(df_j.TimeElapsedTrial, -150, 150, where=df_j.FamCar_EhmiStatus=='green', color='green', alpha=0.1, transform=ax2.get_xaxis_transform())
                        #     ax2.fill_between(df_j.TimeElapsedTrial, -150, 150, where=df_j.FamCar_EhmiStatus=='red', color='red', alpha=0.1, transform=ax2.get_xaxis_transform())

                    # calculated velocity based on trajectory
                    ax2.plot(df_j.TimeElapsedTrial.iloc[:-1],
                            ((df_j[col_name] - df_j[col_name].shift(-1))/(df_j.TimeElapsedTrial - df_j.TimeElapsedTrial.shift(-1))).iloc[:-1], alpha=0.2)

                    ax2.set_ylabel('Velocity (cm/s)', c=COLOR_VEL)
                    ax2.tick_params(axis="y", labelcolor=COLOR_VEL, length=0)
                    if is_fam:
                        axes[k, l].set_title(self.dt_fam[self.dt_fam.sid == sid].iloc[0].astype(str).to_list())
                    else:   
                        axes[k, l].set_title(self.dt[self.dt.sid == sid].iloc[0].astype(str).to_list())
                else:
                    axes[k, l].axis('off')
                sid += 1
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_{agent}_traj_speed_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_{agent}_traj_speed_formal.png', bbox_inches='tight')
        plt.close()

    def plot_trajectory_per_trial(self, df, is_fam=False):
        n_col = 6
        n_row = math.ceil(len(df.sid.unique()) / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*3), squeeze=False)
        sid = 1

        for k in range(n_row):
            for l in range(n_col):
                if sid in df.sid.unique():
                    df_j = df[df.sid == sid]
                    if df_j.Ped_Location_x.iloc[0] < -100:
                        axes[k, l].plot(df_j.TimeElapsedTrial, -df_j.Ped_Location_x, alpha=0.3, c='red')
                        axes[k, l].plot(df_j.TimeElapsedTrial, -df_j.Ped_Location_x_smoothed, label='x(reversed)', c='red')
                    else:
                        axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Location_x, alpha=0.3, c='red')
                        axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Location_x_smoothed, label='x', c='red')
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Location_y, alpha=0.3, c='green')
                    axes[k, l].plot(df_j.TimeElapsedTrial, df_j.Ped_Location_y_smoothed, label='y', c='green')

                    if is_fam:
                        axes[k, l].set_title(self.dt_fam.iloc[sid-1].astype(str).values[1:])
                    else:   
                        axes[k, l].set_title(self.dt.iloc[sid-1].astype(str).values[1:])
                    axes[k, l].set_xlim(0, df.TimeElapsedTrial.max())
                    axes[k, l].legend(loc='upper right')
                else:
                    axes[k, l].axis('off')
                sid += 1 
        fig.tight_layout()

        if self.is_savefig:
            if is_fam:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_ped_traj_fam.png', bbox_inches='tight')
            else:
                plt.savefig(
                    f'{self.fig_dir}/{self.pid}_ped_traj_formal.png', bbox_inches='tight')
        plt.close()

    ################## Preprocessing data ###################
    def get_time_elapsed_for_each_trial(self):    
        # label trials 
        column_order = self.df_all.columns.tolist()

        # create time elapsed for each trial 
        self.df_all['TimeElapsedTrial'] = self.df_all.groupby('Scenario')['TimeElapsed'].transform(
            lambda x: x - x.iloc[0])

        # reorder 
        column_order.insert(1, 'TimeElapsedTrial')
        column_order.remove('pid')
        column_order.insert(3, 'pid')
        column_order.remove('sid')
        column_order.insert(4, 'sid')
        self.df_all = self.df_all[column_order]
    
    def recenter_locations(self, center_x=-16800, center_y=-100):
        for col in self.df_all.columns:
            if 'Location_x' in col:
                self.df_all[col] = self.df_all[col] - center_x
            elif 'Location_y' in col:
                self.df_all[col] = self.df_all[col] - center_y

    def preprocess_eyegaze_data(self):
        objects_envs = [
            'ground', 'build', 'tree', 'tarppost', 'recycle_bin', 'lightpost', 'bench', 'stairs', 
            'pot', 'move', 'lamp', 'drinkingfountain', 'railing', 'chair', 'landscape', 'sign']
        objects_widget = ['isready', 'isreadyfix', 'instruction', 'mainmenu', 'widget']
        objects_podleader = ['pod_leader']
        objects_podfollower = ['pod_follower']
        objects_fam = ['pod_famtask']
        objects_goal = ['goal']
        objects_neighbor = ['neighbor']
        object_dict = {
            'env': objects_envs, 'widget': objects_widget, 'pod_leader': objects_podleader, 
            'pod_follower': objects_podfollower, 'pod_fam': objects_fam, 'goal': objects_goal, 'neighbor': objects_neighbor}
        
        def which_cat(name, object_dict):
            for object_cat, object_list in object_dict.items():
                for obj in object_list:
                    if obj in name:
                        return object_cat 
            return name

        # get actor and component 
        self.df_all['HitActor'] = self.df_all['HitActor'].astype(str).apply(lambda x: x.lower())
        self.df_all.loc[:, ['HitObject']] = self.df_all['HitActor'].apply(lambda x: which_cat(x, object_dict))
        
        # preprocess head and eye gaze data 
        self.match_head_and_eyegaze()
        self.df_all.loc[self.df_all.ConfidenceValue == 0.0, ['HitObject', 'EyeGaze_Rotation_z']] = None
        self.df_all.HitObject = self.df_all.HitObject.replace({'nan': None})
    
    def calculate_eye_angle(self, x, y):
        x, y = np.array(x), np.array(y)
        angle =  - np.arccos(x / np.sqrt(x**2 + y**2)) / np.pi * 180
        angle[np.where(y > 0)] = - 360 - angle[np.where(y > 0)]
        return angle
    
    def convert_rotation(self, angle, angle_min=-270):
        return np.floor(angle / angle_min) * 360 + angle
    
    def match_head_and_eyegaze(self):
        # make the range within -360~0 degree
        self.df_all.loc[:, ['Ped_Rotation_z']] = self.convert_rotation(self.df_all.groupby('Scenario')['Ped_Rotation_z'].transform(
            lambda x: x - np.ceil(x / 360)*360))
        self.df_all.loc[:, ['Ped_Rotation_z']] = self.df_all.groupby(['pid', 'Scenario'])['Ped_Rotation_z'].transform(
            lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous
        self.df_all.loc[:, ['Ped_Rotation_z_smoothed']] = self.df_all.groupby('Scenario')['Ped_Rotation_z'].transform(
            lambda x: self.smooth_array_gf(x, sigma=4))

        self.df_all.loc[:, ['EyeGaze_Rotation_z']] = self.convert_rotation(self.calculate_eye_angle(self.df_all['GazeDirection_x'], self.df_all['GazeDirection_y']))
        self.df_all.loc[self.df_all.ConfidenceValue == 0.0, ['EyeGaze_Rotation_z']] = np.nan  # set the eye gaze rotation to nan if confidence value is 0
        self.df_all.loc[:, ['EyeGaze_Rotation_z']] = self.df_all.groupby('Scenario')['EyeGaze_Rotation_z'].transform(
            lambda x: x.interpolate(method='linear').to_numpy())
        self.df_all.loc[:, ['EyeGaze_Rotation_z']] = self.df_all.groupby(['pid', 'Scenario'])['EyeGaze_Rotation_z'].transform(
            lambda x: np.unwrap(x.ffill().bfill(), period=360))  # unwrap the angles, making it continuous
        self.df_all.loc[:, ['EyeGaze_Rotation_z_smoothed']] = self.df_all.groupby('Scenario')['EyeGaze_Rotation_z'].transform(
            lambda x: self.smooth_array_gf(x, sigma=4))

    def calculate_distance_with_interaction_point(self):
        self.df_all['Ped_Distance'] = np.sqrt(
            self.df_all.Ped_Location_x**2 + self.df_all.Ped_Location_y**2)
        self.df_all['PodLeader_Distance'] = np.sqrt(
            self.df_all.PodLeader_Location_x**2 + self.df_all.PodLeader_Location_y**2)
        self.df_all['PodFollower_Distance'] = np.sqrt(
            self.df_all.PodFollower_Location_x**2 + self.df_all.PodFollower_Location_y**2)
        self.df_all['FamCar_Distance'] = np.sqrt(
            self.df_all.FamCar_Location_x**2 + self.df_all.FamCar_Location_y**2)
    
    def calculate_distance_between_agents(self):
        half_pod_length = 150
        self.df_all['PodLeader_Ped_CenterDistance_x'] = self.df_all['Ped_Location_x'] - self.df_all['PodLeader_Location_x']
        self.df_all['PodLeader_Ped_CenterDistance_x_abs'] = self.df_all['PodLeader_Ped_CenterDistance_x'].abs()
        self.df_all['PodLeader_Ped_CenterDistance_y'] = self.df_all['Ped_Location_y'] - self.df_all['PodLeader_Location_y']
        self.df_all['PodLeader_Ped_CenterDistance_y_abs'] = self.df_all['PodLeader_Ped_CenterDistance_y'].abs()
        self.df_all['PodLeader_Ped_CenterDistance'] = (self.df_all['PodLeader_Ped_CenterDistance_x']**2 + self.df_all['PodLeader_Ped_CenterDistance_y']**2).apply(np.sqrt)

        self.df_all['PodFollower_Ped_CenterDistance_x'] = self.df_all['Ped_Location_x'] - self.df_all['PodFollower_Location_x']
        self.df_all['PodFollower_Ped_CenterDistance_x_abs'] = self.df_all['PodFollower_Ped_CenterDistance_x'].abs()
        self.df_all['PodFollower_Ped_CenterDistance_y'] = self.df_all['Ped_Location_y'] - self.df_all['PodFollower_Location_y']
        self.df_all['PodFollower_Ped_CenterDistance_y_abs'] = self.df_all['PodFollower_Ped_CenterDistance_y'].abs()
        self.df_all['PodFollower_Ped_CenterDistance'] = (self.df_all['PodFollower_Ped_CenterDistance_x']**2 + self.df_all['PodFollower_Ped_CenterDistance_y']**2).apply(np.sqrt)

        self.df_all['FamCar_Ped_CenterDistance_x'] = self.df_all['Ped_Location_x'] - self.df_all['FamCar_Location_x']

    def preprocess_speed(self):
        self.df_all['DiffTime'] = self.df_all.groupby('Scenario')['TimeElapsed'].transform(
            lambda x: np.insert(np.diff(x), 0, np.nan))
        
        # pedestrian x axis, using smoothed trajectories! (instead of the raw one)
        self.df_all['DiffPedLocation_x'] = self.df_all.groupby('Scenario')['Ped_Location_x_smoothed'].transform(
            lambda x: np.insert(np.diff(x), 0, np.nan))
        self.df_all['Ped_Velocity_x'] = self.df_all['DiffPedLocation_x'] / self.df_all['DiffTime']

        # pedestrian y axis 
        self.df_all['DiffPedLocation_y'] = self.df_all.groupby('Scenario')['Ped_Location_y_smoothed'].transform(
            lambda x: np.insert(np.diff(x), 0, np.nan))
        self.df_all['Ped_Velocity_y'] = self.df_all['DiffPedLocation_y'] / self.df_all['DiffTime']
        
        # pedestrian speed in absolute values
        self.df_all['Ped_Velocity_abs'] = np.sqrt(self.df_all.Ped_Velocity_x**2 + self.df_all.Ped_Velocity_y**2)

        # fill empty values with 0
        self.df_all.fillna({'Ped_Velocity_x': 0}, inplace=True)
        self.df_all.fillna({'Ped_Velocity_y': 0}, inplace=True)
        self.df_all.fillna({'Ped_Velocity_abs': 0}, inplace=True)

        # pedestrian forward speed direction
        self.df_all['Ped_Velocity_Rotation'] = self.convert_rotation(
            self.calculate_speed_direction(self.df_all.Ped_Velocity_x.to_numpy(), self.df_all.Ped_Velocity_y.to_numpy()))
        # self.df_all.fillna({'Ped_Velocity_Rotation': 0}, inplace=True) # replace nan as 0 may not be the right way
        self.df_all['Ped_Velocity_Rotation'] = self.df_all.groupby('Scenario')['Ped_Velocity_Rotation'].transform(
            lambda x: np.unwrap(x.ffill().bfill(), period=360))  # unwrap the angles, making it continuous
        self.df_all['Ped_Velocity_Rotation_smoothed'] = self.df_all.groupby('Scenario')['Ped_Velocity_Rotation'].transform(
            lambda x: self.smooth_array_gf(x, sigma=4))

        # get head velocity relative to the pedestrian velocity rotation 
        self.df_all['head_vel_relative_initial'] = self.convert_into_symmetric_angles(
            (self.df_all.Ped_Rotation_z_smoothed - self.df_all.Ped_Velocity_Rotation_smoothed).to_numpy(), angle_range=180)
        self.get_head_velocity_relative_threshold()

        # get final speed direction based on head-vel-relative and y-location
        direction = (np.logical_or(self.df_all['head_vel_relative_initial'].abs() < self.threshold, self.df_all.DiffPedLocation_y < 0)).astype(int)
        direction[direction == 0] = -1
        self.df_all['direction'] = self.fix_short_direction_segments(direction)
        self.df_all['Ped_Velocity'] = self.df_all.Ped_Velocity_abs * self.df_all.direction
        self.df_all['direction'] = (self.df_all.Ped_Velocity > 0).astype(int)
        self.df_all.loc[self.df_all.direction == 0, 'direction'] = -1  # 1 for forward, -1 for backward

        # smooth speed profile
        self.smooth_speed_gf()

        # get body rotation  
        self.df_all['body_rotation'] = self.convert_into_symmetric_angles(self.df_all.Ped_Velocity_Rotation_smoothed - 180*(self.df_all.direction == -1).to_numpy())
        self.df_all['body_rotation_mask'] = self.df_all['Ped_Velocity_smoothed'].abs() < 35  # mask the body rotation when speed is too low
        self.df_all.loc[self.df_all.body_rotation_mask, 'body_rotation'] = np.nan
        self.df_all['body_rotation_int'] = self.df_all.body_rotation.copy()
        self.df_all['body_rotation_int'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_int'].transform(
            lambda x: x.interpolate(method='nearest', limit_direction='both').to_numpy()) # this is for direction where nearest makes more sense
        self.df_all['body_rotation_int'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_int'].transform(
            lambda x: x.ffill().bfill())
        self.df_all['body_rotation_unwrap'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_int'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous
        self.df_all.loc[self.df_all.body_rotation_mask, 'body_rotation_unwrap'] = np.nan 
        self.df_all['body_rotation_unwrap'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_unwrap'].transform(
            lambda x: x.interpolate(method='linear').to_numpy()) # this is reinterpolate by linear 
        self.df_all['body_rotation_unwrap'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_unwrap'].transform(lambda x: x.ffill().bfill())  
        self.df_all['body_rotation_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_unwrap'].transform(lambda x: self.smooth_array_gf(x, sigma=4))

        self.df_all['head_vel_relative'] = self.convert_into_symmetric_angles((self.df_all.Ped_Rotation_z - self.df_all.body_rotation).to_numpy(), angle_range=180)
        self.df_all['head_vel_relative_smoothed'] = self.convert_into_symmetric_angles((self.df_all.Ped_Rotation_z_smoothed - self.df_all.body_rotation_smoothed).to_numpy(), angle_range=180)
        self.df_all['head_vel_relative_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['head_vel_relative_smoothed'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous

        # self.df_all['eye_vel_relative'] = self.convert_into_symmetric_angles((self.df_all.EyeGaze_Rotation_z - (self.df_all.Ped_Velocity_Rotation - 180*(self.df_all.direction == -1))).to_numpy(), angle_range=180)
        self.df_all['eye_vel_relative'] = self.convert_into_symmetric_angles((self.df_all.EyeGaze_Rotation_z - self.df_all.body_rotation).to_numpy(), angle_range=180)
        self.df_all['eye_vel_relative_smoothed'] = self.convert_into_symmetric_angles((self.df_all.EyeGaze_Rotation_z_smoothed - self.df_all.body_rotation_smoothed).to_numpy(), angle_range=180)
        self.df_all['eye_vel_relative_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['eye_vel_relative_smoothed'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous

        self.df_all['eye_head_relative'] = self.convert_into_symmetric_angles((self.df_all.EyeGaze_Rotation_z - self.df_all.Ped_Rotation_z).to_numpy(), angle_range=180)
        self.df_all['eye_head_relative_smoothed'] = self.convert_into_symmetric_angles((self.df_all.EyeGaze_Rotation_z_smoothed - self.df_all.Ped_Rotation_z_smoothed).to_numpy(), angle_range=180)
        self.df_all['eye_head_relative_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['eye_head_relative_smoothed'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous

        # recenter all angles
        self.df_all['Ped_Velocity_Rotation_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['Ped_Velocity_Rotation_smoothed'].transform(self.recenter_angles)
        self.df_all['Ped_Rotation_z_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['Ped_Rotation_z_smoothed'].transform(self.recenter_angles)
        self.df_all['body_rotation_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['body_rotation_smoothed'].transform(self.recenter_angles)
        self.df_all['EyeGaze_Rotation_z_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['EyeGaze_Rotation_z_smoothed'].transform(self.recenter_angles)
        self.df_all['head_vel_relative_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['head_vel_relative_smoothed'].transform(self.recenter_angles)
        self.df_all['eye_vel_relative_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['eye_vel_relative_smoothed'].transform(self.recenter_angles)
        self.df_all['eye_head_relative_smoothed'] = self.df_all.groupby(['pid', 'Scenario'])['eye_head_relative_smoothed'].transform(self.recenter_angles)

        # convert relative angles into sysmetric angles
        self.df_all['head_vel_relative_smoothed'] = self.convert_into_symmetric_angles(self.df_all['head_vel_relative_smoothed'], angle_range=180)
        self.df_all['eye_vel_relative_smoothed'] = self.convert_into_symmetric_angles(self.df_all['eye_vel_relative_smoothed'], angle_range=180)
        self.df_all['eye_head_relative_smoothed'] = self.convert_into_symmetric_angles(self.df_all['eye_head_relative_smoothed'], angle_range=180)

        self.df_all = self.df_all.drop(columns=['DiffTime', 'DiffPedLocation_x'])

    def recenter_angles(self, values):
        center = values.mean()
        n = (center + 180) // 360
        return values - 360*n

    def mask_near_direction_change(self, value, direction_series, window=2):
        change_mask = direction_series.shift() != direction_series
        change_mask.iloc[0] = False  # 第一个元素没有前一个元素，设为False

        # 生成布尔掩码
        mask = np.zeros(len(direction_series), dtype=bool)

        # 前后2行偏移量
        for offset in range(-window, window):  
            mask |= change_mask.shift(offset, fill_value=False)

        # 将value设为NaN
        value[mask] = np.nan
        return value 

    def fix_short_direction_segments(self, direction_series, min_length=15):
        direction = direction_series.to_numpy()
        # 找每个段的开始和结束
        change_idx = np.flatnonzero(np.diff(direction) != 0) + 1
        segment_bounds = np.r_[0, change_idx, len(direction)]
        
        # 逐段处理
        seg_lengths = np.diff(segment_bounds)
        seg_values = direction[segment_bounds[:-1]]
        
        # 找出短段
        short_mask = seg_lengths < min_length
        
        # 修正短段
        for i, short in enumerate(short_mask):
            if short:
                prev_val = seg_values[i - 1] if i > 0 else None
                next_val = seg_values[i + 1] if i < len(seg_values) - 1 else None
                if prev_val == next_val and prev_val is not None:
                    direction[segment_bounds[i]:segment_bounds[i+1]] = prev_val

        return direction

    def convert_into_symmetric_angles(self, angle, angle_range=180):
        return (angle + angle_range) % 360 - angle_range

    def smooth_trajectories_gf(self, sigma=4):
        self.df_all['Ped_Location_x_smoothed'] = self.df_all.groupby('Scenario')['Ped_Location_x'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
        self.df_all['Ped_Location_y_smoothed'] = self.df_all.groupby('Scenario')['Ped_Location_y'].transform(
            lambda x: self.smooth_array_gf(x, sigma))

    def smooth_speed_gf(self, sigma=4):
        self.df_all['Ped_Velocity_x_smoothed'] = self.df_all.groupby('Scenario')['Ped_Velocity_x'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
        self.df_all['Ped_Velocity_y_smoothed'] = self.df_all.groupby('Scenario')['Ped_Velocity_y'].transform(
            lambda x: self.smooth_array_gf(x, sigma)) 
        self.df_all['Ped_Velocity_abs_smoothed'] = self.df_all.groupby('Scenario')['Ped_Velocity_abs'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
        self.df_all['Ped_Velocity_smoothed'] = self.df_all.groupby('Scenario')['Ped_Velocity'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
    
    def smooth_array_gf(self, array, sigma=4):
        return scipy.ndimage.gaussian_filter1d(array, sigma=sigma, axis=0, mode='nearest')
        
    def correct_velocity_rotation_range(self, vx, vy, alpha):
        if vx > 0 and vy < 0:
            return - alpha
        elif vx < 0 and vy < 0:
            return alpha - 180
        elif vx < 0 and vy > 0:
            return - 180 - alpha 
        else:
            return alpha - 360
    
    def calculate_speed_direction(self, arr_vx, arr_vy):
        res = []
        arr_alpha = np.arctan(np.abs(arr_vy) / np.abs(arr_vx)) * 180 / np.pi
        for vx, vy, alpha in zip(arr_vx, arr_vy, arr_alpha):
            res.append(self.correct_velocity_rotation_range(vx, vy, alpha))
        return np.array(res)

    def get_consecutive_trues(self, bool_array, verbose=False):
        # Identify consecutive True groups
        diff = np.diff(np.concatenate(([False], bool_array, [False])).astype(int))
        start_indices = np.where(diff == 1)[0]
        end_indices = np.where(diff == -1)[0]

        # Length of consecutive True sequences
        lengths = end_indices - start_indices
        # print("Consecutive True lengths:", lengths)  # Output: [2, 1, 3]
        if verbose:
            return start_indices, end_indices, lengths
        else:
            return lengths 

    def find_nearest_zero_velocity_before_pod_moving(self, df_j):
        df_j = df_j.reset_index()
        diff_dist = df_j.PodLeader_Distance.diff()
        idx = diff_dist[diff_dist != 0].index[1]

        bool_velocity_near_zero = (df_j.Ped_Velocity_smoothed.abs() < 3).to_numpy()
        _, end, lengths = self.get_consecutive_trues(bool_velocity_near_zero, verbose=True)
        idx_before = np.where(end < idx)[0]
        if len(idx_before) > 0:
            end_last_idx = idx_before[-1]
            last_idx = end[end_last_idx]
            return last_idx - int(lengths[end_last_idx] / 2)
        else:
            return None
        
    def trim_white_to_green(self):
        self.trim_wrong_head_rotation()
        self.trim_wrong_starting_points()

    def trim_wrong_head_rotation(self):
        lengths = self.df_all.groupby('Scenario')['Ped_Rotation_z'].apply(lambda x: len(self.get_consecutive_trues(x > 0)))
        if (lengths > 0).sum() > 0:
            print(lengths[lengths > 0].to_dict())
        for i in lengths[lengths > 0].index.to_numpy():
            df_i = self.df_all[self.df_all.Scenario == i]
            idx_new = self.find_nearest_zero_velocity_before_pod_moving(df_i)
            if idx_new:
                print(f'\nTrimming scenario {i} at index {idx_new} (white to green - rotation)')
                # bc of df.drop_duplicates(), there are missing index
                self.df_all = self.df_all.drop(index=[i for i in df_i.index.to_numpy() if df_i.index[0] <= i <= df_i.index[idx_new]])

    def remove_pod_already_left(self):
        is_there_famcar = self.df_all.groupby('Scenario')['FamCar_Distance'].transform(lambda x: (x < 2000).sum() > 0)
        is_there_podleader = self.df_all.groupby('Scenario')['PodLeader_Distance'].transform(lambda x: (x < 2000).sum() > 0)
        is_there = np.logical_or(is_there_famcar, is_there_podleader)
        print(f'\nRemoving scenarios {self.df_all[~is_there].Scenario.unique()} (no pod)')
        self.df_all = self.df_all[is_there]

    def remove_mistakenly_triggered_scenarios(self):
        self.get_mistakenly_triggered_scenarios()
        self.df_all = self.df_all[self.df_all.Scenario.apply(lambda x: x not in self.list_mistakenly_triggered)]
        
    def get_mistakenly_triggered_scenarios(self):
        self.list_mistakenly_triggered = []
        for j, idx in self.famcar_trigger_points.items():
            if idx == 1:
                self.list_mistakenly_triggered.append(f'F{j}')
        for j, idx in self.podleader_trigger_points.items():
            if idx == 1:
                self.list_mistakenly_triggered.append(str(j))
        print(f'Removing scenarios {self.list_mistakenly_triggered} (mistakenly triggered)')

    def get_pod_trigger_points(self):
        self.famcar_trigger_points = dict()
        self.famcar_trigger_index = dict()

        self.podleader_trigger_points = dict()
        self.podleader_trigger_index = dict()

        for j in self.df_all.Scenario.unique():
            df_j = self.df_all[self.df_all.Scenario == j]
            idx_base = df_j.index.to_numpy()[0]

            if 'F' in j:
                diff_dist = df_j.FamCar_Location_x.diff().reset_index(drop=True)
                idx = diff_dist[diff_dist != 0].index[1]
                j = int(j[1:])
                self.famcar_trigger_points[j] = idx
                self.famcar_trigger_index[j] = idx_base + idx
            else:
                diff_dist = df_j.PodLeader_Location_x.diff().reset_index(drop=True)
                idx = diff_dist[diff_dist != 0].index[1]
                j = int(j)
                self.podleader_trigger_points[j] = idx
                self.podleader_trigger_index[j] = idx_base + idx

        print('\nFamiliarization:')
        print(self.famcar_trigger_points)
        print(self.famcar_trigger_index)

        print('\nFormal experiment:')
        print(self.podleader_trigger_points)
        print(self.podleader_trigger_index)

    def match_columns_for_all_data(self):
        if 'PodLeader_EhmiStatus' not in self.df_all.columns:
            self.df_all['PodLeader_EhmiStatus'] = None
            self.df_all['PodFollower_EhmiStatus'] = None
            for sid in self.dt['sid'].unique():
                is_ehmi = self.dt[(self.dt.sid == sid) & (self.dt.pid == self.pid)].eHMI.iloc[0] == 'yes'
                is_yielding = self.dt[(self.dt.sid == sid) & (self.dt.pid == self.pid)].yielding.iloc[0]
                is_continuous = self.dt[(self.dt.sid == sid) & (self.dt.pid == self.pid)]['traffic flow'].iloc[0] != 0
                if is_ehmi: 
                    if is_yielding:
                        self.df_all.loc[(self.df_all['sid'] == sid) & (self.df_all['PodLeader_Ped_CenterDistance_x'] < 1330) & (self.df_all['PodLeader_Ped_CenterDistance_x'] >= 0), 'PodLeader_EhmiStatus'] = 'green'
                        if is_continuous:
                            self.df_all.loc[(self.df_all['sid'] == sid) & (self.df_all['PodFollower_Ped_CenterDistance_x'] < 1330) & (self.df_all['PodFollower_Ped_CenterDistance_x'] >= 0), 'PodFollower_EhmiStatus'] = 'green'
                    else:
                        self.df_all.loc[(self.df_all['sid'] == sid) & (self.df_all['PodLeader_Ped_CenterDistance_x'] < 1330) & (self.df_all['PodLeader_Ped_CenterDistance_x'] >= 0), 'PodLeader_EhmiStatus'] = 'red'
                        if is_continuous:
                            self.df_all.loc[(self.df_all['sid'] == sid) & (self.df_all['PodFollower_Ped_CenterDistance_x'] < 1330) & (self.df_all['PodFollower_Ped_CenterDistance_x'] >= 0), 'PodFollower_EhmiStatus'] = 'red'
            # self.df_all['FamCar_EhmiStatus'] = None
            # for sid in self.dt_fam.sid.unique():
            #     is_ehmi = self.dt_fam[(self.dt_fam.sid == sid) & (self.dt_fam.pid == self.pid)].eHMI.iloc[0] == 'yes'
            #     is_yielding = self.dt_fam[(self.dt_fam.sid == sid) & (self.dt_fam.pid == self.pid)].yielding.iloc[0]
        else:
            self.df_all.drop(columns=[
                'GazeOrigin_Location_x', 'GazeOrigin_Location_y', 'GazeOrigin_Location_z',
                'PodLeader_PedDistance_x', 'PodLeader_PedDistance_y', 
                'PodFollower_PedDistance_x', 'PodFollower_PedDistance_y',
                'FamCar_PedDistance_x', 'FamCar_PedDistance_y', 'FamCar_EhmiStatus'], inplace=True)

    def get_head_velocity_relative_threshold(self):
        data_ped = self.df_all.head_vel_relative_initial.abs().to_numpy()

        x_vals, density, x_argmax = self.fit_gaussian_mixture(data_ped)
        x_diff_idx = density[x_argmax[0]: x_argmax[1]].argmin() + x_argmax[0]
        x_diff = int(x_vals[x_diff_idx])

        self.threshold = max(min(x_diff, 90), 45)
        print(f'\nSetting head_vel_relative threshold as {self.threshold}')
    
    def fit_gaussian_mixture(self, data):
        # Fit a GMM with 2 components
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data.reshape(-1, 1))  # Reshape data to be 2D for GMM

        # Extract parameters
        weights = gmm.weights_  # Weights of the two components
        means = gmm.means_.flatten()  # Means of the two components
        covariances = gmm.covariances_.flatten() 

        x_vals = np.linspace(min(data), max(data), 1000)
        density = np.zeros_like(x_vals)
        x_argmax = []

        # Plot each Gaussian component and sum them up for the total density
        for weight, mean, covariance in zip(weights, means, covariances):
            std_dev = np.sqrt(covariance)  # Standard deviation
            component_density = weight * norm.pdf(x_vals, mean, std_dev)
            # plt.plot(x_vals, component_density, label=f'Gaussian (mean={mean:.2f}, std={std_dev:.2f})')
            density += component_density
            x_argmax.append(component_density.argmax())
        x_argmax.sort()
        
        return x_vals, density, x_argmax
    
    def trim_wrong_starting_points(self):
        unique_combo = self.df_all[['Scenario', 'sid']].drop_duplicates()
        for j, sid in zip(unique_combo['Scenario'], unique_combo['sid']):
            df_i = self.df_all[self.df_all.Scenario == j]
            # based on starting position 
            if not self.check_starting_points_range(
                df_i.Ped_Location_x_smoothed.iloc[0], df_i.Ped_Location_y_smoothed.iloc[0], sid, True if 'F' in j else False):
                idx_new = self.find_nearest_zero_velocity_before_pod_moving(df_i)
                if idx_new:
                    print(f'Trimming scenario {j} at index {idx_new} (white to green - starting positions)')
                    # bc of df.drop_duplicates(), there are missing index
                    self.df_all = self.df_all.drop(index=[i for i in df_i.index.to_numpy() if df_i.index[0] <= i <= df_i.index[idx_new]])
            # based on last zero velocity
            else:
                idx_new = self.find_nearest_zero_velocity_before_pod_moving(df_i)
                if idx_new:
                    print('Trimming scenario', j, 'at index', idx_new, '(white to green - last zero velocity)')
                    self.df_all = self.df_all.drop(index=[i for i in df_i.index.to_numpy() if df_i.index[0] <= i <= df_i.index[idx_new]])
            # based on eye gazing object
            start_name = None
            for c in df_i.HitObject.astype(str).unique():
                if c != 'nan':
                    if 'start' in c:
                        start_name = c
                        break
            if start_name:
                idx_new3 = df_i[df_i.HitObject.astype(str) == start_name].index.to_numpy()
                idx_new3_start, idx_new3_end = idx_new3[0], idx_new3[-1]
                time_pasted = df_i.TimeElapsed.loc[idx_new3_end] - df_i.TimeElapsed.iloc[0]
                time_remained = df_i.TimeElapsed.iloc[-1] - df_i.TimeElapsed.loc[idx_new3_start]
                trial_start_idx = df_i.index[0]
                if time_pasted < time_remained: # delete forward
                    idx_new3_start = None
                    # delete forward 
                    if idx_new is not None:
                        if df_i.index[idx_new] < idx_new3_end:
                            idx_new3_start = df_i.index[idx_new] + 1
                    else:
                        idx_new3_start = df_i.index[0]
                    if idx_new3_start:
                        print('Trimming scenario', j, 'from', idx_new3_start-trial_start_idx, 'to', idx_new3_end-trial_start_idx, '(white to green - gazing object)')
                        self.df_all = self.df_all.drop(index=np.arange(idx_new3_start, idx_new3_end+1).tolist())
                else: # delete backward 
                    idx_new3_end = df_i.index[-1]
                    print('Trimming scenario', j, 'from', idx_new3_start-trial_start_idx, 'to the end (white to green - gazing object)')
                    self.df_all = self.df_all.drop(index=np.arange(idx_new3_start, idx_new3_end+1).tolist())
                        

    def check_starting_points_range(self, x_init, y_init, sid, is_fam=False):
        '''
        45 degree: x ~ (-350, -250), y ~ (250, 350)
        90 degree: x ~ (-50, 50), y ~ (350, 450)
        135 degree: x ~ (350, 250), y ~ (250, 350)
        '''
        # get angle
        if is_fam:
            angle = 90 
        else:
            angle = self.dt[self.dt.sid==sid].angle.iloc[0]
        # range by angles 
        if angle == 45: 
            if -350 < x_init < -250 and 250 < y_init < 350:
                return True
            else:
                return False
        elif angle == 90:
            if -50 < x_init < 50 and 350 < y_init < 450:
                return True
            else:
                return False
        elif angle == 135: 
            if 250 < x_init < 350 and 250 < y_init < 350:
                return True
            else:
                return False
            
    def remove_trials_with_bad_speed(self, speed_threshold=300): 
        bad_scenarios = self.df_all[self.df_all.Ped_Velocity_smoothed > speed_threshold].Scenario.unique()
        if bad_scenarios.shape[0]:
            print(f'Removing scenarios {bad_scenarios} (bad speed)')
            self.df_all = self.df_all[self.df_all.Scenario.apply(lambda x: x not in bad_scenarios)]

    def remove_collision(self, half_length=150, half_width=80):
        collision_scenarios = self.df_all[
            ((abs((self.df_all.Ped_Location_x_smoothed - self.df_all.PodLeader_Location_x).to_numpy()) < half_length) &
             (abs((self.df_all.Ped_Location_y_smoothed - self.df_all.PodLeader_Location_y).to_numpy()) < half_width)) | 
            ((abs((self.df_all.Ped_Location_x_smoothed - self.df_all.PodFollower_Location_x).to_numpy()) < half_length) & 
             (abs((self.df_all.Ped_Location_y_smoothed - self.df_all.PodFollower_Location_y).to_numpy()) < half_width))].Scenario.unique()
        if collision_scenarios.shape[0]:
            print(f'Removing scenarios {collision_scenarios} (collision)')
            self.df_all = self.df_all[self.df_all.Scenario.apply(lambda x: x not in collision_scenarios)]


def main(args):
    if isinstance(args.filenames, str):
        args.filenames = [args.filenames]
    for filename in args.filenames:
        ita = Interaction(
            filename, 
            is_recenter=args.is_center, 
            is_save_prep=args.is_save_prep, 
            is_plot_intermediate=args.is_plot_intermediate, 
            vis_fam=args.vis_fam,
            suffix=args.suffix)
        if ita.df.shape[0] > 0:
            ita.visualize()
        else:
            print(f'No data for {filename}, skipping visualization.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-d", "--data_dir", type=str, default='data/vrdata/combined')
    parser.add_argument("-f", "--filenames", nargs='+', type=str, default=None)
    parser.add_argument("--is_center", type=bool, default=True)
    parser.add_argument("--is_save_prep", type=bool, default=True)
    parser.add_argument("--is_plot_intermediate", type=bool, default=False)
    parser.add_argument("--vis_fam", default=False, action='store_true')
    parser.add_argument("--suffix", type=str, default='_prep')
    args = parser.parse_args()

    # list all filenames in the data directory
    if args.data_dir:
        args.filenames = [f.split('.')[0] for f in os.listdir(args.data_dir) if f.endswith('.csv')]
    main(args)

## example
# python -m utils.preprocessing -f filenames
# python -m utils.preprocessing -f P1_20241217103615516 P2_202412171419360 P3_20241217152233801 P4_2024121911738427 P5_20241219132245990 P6_2024121915620772 P7_2024121915567133 P8_2024121916394796 > out1-8.txt
# python -m utils.preprocessing -f P9_20251610917686 P10_20251613298921 P11_202516151537305 P12_202516171753723 P13_202517101430120 P14_20251711546819 P15_20251714314880 P16_202517151737922 P17_202517162823137 P18_2025171745643 > out9-18.txt
# python -m utils.preprocessing -f P19_202519171112119 P20_202511010139902 P21_202511011150733 P22_202511013946226 P23_202511014555424 P24_202511015455563 P25_2025110161322537 P26_202511017318590 P27_202511194845442 P28_20251111384223 P29_20251111435533 P30_2025111151424898 P31_202511116142262 > out19-31.txt
# python -m utils.preprocessing -f P32_202511395855284 P33_2025113131228931 P34_2025113135340982 P35_202511315915944 P36_20251131675773 P37_202511317103643 P38_2025114101220373 P39_20251141112424 P40_2025114131911860 P41_202511414230295 P42_202511415655706 P43_2025114162926461 P44_2025114171410580 > out32-44.txt
# python -m utils.preprocessing -f P45_202511610513261 P46_202511611914394 P47_202511614223187 P48_2025116151043950 P49_2025116171511625 > out45-49.txt
# python -m utils.preprocessing -f P50_202511813834529 P51_2025118141418658 > out50-51.txt
# python -m utils.preprocessing -f P50_202511813834529 P51_2025118141418658 --suffix _noremoval > out50-51_noremoval.txt
    
# python -m utils.preprocessing -f P1_20241217103615516 P2_202412171419360 P3_20241217152233801 P4_2024121911738427 P5_20241219132245990 P6_2024121915620772 P7_2024121915567133 P8_2024121916394796 P9_20251610917686 P10_20251613298921 P11_202516151537305 P12_202516171753723 P13_202517101430120 P14_20251711546819 P15_20251714314880 P16_202517151737922 P17_202517162823137 P18_2025171745643 P19_202519171112119 P20_202511010139902 P21_202511011150733 P22_202511013946226 P23_202511014555424 P24_202511015455563 P25_2025110161322537 P26_202511017318590 P27_202511194845442 P28_20251111384223 P29_20251111435533 P30_2025111151424898 P31_202511116142262 P32_202511395855284 P33_2025113131228931 P34_2025113135340982 P35_202511315915944 P36_20251131675773 P37_202511317103643 P38_2025114101220373 P39_20251141112424 P40_2025114131911860 P41_202511414230295 P42_202511415655706 P43_2025114162926461 P44_2025114171410580 P45_202511610513261 P46_202511611914394 P47_202511614223187 P48_2025116151043950 P49_2025116171511625 P50_202511813834529 P51_2025118141418658 2>&1 | tee out_preprocess.txt
      