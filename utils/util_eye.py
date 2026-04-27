import numpy as np
import pandas as pd
import os, shutil, builtins
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from collections import Counter


def getTerminalWidth():
	"""
	Return the current wdith of the terminal/shell
	"""
	return shutil.get_terminal_size()[0]


def dist_angle_arrays_unsigned(vecs1, vecs2):
	# Unsigned distance computed pair-wise between two arrays of unit vectors
	vecs1 = vecs1 / np.linalg.norm(vecs1, axis=1, keepdims=True)
	vecs2 = vecs2 / np.linalg.norm(vecs2, axis=1, keepdims=True) # may get nan if not normalized

	dot = np.einsum("ji,ji->j", vecs1, vecs2)
	return np.arccos(dot)


def dist_angle_vectors_unsigned(vec1, vec2):
	# Signed distance computed between two unit vectors

	if len(vec1.shape) != 2:
		vec1 = vec1[None, :]

	if len(vec2.shape) != 2:
		vec2 = vec2[None, :]

	return dist_angle_arrays_unsigned(vec1, vec2)[0]


def printWarning(*args, header="Warning", bold=True, verbose=1, **kwargs):
	"""
	DOC
	"""
	if header is not None and os.name != "nt" and header != "":
		header = "\033[1;4m{}:\033[m\033[33m".format(header)
		args = header, *args
	printC(*args, color="33", bold=bold, verbose=verbose, **kwargs)


def printNeutral(*args, header=None, bold=True, verbose=2, **kwargs):
	"""
	DOC
	"""
	if header is not None and os.name != "nt" and header != "":
		header = "\033[1;4m{}:\033[m\033[94m".format(header)
		args = header, *args
	printC(*args, color="94", bold=bold, verbose=verbose, **kwargs)


def printC(*args, color="21", bold=False, tab=0, clear=False, verbose=1, exit_=False, sep=" ", **kwargs):
	"""
	DOC
	"""
	# Verbose level
	if verbose != -1 and hasattr(builtins, "verbose") and verbose > builtins.verbose:
		return
	# Clear line before?
	if clear:
		cl = "\r" + " "*getTerminalWidth() + "\r"
	else:
		cl = ""

	if type(tab) == float:
		tab = " "*int(tab)
	else:
		tab = " "*(tab*2)

	if os.name == "nt":
		print("{}{}{}".format(
				cl,
				" "*(tab*2),
				sep.join(map(str, args))
				),
			**kwargs)
	else:
		print("{}{}\033[{}m\033[{}m{}\033[m".format(
				cl,
				tab,
				int(bold),
				color,
				sep.join(map(str, args))
				),
			**kwargs)

	if exit_: exit()



def fix_gen(label_list):
	# Removes unique True and False values surrounded by their complement (would usually disappear when the signal is smoothed)

	for i in range(1, label_list.shape[0]-1):
		lc = label_list[i]
		lm = label_list[i-1]
		lp = label_list[i+1]

		if lc != lm and lc != lp:
			label_list[i] = lm

	for i in [0, label_list.shape[0]-1]:
		lc = label_list[i]

		if i > 0:
			lm = label_list[i-1]
		else: lm = not lc

		if i < (label_list.shape[0]-1):
			lp = label_list[i+1]
		else: lp = not lc

		if lc != lm and lc != lp:
			label_list[i] = lm


def get_yaw_velocity(gp, return_keep=False, outlierSigma=5):
	"""Input data: gaze data as unit vector followed by a timestamp
	Return the angular velocity of gaze in rad/ms
	0: X
	1: Y
	2: Z
	3: timestamp
	"""

	diffT = gp[1:, 3] - gp[:-1, 3]

	distance = dist_angle_arrays_unsigned(gp[1:, :3], gp[:-1, :3])

	velocity = distance/diffT
	velocity = np.append(velocity, [velocity[0]])

	if return_keep:
		# Remove samples farther than X std from the mean
		keep = np.abs((velocity-np.nanmean(velocity))/np.nanstd(velocity)) < outlierSigma
		printNeutral("Removed {} samples more than {} sigmas away from the mean".format((keep==0).sum(), outlierSigma),
			bold=False, verbose=2)

		keep &= np.logical_not(np.isnan(velocity) | np.isinf(velocity))

		return keep, velocity
	else:
		return None, velocity


def remove_short_markers(markers, Nsamples, timestamp, min_dur=80):
	iDisp = 1
	startMarker = 0

	# Remove short fixations
	change = False
	iDisp = 1
	pStart = np.zeros(1)
	pEnd = np.zeros(1)
	while iDisp < (Nsamples-1):
		# print("  \rremove sf", iDisp, end="");
		# Point where sacc ends and fix starts
		if not markers[iDisp] and markers[iDisp+1]:
			# print("  \rremove sf", iDisp, end="");
			startMarker = iDisp
			pStart[:] = timestamp[iDisp]

			# Loop ahead until we find the start of a new saccade
			while iDisp < (Nsamples-1) and markers[iDisp+1]:
				iDisp+=1

			pEnd[:] = timestamp[iDisp]

			if pEnd-pStart < min_dur:
				markers[startMarker: iDisp+1] = False
				change = True
		else:
			iDisp += 1

		# Reset until no small fixations are found
		if iDisp==(Nsamples-1) and change:
			iDisp = 1
			change = False
	return markers


def remove_short_markers_plus_affected_seq(markers, affected_seq, timestamp, min_dur=80):
	Nsamples = markers.shape[0]
	iDisp = 1
	startMarker = 0

	# Remove short fixations
	change = False
	iDisp = 1
	pStart = np.zeros(1)
	pEnd = np.zeros(1)
	while iDisp < (Nsamples-1):
		# print("  \rremove sf", iDisp, end="");
		# Point where sacc ends and fix starts
		if not markers[iDisp] and markers[iDisp+1]:
			# print("  \rremove sf", iDisp, end="");
			startMarker = iDisp
			pStart[:] = timestamp[iDisp]

			# Loop ahead until we find the start of a new saccade
			while iDisp < (Nsamples-1) and markers[iDisp+1]:
				iDisp+=1

			pEnd[:] = timestamp[iDisp]

			if pEnd-pStart < min_dur:
				markers[startMarker: iDisp+1] = False
				affected_seq[startMarker: iDisp+1] = True
				change = True
		else:
			iDisp += 1

		# Reset until no small fixations are found
		if iDisp==(Nsamples-1) and change:
			iDisp = 1
			change = False

	return markers, affected_seq


def parser_fix_VT(timestamp, velocity, threshold=100, minFixationTime=80):
	"""
	Parse gaze data on sphere and output a saccade/fixation label list with a velocity-base algorithm.
	Returns an integer array where 0 = fixations, 1 = saccades
	"""

	threshold = np.deg2rad(threshold)/1000 # Eye threshold rad/ms
	print('fixation threshold is', threshold)

	# Label as part of fixations samples with velocity below threshold
	#	i.e., Fixation == 1, Saccade == 0
	fixationMarkers = np.array(velocity <= threshold, dtype=np.bool)

	fix_gen(fixationMarkers)

	Nsamples = velocity.shape[0]
	fixationMarkers = remove_short_markers(
		fixationMarkers, Nsamples, timestamp, min_dur=minFixationTime)
	
	return fixationMarkers
	

def parser_sac_VT(timestamp, velocity, threshold=100, minSacTime=80):
	"""
	Parse gaze data on sphere and output a saccade/fixation label list with a velocity-base algorithm.
	Returns an integer array where 0 = fixations, 1 = saccades
	"""

	threshold = np.deg2rad(threshold)/1000 # Eye threshold rad/ms
	print('saccades threshold is', threshold)

	# Label as part of fixations samples with velocity below threshold
	#	i.e., Saccades == 1, Others == 0
	sacMarkers = np.array(velocity > threshold, dtype=np.bool)

	fix_gen(sacMarkers)

	Nsamples = velocity.shape[0]
	sacMarkers = remove_short_markers(
        sacMarkers, Nsamples, timestamp, min_dur=minSacTime)

	return sacMarkers


def get_smooth_pursuit_via_angle(
		timestamp, eye_yaw, leader_yaw, follower_yaw, 
		window_size=5, step_size=1, mae_thd=50, thd_corr=0.6, min_dur=100):
    corrs_leader, centers_leader, mae_leader, bestlags_leader = sp_detection_via_angle_matching(eye_yaw, leader_yaw, window_size=window_size, step_size=step_size)
    corrs_follower, centers_follower, mae_follower, bestlags_follower = sp_detection_via_angle_matching(eye_yaw, follower_yaw, window_size=window_size, step_size=step_size)
    mask_leader = (corrs_leader > thd_corr) & (mae_leader < mae_follower) & (mae_leader < mae_thd)
    mask_follower = (corrs_follower > thd_corr) & (mae_follower < mae_leader) & (mae_follower < mae_thd)
    Nsamples = mask_leader.shape[0]
    leader_makers = remove_short_markers(mask_leader, Nsamples, timestamp, min_dur=min_dur)
    follower_makers = remove_short_markers(mask_follower, Nsamples, timestamp, min_dur=min_dur)
    return leader_makers, follower_makers


def get_smooth_pursuit_via_loc(
		timestamp, eye_yaw, loc_x, loc_y, leader_x, leader_y, follower_x, follower_y, 
        window_size=5, step_size=1, lag=3, mae_thd=50, thd_corr=0.6, min_dur=100):
	corrs_leader, centers_leader, mae_leader, bestlags_leader = sp_detection_via_loc_matching(
		eye_yaw, loc_x, loc_y, leader_x, leader_y, window_size=window_size, step_size=step_size, lag=lag)
	corrs_follower, centers_follower, mae_follower, bestlags_follower = sp_detection_via_loc_matching(
		eye_yaw, loc_x, loc_y, follower_x, follower_y, window_size=window_size, step_size=step_size)
	mask_leader = (corrs_leader > thd_corr) & (mae_leader < mae_follower)
	mask_follower = (corrs_follower > thd_corr) & (mae_follower < mae_leader)
	
	Nsamples = mask_leader.shape[0]
	leader_makers = remove_short_markers(mask_leader, Nsamples, timestamp, min_dur=min_dur)
	follower_makers = remove_short_markers(mask_follower, Nsamples, timestamp, min_dur=min_dur)

	return leader_makers, follower_makers, bestlags_leader, bestlags_follower, corrs_leader, corrs_follower
	

def sp_detection_via_angle_matching(gaze_yaw, target_yaw, window_size=10, step_size=1, lag=3):
    """
    Compute correlation between gaze yaw and target yaw over sliding windows.
    
    Parameters
    ----------
    gaze_yaw : array-like
        Time series of gaze yaw angle (deg or rad).
    target_yaw : array-like
        Time series of target yaw angle (deg or rad).
    window_size : int
        Window length in samples (at 20 Hz, 10 samples = 500 ms).
    step_size : int
        Step size in samples (default=1 sample = 50 ms at 20 Hz).
    
    Returns
    -------
    corrs : np.ndarray
        Correlation values for each window (aligned to window center).
    centers : np.ndarray
        Sample indices corresponding to correlation values.
    """
    n = len(gaze_yaw)
    gaze_yaw = np.pad(np.asarray(gaze_yaw), (window_size//2, window_size//2), mode='edge')
    target_yaw = np.pad(np.asarray(target_yaw), (window_size//2, window_size//2), mode='edge')
    
    corrs = []
    centers = []
    maes = []
    bestlags = []
    
    for start in range(-(window_size//2), n - window_size//2, step_size):
        end = start + window_size
        t = target_yaw[start:end]
		
        corr_l = []
        mae_l = []
        # iterate over lags and take the max correlation
        for l in range(-lag, lag+1):
            start_l = start + l
            end_l = end + l
            if start_l < - (window_size//2) or end_l > n + (window_size//2):
                r = np.nan
                mae = np.nan
            else:
                g = gaze_yaw[start_l:end_l]
            
                if np.std(g) > 1e-6 and np.std(t) > 1e-6:
                    r, _ = pearsonr(g, t)
                    mae = np.mean(np.abs(g - t))
                else:
                    r = np.nan  # correlation undefined if one signal is flat
                    mae = np.nan
                    
            corr_l.append(r)
            mae_l.append(mae)
        if np.isnan(corr_l).all():
            l_best = np.nan
            r_best = np.nan
            mae_best = np.nan
        else:
            l_best = np.nanargmax(corr_l)
            r_best = corr_l[l_best]
            mae_best = mae_l[l_best]
        
        corrs.append(r_best)
        centers.append(start + window_size // 2)
        maes.append(mae_best)
        bestlags.append(l_best - lag)
		
    return np.array(corrs), np.array(centers), np.array(maes), np.array(bestlags)


def sp_detection_via_loc_matching(gaze_yaw, loc_x, loc_y, target_x, target_y, window_size=10, step_size=1, lag=3):
    """
    Compute correlation between gaze yaw and target yaw over sliding windows.
    
    Parameters
    ----------
    gaze_yaw : array-like
        Time series of gaze yaw angle (deg or rad).
    target_yaw : array-like
        Time series of target yaw angle (deg or rad).
    window_size : int
        Window length in samples (at 20 Hz, 10 samples = 500 ms).
    step_size : int
        Step size in samples (default=1 sample = 50 ms at 20 Hz).
    
    Returns
    -------
    corrs : np.ndarray
        Correlation values for each window (aligned to window center).
    centers : np.ndarray
        Sample indices corresponding to correlation values.
    """
    n = len(gaze_yaw)
    delta_y = loc_y - target_y
    delta_x = delta_y / np.tan(np.deg2rad(gaze_yaw))
    mask = (np.abs(gaze_yaw) % 90 - 1) < 1e-3
    delta_x[mask] = 0
    gaze_x = loc_x - delta_x

    gaze_x = np.pad(np.asarray(gaze_x), (window_size//2, window_size//2), mode='edge')
    target_x = np.pad(np.asarray(target_x), (window_size//2, window_size//2), mode='edge')
    
    corrs = []
    centers = []
    maes = []
    bestlags = []
    
    for start in range(-(window_size//2), n - window_size//2, step_size):
        end = start + window_size
        t = target_x[start:end]
		
        corr_l = []
        mae_l = []
        # iterate over lags and take the max correlation
        for l in range(-lag, lag+1):
            start_l = start + l
            end_l = end + l
            if start_l < - (window_size//2) or end_l > n + (window_size//2):
                r = np.nan
                mae = np.nan
            else:
                g = gaze_x[start_l:end_l]
            
                if np.std(g) > 1e-6 and np.std(t) > 1e-6:
                    r, _ = pearsonr(g, t)
                    mae = np.mean(np.abs(g - t))
                else:
                    r = np.nan  # correlation undefined if one signal is flat
                    mae = np.nan
                    
            corr_l.append(r)
            mae_l.append(mae)
        if np.isnan(corr_l).all():
            l_best = np.nan
            r_best = np.nan
            mae_best = np.nan
        else:
            l_best = np.nanargmax(corr_l)
            r_best = corr_l[l_best]
            mae_best = mae_l[l_best]
        
        corrs.append(r_best)
        centers.append(start + window_size // 2)
        maes.append(mae_best)
        bestlags.append(l_best - lag)
		
    return np.array(corrs), np.array(centers), np.array(maes), np.array(bestlags)


def fill_target_with_neighbors(target, event_list, timestamp, min_dur=100):
	Nsamples = event_list[0].shape[0]
	
	iDisp = 1
	startMarker = 0

	# Remove short fixations
	change = False
	iDisp = 1
	pStart = np.zeros(1)
	pEnd = np.zeros(1)
	while iDisp < (Nsamples-1):
		# print("  \rremove sf", iDisp, end="");
		# Point where sacc ends and fix starts
		if not target[iDisp] and target[iDisp+1]:
			# print("  \rremove sf", iDisp, end="");
			startMarker = iDisp
			pStart[:] = timestamp[iDisp]
			
			# Loop ahead until we find the start of a new saccade
			while iDisp < (Nsamples-1) and target[iDisp+1]:
				iDisp+=1

			pEnd[:] = timestamp[iDisp]

			if pEnd-pStart < min_dur:
				neighbor_left = startMarker
				neighbor_right = iDisp + 1
				if neighbor_left > 0 and neighbor_right < Nsamples:
					for i, event in enumerate(event_list):
						if event[neighbor_left] == True and event[neighbor_right] == True: 
							event_list[i][startMarker+1:iDisp+1] = True
							target[startMarker+1: iDisp+1] = False
							change = True
		else:
			iDisp += 1

		# Reset until no small fixations are found
		if iDisp==(Nsamples-1) and change:
			iDisp = 1
			change = False

	return event_list, target


def extract_fix_sac_sp(data, thd_fix=30, thd_sac=100, max_dur_blink=200, min_dur_event=100, smooth_method="g", filter_sigma=4, filter_win=9, filter_poly=2, via='loc'):
	'''
	input data: [gaze_x, gaze_y, gaze_z, time, conf_val, eye_yaw, leader_yaw, follower_yaw]
	or [gaze_x, gaze_y, gaze_z, time, conf_val, eye_yaw, loc_x, loc_y, leader_x, leader_y, follower_x, follower_y]
	output_data: 
	 
	'''
	
	data[..., 3] = data[..., 3] * 1000 # turn s into ms
	
	mask_cv = data[:, 4] == 1
	data_filtered = data[mask_cv]  # remove low confidence samples
	keep, velocity = get_yaw_velocity(data_filtered[..., :4], return_keep=True)
	mask_keep = np.zeros_like(mask_cv)
	mask_keep[mask_cv] = keep
	velocity_keep = velocity[keep]
	data_keep = data[mask_keep]
	
	if smooth_method == "g":
		velocity_smoothed = gaussian_filter1d(velocity_keep, sigma=filter_sigma)
	elif smooth_method == "s":
		velocity_smoothed = savgol_filter(velocity_keep, filter_win, filter_poly)
	else:
		raise ValueError("Unknown smoothing method: {}".format(smooth_method))
	
	label_list_fix = parser_fix_VT(data_keep[:, 3], velocity_smoothed, threshold=thd_fix, minFixationTime=min_dur_event).astype(bool)
	label_list_sac = parser_sac_VT(data_keep[:, 3], velocity_smoothed, threshold=thd_sac, minSacTime=min_dur_event).astype(bool)
	
	ind_fix = np.zeros_like(mask_cv, dtype=bool)
	ind_fix[mask_keep] = label_list_fix
	ind_sac = np.zeros_like(mask_cv, dtype=bool)
	ind_sac[mask_keep] = label_list_sac
	
	if via == 'angle':
		sp_leader, sp_follower = get_smooth_pursuit_via_angle(
			data_keep[..., 3], data_keep[:, -3], data_keep[:, -2], data_keep[:, -1], min_dur=min_dur_event)
	elif via == 'loc':
		sp_leader, sp_follower = get_smooth_pursuit_via_loc(
            data_keep[..., 3], data_keep[:, -7], data_keep[:, -6], data_keep[:, -5], data_keep[:, -4],
            data_keep[:, -3], data_keep[:, -2], data_keep[:, -1], min_dur=min_dur_event)
	else:
		raise NotImplementedError("Unknown SP detection method: {}".format(via))
	
	label_list_sp_leader = np.logical_and(np.logical_and(~label_list_fix, ~label_list_sac), sp_leader)
	label_list_sp_follower = np.logical_and(np.logical_and(~label_list_fix, ~label_list_sac), sp_follower)
	ind_sp_leader = np.zeros_like(mask_cv, dtype=bool)
	ind_sp_leader[mask_keep] = label_list_sp_leader
	ind_sp_follower = np.zeros_like(mask_cv, dtype=bool)
	ind_sp_follower[mask_keep] = label_list_sp_follower
	ind_att = np.logical_or(np.logical_or(ind_sp_leader, ind_sp_follower), ind_fix)
	ind_noise = np.logical_and(np.logical_not(np.logical_or(ind_att, ind_sac)), mask_cv)

    # fill noise with neighbors
	event_list, ind_noise = fill_target_with_neighbors(
		ind_noise, [ind_fix, ind_sac, ind_sp_leader, ind_sp_follower], data[:, 3], min_dur=min_dur_event)
	ind_fix, ind_sac, ind_sp_leader, ind_sp_follower = event_list
	# fill blink with neighbors
	event_list, mask_cv_updated = fill_target_with_neighbors(
		~mask_cv, [ind_fix, ind_sac, ind_sp_leader, ind_sp_follower, ind_noise], data[:, 3], min_dur=max_dur_blink)
	ind_fix, ind_sac, ind_sp_leader, ind_sp_follower, ind_noise = event_list

	ind_att = np.logical_or(np.logical_or(ind_sp_leader, ind_sp_follower), ind_fix)

	velocity_recover = np.full_like(mask_cv, np.nan, dtype=float)
	velocity_recover[mask_keep] = velocity_smoothed
	
	return ind_fix, ind_sac, ind_sp_leader, ind_sp_follower, ind_att, ind_noise, velocity_recover, mask_keep


def assign_fixation_to_object(fix, hit):
	fix_assign = np.zeros_like(hit)
	
	Nsamples = fix.shape[0]
	iDisp = 1
	startMarker = 0

	# Remove short fixations
	iDisp = 1
	# pStart = np.zeros(1)
	# pEnd = np.zeros(1)
	while iDisp < (Nsamples-1):
		# print("  \rremove sf", iDisp, end="");
		# Point where sacc ends and fix starts
		if not fix[iDisp] and fix[iDisp+1]:
			# print("  \rremove sf", iDisp, end="");
			startMarker = iDisp
			
			# Loop ahead until we find the start of a new saccade
			while iDisp < (Nsamples-1) and fix[iDisp+1]:
				iDisp+=1

			hit_list = hit[startMarker+1:iDisp+1]
			hit_label = Counter(hit_list).most_common(1)[0][0]
			
			fix_assign[startMarker+1:iDisp+1] = hit_label
		else:
			iDisp += 1

	return fix_assign


def extract_fix_sac_sp_df(df, thd_fix=30, thd_sac=100, max_dur_blink=400, min_dur_event=100, smooth_method="g", filter_sigma=4, filter_win=9, filter_poly=2):
	'''
	input data: [gaze_x, gaze_y, gaze_z, time, conf_val, eye_yaw, loc_x, loc_y, leader_x, leader_y, follower_x, follower_y]
	output_data: 
	 
	'''
	data = df.loc[..., [
            'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 
            'TimeElapsedTrial', 'ConfidenceValue',
			'hit_leader', 'hit_follower', 'hit_goal', 'hit_others', 
            'eye_yaw', 'loc_x', 'loc_y', 'leader_x', 'leader_y', 'follower_x', 'follower_y']].values
	data[..., 3] = data[..., 3] * 1000 # turn s into ms
	mask_cv = data[:, 4] == 1
	data_filtered = data[mask_cv]  # remove low confidence samples
	keep, velocity = get_yaw_velocity(data_filtered[..., :4], return_keep=True)
	mask_keep = np.zeros_like(mask_cv)
	mask_keep[mask_cv] = keep
	velocity_keep = velocity[keep]
	data_keep = data[mask_keep]
	
	if smooth_method == "g":
		velocity_smoothed = gaussian_filter1d(velocity_keep, sigma=filter_sigma)
	elif smooth_method == "s":
		velocity_smoothed = savgol_filter(velocity_keep, filter_win, filter_poly)
	else:
		raise ValueError("Unknown smoothing method: {}".format(smooth_method))
	
	label_list_fix = parser_fix_VT(data_keep[:, 3], velocity_smoothed, threshold=thd_fix, minFixationTime=min_dur_event).astype(bool)
	label_list_sac = parser_sac_VT(data_keep[:, 3], velocity_smoothed, threshold=thd_sac, minSacTime=min_dur_event).astype(bool)
	
	ind_fix = np.zeros_like(mask_cv, dtype=bool)
	ind_fix[mask_keep] = label_list_fix
	ind_sac = np.zeros_like(mask_cv, dtype=bool)
	ind_sac[mask_keep] = label_list_sac
	
	sp_leader, sp_follower, bestlags_leader, bestlags_follower, corrs_leader, corrs_follower = get_smooth_pursuit_via_loc(
            data_keep[..., 3], data_keep[:, -7], data_keep[:, -6], data_keep[:, -5], data_keep[:, -4],
            data_keep[:, -3], data_keep[:, -2], data_keep[:, -1], min_dur=min_dur_event)
		
	label_list_sp_leader = np.logical_and(np.logical_and(~label_list_fix, ~label_list_sac), sp_leader)
	label_list_sp_follower = np.logical_and(np.logical_and(~label_list_fix, ~label_list_sac), sp_follower)
	ind_sp_leader = np.zeros_like(mask_cv, dtype=bool)
	ind_sp_leader[mask_keep] = label_list_sp_leader
	ind_sp_follower = np.zeros_like(mask_cv, dtype=bool)
	ind_sp_follower[mask_keep] = label_list_sp_follower
	ind_att = np.logical_or(np.logical_or(ind_sp_leader, ind_sp_follower), ind_fix)
	ind_noise = np.logical_and(np.logical_not(np.logical_or(ind_att, ind_sac)), mask_cv)
	
	# assign fixation to object
	hit = df.hit_leader.values*1 + df.hit_follower.values*2 + df.hit_goal.values*3 + df.hit_others.values*4
	fix_assign = assign_fixation_to_object(ind_fix, hit)
	ind_fix_leader = fix_assign == 1
	ind_fix_follower = fix_assign == 2
	ind_fix_goal = fix_assign == 3
	ind_fix_others = fix_assign == 4

    # fill noise with neighbors
	event_list, ind_noise = fill_target_with_neighbors(
		ind_noise, [ind_fix_leader, ind_fix_follower, ind_fix_goal, ind_fix_others, 
			  ind_sac, ind_sp_leader, ind_sp_follower], data[:, 3], min_dur=min_dur_event)
	ind_fix_leader, ind_fix_follower, ind_fix_goal, ind_fix_others, ind_sac, ind_sp_leader, ind_sp_follower = event_list
	# fill blink with neighbors
	event_list, mask_cv_updated = fill_target_with_neighbors(
		~mask_cv, [ind_fix_leader, ind_fix_follower, ind_fix_goal, ind_fix_others, 
			 ind_sac, ind_sp_leader, ind_sp_follower, ind_noise], data[:, 3], min_dur=max_dur_blink)
	ind_fix_leader, ind_fix_follower, ind_fix_goal, ind_fix_others, ind_sac, ind_sp_leader, ind_sp_follower, ind_noise = event_list

	ind_fix = ind_fix_leader | ind_fix_follower | ind_fix_goal | ind_fix_others
	ind_sp = ind_sp_leader | ind_sp_follower
	ind_att = ind_fix | ind_sp

	# ind_att_leader = ind_fix_leader | ind_sp_leader
	# ind_att_follower = ind_fix_follower | ind_sp_follower
	ind_att_pod = ind_fix_leader | ind_fix_follower | ind_sp_leader | ind_sp_follower
	ind_att_nonpod = ind_fix_goal | ind_fix_others

	velocity_recover = np.empty(mask_cv.shape)
	velocity_recover[:] = np.nan
	velocity_recover[mask_keep] = velocity_smoothed

	bestlags_leader_recover = np.empty(mask_cv.shape)
	bestlags_leader_recover[:] = np.nan
	bestlags_leader_recover[mask_keep] = bestlags_leader

	bestlags_follower_recover = np.empty(mask_cv.shape)
	bestlags_follower_recover[:] = np.nan
	bestlags_follower_recover[mask_keep] = bestlags_follower

	corrs_leader_recover = np.empty(mask_cv.shape)
	corrs_leader_recover[:] = np.nan
	corrs_leader_recover[mask_keep] = corrs_leader

	corrs_follower_recover = np.empty(mask_cv.shape)
	corrs_follower_recover[:] = np.nan
	corrs_follower_recover[mask_keep] = corrs_follower
	
	df.loc[:, ['ind_fix']] = ind_fix.astype(int)
	df.loc[:, ['ind_sac']] = ind_sac.astype(int)
	df.loc[:, ['ind_sp']] = ind_sp.astype(int)
	df.loc[:, ['ind_att']] = ind_att.astype(int)
	df.loc[:, ['ind_noise']] = ind_noise.astype(int)
	
	df.loc[:, ['ind_fix_leader']] = ind_fix_leader.astype(int)
	df.loc[:, ['ind_fix_follower']] = ind_fix_follower.astype(int)
	df.loc[:, ['ind_fix_goal']] = ind_fix_goal.astype(int)
	df.loc[:, ['ind_fix_others']] = ind_fix_others.astype(int)
	df.loc[:, ['ind_sp_leader']] = ind_sp_leader.astype(int)
	df.loc[:, ['ind_sp_follower']] = ind_sp_follower.astype(int)

	# df.loc[:, ['ind_att_leader']] = ind_att_leader.astype(int)
	# df.loc[:, ['ind_att_follower']] = ind_att_follower.astype(int)
	df.loc[:, ['ind_att_pod']] = ind_att_pod.astype(int)
	df.loc[:, ['ind_att_nonpod']] = ind_att_nonpod.astype(int)
	
	df.loc[:, ['mask_keep']] = mask_keep.astype(int)
	df.loc[:, ['eye_yaw_vel']] = velocity_recover
	df.loc[:, ['bestlag_sp_leader']] = bestlags_leader_recover
	df.loc[:, ['bestlag_sp_follower']] = bestlags_follower_recover
	df.loc[:, ['corr_sp_leader']] = corrs_leader_recover
	df.loc[:, ['corr_sp_follower']] = corrs_follower_recover
	
	return df


def get_rid_of_singe_value(hit_masked, timestamp):

	# fill those single element with same neighbors
	right = hit_masked[2:]
	mid = hit_masked[1:-1]
	left = hit_masked[:-2]
	diff = right - left
	mask = (diff == 0) & (mid != right)
	hit_masked[1:-1][mask] = right[mask]
	
	# fill those single element with different neighbors
	# for the first item
	mask_first = (hit_masked[1] == hit_masked[2]) & (hit_masked[0] != hit_masked[1])
	if mask_first: hit_masked[0] = hit_masked[1]
	# for the last item
	mask_last = (hit_masked[-2] == hit_masked[-3]) & (hit_masked[-1] != hit_masked[-2])
	if mask_last: hit_masked[-1] = hit_masked[-2]
	# for others
	# fill right when tdiff=1, fill left when tdiff=0
	tdiff_mask = timestamp[2:] - timestamp[1:-1] < timestamp[1:-1] - timestamp[:-2]
	right = hit_masked[2:]
	mid = hit_masked[1:-1]
	left = hit_masked[:-2]
	mask = (right != left) & (mid != right) & (mid != left)
	hit_masked[1:-1][mask & tdiff_mask] = right[mask & tdiff_mask]
	hit_masked[1:-1][mask & ~tdiff_mask] = left[mask & ~tdiff_mask]

	return hit_masked


def extract_att_sac_sp_df_using_hit(df, thd_sac=100, min_dur_event=100, max_dur_blink=400, smooth_method="g", filter_sigma=4, filter_win=9, filter_poly=2):
	'''
	input data: [gaze_x, gaze_y, gaze_z, time, conf_val, eye_yaw, leader_yaw, follower_yaw]
	or [gaze_x, gaze_y, gaze_z, time, conf_val, eye_yaw, loc_x, loc_y, leader_x, leader_y, follower_x, follower_y]
	output_data: 
	 
	'''
	data = df.loc[..., [
            'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 
            'TimeElapsedTrial', 'ConfidenceValue',
			'hit_leader', 'hit_follower', 'hit_goal', 'hit_others', 'eye_yaw']].values
	data[..., 3] = data[..., 3] * 1000 # turn s into ms
	mask_cv = data[:, 4] == 1
	data_filtered = data[mask_cv]  # remove low confidence samples
	keep, velocity = get_yaw_velocity(data_filtered[..., :4], return_keep=True)
	mask_keep = np.zeros_like(mask_cv)
	mask_keep[mask_cv] = keep
	velocity_keep = velocity[keep]
	data_keep = data[mask_keep]
	
	if smooth_method == "g":
		velocity_smoothed = gaussian_filter1d(velocity_keep, sigma=filter_sigma)
	elif smooth_method == "s":
		velocity_smoothed = savgol_filter(velocity_keep, filter_win, filter_poly)
	else:
		raise ValueError("Unknown smoothing method: {}".format(smooth_method))
	
	label_list_sac = parser_sac_VT(data_keep[:, 3], velocity_smoothed, threshold=thd_sac, minSacTime=min_dur_event).astype(bool)
	ind_sac = np.zeros_like(mask_cv, dtype=bool)
	ind_sac[mask_keep] = label_list_sac
	
	# assign fixation to object
	ind_att_noise = mask_keep & (~ind_sac)
	hit = df.hit_leader.values*1 + df.hit_follower.values*2 + df.hit_goal.values*3 + df.hit_others.values*4
	hit_att_noise = hit[ind_att_noise]

	# get rid of a singe element
	hit_att_noise = get_rid_of_singe_value(hit_att_noise, data[ind_att_noise, 3])
	hit_att_noise = get_rid_of_singe_value(hit_att_noise, data[ind_att_noise, 3]) # repeat to make sure
	ind_att_leader = np.zeros_like(ind_att_noise, dtype=bool)
	ind_att_follower = np.zeros_like(ind_att_noise, dtype=bool)
	ind_att_goal = np.zeros_like(ind_att_noise, dtype=bool)
	ind_att_others = np.zeros_like(ind_att_noise, dtype=bool)
	ind_att_leader[ind_att_noise] = hit_att_noise == 1
	ind_att_follower[ind_att_noise] = hit_att_noise == 2
	ind_att_goal[ind_att_noise] = hit_att_noise == 3
	ind_att_others[ind_att_noise] = hit_att_noise == 4
	ind_noise = ind_att_noise & ~(ind_att_leader | ind_att_follower | ind_att_goal | ind_att_others)
	
	# fill blink with neighbors
	ind_nodata = ~mask_keep
	event_list, ind_nodata = fill_target_with_neighbors(
		ind_nodata, [ind_att_leader, ind_att_follower, ind_att_goal, ind_att_others, ind_sac, ind_noise], 
		data[:, 3], min_dur=max_dur_blink)
	ind_att_leader, ind_att_follower, ind_att_goal, ind_att_others, ind_sac, ind_noise = event_list
	ind_att = ind_att_leader | ind_att_follower | ind_att_goal | ind_att_others

	# mask short sequences as noise
	ind_att_leader, ind_noise = remove_short_markers_plus_affected_seq(ind_att_leader, ind_noise, data[..., 3], min_dur=min_dur_event)
	ind_att_follower, ind_noise = remove_short_markers_plus_affected_seq(ind_att_follower, ind_noise, data[..., 3], min_dur=min_dur_event)
	ind_att_goal, ind_noise = remove_short_markers_plus_affected_seq(ind_att_goal, ind_noise, data[..., 3], min_dur=min_dur_event)
	ind_att_others, ind_noise = remove_short_markers_plus_affected_seq(ind_att_others, ind_noise, data[..., 3], min_dur=min_dur_event)

	# store results
	velocity_recover = np.empty(mask_cv.shape)
	velocity_recover[:] = np.nan
	velocity_recover[mask_keep] = velocity_smoothed
	
	df.loc[:, ['ind_att']] = ind_att.astype(int)
	df.loc[:, ['ind_sac']] = ind_sac.astype(int)
	df.loc[:, ['ind_noise']] = ind_noise.astype(int)
	
	df.loc[:, ['ind_att_leader']] = ind_att_leader.astype(int)
	df.loc[:, ['ind_att_follower']] = ind_att_follower.astype(int)
	df.loc[:, ['ind_att_goal']] = ind_att_goal.astype(int)
	df.loc[:, ['ind_att_others']] = ind_att_others.astype(int)
	df.loc[:, ['ind_att_pod']] = (ind_att_leader | ind_att_follower).astype(int)
	df.loc[:, ['ind_att_nonpod']] = (ind_att_goal | ind_att_others).astype(int)
	
	df.loc[:, ['mask_keep']] = mask_keep.astype(int)
	df.loc[:, ['eye_yaw_vel']] = velocity_recover
	
	return df


if __name__ == '__main__':

    dfs = pd.read_csv('data/dfs_combined.csv')
    dfs['leader_yaw'] = np.rad2deg(np.atan2((dfs['PodLeader_Location_y'] - dfs['Ped_Location_y']).values, (dfs['PodLeader_Location_x'] - dfs['Ped_Location_x']).values))
    dfs['follower_yaw'] = np.rad2deg(np.atan2((dfs['PodFollower_Location_y'] - dfs['Ped_Location_y']).values, (dfs['PodFollower_Location_x'] - dfs['Ped_Location_x']).values))
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
            'HitObject_env': 'hit_env', 'HitObject_goal': 'hit_goal', 'HitObject_neighbor': 'hit_neighbor', 
            'HitObject_pod_leader': 'hit_leader', 'HitObject_pod_follower': 'hit_follower'
            }, inplace=True)
    dfs10 = dfs[(dfs['pid'] == 10) & (dfs['sid'] == 11)]

    via = 'loc'
    thd_fix=30
    thd_sac=100
    max_dur_blink=400
    min_dur_event=100
    smooth_method="g"
    filter_sigma=4
    filter_win=9
    filter_poly=2
	
    mask_cv = dfs10['ConfidenceValue'] == 1

    dfs10 = extract_fix_sac_sp_df(dfs10, 
		thd_fix=thd_fix, thd_sac=thd_sac, max_dur_blink=max_dur_blink, min_dur_event=min_dur_event)
	
    print('done')
	

def main1(dfs10):
	if via == 'angle':
		fix, sac, sp_leader, sp_follower, att, noise, eye_vel, mask_keep = extract_fix_sac_sp(dfs10[[
            'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 
            'TimeElapsedTrial', 'ConfidenceValue',
            'eye_yaw', 'leader_yaw', 'follower_yaw']].values, 
		via=via, thd_fix=thd_fix, thd_sac=thd_sac, max_dur_blink=max_dur_blink, min_dur_event=min_dur_event,)
	elif via == 'loc':
		fix, sac, sp_leader, sp_follower, att, noise, eye_vel, mask_keep = extract_fix_sac_sp(dfs10[[
            'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 
            'TimeElapsedTrial', 'ConfidenceValue',
            'eye_yaw', 'loc_x', 'loc_y', 'leader_x', 'leader_y', 'follower_x', 'follower_y']].values, 
		via=via, thd_fix=thd_fix, thd_sac=thd_sac, max_dur_blink=max_dur_blink, min_dur_event=min_dur_event,)

    # plot 
	fix = dfs10.ind_fix.values.astype(bool)
	fix_leader = dfs10.ind_fix_leader.values.astype(bool)
	fix_follower = dfs10.ind_fix_follower.values.astype(bool)
	fix_goal = dfs10.ind_fix_goal.values.astype(bool)
	fix_others = dfs10.ind_fix_others.values.astype(bool)
	sac = dfs10.ind_sac.values.astype(bool)
	sp_leader = dfs10.ind_sp_leader.values.astype(bool)
	sp_follower = dfs10.ind_sp_follower.values.astype(bool)
	att = dfs10.ind_att.values.astype(bool)
	noise = dfs10.ind_noise.values.astype(bool)
	# corr_leader = dfs10.corr_sp_leader.values
	# corr_follower = dfs10.corr_sp_follower.values

	plt.figure(figsize=(10, 3))
	ax1 = plt.subplot(111)
	mask = dfs10.hit_leader == True
	ax1.scatter(dfs10.TimeElapsedTrial[mask], dfs10.hit_leader[mask], label='leader', s=2)
	mask = dfs10.hit_follower == True
	ax1.scatter(dfs10.TimeElapsedTrial[mask], dfs10.hit_follower[mask], label='follower', s=2)
	mask = dfs10.hit_goal == True
	ax1.scatter(dfs10.TimeElapsedTrial[mask], dfs10.hit_goal[mask], label='goal', s=2)
	mask = dfs10.hit_others == True
	ax1.scatter(dfs10.TimeElapsedTrial[mask], dfs10.hit_others[mask], label='others', s=2)
	nohit = (dfs10[['hit_others', 'hit_goal', 'hit_leader', 'hit_follower']].sum(axis=1) == 0)
	nohit = np.logical_and(nohit, dfs10.mask_keep.values)
	plt.scatter(dfs10.TimeElapsedTrial[nohit], empty[nohit], label='empty', s=2, alpha=0.2, c='k')

	ax1.scatter(dfs10.TimeElapsedTrial[fix], fix[fix]*1.1, label='Fixation', s=2, c='brown')
	ax1.scatter(dfs10.TimeElapsedTrial[fix_leader], fix_leader[fix_leader]*1.07, s=2, c='blue')
	ax1.scatter(dfs10.TimeElapsedTrial[fix_follower], fix_follower[fix_follower]*1.07, s=2, c='orange')
	ax1.scatter(dfs10.TimeElapsedTrial[fix_goal], fix_goal[fix_goal]*1.07, s=2, c='green')
	ax1.scatter(dfs10.TimeElapsedTrial[fix_others], fix_others[fix_others]*1.07, s=2, c='red')
	ax1.scatter(dfs10.TimeElapsedTrial[sac], sac[sac]*1.2, label='Saccades', s=2, c='olive')
	ax1.scatter(dfs10.TimeElapsedTrial[sp_leader], sp_leader[sp_leader]*1.3, label='SP leader', s=2, c='orange')
	ax1.scatter(dfs10.TimeElapsedTrial[sp_follower], sp_follower[sp_follower]*1.3, label='SP follower', s=2, c='salmon')
	# ax1.scatter(dfs10.TimeElapsedTrial, corr_leader*0.9+1, s=2, c='orange')
	# ax1.scatter(dfs10.TimeElapsedTrial, corr_follower*0.9+1, s=2, c='salmon')
	ax1.scatter(dfs10.TimeElapsedTrial[att], att[att]*1.5, label='Attention', s=2, c='red')
	ax1.scatter(dfs10.TimeElapsedTrial[noise], noise[noise]*1.6, label='Noise', s=2, c='gray')
	empty = ~(sac | att | noise)
	plt.scatter(dfs10.TimeElapsedTrial[empty], empty[empty]*1.7, label='empty', s=2, c='k')

	ax1.legend(loc='upper right')
	ax1.set_ylim(0, 2)

	ax2 = ax1.twinx()
	ax2.plot(dfs10.TimeElapsedTrial, dfs10.head_yaw, label='head', color='black', linewidth=1)
	ax2.plot(dfs10.TimeElapsedTrial, dfs10.eye_yaw, label='eye', color='blue', linewidth=1)
	ax2.plot(dfs10.TimeElapsedTrial, dfs10.leader_yaw, label='leader_yaw', color='orange', linewidth=1)
	ax2.plot(dfs10.TimeElapsedTrial, dfs10.follower_yaw, label='follower_yaw', color='salmon', linewidth=1)
	plt.savefig('fig_eye_event_detection_example.png', dpi=500)

    # plot 
	plt.figure(figsize=(6, 4))
	plt.hist(eye_vel[fix], bins=20, range=(np.nanmin(eye_vel), np.nanmax(eye_vel)), label='fix', alpha=0.4, color='brown')
	plt.hist(eye_vel[sp_leader], bins=20, range=(np.nanmin(eye_vel), np.nanmax(eye_vel)), label='sp_leader', alpha=0.4, color='orange')
	plt.hist(eye_vel[sp_follower], bins=20, range=(np.nanmin(eye_vel), np.nanmax(eye_vel)), label='sp_follower', alpha=0.4, color='salmon')
	plt.hist(eye_vel[sac], bins=20, range=(np.nanmin(eye_vel), np.nanmax(eye_vel)), label='sac', alpha=0.4, color='olive')
	plt.legend()
	plt.savefig('fig_eye_event_detection_hist.png', dpi=500)