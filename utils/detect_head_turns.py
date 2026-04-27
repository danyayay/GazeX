"""
detect_head_turns.py

Head look event detector for VR pedestrian experiment data.

A **look** is one outward head excursion (departure + optional return):
the head moves away from a local baseline, and either returns to within
excursion_threshold degrees of that baseline, or makes a decisive new
outward movement (high-velocity same-direction departure).

Algorithm
---------
1. Compute angular velocity via central differences.
2. Find velocity zero-crossings to get reversal frames.
3. Walk through reversals, tracking peak excursion from a FROZEN onset baseline:
   a. Return to near-baseline (|angle - onset| < threshold) AND excursion
      has qualified (>= threshold) -> emit look, reset.
   b. Return by >50% of peak excursion (return_ratio) -> emit look, reset.
   c. High-velocity outward departure (>= new_look_vel deg/s, same direction
      as current look, excursion already qualified) -> forced split.
   d. Sub-threshold near-baseline reversals do NOT reset the baseline —
      oscillatory exploratory motion accumulates into one look.
   e. Trial end -> emit whatever was accumulated.

This handles:
  - Oscillatory exploratory checks that collectively reach threshold -> one look
  - Large out-and-back arcs -> one look
  - Multiple successive large arcs (separated by valley returns) -> separate looks

Inputs:
    A single-trial DataFrame (already subset by pid + sid). Required columns:
        TimeElapsedTrial           – float, seconds, monotonically increasing
        head_vel_relative_smoothed – float, degrees, head yaw rel. to body
        body_rotation_mask         – bool, True when body direction unreliable

Outputs:
    pd.DataFrame with one row per detected look. Columns:
        pid             – participant ID
        sid             – session/trial ID
        start_time      – seconds (trial-relative)
        end_time        – seconds (trial-relative)
        duration        – seconds
        baseline_angle  – degrees, head angle at look onset (frozen reference)
        peak_angle      – degrees, maximum angular displacement from baseline
        excursion       – abs(peak_angle - baseline_angle), degrees
        direction       – "left" | "right"
        complete        – bool, True if head returned to baseline before trial end
        body_unreliable – bool, True if >50% of event frames have unreliable
                          body direction estimate

Usage:
    from utils.detect_head_turns import detect_head_turns

    all_events = []
    for _, df_trial in df_all.groupby(["pid", "sid"]):
        events = detect_head_turns(df_trial)
        if not events.empty:
            all_events.append(events)

    df_events = pd.concat(all_events, ignore_index=True)
    turns_per_trial = df_events.groupby(["pid", "sid"]).size().reset_index(name="n_looks")
"""

from __future__ import annotations

import numpy as np
import pandas as pd


COLUMNS = [
    "pid", "sid",
    "start_time", "end_time", "duration",
    "baseline_angle", "peak_angle", "excursion",
    "direction", "complete", "body_unreliable",
]


def detect_head_turns(
    df_trial: pd.DataFrame,
    *,
    excursion_threshold: float = 10.0,
    return_ratio: float = 0.5,
    new_look_vel: float = 3.0,
    min_dur_s: float = 0.15,
    body_mask_max_frac: float = 0.5,
    sampling_rate: float = 20.0,
) -> pd.DataFrame:
    """
    Detect discrete head look events within a single trial.

    Parameters
    ----------
    df_trial : pd.DataFrame
        Single-trial slice (one pid+sid).
    excursion_threshold : float
        Minimum peak excursion (degrees) from onset baseline to qualify as
        a look. Also used as the return threshold. Default 10 deg.
    return_ratio : float
        A look also ends when the head returns more than this fraction of
        peak excursion back toward baseline. Default 0.5 (50%).
    new_look_vel : float
        If the outgoing velocity after a reversal exceeds this (deg/s) in
        the same direction as the current look, AND the current look has
        already qualified, force a split. Default 3.0 deg/s.
    min_dur_s : float
        Minimum look duration in seconds. Default 0.15 s.
    body_mask_max_frac : float
        Fraction of event frames where body_rotation_mask is True above
        which the event is flagged body_unreliable. Default 0.5.
    sampling_rate : float
        Nominal sampling rate in Hz. Default 20.

    Returns
    -------
    pd.DataFrame
        One row per detected look. Empty DataFrame (correct columns) if none.
    """
    empty = pd.DataFrame(columns=COLUMNS)

    df = df_trial.reset_index(drop=True)
    n = len(df)
    if n < 3:
        return empty

    pid = df["pid"].iloc[0]
    sid = df["sid"].iloc[0]

    time = df["TimeElapsedTrial"].to_numpy(dtype=float)
    head = df["head_vel_relative_smoothed"].to_numpy(dtype=float)
    body_mask = df["body_rotation_mask"].to_numpy(dtype=bool)

    # Angular velocity via central differences (deg/s)
    vel = np.empty(n, dtype=float)
    vel[1:-1] = (head[2:] - head[:-2]) / 2.0 * sampling_rate
    vel[0]    = (head[1]  - head[0])          * sampling_rate
    vel[-1]   = (head[-1] - head[-2])          * sampling_rate

    # Velocity zero-crossings = direction reversals
    reversal_frames = [0]
    for i in range(1, n):
        if vel[i - 1] * vel[i] < 0:
            reversal_frames.append(i)
    reversal_frames.append(n)

    events = []
    look_start = 0
    onset_baseline = head[0]
    peak_angle = head[0]

    def _emit(end_frame: int, complete: bool) -> None:
        seg_head = head[look_start:end_frame + 1]
        seg_time = time[look_start:end_frame + 1]
        seg_body = body_mask[look_start:end_frame + 1]
        excursions = seg_head - onset_baseline
        peak_local = int(np.argmax(np.abs(excursions)))
        excursion = float(abs(excursions[peak_local]))
        direction = "left" if excursions[peak_local] < 0 else "right"
        duration = float(seg_time[-1] - seg_time[0])
        if excursion >= excursion_threshold and duration >= min_dur_s:
            body_unreliable = bool(len(seg_body) > 0 and seg_body.mean() > body_mask_max_frac)
            events.append({
                "pid":             pid,
                "sid":             sid,
                "start_time":      float(seg_time[0]),
                "end_time":        float(seg_time[-1]),
                "duration":        duration,
                "baseline_angle":  float(onset_baseline),
                "peak_angle":      float(seg_head[peak_local]),
                "excursion":       excursion,
                "direction":       direction,
                "complete":        complete,
                "body_unreliable": body_unreliable,
            })

    for seg_end in reversal_frames[1:]:
        seg_end_frame = min(seg_end, n - 1)
        current_angle = head[seg_end_frame]

        # Update running peak
        if abs(current_angle - onset_baseline) > abs(peak_angle - onset_baseline):
            peak_angle = current_angle
        peak_excursion = abs(peak_angle - onset_baseline)

        near_baseline = abs(current_angle - onset_baseline) < excursion_threshold
        returned_from_peak = (
            abs(current_angle - peak_angle) / peak_excursion > return_ratio
            if peak_excursion > 0 else False
        )

        # Forced split: high-velocity same-direction outgoing stroke
        v_out = vel[seg_end_frame] if seg_end_frame < n else 0.0
        look_sign = np.sign(peak_angle - onset_baseline) if peak_excursion > 0 else 0
        same_dir = (look_sign != 0) and (np.sign(v_out) == look_sign)
        forced_split = (
            same_dir
            and abs(v_out) >= new_look_vel
            and peak_excursion >= excursion_threshold
            and not near_baseline
            and not returned_from_peak
        )

        if returned_from_peak or forced_split or seg_end == n:
            complete = returned_from_peak and seg_end < n
            _emit(seg_end_frame, complete)
            look_start = seg_end_frame
            onset_baseline = current_angle
            peak_angle = current_angle

        elif near_baseline and peak_excursion >= excursion_threshold:
            # Qualified look returned to near-baseline
            _emit(seg_end_frame, complete=True)
            look_start = seg_end_frame
            onset_baseline = current_angle
            peak_angle = current_angle
        # else: sub-threshold near-baseline reversal — keep frozen baseline,
        # keep accumulating (oscillatory phase belongs to the same look)

    if not events:
        return empty

    return pd.DataFrame(events, columns=COLUMNS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect head turns in a trial CSV")
    parser.add_argument("--csv_path", default="data/dfs_combined.csv", type=str)
    parser.add_argument("--excursion_threshold", default=10.0, type=float,
                        help="Minimum peak prominence in degrees")
    args = parser.parse_args()

    df_all = pd.read_csv(args.csv_path)
    all_events = []
    for _, df_trial in df_all.groupby(["pid", "sid"]):
        evs = detect_head_turns(df_trial, excursion_threshold=args.excursion_threshold)
        if not evs.empty:
            all_events.append(evs)
    df = pd.concat(all_events, ignore_index=True)
    df.to_csv(f"head_turn_events__{args.excursion_threshold}.csv", index=False)
