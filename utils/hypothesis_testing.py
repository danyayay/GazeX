"""Hypothesis testing for trajectory prediction model comparison.

Edit MODELS and COMPARISONS below, then run:
    python utils/hypothesis_testing.py --metric both --n_bootstrap 10000

Per-trial errors are right-skewed, so we use:
  - Paired bootstrap test (primary)
  - Wilcoxon signed-rank test (sensitivity, non-parametric)
  - Permutation test on mean difference (additional power)

Negative Diff(A-B) = model A has lower error = A is better.
"""

import argparse
import os
import sys
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import permutation_test as scipy_permutation_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metric import compute_rmse_over_sample, compute_rmse_over_sample_minK


# ── User configuration ─────────────────────────────────────────────────────────
# Each value is either a single NPZ path or a list of paths (multiple runs).
# Multiple runs: per-trial errors are averaged across runs before testing.
MODELS = {
    # "baseline": "logs/indiv_time_o40_p40_s4/sqlite_final/20250918_093737_LSTMse/test_best.npz",
    "baseline": "logs/indiv_time_o40_p40_s4/sqlite_final/20250918_093737_LSTMse/test_last_1_True.npz",
    "eye_vislet": "logs/indiv_time_o40_p40_s4/20260416_105242_MultiModalLSTM/20260417_100136_MultiModalLSTM/test_last_1_True.npz",
    # "eye_vislet": "logs/indiv_time_o40_p40_s4/20260416_105242_MultiModalLSTM/test_best_20_False.npz",
}

# Each tuple (A, B): tests H1 = model A has lower ADE/FDE than model B.
COMPARISONS = [
    ("eye_vislet", "baseline"),
]
# ──────────────────────────────────────────────────────────────────────────────


def load_model_predictions(path: str, labels_ref: np.ndarray | None = None):
    """Load y_preds and y_labels from an NPZ file.

    Args:
        path: Path to .npz file with 'y_preds' and optionally 'y_labels' (or 'y').
        labels_ref: Reference labels to validate against (same test set check).

    Returns:
        y_preds: (N,T,2) deterministic or (K,N,T,2) stochastic, float32.
        y_labels: (N,T,2), float32.
    """
    data = np.load(path, allow_pickle=True)
    y_preds = data['y_preds'].astype(np.float32)

    if 'y_labels' in data:
        y_labels = data['y_labels'].astype(np.float32)
    elif 'y' in data:
        y_labels = data['y'].astype(np.float32)[..., 1:3]  # strip PSID col
    else:
        if labels_ref is None:
            raise ValueError(f"{path} has no 'y_labels' or 'y' key and no --labels file was given")
        y_labels = labels_ref

    if labels_ref is not None:
        assert y_labels.shape == labels_ref.shape and np.allclose(y_labels, labels_ref, atol=1e-3), \
            f"y_labels in {path} do not match reference labels (shape {y_labels.shape} vs {labels_ref.shape})"

    return y_preds, y_labels


def _per_trial_ade_single(y_preds: np.ndarray, y_labels: np.ndarray) -> np.ndarray:
    """Per-trial ADE for one NPZ. Returns shape (N,)."""
    if y_preds.ndim == 3:
        return compute_rmse_over_sample(y_preds, y_labels)
    elif y_preds.ndim == 4:
        ade, _ = compute_rmse_over_sample_minK(y_preds, y_labels)
        return ade
    else:
        raise ValueError(f"Unexpected y_preds shape: {y_preds.shape}")


def _per_trial_fde_single(y_preds: np.ndarray, y_labels: np.ndarray) -> np.ndarray:
    """Per-trial FDE for one NPZ. Returns shape (N,).

    For stochastic models uses the same min-K sample selection as ADE
    (sample with lowest mean ADE across the trajectory).
    """
    if y_preds.ndim == 3:
        return np.sqrt(np.sum(
            (y_preds[:, -1, :2] - y_labels[:, -1, :2]) ** 2, axis=-1
        ))
    elif y_preds.ndim == 4:
        labels_exp = np.expand_dims(y_labels, 0)  # (1,N,T,2)
        ade_all = np.mean(
            np.sqrt(np.sum((y_preds[..., :2] - labels_exp[..., :2]) ** 2 + 1e-6, axis=-1)),
            axis=-1
        )  # (K,N)
        idx = np.argmin(ade_all, axis=0)  # (N,) best sample per trial
        best_final = y_preds[idx, np.arange(y_preds.shape[1]), -1, :2]  # (N,2)
        return np.sqrt(np.sum((best_final - y_labels[:, -1, :2]) ** 2, axis=-1))
    else:
        raise ValueError(f"Unexpected y_preds shape: {y_preds.shape}")


def compute_per_trial_errors(
    paths_or_path,
    labels_ref: np.ndarray | None = None
):
    """Compute per-trial ADE and FDE for a model (one or multiple runs).

    If multiple runs are given, per-trial errors are averaged across runs
    before returning.

    Returns:
        ade: (N,) mean per-trial ADE
        fde: (N,) mean per-trial FDE
        labels_ref: the resolved labels array (from first run if not supplied)
    """
    paths = [paths_or_path] if isinstance(paths_or_path, str) else list(paths_or_path)
    ade_runs, fde_runs = [], []
    labels_ref_local = labels_ref

    for path in paths:
        y_preds, y_labels = load_model_predictions(path, labels_ref_local)
        if labels_ref_local is None:
            labels_ref_local = y_labels
        ade_runs.append(_per_trial_ade_single(y_preds, y_labels))
        fde_runs.append(_per_trial_fde_single(y_preds, y_labels))

    ade = np.mean(np.stack(ade_runs, axis=0), axis=0)  # (N,)
    fde = np.mean(np.stack(fde_runs, axis=0), axis=0)  # (N,)
    return ade, fde, labels_ref_local


def paired_bootstrap_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test on mean difference of per-trial errors.

    H1: mean(errors_a) < mean(errors_b)  (model A is better)
    One-tailed p-value via shift-to-null method.

    Returns dict with: observed_diff, ci_low, ci_high, p_value, n_bootstrap.
    """
    assert len(errors_a) == len(errors_b), \
        f"Trial count mismatch: {len(errors_a)} vs {len(errors_b)}"
    N = len(errors_a)
    rng = np.random.default_rng(seed)

    diff = errors_a - errors_b            # (N,)
    observed_diff = diff.mean()

    indices = rng.integers(0, N, size=(n_bootstrap, N))
    boot_diffs = diff[indices].mean(axis=1)   # (B,)

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    # Shift bootstrap to be centred at 0 (null boundary), count >= 0
    p_value = float(np.mean(boot_diffs - observed_diff >= 0))

    return {
        'observed_diff': float(observed_diff),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': p_value,
        'n_bootstrap': n_bootstrap,
    }


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alternative: str = 'less',
) -> dict:
    """Wilcoxon signed-rank test on paired per-trial errors.

    H1 (alternative='less'): median(errors_a - errors_b) < 0 (A is better).

    Returns dict with: statistic, p_value.
    """
    try:
        stat, p = wilcoxon(
            errors_a - errors_b,
            alternative=alternative,
            zero_method='wilcox',
        )
    except ValueError:
        print("  [Warning] Wilcoxon: all differences are zero (identical predictions). p=1.0")
        stat, p = np.nan, 1.0
    return {'statistic': float(stat) if not np.isnan(stat) else np.nan, 'p_value': float(p)}


def permutation_test_mean(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test on difference of means.

    H1: mean(errors_a) < mean(errors_b)  (A is better).

    Returns dict with: p_value.
    """
    def statistic(a, b):
        return np.mean(a) - np.mean(b)

    result = scipy_permutation_test(
        (errors_a, errors_b),
        statistic,
        permutation_type='samples',
        n_resamples=n_permutations,
        alternative='less',
        random_state=seed,
    )
    return {'p_value': float(result.pvalue)}


def run_comparison(
    name_a: str,
    name_b: str,
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Run all three tests for one A-vs-B comparison."""
    boot = paired_bootstrap_test(errors_a, errors_b, n_bootstrap=n_bootstrap, seed=seed)
    wilc = wilcoxon_test(errors_a, errors_b)
    perm = permutation_test_mean(errors_a, errors_b, n_permutations=n_bootstrap, seed=seed)
    return {
        'name_a': name_a,
        'name_b': name_b,
        'n': len(errors_a),
        **{f'boot_{k}': v for k, v in boot.items()},
        'wilcox_p': wilc['p_value'],
        'perm_p': perm['p_value'],
    }


def _sig_marker(p: float, alpha: float) -> str:
    if p < alpha / 50:
        return '***'
    elif p < alpha / 5:
        return '**'
    elif p < alpha:
        return '*'
    return 'n.s.'


def _fmt_p(p: float) -> str:
    if p < 0.0001:
        return '<0.0001'
    return f'{p:.4f}'


def format_results_table(
    model_names: list,
    model_errors_ade: dict,
    model_errors_fde: dict,
    comparison_results: list,
    alpha: float = 0.05,
    metrics: str = 'both',
) -> None:
    """Print summary and comparison tables to stdout."""
    N = len(next(iter(model_errors_ade.values())))

    # ── Per-model summary ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Per-Model Error Summary  (N={N} trials)")
    print(f"{'='*80}")
    if metrics in ('ade', 'both'):
        header = f"{'Model':<20} {'ADE Mean':>10} {'ADE Std':>10} {'ADE Median':>12}"
        if metrics == 'both':
            header += f"  {'FDE Mean':>10} {'FDE Std':>10} {'FDE Median':>12}"
        print(header)
        print('-' * len(header))
        for name in model_names:
            ade = model_errors_ade[name]
            row = f"{name:<20} {ade.mean():>10.4f} {ade.std():>10.4f} {np.median(ade):>12.4f}"
            if metrics == 'both':
                fde = model_errors_fde[name]
                row += f"  {fde.mean():>10.4f} {fde.std():>10.4f} {np.median(fde):>12.4f}"
            print(row)
    elif metrics == 'fde':
        header = f"{'Model':<20} {'FDE Mean':>10} {'FDE Std':>10} {'FDE Median':>12}"
        print(header)
        print('-' * len(header))
        for name in model_names:
            fde = model_errors_fde[name]
            print(f"{name:<20} {fde.mean():>10.4f} {fde.std():>10.4f} {np.median(fde):>12.4f}")

    # ── Comparison tables ──────────────────────────────────────────────────────
    def _print_comparison_table(results, metric_label, error_dict):
        print(f"\n{'='*80}")
        print(f"{metric_label} Comparisons  (H1: A < B, i.e. A is better — negative Diff(A-B) favours A)")
        print(f"{'='*80}")
        col_w = [20, 20, 10, 26, 10, 10, 10, 5]
        hdr = (f"{'Model A':<{col_w[0]}} {'Model B':<{col_w[1]}} "
               f"{'Diff(A-B)':>{col_w[2]}} {'Boot 95% CI':^{col_w[3]}} "
               f"{'Boot p':>{col_w[4]}} {'Wilcox p':>{col_w[5]}} "
               f"{'Perm p':>{col_w[6]}} {'Sig':>{col_w[7]}}")
        print(hdr)
        print('-' * len(hdr))
        for r in results:
            diff_str = f"{r['boot_observed_diff']:+.4f}"
            ci_str = f"[{r['boot_ci_low']:+.4f}, {r['boot_ci_high']:+.4f}]"
            sig = _sig_marker(r['boot_p_value'], alpha)
            print(
                f"{r['name_a']:<{col_w[0]}} {r['name_b']:<{col_w[1]}} "
                f"{diff_str:>{col_w[2]}} {ci_str:^{col_w[3]}} "
                f"{_fmt_p(r['boot_p_value']):>{col_w[4]}} "
                f"{_fmt_p(r['wilcox_p']):>{col_w[5]}} "
                f"{_fmt_p(r['perm_p']):>{col_w[6]}} "
                f"{sig:>{col_w[7]}}"
            )
        n_tests = len(results)
        print(f"\nSignificance (bootstrap p): * p<{alpha:.2f}  ** p<{alpha/5:.3f}  *** p<{alpha/50:.4f}")
        if n_tests > 1:
            print(f"No multiple-comparisons correction applied. "
                  f"Bonferroni threshold for {n_tests} tests: α/{n_tests}={alpha/n_tests:.4f}")

    if metrics in ('ade', 'both'):
        ade_results = []
        for r in comparison_results:
            ea = error_dict_ade[r['name_a']]
            eb = error_dict_ade[r['name_b']]
            ade_results.append(run_comparison(
                r['name_a'], r['name_b'], ea, eb,
                n_bootstrap=r['n_bootstrap'], seed=r['seed']
            ))
        _print_comparison_table(ade_results, 'ADE', model_errors_ade)

    if metrics in ('fde', 'both'):
        fde_results = []
        for r in comparison_results:
            ea = error_dict_fde[r['name_a']]
            eb = error_dict_fde[r['name_b']]
            fde_results.append(run_comparison(
                r['name_a'], r['name_b'], ea, eb,
                n_bootstrap=r['n_bootstrap'], seed=r['seed']
            ))
        _print_comparison_table(fde_results, 'FDE', model_errors_fde)


def main():
    parser = argparse.ArgumentParser(
        description='Hypothesis testing for trajectory prediction model comparison'
    )
    parser.add_argument('--labels', type=str, default='data/indiv_time_o40_p40_s4/test.npz',
                        help='Path to external labels NPZ (key: y_labels or y). '
                             'If omitted, labels are read from each model\'s NPZ.')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap / permutation resamples (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for markers (default: 0.05)')
    parser.add_argument('--two_sided', action='store_true',
                        help='Use two-sided tests instead of one-tailed (H1: A < B)')
    parser.add_argument('--metric', choices=['ade', 'fde', 'both'], default='both',
                        help='Which metric(s) to test (default: both)')
    args = parser.parse_args()

    if not MODELS:
        parser.error("Edit MODELS dict at the top of this script before running.")
    if not COMPARISONS:
        parser.error("Edit COMPARISONS list at the top of this script before running.")

    # Load external labels if provided
    labels_ref = None
    if args.labels:
        lab_data = np.load(args.labels, allow_pickle=True)
        key = 'y_labels' if 'y_labels' in lab_data else 'y'
        labels_ref = lab_data[key].astype(np.float32)
        if labels_ref.ndim == 3 and labels_ref.shape[-1] > 2:
            labels_ref = labels_ref[..., 1:3]  # strip PSID col if present
        print(f"Loaded external labels: {labels_ref.shape} from {args.labels}")

    # Compute per-trial errors for each model
    print("\nLoading model predictions...")
    model_names = list(MODELS.keys())
    model_errors_ade = {}
    model_errors_fde = {}
    labels_resolved = labels_ref

    for name, paths in MODELS.items():
        print(f"  {name}: {paths if isinstance(paths, str) else f'{len(paths)} runs'}")
        ade, fde, labels_resolved = compute_per_trial_errors(paths, labels_resolved)
        model_errors_ade[name] = ade
        model_errors_fde[name] = fde
        print(f"    ADE mean={ade.mean():.4f}  FDE mean={fde.mean():.4f}  N={len(ade)}")

    # Validate all models have same N
    ns = {name: len(v) for name, v in model_errors_ade.items()}
    if len(set(ns.values())) > 1:
        raise AssertionError(
            f"Trial count mismatch across models: {ns}. "
            "All models must be evaluated on the same test set."
        )

    # Build comparison metadata (needed for format_results_table)
    comparison_meta = [
        {'name_a': a, 'name_b': b, 'n_bootstrap': args.n_bootstrap, 'seed': args.seed}
        for a, b in COMPARISONS
    ]
    for cm in comparison_meta:
        assert cm['name_a'] in MODELS, f"'{cm['name_a']}' not in MODELS"
        assert cm['name_b'] in MODELS, f"'{cm['name_b']}' not in MODELS"

    # Run tests and print tables
    # We pass error dicts via closure-like globals for format_results_table
    global error_dict_ade, error_dict_fde
    error_dict_ade = model_errors_ade
    error_dict_fde = model_errors_fde

    format_results_table(
        model_names=model_names,
        model_errors_ade=model_errors_ade,
        model_errors_fde=model_errors_fde,
        comparison_results=comparison_meta,
        alpha=args.alpha,
        metrics=args.metric,
    )


if __name__ == '__main__':
    main()
