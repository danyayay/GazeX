"""Hyperparameter optimization for trajectory prediction models.

Supported models: multimodallstm
Example: python tune.py --config_filename data/config/multimodallstm.yaml --n_trials 100
"""

import os
import math
import shutil
import logging
import argparse
from typing import Any, Tuple, Callable

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from model.supervisor import Supervisor
from utils.util import seed_everything, setattr_seq_len, setattr_input_dim, setattr_output_dim, setattr_aux_info, save_config_file, get_device
from utils.hyperparameter_space import get_hyperparameter_space
from utils.config import load_runtime_args

logger = logging.getLogger(__name__)


def _remove_trial_logs(log_dir: str) -> None:
    """Remove trial logs if they exist."""
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        logger.debug(f"Removed trial logs: {log_dir}")


def _create_objective_function(
    base_args: argparse.Namespace,
    device: torch.device,
    callback: Any,
    hyperparameter_space: Any,
) -> Callable:
    """Create objective function for Optuna optimization."""
    best_state = {"value": float('inf')}

    def objective(trial: optuna.Trial) -> float:
        trial_params = hyperparameter_space.suggest_hyperparameters(trial, base_args)
        args = argparse.Namespace(**vars(base_args))
        hyperparameter_space.apply_hyperparameters(trial_params, args)
        args = setattr_aux_info(args)

        from utils.init_model import init_model

        def _make_supervisor():
            model = init_model(args, device)
            return Supervisor(
                model=model,
                callback=callback,
                device=device,
                ckpt_path=args.ckpt_path,
                args=args,
            )

        supervisor = _make_supervisor()
        for attempt in range(2):
            try:
                # On retry, pass trial=None to avoid re-reporting already-reported
                # Optuna pruning steps, which raises "step N is already reported".
                supervisor.train(n_epochs=args.n_epochs, trial=trial if attempt == 0 else None)
                break
            except optuna.exceptions.TrialPruned:
                _remove_trial_logs(supervisor.log_dir)
                raise
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Trial {trial.number} training failed, retrying: {e}")
                    _remove_trial_logs(supervisor.log_dir)
                    supervisor = _make_supervisor()
                    continue
                logger.error(f"Trial {trial.number} failed after retry: {e}")
                _remove_trial_logs(supervisor.log_dir)
                return float('nan')

        min_val_loss = supervisor.min_val_loss
        if min_val_loss is None or math.isnan(min_val_loss):
            logger.warning(f"Trial {trial.number} returned invalid loss")
            _remove_trial_logs(supervisor.log_dir)
            return float('nan')

        trial.set_user_attr("best_val_loss", min_val_loss)
        if hasattr(supervisor, 'min_val_fde'):
            trial.set_user_attr("best_val_fde", supervisor.min_val_fde)

        if min_val_loss < best_state["value"]:
            best_state["value"] = min_val_loss
            save_config_file(supervisor.log_dir, args)
        else:
            _remove_trial_logs(supervisor.log_dir)

        logger.info(f"Trial {trial.number} finished with val_loss: {min_val_loss:.4f}")
        return min_val_loss

    return objective


def _setup_optuna_study(base_args: argparse.Namespace) -> Tuple[optuna.Study, str]:
    """Create and setup Optuna study."""
    os.makedirs('logs', exist_ok=True)
    storage_url = f'sqlite:///logs/db.sqlite_{base_args.model}'
    study_name = f"{base_args.model}_{base_args.use_headeye[0]}_new"
    
    sampler = TPESampler(seed=base_args.random_seed)
    pruner = MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction='minimize'
    )
    
    if len(study.trials) == 0:
        hyperparameter_space = get_hyperparameter_space(base_args.model)
        initial_params = hyperparameter_space.get_initial_trial_params()
        study.enqueue_trial(initial_params)
        logger.info(f"Enqueued initial trial with params: {list(initial_params.keys())}")
    
    return study, study_name


def main() -> None:
    """Main hyperparameter tuning pipeline."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for trajectory prediction models',
        epilog=(
            'Examples:\n'
            '  python tune.py --config_filename data/config/multimodallstm.yaml --n_trials 50'
        )
    )
    parser.add_argument('--config_filename', type=str, default='data/config/multimodallstm.yaml',
                        help='YAML config file; '
                             'seeds data/arch/training fixed params)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')

    cli = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    device = get_device()

    # --- Build base_args ---
    # file.  Legacy models build args from CLI as before.
    if cli.config_filename is not None:
        base_args = load_runtime_args(cli.config_filename)
        # Allow CLI overrides for tuning-specific knobs
        base_args.n_trials = cli.n_trials
        # Derive len_obs / len_pred and input/output dims from YAML-loaded args
        from utils.util import setattrs
        base_args = setattrs(base_args)
    else:
        base_args = cli
        base_args.batch_size = 64
        base_args.lr = 1e-3
        seed_everything(base_args.random_seed)
        base_args = setattr_seq_len(base_args)
        base_args = setattr_input_dim(base_args)
        base_args = setattr_output_dim(base_args)

    seed_everything(base_args.random_seed)
    logger.info(f"Starting hyperparameter tuning for {base_args.model}")
    logger.info(f"Device: {device}")

    hyperparameter_space = get_hyperparameter_space(base_args.model)
    from utils.logger import TrainingLogger
    callback = TrainingLogger(log_interval=50)

    objective = _create_objective_function(
        base_args=base_args,
        device=device,
        callback=callback,
        hyperparameter_space=hyperparameter_space,
    )

    study, study_name = _setup_optuna_study(base_args)

    logger.info(f"Starting optimization with {cli.n_trials} trials")
    study.optimize(objective, n_trials=cli.n_trials)

    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value: {best_trial.value:.4f}")
    logger.info(f"Best params: {best_trial.params}")


if __name__ == '__main__':
    main()
