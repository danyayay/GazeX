"""Main entry point for trajectory prediction model training and evaluation.

Usage:
    python run.py --config_filename data/config/multimodallstm.yaml
"""

import os
import logging
import argparse

import torch
import pandas as pd

from model.supervisor import Supervisor
from utils.logger import TrainingLogger
from utils.util import seed_everything, get_config_file, save_config_file, setattrs, get_device
from utils.init_model import init_model
from utils.eval import test_stochastic_preds

logger = logging.getLogger(__name__)


def main(args):
    callback = TrainingLogger(log_interval=10)
    seed_everything(args.random_seed)
    device = get_device()

    args = setattrs(args)
    model = init_model(args, device=device)

    supervisor = Supervisor(
        model=model, callback=callback,
        device=device, ckpt_path=args.ckpt_path, args=args
    )

    if args.train:
        metrics = supervisor.train(args.n_epochs)
        min_val_loss, min_val_fde, min_train_ade, min_train_fde, min_test_ade, min_test_fde, best_epoch = metrics
        logger.info(f'Best epoch: {best_epoch}, val_ADE={min_val_loss:.4f}, val_FDE={min_val_fde:.4f}')

        if args.is_save_config_file:
            save_config_file(supervisor.log_dir, args)

        if not args.is_deter:
            best_ckpt_path = os.path.join(supervisor.ckpt_dir, 'best_model.pth')
            test_stochastic_preds(args, model, callback, device, best_ckpt_path)

    else:
        if args.is_deter:
            supervisor.test(supervisor.epoch, False)
            return None
        else:
            return test_stochastic_preds(args, model, callback, device, args.ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or evaluate trajectory prediction models from YAML config'
    )
    # parser.add_argument('--config_filename', default='data/config/multimodallstm.yaml', type=str)
    parser.add_argument('--config_filename', default='logs/indiv_time_o40_p40_s4/20260420_020822_MultiModalLSTM/config.yaml', type=str)
    args = parser.parse_args()
    args = get_config_file(args.config_filename, args)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.train:
        main(args)

    else:
        # Evaluation mode: run across multiple seeds and aggregate
        results_df = pd.DataFrame(columns=['seed', 'mu', 'min1', 'min5', 'min20'])

        for random_seed in args.random_seeds:
            logger.info(f'Evaluating with seed: {random_seed}')
            setattr(args, 'random_seed', random_seed)

            df = main(args)
            if df is None:
                continue

            df = pd.concat([df, df.mean().to_frame().T], ignore_index=True)
            df['seed'] = random_seed
            df['horizon'] = (df.index + 1).astype(int)
            df = pd.concat([df.iloc[4::5], df.iloc[-1:]], ignore_index=True)

            results_df = df.copy() if results_df.shape[0] == 0 else pd.concat([results_df, df])

        logger.info('=' * 60)
        logger.info('########### Averaging results across seeds:')
        for col in ['mu', 'min1', 'min5', 'min20']:
            if col in results_df.columns:
                mean = results_df.groupby('horizon')[col].agg(['mean'])
                df_string = mean.to_csv(sep='\t', index=False)
                logger.info(f'\n### {col} avg. ###\n' + df_string)

                std = results_df.groupby('horizon')[col].agg(['std'])
                df_string = std.to_csv(sep='\t', index=False)
                logger.info(f'\n### {col} std. ###' + df_string)