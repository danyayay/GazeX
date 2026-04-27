import os
import time
import math
import logging
from typing import Optional, Tuple, Any

import numpy as np

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from utils.metric import compute_ade_fde_np, compute_minK_ade_fde_np, compute_ade_fde_torch, compute_rmse_over_horizon, \
    compute_rmse_over_horizon_minK, compute_minK_ade_fde_torch, compute_nll_loss_2d
from utils.load_data import load_dataset, get_dataloaders, StandardScaler, StandardScalerAux
from utils.util import reverse_delta_to_abs, reverse_offset_to_abs, get_input_group_indices
import optuna

logger = logging.getLogger(__name__)

TRAJ_REL_DELTA = 'rel_delta'
TRAJ_REL_TO_ORIGIN = 'rel_to_origin'
TRAJ_REL_TO_T = 'rel_to_t'


class Supervisor:
    def __init__(self, model: torch.nn.Module, callback: Any, device: torch.device, ckpt_path: str, args: Any) -> None:
        self.model = model
        self.device = device
        self.callback = callback
        self.ckpt_path = ckpt_path

        self._initialize_config(args)
        self._initialize_logging(args)
        self._initialize_data(args)
        self._initialize_optimizer(args)

        # Ensure model is on the correct device before loading checkpoint or training
        if not self._is_model_on_device():
            self.model = self.model.to(self.device)

        if self.ckpt_path:
            self.resume_training()
        else:
            self.epoch = -1

    def _is_model_on_device(self) -> bool:
        """Check if all model parameters are on the target device."""
        try:
            params = next(self.model.parameters())
            return params.device == self.device
        except StopIteration:
            # Model has no parameters
            return True

    def _initialize_config(self, args: Any) -> None:
        self.len_obs = args.len_obs
        self.len_pred = args.len_pred
        self.model_name = self.model.__class__.__name__
        SUPPORTED_MODELS = {'MultiModalLSTM'}
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f'Supervisor supports only {SUPPORTED_MODELS}, got: {self.model_name}')
        self.traj_format = args.traj_format
        self.is_deter = args.is_deter
        self.is_infer_mu = args.is_infer_mu
        self.n_samples = args.num_samples  # number of stochastic samples at inference
        self.kl_warmup_end = getattr(args, 'kl_warmup_end', 0)

    def _initialize_logging(self, args: Any) -> None:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.log_dir = os.path.join(
            'logs', args.data_dir.split('/')[1],
            f"{timestamp}_{self.model_name}")
        os.makedirs(self.log_dir, exist_ok=True)

        self._writer = SummaryWriter(self.log_dir)
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        logger.info(f'Log directory: {self.log_dir}')
        logger.info(f'Using device: {self.device}')
        logger.info(args)

    def _initialize_data(self, args: Any) -> None:
        data = load_dataset(
            args.data_dir, base_motion=args.base_motion, target=args.target,
            use_headeye=args.use_headeye, use_pod=args.use_pod,
            use_expt=args.use_expt, use_person=args.use_person, aux_format=args.aux_format)

        self.aux_format = args.aux_format
        self.ts_names = data['ts_names']
        self.predict_not_only_traj = len(args.target) > 1
        self.target_indices = data['target_indices']
        self.is_normalize_ts = args.is_normalize_ts
        self.is_normalize_aux = args.is_normalize_aux

        if self.is_normalize_ts:
            group_indices_dict, _ = get_input_group_indices(self.ts_names)
            print('Group indices dict for normalization:', group_indices_dict)
            self.scaler_torch = StandardScaler(
                data['x_train'], group_indices_dict=group_indices_dict,
                target_indices=self.target_indices, device=self.device,
                ts_names=self.ts_names)

        if self.is_normalize_aux and self.aux_format == 'raw':
            self.scaler_aux = StandardScalerAux(data['aux_train'], device=self.device)

        self.data_loader = get_dataloaders(data, args.batch_size, args.traj_format, is_pgm=False)

    def _initialize_optimizer(self, args: Any) -> None:
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, eps=1e-8)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=args.lr_milestones, gamma=args.gamma)

    def _prepare_batch(self, batch):
        """Move batch to device, normalize, and slice y to prediction targets."""
        x, y_full_, x_rel, y_rel_full_, aux = batch
        x = x.to(self.device)
        y_full_ = y_full_.to(self.device)
        x_rel = x_rel.to(self.device)
        y_rel_full_ = y_rel_full_.to(self.device)
        aux = aux.to(self.device)

        if self.is_normalize_ts:
            x_rel = self.scaler_torch.transform(x_rel)
        if self.is_normalize_aux and self.aux_format == 'raw':
            aux = self.scaler_aux.transform(aux)

        # y_full_ has the same columns as x; slice to prediction targets only.
        # target_indices are used by the loss and metrics.
        y = y_full_[..., self.target_indices]
        y_rel = y_rel_full_[..., self.target_indices]
        return x, x_rel, aux, y, y_rel

    def _to_absolute_traj(self, traj: torch.Tensor, x: torch.Tensor, is_test: bool) -> torch.Tensor:
        """Convert relative trajectory predictions back to absolute coordinates."""
        if self.traj_format == TRAJ_REL_DELTA:
            return reverse_delta_to_abs(traj, x[:, -1, :2])
        elif self.traj_format == TRAJ_REL_TO_ORIGIN:
            ref = x[:, -1, :2] if is_test else x[:, 0, :2]
            return reverse_offset_to_abs(traj, ref)
        elif self.traj_format == TRAJ_REL_TO_T:
            return reverse_offset_to_abs(traj, x[:, -1, :2])
        return traj  # absolute format: no conversion needed

    def _to_absolute_point(self, raw_out: torch.Tensor, x: torch.Tensor, is_test: bool) -> torch.Tensor:
        """Convert a raw point prediction (B, T, D) to absolute coordinates.

        Applies inverse normalization (if applicable) then converts the trajectory
        dimensions (:2) from relative to absolute. Non-trajectory dims (2:) are
        passed through unchanged.
        """
        if self.is_normalize_ts and self.predict_not_only_traj:
            raw_out = self.scaler_torch.inverse_transform(raw_out)
        traj_abs = self._to_absolute_traj(raw_out[..., :2], x, is_test)
        if self.predict_not_only_traj:
            return torch.cat([traj_abs, raw_out[..., 2:]], dim=-1)
        return traj_abs

    def _to_absolute_samples(self, samples: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Convert stochastic samples (K, B, T, 2) to absolute coordinates."""
        ref = x[:, -1, :2]
        if self.traj_format == TRAJ_REL_DELTA:
            fn = reverse_delta_to_abs
        elif self.traj_format in (TRAJ_REL_TO_ORIGIN, TRAJ_REL_TO_T):
            fn = reverse_offset_to_abs
        else:
            return samples
        return torch.vmap(fn, in_dims=(0, None))(samples, ref)

    def _forward_train(self, x_rel, aux, y_rel=None):
        """Forward pass in training mode.

        Returns (raw_out, raw_out_distr) where raw_out_distr is None for
        deterministic models and (sigma_x, sigma_y[, rho]) for stochastic.
        y_rel is passed through to models that require future context (e.g.
        TrajectronPP posterior encoder); ignored by models that don't use it.
        """
        if self.is_deter:
            return self.model(x_rel, aux), None
        return self.model(x_rel, aux)

    def _forward_eval(self, x_rel, aux, is_infer_mu: bool):
        """Forward pass in eval mode; MultiModalLSTM always returns a single tensor here."""
        if self.is_deter:
            return self.model(x_rel, aux)
        return self.model(x_rel, aux, num_samples=self.n_samples, is_infer_mu=is_infer_mu)

    def resume_training(self) -> None:
        # Move model to device before loading checkpoint
        self.model = self.model.to(self.device)
        
        checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        
        # Move model to device again after loading state dict to ensure all parameters are on correct device
        self.model = self.model.to(self.device)
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch']
        logger.info(f'Resume from epoch {self.epoch}')

    def train(self, n_epochs: int, trial: Optional[Any] = None) -> Tuple[float, float, float, float, float, float, int]:
        self.model = self.model.to(self.device)
        self.min_val_loss = float('inf')
        self.min_val_fde = float('inf')
        self.min_train_loss = float('inf')
        self.min_train_fde = float('inf')
        min_test_ade, min_test_fde = float('inf'), float('inf')
        best_epoch = self.epoch

        logger.info(f'Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

        for epoch in range(self.epoch + 1, n_epochs):
            ade_train, fde_train, n_seen = 0.0, 0.0, 0
            start_time = time.time()
            self.model.train()

            for i, batch in enumerate(self.data_loader['train_loader']):
                x, x_rel, aux, y, y_rel = self._prepare_batch(batch)
                self.optimizer.zero_grad()

                raw_out, raw_out_distr = self._forward_train(x_rel, aux, y_rel=y_rel)

                if self.is_deter:
                    # ADE on absolute coords is both the loss and the monitoring metric.
                    pred = self._to_absolute_point(raw_out, x, is_test=False)
                    loss, fde = compute_ade_fde_torch(pred, y)
                    ade = loss
                else:
                    # NLL on raw relative outputs (no postprocessing needed before loss).
                    loss = compute_nll_loss_2d(raw_out, raw_out_distr, y_rel, diagonal_cov=self.model.diagonal_cov)
                    if hasattr(self.model, 'kl_loss'):
                        if self.kl_warmup_end > 0:
                            kl_factor = min(1.0, epoch / self.kl_warmup_end)
                        else:
                            kl_factor = 1.0
                        loss = loss + self.model.kl_loss(annealing_factor=kl_factor)
                    # ADE/FDE are monitoring metrics only — detach to avoid unnecessary grad.
                    with torch.no_grad():
                        pred = self._to_absolute_point(raw_out, x, is_test=False)
                        ade, fde = compute_ade_fde_torch(pred, y)

                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                n_seen += x.size(0)
                ade_train += ade.detach().cpu().item() * x.size(0)
                fde_train += fde.detach().cpu().item() * x.size(0)

                if i % 50 == 0:
                    logger.info(
                        f'Epoch {epoch}, iter {i}, '
                        f'ADE={ade.detach().cpu().item():.4f}, FDE={fde.detach().cpu().item():.4f}, '
                        f'loss={loss.detach().cpu().item():.4f}'
                    )

            self.lr_scheduler.step()
            ade_train /= max(n_seen, 1)
            fde_train /= max(n_seen, 1)
            self.min_train_loss = min(self.min_train_loss, ade_train)
            self.min_train_fde = min(self.min_train_fde, fde_train)

            self._writer.add_scalar('focus/train_lr', self.lr_scheduler.get_last_lr()[0], epoch)
            self._writer.add_scalar('focus/train_ADE', ade_train, epoch)
            self._writer.add_scalar('focus/train_FDE', fde_train, epoch)
            logger.info(
                f'Epoch {epoch}, train_ADE={ade_train:.4f}, train_FDE={fde_train:.4f}, '
                f'lr={self.lr_scheduler.get_last_lr()[0]:.6f}, time={time.time()-start_time:.2f}s'
            )

            ade_val, fde_val = self._run_val(epoch)
            best_epoch_until_now = ade_val < self.min_val_loss
            if best_epoch_until_now:
                self.min_val_loss = ade_val
                self.min_val_fde = fde_val
                best_epoch = epoch
                self._save_model(epoch, 'best_model')
            else:
                self._save_model(epoch, 'last_model')

            ade_test, fde_test, _ = self.test(epoch, best_epoch_until_now, is_infer_mu=self.is_infer_mu)
            if best_epoch_until_now:
                min_test_ade, min_test_fde = ade_test, fde_test

            if np.isnan(ade_train) or math.isnan(ade_train):
                break

            if trial is not None:
                trial.report(ade_val, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return (
            float(self.min_val_loss),
            float(self.min_val_fde),
            float(self.min_train_loss),
            float(self.min_train_fde),
            float(min_test_ade),
            float(min_test_fde),
            int(best_epoch),
        )

    def _run_val(self, epoch: int) -> Tuple[float, float]:
        ade_list, fde_list, n_seen = [], [], 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loader['val_loader']:
                x, x_rel, aux, y, _ = self._prepare_batch(batch)
                raw_out = self._forward_eval(x_rel, aux, is_infer_mu=self.is_infer_mu)
                if self.is_deter or self.is_infer_mu:
                    pred = self._to_absolute_point(raw_out, x, is_test=False)
                    ade, fde = compute_ade_fde_torch(pred[..., :2], y[..., :2])
                else:
                    preds = self._to_absolute_samples(raw_out[..., :2], x)
                    ade, fde = compute_minK_ade_fde_torch(preds, y[..., :2])
                n_seen += x.shape[0]
                ade_list.append(ade.detach().item() * x.shape[0])
                fde_list.append(fde.detach().item() * x.shape[0])

        ade_val = np.sum(ade_list) / max(n_seen, 1)
        fde_val = np.sum(fde_list) / max(n_seen, 1)
        self._writer.add_scalar('focus/val_ADE', ade_val, epoch)
        self._writer.add_scalar('focus/val_FDE', fde_val, epoch)
        logger.info(f'Epoch {epoch}, val_ADE={ade_val:.4f}, val_FDE={fde_val:.4f}')
        return ade_val, fde_val

    def _save_model(self, epoch: int, file_name: str) -> None:
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }, f'{self.ckpt_dir}/{file_name}.pth')

    def test(self, epoch: int = 0, best_epoch_until_now: bool = False, is_infer_mu: bool = False) -> Tuple[float, float, np.ndarray]:
        start_time = time.time()
        is_point = self.is_deter or is_infer_mu

        x_list, y_list, pred_list, aux_list = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loader['test_loader']:
                x, x_rel, aux, y, _ = self._prepare_batch(batch)
                raw_out = self._forward_eval(x_rel, aux, is_infer_mu=is_infer_mu)
                if is_point:
                    pred = self._to_absolute_point(raw_out, x, is_test=True)
                else:
                    pred = self._to_absolute_samples(raw_out[..., :2], x)  # (K, B, T, 2)
                x_list.append(x.cpu().numpy())
                y_list.append(y.cpu().numpy())
                pred_list.append(pred.cpu().numpy())
                aux_list.append(aux.cpu().numpy())

        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        aux_all = np.concatenate(aux_list, axis=0)
        # point: (N, T, D); stochastic: (K, N, T, 2)
        preds = np.concatenate(pred_list, axis=0 if is_point else 1)

        y_traj = y_all[..., :2]
        p_traj = preds[..., :2] if is_point else preds  # stochastic already (K, N, T, 2)

        if is_point:
            test_ade, test_fde = compute_ade_fde_np(p_traj, y_traj)
            rmse = compute_rmse_over_horizon(p_traj, y_traj)
        else:
            test_ade, test_fde = compute_minK_ade_fde_np(p_traj, y_traj)
            rmse = compute_rmse_over_horizon_minK(p_traj, y_traj)

        msg = f'test_ADE={test_ade:.4f}, test_FDE={test_fde:.4f}'
        if self.predict_not_only_traj and is_point:
            ade_eye, fde_eye = compute_ade_fde_np(preds[..., 2:], y_all[..., 2:])
            msg += f', test_ADE_eye={ade_eye:.4f}, test_FDE_eye={fde_eye:.4f}'
        msg += f', time={time.time()-start_time:.2f}s'
        logger.info(msg)

        self._writer.add_scalar('focus/test_ADE', max(test_ade, 100), epoch)
        self._writer.add_scalar('focus/test_FDE', max(test_fde, 200), epoch)

        print('Horizon, RMSE:')
        for i in range(4, rmse.shape[0], 5):
            print(f'{i+1},\t{rmse[i]:.4f}')
        print(f'ADE,\t{test_ade:.4f}\n')

        if best_epoch_until_now:
            np.savez(f'{self.log_dir}/test_best_{self.n_samples}_{is_infer_mu}.npz',
                x_labels=x_all, y_labels=y_all, y_preds=preds)
        np.savez(f'{self.log_dir}/test_last_{self.n_samples}_{is_infer_mu}.npz',
            x_labels=x_all, y_labels=y_all, y_preds=preds)
        np.savez(f'{self.log_dir}/test_aux_{self.n_samples}_{is_infer_mu}.npz', aux_labels=aux_all)

        return float(test_ade), float(test_fde), rmse
