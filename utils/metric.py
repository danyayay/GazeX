import numpy as np
import torch


def compute_ade_fde_np(preds, labels, eps=1e-6):
    # preds/outputs.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 3 and len(labels.shape) == 3, "data must have 3 dimensions"
    ade = np.mean(np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1)))
    fde = np.mean(np.sqrt(np.sum((preds[:, -1, :] - labels[:, -1, :]) ** 2 + eps, axis=-1)))
    return ade, fde


def compute_minK_ade_fde_np(preds, labels, eps=1e-6):
    # preds.shape = (num_samples, batch_size, len_future, output_dim)
    # labels.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 4 and len(labels.shape) == 3, "preds must have 4 dimensions, labels must have 3 dimensions"
    labels = np.expand_dims(labels, axis=0)  # shape: (1, batch_size, len_future, output_dim)
    ade = np.mean(np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1)), axis=-1) # (num_samples, batch_size)
    fde = np.sqrt(np.sum((preds[:, :, -1, :] - labels[:, :, -1, :]) ** 2 + eps, axis=-1)) # (num_samples, batch_size)
    idx = np.argmin(ade, axis=0) # (batch_size,)
    return ade[idx, np.arange(preds.shape[1])].mean(), fde[idx, np.arange(preds.shape[1])].mean()


def compute_ade_fde_np_mx(preds_mx, labels, eps=1e-6):
    # preds_mx.shape = (batch_size, len_future, output_dim, n_type)
    # outputs.shape = (batch_size, len_future, output_dim)
    assert len(preds_mx.shape) == 4 and len(labels.shape) == 3, "preds_mx must have 4 dimensions, labels must have 3 dimensions"
    labels = labels[:, :, :, np.newaxis]  # shape: (batch_size, len_future, output_dim, 1)
    ade = np.mean(np.sqrt(np.sum((preds_mx - labels) ** 2 + eps, axis=-2)), axis=(0, 1)) # keep n_type dimension
    fde = np.mean(np.sqrt(np.sum((preds_mx[:, -1, :] - labels[:, -1, :]) ** 2 + eps, axis=-2)), axis=0) # keep n_type dimension
    ade_mean, ade_std = np.mean(ade), np.std(ade)
    fde_mean, fde_std = np.mean(fde), np.std(fde)
    return ade_mean, ade_std, fde_mean, fde_std


def compute_ade_fde_torch(preds, labels, eps=1e-6, verbose=False):
    # preds/outputs.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 3 and len(labels.shape) == 3, "data must have 3 dimensions"
    diff = (preds - labels) ** 2
    ade = torch.mean(torch.sqrt(torch.sum(diff + eps, dim=-1)))
    fde = torch.mean(torch.sqrt(torch.sum(diff[:, -1, :] + eps, dim=-1)))
    if verbose: print(f'diff.max={diff.max():.4f}, diff.min={diff.min():.4f}, #na={diff.isnan().sum()}, #inf={(diff == torch.inf).sum()}')
    # assert ((preds - labels) != 0).all(), "existing error=0 in at least one ADE/FDE calculation"
    return ade, fde


def compute_ade_fde_torch_mx(preds_mx, labels, eps=1e-6):
    # preds_mx.shape = (batch_size, len_future, output_dim, n_type)
    # outputs.shape = (batch_size, len_future, output_dim)
    assert len(preds_mx.shape) == 4 and len(labels.shape) == 3, "preds_mx must have 4 dimensions, labels must have 3 dimensions"
    labels = labels.unsqueeze(-1)  # shape: (batch_size, len_future, output_dim, 1)
    ade = torch.mean(torch.sqrt(torch.sum((preds_mx - labels) ** 2 + eps, dim=-2)), dim=(0, 1)) # keep n_type dimension
    fde = torch.mean(torch.sqrt(torch.sum((preds_mx[:, -1, :, :] - labels[:, -1, :, :]) ** 2 + eps, dim=-2)), dim=0) # keep n_type dimension
    ade_mean, ade_std = torch.mean(ade), torch.std(ade)
    fde_mean, fde_std = torch.mean(fde), torch.std(fde)
    return ade_mean, ade_std, fde_mean, fde_std


def compute_minK_ade_fde_torch(preds, labels, eps=1e-6):
    # preds.shape = (num_samples, batch_size, len_future, output_dim)
    # labels.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 4 and len(labels.shape) == 3, "preds must have 4 dimensions, labels must have 3 dimensions"
    labels = labels.unsqueeze(0)  # shape: (1, batch_size, len_future, output_dim)
    ade = torch.mean(torch.sqrt(torch.sum((preds - labels) ** 2 + eps, dim=-1)), dim=-1)  # (num_samples, batch_size)
    fde = torch.sqrt(torch.sum((preds[:, :, -1, :] - labels[:, :, -1, :]) ** 2 + eps, dim=-1)) # (num_samples, batch_size)
    idx = torch.argmin(ade, dim=0) # (batch_size,)
    return ade[idx, np.arange(preds.shape[1])].mean(), fde[idx, np.arange(preds.shape[1])].mean()


def compute_rmse_over_horizon(preds, labels):
    rmse = np.mean(np.sqrt(np.sum((preds - labels) ** 2, axis=-1)), axis=0)
    return rmse


def compute_rmse_over_horizon_minK(preds, labels, eps=1e-6):
    # preds.shape = (num_samples, batch_size, len_future, output_dim)
    # labels.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 4 and len(labels.shape) == 3, "preds must have 4 dimensions, labels must have 3 dimensions"
    print(f'Calculating the min_{preds.shape[0]}...')
    labels = np.expand_dims(labels, axis=0)  # shape: (1, batch_size, len_future, output_dim)
    rmse_all = np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1)) # (num_samples, batch_size, len_pred)
    ade = np.mean(rmse_all, axis=-1) # (num_samples, batch_size)
    idx = np.argmin(ade, axis=0) # (batch_size,)
    rmse = rmse_all[idx, np.arange(preds.shape[1])] # (num_samples, len_future)
    return rmse.mean(axis=0)


def compute_rmse_over_horizon_avgK(preds, labels, eps=1e-6):
    # preds.shape = (num_samples, batch_size, len_future, output_dim)
    # labels.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 4 and len(labels.shape) == 3, "preds must have 4 dimensions, labels must have 3 dimensions"
    print(f'Calculating the avg_{preds.shape[0]}...')
    preds = np.mean(preds, axis=0)  # average over num_samples (batch_size, len_pred, output_dim)
    rmse_all = np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1))  # (batch_size, len_pred)
    rmse = np.mean(rmse_all, axis=0)  # (len_pred,)
    return rmse, np.mean(rmse_all)


def compute_rmse_over_sample(preds, labels, eps=1e-6):
    if preds.shape[-1] > 2: preds = preds[..., :2]
    if labels.shape[-1] > 2: labels = labels[..., :2]
    rmse = np.mean(np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1)), axis=1)
    return rmse


def compute_mape_over_sample(preds, labels, eps=1e-6):
    if preds.shape[-1] > 2: preds = preds[..., :2]
    if labels.shape[-1] > 2: labels = labels[..., :2]
    rmse = np.mean(np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1)) / np.sqrt(np.sum(labels**2 + eps, axis=-1)), axis=1)
    return rmse


def compute_loss_over_sample(preds, labels, metric, eps=1e-6):
    if metric == 'rmse':
        return compute_rmse_over_sample(preds, labels, eps=eps)
    elif metric == 'mape':
        return compute_mape_over_sample(preds, labels, eps=eps)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented. Supported metrics: 'rmse', 'mape'.")


def compute_rmse_over_sample_minK(preds, labels, eps=1e-6):
    # preds.shape = (num_samples, batch_size, len_future, output_dim)
    # labels.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 4 and len(labels.shape) == 3, "preds must have 4 dimensions, labels must have 3 dimensions"
    labels = np.expand_dims(labels, axis=0)  # shape: (1, batch_size, len_future, output_dim)
    rmse_all = np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1))  # (num_samples, batch_size, len_pred)
    ade = np.mean(rmse_all, axis=-1)  # (num_samples, batch_size)
    idx = np.argmin(ade, axis=0)  # (batch_size,)
    rmse = ade[idx, np.arange(preds.shape[1])]  # (num_samples,)
    return rmse, idx


def compute_mape_over_sample_minK(preds, labels, eps=1e-6):
    # preds.shape = (num_samples, batch_size, len_future, output_dim)
    # labels.shape = (batch_size, len_future, output_dim)
    assert len(preds.shape) == 4 and len(labels.shape) == 3, "preds must have 4 dimensions, labels must have 3 dimensions"
    if preds.shape[-1] > 2: preds = preds[..., :2]
    if labels.shape[-1] > 2: labels = labels[..., :2]
    labels = np.expand_dims(labels, axis=0)  # shape: (1, batch_size, len_future, output_dim)
    mape_all = np.sqrt(np.sum((preds - labels) ** 2 + eps, axis=-1)) / np.sqrt(np.sum(labels**2 + eps, axis=-1))  # (num_samples, batch_size, len_pred)
    mape_mean = np.mean(mape_all, axis=-1)  # (num_samples, batch_size)
    idx = np.argmin(mape_mean, axis=0)  # (batch_size,)
    mape = mape_mean[idx, np.arange(preds.shape[1])]  # (num_samples,)
    return mape, idx

def compute_nll_loss_2d(out: torch.Tensor, out_distr: torch.Tensor, labels: torch.Tensor, 
        diagonal_cov: bool = True, eps: float = 1e-6) -> torch.Tensor:
        """Compute negative log likelihood for 2D output with learned covariance.
        
        Args:
            out: Predicted mean trajectory, shape (batch, seq_len, 2)
            out_distr: Distribution parameters (variance/sigma), shape (batch, seq_len, 2 or 3)
            labels: Ground truth trajectory, shape (batch, seq_len, 2)
            diagonal_cov: Whether to use diagonal covariance (True) or full with correlation (False)
            eps: Small epsilon for numerical stability
            
        Returns:
            NLL loss (scalar)
        """
        x, y = labels[..., 0], labels[..., 1]
        mu_x, mu_y = out[..., 0], out[..., 1]
        sigma_x, sigma_y = out_distr[..., 0], out_distr[..., 1]
        if not diagonal_cov: rho = out_distr[..., 2]

        norm_x = (x - mu_x) / (sigma_x + eps)
        norm_y = (y - mu_y) / (sigma_y + eps)
        if diagonal_cov:
            z = norm_x**2 + norm_y**2
            denom = torch.Tensor([2 * (1 + eps)]).to(out.device)
        else:
            z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
            denom = 2 * (1 - rho**2 + eps)
        nll = torch.log(2 * torch.pi * sigma_x * sigma_y * torch.sqrt(denom)) + z / denom

        return nll.mean()