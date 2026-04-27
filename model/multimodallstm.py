import numpy as np
import torch
import torch.nn as nn
from utils.util import init_weights, get_input_group_indices
from model.module import HiddenLSTM, HiddenDense, OutputDense
from pytorch_tcn import TCN


class MultiModalLSTM(nn.Module):
    def __init__(
            self, output_dim, diagonal_cov, target_dim, len_pred,
            ts_names, dense_hidden_dim, dense_n_layers,
            motion_hidden_dim, motion_n_layers,
            headeye_hidden_dim, headeye_n_layers, headeye_encoder_type='lstm',
            pod_hidden_dim=64, pod_n_layers=1, pod_encoder_type='lstm', pod_comb='max',
            is_embed_ts=False, embed_dim_motion=0, embed_dim_headeye=0, embed_dim_pod=0,
            is_embed_aux=False, aux_dim=0, aux_hidden_dim=0,
            device='cpu', with_reg=False, dropout=0.0,
            tcn_kernel_size=None, tcn_dropout=None, tcn_embedding_shape=None,
            is_deter=False):
        super(MultiModalLSTM, self).__init__()
        self.output_dim = output_dim
        self.diagonal_cov = diagonal_cov
        self.target_dim = target_dim
        self.len_pred = len_pred
        self.is_deter = is_deter
        self.device = device

        self.ts_names = ts_names
        self.cat_indices_dict, self.cat_dim_dict = get_input_group_indices(self.ts_names)
        self.is_embed_ts = is_embed_ts
        
        # motion must be included
        motion_dim = self.cat_dim_dict['motion']
        if self.is_embed_ts:
            self.motion_embed_fn = nn.Sequential(
                nn.Linear(motion_dim, embed_dim_motion),
                nn.ReLU(),
            )
            motion_dim = embed_dim_motion 
        self.motion_encoder = HiddenLSTM(motion_dim, motion_hidden_dim, motion_n_layers)
        dec_dim = motion_hidden_dim

        # headeye is optional
        if 'headeye' in self.cat_dim_dict:
            headeye_dim = self.cat_dim_dict['headeye']
            if self.is_embed_ts:
                self.headeye_embed_fn = nn.Sequential(
                    nn.Linear(headeye_dim, embed_dim_headeye),
                    nn.ReLU(),
                )
                headeye_dim = embed_dim_headeye
            if headeye_encoder_type == 'lstm':
                self.headeye_encoder = HiddenLSTM(headeye_dim, headeye_hidden_dim, headeye_n_layers)
                dec_dim += headeye_hidden_dim
            elif headeye_encoder_type == 'tcn':
                self.headeye_encoder = TCN(
                    num_inputs=headeye_dim, num_channels=[headeye_hidden_dim]*headeye_n_layers, 
                    kernel_size=tcn_kernel_size, dropout=tcn_dropout, 
                    causal=False, input_shape='NLC', embedding_shapes=tcn_embedding_shape)
                dec_dim += headeye_hidden_dim

        # pod is optional
        if 'pod' in self.cat_dim_dict:
            pod_dim = int(self.cat_dim_dict['pod'] // 2)
            self.indiv_pod_dim = pod_dim
            if self.is_embed_ts:
                self.pod_embed_fn = nn.Sequential(
                    nn.Linear(pod_dim, embed_dim_pod),
                    nn.ReLU(),
                )
                pod_dim = embed_dim_pod
            if pod_encoder_type == 'lstm':
                self.pod_encoder = HiddenLSTM(pod_dim, pod_hidden_dim, pod_n_layers)
                dec_dim += pod_hidden_dim
            self.pod_comb = pod_comb
        
        # auxiliary input is optional
        
        self.use_aux = aux_dim > 0
        if self.use_aux:
            self.aux_dim = aux_dim
            self.is_embed_aux = is_embed_aux
            if self.is_embed_aux:
                self.aux_embed_fn = nn.Sequential(
                    nn.Linear(aux_dim, aux_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                dec_dim += aux_hidden_dim
            else:
                dec_dim += aux_dim

        # decoder
        self.hidden_dense = HiddenDense(
            dec_dim, dense_hidden_dim, dense_hidden_dim, dense_n_layers,
            with_reg=with_reg, dropout=dropout)
        self.output_dense = OutputDense(dense_hidden_dim, output_dim*len_pred)
            
        self.apply(init_weights)

    def _combine_pod(self, leader: torch.Tensor, follower: torch.Tensor) -> torch.Tensor:
        """Combine leader and follower POD tensors according to self.pod_comb."""
        if self.pod_comb == 'mean':
            return torch.mean(torch.stack([leader, follower]), dim=0)
        elif self.pod_comb == 'max':
            return torch.max(torch.stack([leader, follower]), dim=0)[0]
        elif self.pod_comb == 'sum':
            return torch.sum(torch.stack([leader, follower]), dim=0)
        elif self.pod_comb == 'min':
            idx = torch.min(torch.stack([leader.abs(), follower.abs()]), dim=0)[1]
            return torch.gather(torch.stack([leader, follower]), 0, idx.unsqueeze(0)).squeeze(0)
        raise ValueError(f'Unknown pod_comb: {self.pod_comb}')

    def forward(self, data_ts, data_aux, num_samples=1, is_infer_mu=True):
        # motion input
        input_motion = data_ts[:, :, self.cat_indices_dict['motion']]  # (batch_size, len_obs, input_dim-2)
        if self.is_embed_ts:
            input_motion = self.motion_embed_fn(input_motion)
        encoded_motion = self.motion_encoder(input_motion)
        encoded = [encoded_motion]

        if 'headeye' in self.cat_dim_dict:
            input_headeye = data_ts[:, :, self.cat_indices_dict['headeye']]  # (batch_size, len_obs, 2)
            if self.is_embed_ts:
                input_headeye = self.headeye_embed_fn(input_headeye)
            encoded_headeye = self.headeye_encoder(input_headeye)
            encoded.append(encoded_headeye)

        if 'pod' in self.cat_dim_dict:
            indices = self.cat_indices_dict['pod']
            pod_leader   = data_ts[:, :, indices[:self.indiv_pod_dim]]
            pod_follower = data_ts[:, :, indices[self.indiv_pod_dim:]]
            if self.is_embed_ts:
                pod_leader   = self.pod_embed_fn(pod_leader)
                pod_follower = self.pod_embed_fn(pod_follower)
            encoded_pod = self.pod_encoder(self._combine_pod(pod_leader, pod_follower))
            encoded.append(encoded_pod)
        
        # concatenate encoded features
        encoded = torch.concat(encoded, dim=-1)[:, -1, :]

        # auxiliary input
        if self.use_aux:
            encoded_aux = data_aux
            if self.is_embed_aux:
                encoded_aux = self.aux_embed_fn(encoded_aux.float())
            encoded = torch.cat([encoded, encoded_aux], dim=-1)

        # decoder 
        out = self.hidden_dense(encoded)
        out = self.output_dense(out)
        out = out.view(-1, self.len_pred, self.output_dim)

        if not self.is_deter:
            if self.diagonal_cov:
                mu_x, mu_y, log_sigma_x, log_sigma_y = out.split(1, dim=-1) # (batch_size, len_pred, 1)
            else:
                mu_x, mu_y, log_sigma_x, log_sigma_y, tanh_rho = out.split(1, dim=-1) # (batch_size, len_pred, 1)
                rho = torch.tanh(tanh_rho)
            sigma_x = torch.exp(log_sigma_x)     # ensures > 0
            sigma_y = torch.exp(log_sigma_y)     # ensures > 0

            if self.training or is_infer_mu:
                out = torch.cat([mu_x, mu_y], dim=-1)
                if self.diagonal_cov:
                    out_distr = torch.cat([sigma_x, sigma_y], dim=-1)
                else:
                    out_distr = torch.cat([sigma_x, sigma_y, rho], dim=-1)
            else:
                batch_size = data_ts.size(0)
                eps_x = torch.randn(num_samples, batch_size, 1, 1, device=self.device) # (num_samples, batch_size, 1, 1)
                eps_y = torch.randn(num_samples, batch_size, 1, 1, device=self.device) # (num_samples, batch_size, 1, 1)
                x_samples = mu_x + sigma_x * eps_x # (num_samples, batch_size, len_pred, 1)
                if self.diagonal_cov:
                    y_samples = mu_y + sigma_y * eps_y # (num_samples, batch_size, len_pred, 1)
                else:
                    y_samples = mu_y + sigma_y * (rho.unsqueeze(0) * eps_x + torch.sqrt(1 - rho.unsqueeze(0)**2 + 1e-6) * eps_y)
                out = torch.concat([x_samples, y_samples], dim=-1) # (num_samples, batch_size, len_pred, 2)

        if self.training:
            if self.is_deter:
                return out
            else:
                return out, out_distr
        else:
            return out
            # return out[:, -1, :] ## only for SHAP output 