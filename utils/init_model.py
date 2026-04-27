from model.multimodallstm import MultiModalLSTM
from utils.util import get_input_group_indices


def init_model(args, device):
    print('Initializing model...')
    if args.model == 'multimodallstm':
        model = MultiModalLSTM(
            output_dim=args.output_dim,
            diagonal_cov=args.diagonal_cov,
            target_dim=args.target_dim,
            len_pred=args.len_pred,
            ts_names=args.ts_names,
            dense_hidden_dim=args.dense_hidden_dim,
            dense_n_layers=args.dense_n_layers,
            motion_hidden_dim=args.motion_hidden_dim,
            motion_n_layers=args.motion_n_layers,
            headeye_hidden_dim=args.headeye_hidden_dim,
            headeye_n_layers=args.headeye_n_layers,
            headeye_encoder_type=args.headeye_encoder_type,
            pod_hidden_dim=args.pod_hidden_dim,
            pod_n_layers=args.pod_n_layers,
            pod_encoder_type=args.pod_encoder_type,
            pod_comb=args.pod_comb,
            is_embed_ts=args.is_embed_ts,
            is_embed_aux=args.is_embed_aux,
            aux_dim=args.aux_dim,
            device=device,
            with_reg=args.with_reg,
            dropout=args.dropout,
            is_deter=args.is_deter
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Supported models: multimodallstm")
    
    print(f"Model: {model}")
    return model 