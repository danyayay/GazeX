"""Neural network models for trajectory prediction.

This module contains the core trajectory prediction models used in the research:

- **MultiModalLSTM**: Multi-modal LSTM encoder with Gaussian output (main model)
- **SocialVAE**: Social-aware Variational Autoencoder
- **SocialCVAE**: Social-aware Conditional VAE with optional coarse variant (SocialCVAEcoarse)

Supporting modules:
- **module.py**: Utility classes (HiddenLSTM, HiddenDense, OutputDense,
  CategoricalEmbedding, AutoregressiveLSTMDecoder)
- **supervisor.py**: Training orchestration and supervision

Usage:
    from model.multimodallstm import MultiModalLSTM
    from model.supervisor import Supervisor
"""

__all__ = ['MultiModalLSTM', 'Supervisor']
