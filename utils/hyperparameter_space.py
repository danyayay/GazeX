"""Hyperparameter space definitions for Optuna trials.

This module defines model-specific hyperparameter search spaces using dataclasses
and a registry pattern. This replaces massive if-elif chains in the objective
function with clean, composable configuration objects.

Usage:
    # Get hyperparameter space for a model
    space = HYPERPARAMETER_REGISTRY['multimodallstm']
    
    # Apply hyperparameters to args
    space.apply_hyperparameters(trial, args)
    
    # Get initial trial parameters
    initial_params = space.get_initial_trial_params()

Classes:
    HyperparameterSpace: Base class for all hyperparameter spaces
    MultiModalLSTMHyperparams: LSTM with Stochastic Embedding hyperparameters
    SocialCVAEHyperparams: Social-aware CVAE hyperparameters
    SocialCVAECoarseHyperparams: Social CVAE coarse variant hyperparameters
    SocialVAEHyperparams: Social-aware VAE hyperparameters
    HYPERPARAMETER_REGISTRY: Global registry of all model hyperparameter spaces

Attributes:
    HYPERPARAMETER_REGISTRY: Dict[str, HyperparameterSpace]
        Maps model names to their hyperparameter space objects
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import optuna


class HyperparameterSpace(ABC):
    """Abstract base class for hyperparameter search spaces.
    
    Each model's hyperparameter space is defined as a subclass that implements
    the abstract methods to generate trial suggestions and apply them to args.
    
    Attributes:
        model_name: Name of the model this space applies to
        common_hyperparams: Dict of common hyperparameters (batch_size, lr)
    """
    
    def __init__(self, model_name: str):
        """Initialize hyperparameter space.
        
        Args:
            model_name: Name of the model this space applies to
        """
        self.model_name = model_name
        self.common_hyperparams = {
            'batch_size': (32, 256),
            'lr': (0.0001, 0.01),
        }
    
    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial, args: Any) -> Dict[str, Any]:
        """Generate hyperparameter suggestions for this trial.
        
        Args:
            trial: Optuna trial object
            args: Arguments object
            
        Returns:
            Dictionary of hyperparameter names to suggested values
        """
        pass
    
    @abstractmethod
    def apply_hyperparameters(self, trial_params: Dict[str, Any], args: Any) -> None:
        """Apply suggested hyperparameters to args object.
        
        Args:
            trial_params: Dictionary from suggest_hyperparameters()
            args: Arguments object to modify in-place
        """
        pass
    
    @abstractmethod
    def get_initial_trial_params(self) -> Dict[str, Any]:
        """Get initial trial parameters for this model.
        
        Returns:
            Dictionary of initial hyperparameter values
        """
        pass
    
    def _suggest_common_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest common hyperparameters (batch_size, lr).
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary with 'batch_size' and 'lr' keys
        """
        batch_size_range, lr_range = self.common_hyperparams['batch_size'], self.common_hyperparams['lr']
        return {
            'batch_size': trial.suggest_int('batch_size', batch_size_range[0], batch_size_range[1]),
            'lr': trial.suggest_float('lr', lr_range[0], lr_range[1]),
        }


class MultiModalLSTMHyperparams(HyperparameterSpace):
    """LSTM with Stochastic Embedding hyperparameter space.
    
    Suggests hyperparameters for:
    - Dense layers (hidden_dim, n_layers)
    - Motion encoder (hidden_dim, n_layers)
    - Optional head/eye encoder (TCN or LSTM)
    - Optional POD encoder
    - Optional auxiliary information encoding
    """
    
    def __init__(self):
        super().__init__('multimodallstm')
    
    def suggest_hyperparameters(self, trial: optuna.Trial, args: Any) -> Dict[str, Any]:
        """Suggest MultiModalLSTM hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameter suggestions
        """
        params = self._suggest_common_hyperparams(trial)
        
        # Dense layers
        params['dense_hidden_dim'] = trial.suggest_int('dense_hidden_dim', 64, 256)
        params['dense_n_layers'] = trial.suggest_int('dense_n_layers', 1, 3)
        params['with_reg'] = False
        params['dropout'] = 0

        # Motion encoder
        params['motion_hidden_dim'] = trial.suggest_int('motion_hidden_dim', 64, 256)
        params['motion_n_layers'] = trial.suggest_int('motion_n_layers', 1, 3)
        
        # Embedding dimensions (may be 0 if not using embeddings)
        params['embed_dim_motion'] = trial.suggest_int('embed_dim_motion', 8, 128)
        params['embed_dim_pod'] = trial.suggest_int('embed_dim_pod', 8, 128)
        params['pod_hidden_dim'] = trial.suggest_int('pod_hidden_dim', 10, 128)
        params['pod_n_layers'] = trial.suggest_int('pod_n_layers', 1, 3)

        if args.use_headeye:
            params['embed_dim_headeye'] = trial.suggest_int('embed_dim_headeye', 8, 128)
            params['headeye_hidden_dim'] = trial.suggest_int('headeye_hidden_dim', 10, 128)
            params['headeye_n_layers'] = trial.suggest_int('headeye_n_layers', 1, 3)
        else: # Not using head/eye embedding
            params['embed_dim_headeye'] = 0  
            params['headeye_hidden_dim'] = 0
            params['headeye_n_layers'] = 0
        
        # Auxiliary information
        if args.use_person:
            params['aux_format'] = trial.suggest_categorical('aux_format', ['raw', 'onehot'])
            params['aux_hidden_dim'] = trial.suggest_int('aux_hidden_dim', 5, 64)
        else:
            params['aux_format'] = None
            params['aux_hidden_dim'] = 0
        
        return params
    
    def apply_hyperparameters(self, trial_params: Dict[str, Any], args: Any) -> None:
        """Apply MultiModalLSTM hyperparameters to args.
        
        Args:
            trial_params: Dictionary from suggest_hyperparameters()
            args: Arguments object to modify in-place
        """
        for key, value in trial_params.items():
            setattr(args, key, value)
    
    def get_initial_trial_params(self) -> Dict[str, Any]:
        """Get initial trial parameters for MultiModalLSTM.
        
        Returns:
            Dictionary of recommended initial hyperparameters
        """
        return {
            'dense_hidden_dim': 211,
            'dense_n_layers': 2,
            'with_reg': False,
            'dropout': 0,
            'batch_size': 42,
            'lr': 0.0015,
            'motion_hidden_dim': 190,
            'motion_n_layers': 3,
            'embed_dim_motion': 16,
            'embed_dim_headeye': 16,
            'embed_dim_pod': 16,
            'headeye_hidden_dim': 64,
            'headeye_n_layers': 2,
            'pod_hidden_dim': 64,
            'pod_n_layers': 2,
            'aux_format': None,
            'aux_hidden_dim': 0
        }

# Global registry of all hyperparameter spaces
HYPERPARAMETER_REGISTRY: Dict[str, HyperparameterSpace] = {
    'multimodallstm': MultiModalLSTMHyperparams()
}


def get_hyperparameter_space(model_name: str) -> HyperparameterSpace:
    """Get hyperparameter space for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        HyperparameterSpace object for the model
        
    Raises:
        ValueError: If model_name is not in registry
    """
    if model_name not in HYPERPARAMETER_REGISTRY:
        supported = ', '.join(HYPERPARAMETER_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {supported}"
        )
    return HYPERPARAMETER_REGISTRY[model_name]
