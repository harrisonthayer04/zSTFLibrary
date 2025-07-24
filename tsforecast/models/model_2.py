"""
Model 2: LSTMAttention with different hyperparameters.

This model uses the exact same architecture as Model 1 (LSTMAttention),
but with different hyperparameter configurations. The main difference is
typically in the window lengths or hidden dimensions used during training.
"""

from .model_1 import LSTMAttention

# Model 2 is identical to Model 1 in architecture
# The differences are in the configuration (see experiments/configs/model_2_config.yaml)
__all__ = ['LSTMAttention'] 