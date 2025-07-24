import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseForecastModel(nn.Module, ABC):
    """
    Abstract base class for all forecasting models.
    
    It enforces a common interface for model initialization, forward pass,
    and checkpointing, allowing them to be used interchangeably by the
    training and evaluation pipelines.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the model. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def save(self, file_path: str):
        """
        Saves the model's state dictionary and hyperparameters to a file.
        
        Args:
            file_path (str): The path to save the checkpoint file.
        """
        checkpoint = {
            'hyperparameters': self.hparams,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load(cls, file_path: str, device: str = 'cpu'):
        """
        Loads a model from a checkpoint file.
        
        Args:
            file_path (str): The path to the checkpoint file.
            device (str): The device to load the model onto ('cpu' or 'cuda').
        
        Returns:
            An instance of the model, loaded with pretrained weights.
        """
        checkpoint = torch.load(file_path, map_location=device)
        model = cls(**checkpoint['hyperparameters'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

    @property
    def hparams(self):
        """
        Returns the hyperparameters of the model. Must be implemented by subclasses
        that wish to use the default save/load mechanism.
        
        This should return a dictionary of the arguments passed to __init__.
        """
        raise NotImplementedError("Subclasses must implement the 'hparams' property.") 