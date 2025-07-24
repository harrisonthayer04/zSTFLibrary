import torch
import torch.nn as nn

def mean_absolute_error(y_true, y_pred):
    return nn.L1Loss()(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    return nn.MSELoss()(y_true, y_pred) 