import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import csv
import math
from ..models.base import BaseForecastModel
from .metrics import mean_absolute_error, mean_squared_error

class Trainer:
    """
    A generic trainer for time series forecasting models.
    """
    def __init__(self, model: BaseForecastModel, config: dict, run_dir: str):
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.model_name = config['model']['name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        self.criterion_mae = mean_absolute_error
        self.criterion_mse = mean_squared_error

    def _get_collator(self):
        """Returns the correct collate_fn based on model type."""
        # Model 5 is the only standard one so far.
        if self.model_name == 'model_5':
            return None # Use default collate
        else:
            # All other models are few-shot
            def few_shot_collate(batch):
                query_past, query_future, support_past, support_future = zip(*batch)
                return (torch.stack(query_past), torch.stack(support_past), 
                        torch.stack(support_future)), torch.stack(query_future)
            return few_shot_collate

    def train(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=self._get_collator()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=self._get_collator()
        )

        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            self.model.train()
            train_loss = 0.0
            for i, batch in enumerate(train_loader):
                # Unpack inputs depending on model type
                if self.model_name == 'model_5':
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    model_inputs = (inputs,)
                else:
                    inputs, targets = batch
                    query_past, support_past, support_future = inputs
                    model_inputs = (query_past.to(self.device), 
                                    support_past.to(self.device), 
                                    support_future.to(self.device))
                
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(*model_inputs)
                loss = self.criterion_mae(predictions, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['mae'] # Use MAE for checkpointing
            
            end_time = time.time()
            epoch_mins = (end_time - start_time) / 60
            
            print(f"Epoch {epoch+1:03}/{self.config['training']['epochs']} | "
                  f"Time: {epoch_mins:.2f}m | "
                  f"Train MAE: {avg_train_loss:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.6f} | "
                  f"Val MSE: {val_metrics['mse']:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.run_dir, "best_model.pth")
                self.model.save(checkpoint_path)
                print(f"-> Saved best model to {checkpoint_path}")
        
        return best_val_loss

    def evaluate(self, data_loader):
        self.model.eval()
        total_mae = 0.0
        total_mse = 0.0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if self.model_name == 'model_5':
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    model_inputs = (inputs,)
                else:
                    inputs, targets = batch
                    query_past, support_past, support_future = inputs
                    model_inputs = (query_past.to(self.device), 
                                    support_past.to(self.device), 
                                    support_future.to(self.device))
                
                targets = targets.to(self.device)

                predictions = self.model(*model_inputs)
                total_mae += self.criterion_mae(predictions, targets).item()
                total_mse += self.criterion_mse(predictions, targets).item()
                
        avg_mae = total_mae / len(data_loader)
        avg_mse = total_mse / len(data_loader)
        return {'mae': avg_mae, 'mse': avg_mse} 