import yaml
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from tsforecast.data.loaders import FewShotDataset, SlidingWindowDataset
from tsforecast.training.trainer import Trainer 
from tsforecast.models import model_1, model_3, model_4, model_5

# Map model names from config to actual model classes
MODEL_MAP = {
    'model_1': model_1.LSTMAttention,
    'model_3': model_3.SiameseLSTM,
    'model_4': model_4.SiameseMultiResLSTM,
    'model_5': model_5.TransformerForecaster,
}

def load_test_data(config: dict):
    """Loads and preprocesses test data based on the config."""
    data_path = config['data']['path']
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"The data path '{data_path}' specified in the config does not exist."
        )

    # This is a placeholder. You should replace this with your actual test data loading.
    # For now, we just load the validation set as a stand-in for the test set.
    print(f"Loading test data from: {data_path} (using validation set as a placeholder)...")
    
    try:
        val_query = np.load(os.path.join(data_path, "validation_dataset.npy"))
        val_support = np.load(os.path.join(data_path, "validation_support_dataset.npy"))
        val_neighbors = np.load(os.path.join(data_path, "val_query_neighbors.npy"))
    except FileNotFoundError as e:
        print(f"Error loading test data files: {e}")
        raise

    # Reshape for preprocessor
    if val_query.ndim == 2:
        val_query = val_query.reshape(*val_query.shape, 1)
        val_support = val_support.reshape(*val_support.shape, 1)

    # Note: In a real scenario, you would load a *fitted* preprocessor from the training run.
    # This is simplified for this example. We refit it here.
    from tsforecast.data.preprocess import DataPreprocessor
    preprocessor = DataPreprocessor()
    # Dummy fitting on training data to get the scaler
    train_query_dummy = np.load(os.path.join(data_path, "training_dataset.npy"))
    if train_query_dummy.ndim == 2:
        train_query_dummy = train_query_dummy.reshape(*train_query_dummy.shape, 1)
    preprocessor.fit(train_query_dummy)

    val_query = preprocessor.transform(val_query)
    val_support = preprocessor.transform(val_support)
    
    print("Test data loaded.")
    return val_query, val_support, val_neighbors

def main():
    parser = argparse.ArgumentParser(description="Test a trained time series forecasting model.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the experiment run directory.')
    args = parser.parse_args()

    # 1. Load Config and Model
    config_path = os.path.join(args.run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        # Fallback for hyperparameter search runs
        config_path = os.path.join(os.path.dirname(args.run_dir), 'config.yaml')
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    model_class = MODEL_MAP[model_name]
    
    checkpoint_path = os.path.join(args.run_dir, 'best_model.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_class.load(checkpoint_path, device=device)
    print(f"Loaded model from {checkpoint_path}")

    # 2. Load Data
    test_query, test_support, test_neighbors = load_test_data(config)
    past_len = config['data']['past_len']
    future_len = config['data']['future_len']

    if model_name == 'model_5':
        test_series = test_query.reshape(-1, test_query.shape[-1])
        test_dataset = SlidingWindowDataset(test_series, past_len, future_len)
    else:
        test_dataset = FewShotDataset(test_query, test_support, test_neighbors, past_len, future_len)

    # 3. Evaluate
    # We can reuse the collate_fn from the Trainer for consistency
    trainer_for_collate = Trainer(model, config, "")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=trainer_for_collate._get_collator()
    )

    print("\nEvaluating model on the test set...")
    # The trainer's evaluate method is perfect for this
    test_metrics = trainer_for_collate.evaluate(test_loader)
    
    print("\n" + "="*30)
    print("      Test Results")
    print("="*30)
    print(f"  -> Test MAE: {test_metrics['mae']:.6f}")
    print(f"  -> Test MSE: {test_metrics['mse']:.6f}")
    print("="*30)

if __name__ == '__main__':
    main() 