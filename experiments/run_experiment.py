import yaml
import argparse
import os
import itertools
import pandas as pd
from datetime import datetime
import numpy as np

from tsforecast.data.preprocess import DataPreprocessor
from tsforecast.data.loaders import FewShotDataset, SlidingWindowDataset
from tsforecast.training.trainer import Trainer
from tsforecast.models import model_1, model_3, model_4, model_5

# Map model names from config to actual model classes
MODEL_MAP = {
    'model_1': model_1.LSTMAttention,
    # Model 2 is just a config change of Model 1
    'model_3': model_3.SiameseLSTM,
    'model_4': model_4.SiameseMultiResLSTM,
    'model_5': model_5.TransformerForecaster,
}

def load_data(config: dict):
    """Loads and preprocesses data based on the config."""
    data_path = config['data']['path']
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"The data path '{data_path}' specified in the config does not exist. "
            "Please ensure you have placed your data in the correct directory "
            "and updated the `data.path` in your YAML config file."
        )

    print(f"Loading data from: {data_path}")
    
    # Few-shot data
    try:
        train_query = np.load(os.path.join(data_path, "training_dataset.npy"))
        train_support = np.load(os.path.join(data_path, "training_support_dataset.npy"))
        train_neighbors = np.load(os.path.join(data_path, "train_query_neighbors.npy"))
        val_query = np.load(os.path.join(data_path, "validation_dataset.npy"))
        val_support = np.load(os.path.join(data_path, "validation_support_dataset.npy"))
        val_neighbors = np.load(os.path.join(data_path, "val_query_neighbors.npy"))
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure the data directory contains all required .npy files for few-shot learning.")
        raise

    # Check data dimensions - should be (N, seq_len, n_features)
    if train_query.ndim != 3:
        raise ValueError(f"Expected 3D array for training data, but got {train_query.ndim}D. "
                         "Data should be in shape (n_samples, seq_len, n_features).")
    
    print(f"Data shapes:")
    print(f"  Train query: {train_query.shape}")
    print(f"  Train support: {train_support.shape}")
    print(f"  Val query: {val_query.shape}")
    print(f"  Val support: {val_support.shape}")

    # Preprocessing
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_query)
    train_query = preprocessor.transform(train_query)
    train_support = preprocessor.transform(train_support)
    val_query = preprocessor.transform(val_query)
    val_support = preprocessor.transform(val_support)
    
    print("Data loaded and preprocessed.")
    return (train_query, train_support, train_neighbors,
            val_query, val_support, val_neighbors)


def run_single_experiment(config: dict, run_dir: str):
    """Runs one full training experiment."""
    print("\n" + "="*50)
    print(f"Running experiment with config: {config['model']['params']}")
    print("="*50)

    # 1. Load Data
    # This function loads all the files needed for few-shot models.
    # We will then adapt the data based on the specific model's needs.
    (train_query, train_support, train_neighbors,
     val_query, val_support, val_neighbors) = load_data(config)
    
    # 2. Create Datasets
    past_len = config['data']['past_len']
    future_len = config['data']['future_len']
    model_name = config['model']['name']

    if model_name == 'model_5':
        # For Model 5, we treat the query data as one long series and ignore the support set.
        train_series = train_query.reshape(-1, train_query.shape[-1])
        val_series = val_query.reshape(-1, val_query.shape[-1])
        train_dataset = SlidingWindowDataset(train_series, past_len, future_len)
        val_dataset = SlidingWindowDataset(val_series, past_len, future_len)
    else:
        # Few-shot dataset for models 1-4
        train_dataset = FewShotDataset(train_query, train_support, train_neighbors, past_len, future_len)
        val_dataset = FewShotDataset(val_query, val_support, val_neighbors, past_len, future_len)

    # 3. Initialize Model
    model_class = MODEL_MAP[model_name]
    model = model_class(**config['model']['params'])

    # 4. Train Model
    trainer = Trainer(model, config, run_dir)
    val_loss = trainer.train(train_dataset, val_dataset)
    
    print(f"Experiment finished. Final validation loss: {val_loss:.6f}")
    return val_loss

def run_hyperparameter_search(config: dict, base_run_dir: str):
    """Runs a grid search over hyperparameters."""
    print("\n" + "#"*60)
    print("Starting hyperparameter grid search...")
    print("#"*60 + "\n")

    search_params = config['hyper_search']['params']
    keys, values = zip(*search_params.items())
    
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Found {len(param_combinations)} parameter combinations to test.")

    results = []
    
    for i, params in enumerate(param_combinations):
        # Create a unique directory for this run
        run_name = "run_" + "_".join([f"{k}_{v}" for k, v in params.items()])
        run_dir = os.path.join(base_run_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Deep copy to avoid mutating original
        import copy
        current_config = copy.deepcopy(config)
        # Distribute search parameters: if key exists in model->params update there
        for k, v in params.items():
            if k in current_config['model']['params']:
                current_config['model']['params'][k] = v
            elif k == 'learning_rate':
                current_config.setdefault('training', {})['learning_rate'] = v
            else:
                # default: assume it's a model hyperparameter
                current_config['model']['params'][k] = v
        
        # Save the specific config for this run
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.dump(current_config, f)

        val_loss = run_single_experiment(current_config, run_dir)
        
        result_row = params.copy()
        result_row['validation_loss'] = val_loss
        results.append(result_row)

        # Save results incrementally
        results_df = pd.DataFrame(results).sort_values(by='validation_loss')
        results_df.to_csv(os.path.join(base_run_dir, 'search_results.csv'), index=False)
        
        print("\n" + "-"*60)
        print("Current leaderboard:")
        print(results_df)
        print("-"*60 + "\n")

    print("Hyperparameter search finished.")


def main():
    parser = argparse.ArgumentParser(description="Run a time series forecasting experiment.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create a directory for this experiment run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join("experiments", "runs", f"{config['model']['name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    if 'hyper_search' in config:
        run_hyperparameter_search(config, run_dir)
    else:
        run_single_experiment(config, run_dir)


if __name__ == '__main__':
    main() 