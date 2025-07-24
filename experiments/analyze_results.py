#!/usr/bin/env python3
"""
Analyze hyperparameter search results from all models.
"""
import os
import pandas as pd
import glob
from datetime import datetime

def find_latest_results():
    """Find the most recent search results for each model."""
    runs_dir = "runs"
    model_results = {}
    
    for model_num in range(1, 6):
        # Find all runs for this model
        pattern = os.path.join(runs_dir, f"model_{model_num}_*", "search_results.csv")
        result_files = glob.glob(pattern)
        
        if result_files:
            # Get the most recent one
            latest_file = max(result_files, key=os.path.getmtime)
            model_results[f"model_{model_num}"] = latest_file
            
    return model_results

def analyze_model_results(csv_path, model_name):
    """Analyze results for a single model."""
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*60}")
    
    # Best configuration
    best_idx = df['validation_loss'].idxmin()
    best_config = df.iloc[best_idx]
    
    print(f"\nBest configuration (lowest validation loss):")
    print(f"  Validation Loss: {best_config['validation_loss']:.6f}")
    print(f"  Parameters:")
    for col in df.columns:
        if col != 'validation_loss':
            print(f"    {col}: {best_config[col]}")
    
    # Statistics
    print(f"\nStatistics across all {len(df)} configurations:")
    print(f"  Min validation loss: {df['validation_loss'].min():.6f}")
    print(f"  Max validation loss: {df['validation_loss'].max():.6f}")
    print(f"  Mean validation loss: {df['validation_loss'].mean():.6f}")
    print(f"  Std validation loss: {df['validation_loss'].std():.6f}")
    
    # Top 5 configurations
    print(f"\nTop 5 configurations:")
    top_5 = df.nsmallest(5, 'validation_loss')
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n  {i}. Loss: {row['validation_loss']:.6f}")
        for col in df.columns:
            if col != 'validation_loss':
                print(f"     {col}: {row[col]}")
    
    return best_config

def compare_models(all_results):
    """Compare best results across all models."""
    print(f"\n{'='*80}")
    print("COMPARISON OF BEST RESULTS ACROSS ALL MODELS")
    print(f"{'='*80}")
    
    model_names = {
        'model_1': 'LSTMAttention',
        'model_2': 'LSTMAttention-96',
        'model_3': 'SiameseLSTM', 
        'model_4': 'SiameseMultiResLSTM',
        'model_5': 'TransformerForecaster'
    }
    
    comparison_data = []
    for model_id, (best_config, csv_path) in all_results.items():
        comparison_data.append({
            'Model': f"{model_id} ({model_names.get(model_id, 'Unknown')})",
            'Best Val Loss': best_config['validation_loss'],
            'Config': ', '.join([f"{k}={v}" for k, v in best_config.items() if k != 'validation_loss'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Best Val Loss')
    
    print("\nModel Rankings (by validation loss):")
    for i, row in comparison_df.iterrows():
        print(f"\n{i+1}. {row['Model']}")
        print(f"   Validation Loss: {row['Best Val Loss']:.6f}")
        print(f"   Config: {row['Config']}")
    
    # Save comparison to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_file = f"model_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")

def main():
    """Main analysis function."""
    print("Searching for hyperparameter search results...")
    
    # Find latest results
    model_results = find_latest_results()
    
    if not model_results:
        print("No search results found in experiments/runs/")
        return
    
    print(f"Found results for {len(model_results)} models:")
    for model, path in model_results.items():
        print(f"  {model}: {path}")
    
    # Analyze each model
    all_best_configs = {}
    for model, csv_path in model_results.items():
        best_config = analyze_model_results(csv_path, model)
        all_best_configs[model] = (best_config, csv_path)
    
    # Compare models
    if len(all_best_configs) > 1:
        compare_models(all_best_configs)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 