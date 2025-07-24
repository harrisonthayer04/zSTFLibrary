#!/usr/bin/env python3
"""
Parallel hyperparameter optimization for all models.
This script runs hyperparameter search for all 5 models concurrently.
"""
import os
import sys
import subprocess
import multiprocessing
from datetime import datetime
import time

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_model_search(model_name, config_path):
    """Run hyperparameter search for a single model."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting hyperparameter search for {model_name}...")
    
    # Use the virtual environment's Python
    python_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'venv', 'bin', 'python')
    
    cmd = [
        python_path,
        'run_experiment.py',
        '--config', config_path
    ]
    
    try:
        # Run the experiment - set working directory to experiments folder
        experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        experiments_dir = os.path.join(experiments_dir, 'experiments')
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=experiments_dir)
        
        if result.returncode == 0:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model_name} search completed successfully!")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model_name} search failed!")
            print(f"Error: {result.stderr}")
            
        return model_name, result.returncode
        
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error running {model_name}: {str(e)}")
        return model_name, -1

def main():
    """Main function to run all model searches in parallel."""
    print("="*80)
    print("PARALLEL HYPERPARAMETER OPTIMIZATION FOR ALL MODELS")
    print("="*80)
    
    # Define models and their search configs
    models = [
        ("Model 1 (LSTMAttention)", "configs/model_1_search.yaml"),
        ("Model 2 (LSTMAttention-96)", "configs/model_2_search.yaml"),
        ("Model 3 (SiameseLSTM)", "configs/model_3_search.yaml"),
        ("Model 4 (SiameseMultiResLSTM)", "configs/model_4_search.yaml"),
        ("Model 5 (TransformerForecaster)", "configs/model_5_search.yaml")
    ]
    
    # Get number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    # Use at most 5 processes (one per model) or number of cores - 1
    num_processes = min(5, max(1, num_cores - 1))
    
    print(f"\nSystem has {num_cores} CPU cores")
    print(f"Running {num_processes} models in parallel")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*80)
    
    start_time = time.time()
    
    # Create a process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Submit all jobs
        results = []
        for model_name, config_path in models:
            result = pool.apply_async(run_model_search, (model_name, config_path))
            results.append(result)
        
        # Wait for all jobs to complete
        pool.close()
        pool.join()
        
        # Collect results
        final_results = []
        for result in results:
            final_results.append(result.get())
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal time: {duration/60:.2f} minutes")
    print("\nResults:")
    for model_name, return_code in final_results:
        status = "SUCCESS" if return_code == 0 else "FAILED"
        print(f"  {model_name}: {status}")
    
    print(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print location of results
    print("\n" + "-"*80)
    print("Results saved in: experiments/runs/")
    print("Look for directories named: model_X_YYYYMMDD_HHMMSS/search_results.csv")
    print("="*80)

if __name__ == "__main__":
    main() 