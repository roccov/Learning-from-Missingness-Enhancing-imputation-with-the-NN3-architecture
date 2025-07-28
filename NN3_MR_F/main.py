# =============================================================================
# SIMPLIFIED MAIN EXECUTION FILE - PYTORCH ENSEMBLE ONLY
# =============================================================================

import os
import warnings
import pandas as pd
from config import Config
from pipeline import SimplifiedSlidingWindowPipeline
from plotting_utils import SimplifiedTrainingPlotter

warnings.filterwarnings('ignore')

def main():
    """Main execution function for simplified PyTorch ensemble pipeline."""
    
    print("=" * 80)
    print("SIMPLIFIED DEEP LEARNING PIPELINE - PYTORCH ENSEMBLE ONLY")
    print("=" * 80)
    
    # Load data
    print("\n1. LOADING DATA...")
    folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(folder, Config.DATA_FILENAME)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        print(f"Please ensure {Config.DATA_FILENAME} is in the same directory as this script.")
        return
    
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Verify required columns exist
    required_columns = [Config.DATE_COLUMN, Config.TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Print configuration
    print(f"\n2. CONFIGURATION:")
    print(f"✓ Date range: {df[Config.DATE_COLUMN].min()} to {df[Config.DATE_COLUMN].max()}")
    print(f"✓ Target column: {Config.TARGET_COLUMN}")
    print(f"✓ Train/Val/Test split: {Config.TRAIN_YEARS}/{Config.VAL_YEARS}/{Config.TEST_YEARS} years")
    print(f"✓ Feature selection: Random Forest ({Config.MIN_FEATURES}-{Config.MAX_FEATURES} features)")
    print(f"✓ Model: PyTorch Ensemble ({Config.DEFAULT_N_ESTIMATORS} models)")
    print(f"✓ Architecture: {Config.HIDDEN_LAYERS}")
    print(f"✓ Hyperparameter tuning: {'Grid Search' if Config.ENABLE_HYPERPARAMETER_TUNING else 'Disabled'}")
    if Config.ENABLE_HYPERPARAMETER_TUNING:
        print(f"✓ Grid search parameters: {Config.GRID_SEARCH_PARAMS}")
    
    # Initialize and run pipeline
    print(f"\n3. INITIALIZING PIPELINE...")
    pipeline = SimplifiedSlidingWindowPipeline()
    
    # Initialize plotter
    output_folder = os.path.join(folder, Config.OUTPUT_FOLDER)
    os.makedirs(output_folder, exist_ok=True)
    plotter = SimplifiedTrainingPlotter(output_folder)
    
    # Run the pipeline
    print(f"\n4. STARTING SLIDING WINDOW ANALYSIS...")
    print("-" * 60)
    results = pipeline.run(df, plotter)
    
    # Save results
    print(f"\n5. SAVING RESULTS...")
    _save_results(results, output_folder)
    
    # Generate final plots
    print(f"\n6. GENERATING FINAL PLOTS...")
    _generate_final_plots(results, plotter)
    
    # Print final summary
    _print_final_summary(results, output_folder)
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

def _save_results(results: dict, output_folder: str):
    """Save all results to files."""
    
    # Save predictions with permno and date
    predictions_data = {
        'date': results['dates'],
        'actual_values': results['actual_values'],
        'predictions': results['predictions'],
        'residuals': [a - p for a, p in zip(results['actual_values'], results['predictions'])]
    }
    
    # Add permno if it exists
    if 'permnos' in results and any(p is not None for p in results['permnos']):
        predictions_data['permno'] = results['permnos']
        print("✓ Including permno data in predictions file")
    else:
        print("✓ No permno data found in dataset")
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_path = os.path.join(output_folder, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved: {predictions_path}")
    
    # Save window metrics
    metrics_summary = []
    for metrics in results['metrics_by_window']:
        metrics_summary.append({
            'window': metrics['window'],
            'train_period': metrics['train_period'],
            'val_period': metrics['val_period'],
            'test_period': metrics['test_period'],
            'train_r2': round(metrics['train_r2'], 4),
            'test_r2': round(metrics['r2'], 4),
            'rmse': round(metrics['rmse'], 6),
            'mae': round(metrics['mae'], 6),
            'n_features': metrics['n_features_selected'],
            'n_train': metrics['n_train'],
            'n_val': metrics['n_val'],
            'n_test': metrics['n_test']
        })
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_path = os.path.join(output_folder, 'window_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Window metrics saved: {metrics_path}")
    
    # Save feature selection evolution
    if results['feature_selection_evolution']:
        feature_evolution = []
        for fs_info in results['feature_selection_evolution']:
            feature_evolution.append({
                'window': fs_info['window'],
                'n_selected': fs_info['n_selected'],
                'selection_ratio': round(fs_info['n_selected'] / fs_info['n_total'], 3)
            })
        
        feature_df = pd.DataFrame(feature_evolution)
        feature_path = os.path.join(output_folder, 'feature_evolution.csv')
        feature_df.to_csv(feature_path, index=False)
        print(f"✓ Feature evolution saved: {feature_path}")
    
    # Save overall results summary
    overall_metrics = results['overall_metrics']
    with open(os.path.join(output_folder, 'overall_results.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PYTORCH ENSEMBLE PIPELINE RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"Overall Test R²: {overall_metrics['overall_r2']:.4f}\n")
        f.write(f"Overall Train R²: {overall_metrics['avg_train_r2']:.4f}\n")
        f.write(f"Overall RMSE: {overall_metrics['overall_rmse']:.6f}\n")
        f.write(f"Overall MAE: {overall_metrics['overall_mae']:.6f}\n\n")
        
        f.write("WINDOW AVERAGES:\n")
        f.write(f"Average Test R²: {overall_metrics['avg_window_r2']:.4f}\n")
        f.write(f"Average Train R²: {overall_metrics['avg_train_r2']:.4f}\n")
        f.write(f"Average epochs: {overall_metrics['avg_epochs']:.1f}\n")
        f.write(f"Average features: {overall_metrics['avg_features_selected']:.1f}\n")
        f.write(f"Feature reduction: {overall_metrics['feature_reduction_ratio']:.1%}\n\n")
        
        f.write("R² BY WINDOW:\n")
        train_r2_values = overall_metrics['train_r2_by_window']
        test_r2_values = overall_metrics['test_r2_by_window']
        for i, (train_r2, test_r2) in enumerate(zip(train_r2_values, test_r2_values)):
            f.write(f"Window {i+1:2d}: Train R² = {train_r2:.4f} | Test R² = {test_r2:.4f}\n")
        
        f.write(f"\nCONFIGURATION:\n")
        f.write(f"Training years: {Config.TRAIN_YEARS}\n")
        f.write(f"Validation years: {Config.VAL_YEARS}\n")
        f.write(f"Test years: {Config.TEST_YEARS}\n")
        f.write(f"Architecture: {Config.HIDDEN_LAYERS}\n")
        f.write(f"Ensemble size: {Config.DEFAULT_N_ESTIMATORS}\n")
        f.write(f"Min features: {Config.MIN_FEATURES}\n")
        f.write(f"Max features: {Config.MAX_FEATURES}\n")
        f.write(f"Hyperparameter tuning: {Config.ENABLE_HYPERPARAMETER_TUNING}\n")
    
    print(f"✓ Overall results saved: {os.path.join(output_folder, 'overall_results.txt')}")

def _generate_final_plots(results: dict, plotter):
    """Generate all final plots."""
    
    # Extract R² values for plotting
    train_r2_values = results['overall_metrics']['train_r2_by_window']
    test_r2_values = results['overall_metrics']['test_r2_by_window']
    
    # Generate summary plots with R² bar charts
    plotter.plot_summary_at_end(
        results['training_history'], 
        results['metrics_by_window'],
        train_r2_values,
        test_r2_values
    )
    
    # Plot predictions vs actual values
    plotter.plot_predictions_vs_actual(
        results['dates'], 
        results['actual_values'], 
        results['predictions']
    )
    
    print(f"✓ All plots saved to: {plotter.plots_folder}")

def _print_final_summary(results: dict, output_folder: str):
    """Print final summary."""
    overall_metrics = results['overall_metrics']
    
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total windows processed: {overall_metrics['n_windows']}")
    print(f"Overall Test R²: {overall_metrics['overall_r2']:.4f}")
    print(f"Overall Train R²: {overall_metrics['avg_train_r2']:.4f}")
    print(f"Overall RMSE: {overall_metrics['overall_rmse']:.6f}")
    print(f"Average features per window: {overall_metrics['avg_features_selected']:.1f}")
    print(f"Average epochs per window: {overall_metrics['avg_epochs']:.1f}")
    print(f"")
    print(f"Best Test R² Window: {max(overall_metrics['test_r2_by_window']):.4f}")
    print(f"Worst Test R² Window: {min(overall_metrics['test_r2_by_window']):.4f}")
    print(f"R² Standard Deviation: {np.std(overall_metrics['test_r2_by_window']):.4f}")
    print(f"")
    print(f"Results directory: {output_folder}")
    print(f"Plots directory: {os.path.join(output_folder, 'plots')}")

if __name__ == "__main__":
    import numpy as np  # Import for final summary calculations
    main()