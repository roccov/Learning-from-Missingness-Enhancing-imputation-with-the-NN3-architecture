# =============================================================================
# SIMPLIFIED SLIDING WINDOW PIPELINE - PYTORCH ENSEMBLE ONLY
# =============================================================================

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import Config
from models import PyTorchEnsembleRegressor
from feature_selection import RandomForestFeatureSelector
from preprocessing import DataPreprocessor
from hyperparameter_tuning import SimpleHyperparameterOptimizer

class SimplifiedSlidingWindowPipeline:
    """Simplified pipeline for PyTorch ensemble only."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_selector = RandomForestFeatureSelector(
            min_features=Config.MIN_FEATURES,
            max_features=Config.MAX_FEATURES
        )
        self.preprocessor = DataPreprocessor()
        self.optimizer = SimpleHyperparameterOptimizer(self.device)
        
        # Current state
        self.current_selected_features = []
        self.best_params = self._get_default_params()
        
        print(f"Using device: {self.device}")
        print(f"Fixed architecture: {Config.HIDDEN_LAYERS}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': Config.DEFAULT_N_ESTIMATORS,
            'hidden_units': Config.HIDDEN_LAYERS,
            'dropout_rate': Config.DEFAULT_DROPOUT_RATE,
            'learning_rate': Config.DEFAULT_LEARNING_RATE,
            'batch_size': Config.DEFAULT_BATCH_SIZE,
            'epochs': Config.DEFAULT_EPOCHS,
            'patience': Config.DEFAULT_PATIENCE,
            'weight_decay': Config.DEFAULT_WEIGHT_DECAY
        }
    
    def run(self, df: pd.DataFrame, plotter: Optional[object] = None) -> Dict[str, Any]:
        """Run the simplified sliding window pipeline."""
        # Prepare data
        df = df.copy()
        df[Config.DATE_COLUMN] = pd.to_datetime(df[Config.DATE_COLUMN])
        df_sorted = df.sort_values(by=Config.DATE_COLUMN)
        
        # Get feature columns
        feature_columns = self.preprocessor.prepare_features(df_sorted)
        self.current_selected_features = feature_columns.copy()
        
        # Initialize results storage
        results = {
            'predictions': [],
            'actual_values': [],
            'metrics_by_window': [],
            'feature_selection_evolution': [],
            'dates': [],
            'training_history': [],
            'train_r2_by_window': [],  
            'test_r2_by_window': []    
        }
        
        # Setup sliding window
        min_date = df_sorted[Config.DATE_COLUMN].min()
        max_date = df_sorted[Config.DATE_COLUMN].max()
        current_start = min_date
        window_count = 0
        
        print(f"Period: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
        print(f"Training window: {Config.TRAIN_YEARS} years")
        print(f"Validation window: {Config.VAL_YEARS} years") 
        print(f"Test window: {Config.TEST_YEARS} years")
        
        # Main sliding window loop
        while True:
            # Define window periods with proper year boundaries
            # Train: current_start to start of next year (exclusive)
            train_start_year = current_start.year
            train_end_year = train_start_year + Config.TRAIN_YEARS
            train_end = pd.Timestamp(f'{train_end_year}-01-01')
            
            # Val: train_end to start of test year (exclusive)
            val_start = train_end
            val_end_year = train_end_year + Config.VAL_YEARS
            val_end = pd.Timestamp(f'{val_end_year}-01-01')
            
            # Test: val_end to start of next year (exclusive)
            test_start = val_end
            test_end_year = val_end_year + Config.TEST_YEARS
            test_end = pd.Timestamp(f'{test_end_year}-01-01')
            
            if test_end > max_date:
                break
            
            # Split data into train/val/test with < boundaries
            train_mask = ((df_sorted[Config.DATE_COLUMN] >= current_start) & 
                         (df_sorted[Config.DATE_COLUMN] < train_end))
            val_mask = ((df_sorted[Config.DATE_COLUMN] >= val_start) & 
                       (df_sorted[Config.DATE_COLUMN] < val_end))
            test_mask = ((df_sorted[Config.DATE_COLUMN] >= test_start) & 
                        (df_sorted[Config.DATE_COLUMN] < test_end))
            
            train_data = df_sorted[train_mask]
            val_data = df_sorted[val_mask]
            test_data = df_sorted[test_mask]
            
            # Skip if insufficient data
            if len(train_data) < 100 or len(val_data) < 50 or len(test_data) < 10:
                current_start = pd.Timestamp(f'{train_start_year + 1}-01-01')
                continue
            
            print(f"\n=== Window {window_count + 1} ===")
            print(f"Train: {current_start.strftime('%Y-%m')} to {(train_end - pd.Timedelta(days=1)).strftime('%Y-%m')} ({len(train_data)} samples)")
            print(f"Val:   {val_start.strftime('%Y-%m')} to {(val_end - pd.Timedelta(days=1)).strftime('%Y-%m')} ({len(val_data)} samples)")
            print(f"Test:  {test_start.strftime('%Y-%m')} to {(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')} ({len(test_data)} samples)")
            
            # Process this window
            window_results = self._process_window(
                train_data, val_data, test_data, feature_columns, window_count
            )
            
            # Store results
            results['predictions'].extend(window_results['predictions'])
            results['actual_values'].extend(window_results['actual_values'])
            results['metrics_by_window'].append(window_results['metrics'])
            results['feature_selection_evolution'].append(window_results['feature_info'])
            results['dates'].extend(test_data[Config.DATE_COLUMN].tolist())
            results['training_history'].append(window_results['training_info'])
            
            # Store R2 values for plotting
            results['train_r2_by_window'].append(window_results['metrics']['train_r2'])
            results['test_r2_by_window'].append(window_results['metrics']['r2'])
            
            # Store permno if it exists
            if 'permno' in test_data.columns:
                if 'permnos' not in results:
                    results['permnos'] = []
                results['permnos'].extend(test_data['permno'].tolist())
            else:
                if 'permnos' not in results:
                    results['permnos'] = []
                results['permnos'].extend([None] * len(test_data))
            
            # Print window results immediately
            train_r2 = window_results['metrics']['train_r2']
            test_r2 = window_results['metrics']['r2']
            print(f"*** R² TRAIN: {train_r2:.4f} | R² TEST: {test_r2:.4f} ***")
            print(f"RMSE: {window_results['metrics']['rmse']:.6f} | Features: {len(self.current_selected_features)}")
            
            # Immediate plotting if available
            if plotter is not None:
                try:
                    plotter.plot_individual_window_immediately(
                        window_count + 1, 
                        window_results['training_info'], 
                        window_results['metrics']
                    )
                except Exception as e:
                    print(f"Warning: Could not generate plot for window {window_count + 1}: {e}")
            
            current_start = pd.Timestamp(f'{train_start_year + 1}-01-01')
            window_count += 1
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results, window_count, len(feature_columns))
        results['overall_metrics'] = overall_metrics
        
        self._print_final_results(overall_metrics, results['train_r2_by_window'], results['test_r2_by_window'])
        
        return results
    
    def _process_window(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                       test_data: pd.DataFrame, feature_columns: List[str], 
                       window_count: int) -> Dict[str, Any]:
        """Process a single window with CORRECT preprocessing order."""
        
        # Prepare raw data for all splits
        X_train_raw = train_data[feature_columns].fillna(train_data[feature_columns].median())
        y_train_raw = train_data[Config.TARGET_COLUMN]
        
        X_val_raw = val_data[feature_columns].fillna(X_train_raw.median())
        y_val_raw = val_data[Config.TARGET_COLUMN]
        
        X_test_raw = test_data[feature_columns].fillna(X_train_raw.median())
        y_test_raw = test_data[Config.TARGET_COLUMN]
        
        # STEP 1: Winsorization - FIT on train, APPLY to all
        print("Applying winsorization (fit on train, apply to all)...")
        X_train_winsorized, y_train_winsorized = self.preprocessor.winsorize_data(
            X_train_raw, y_train_raw, is_training=True
        )
        
        X_val_winsorized, y_val_winsorized = self.preprocessor.winsorize_data(
            X_val_raw, y_val_raw, is_training=False
        )
        
        X_test_winsorized, y_test_winsorized = self.preprocessor.winsorize_data(
            X_test_raw, y_test_raw, is_training=False
        )
        
        # STEP 2: Normalization - FIT on train, APPLY to all
        print("Applying normalization (fit on train, apply to all)...")
        X_train_normalized = self.preprocessor.normalize_data(
            X_train_winsorized, is_training=True
        )
        
        X_val_normalized = self.preprocessor.normalize_data(
            X_val_winsorized, is_training=False
        )
        
        X_test_normalized = self.preprocessor.normalize_data(
            X_test_winsorized, is_training=False
        )
        
        # STEP 3: Feature selection on PREPROCESSED data (if needed)
        if window_count % Config.FEATURE_SELECTION_FREQUENCY == 0:
            print("Running feature selection on preprocessed data...")
            
            self.current_selected_features = self.feature_selector.select_features(
                X_train_normalized, y_train_winsorized.values, feature_columns
            )
            print(f"Selected {len(self.current_selected_features)} features")
            
            # Get feature indices for subsetting
            feature_indices = [i for i, feature in enumerate(feature_columns) 
                             if feature in self.current_selected_features]
            
            # Subset all data to selected features
            X_train_final = X_train_normalized[:, feature_indices]
            X_val_final = X_val_normalized[:, feature_indices]
            X_test_final = X_test_normalized[:, feature_indices]
            
        else:
            # Use previously selected features
            print(f"Using previously selected {len(self.current_selected_features)} features")
            
            # Get feature indices for subsetting
            feature_indices = [i for i, feature in enumerate(feature_columns) 
                             if feature in self.current_selected_features]
            
            # Subset all data to selected features
            X_train_final = X_train_normalized[:, feature_indices]
            X_val_final = X_val_normalized[:, feature_indices]
            X_test_final = X_test_normalized[:, feature_indices]
        
        # Hyperparameter tuning (if needed)
        if (Config.ENABLE_HYPERPARAMETER_TUNING and 
            window_count % Config.TUNE_FREQUENCY == 0):
            print("Running grid search hyperparameter optimization...")
            self.best_params = self.optimizer.optimize_hyperparameters(
                X_train_final, y_train_winsorized.values,
                X_val_final, y_val_winsorized.values
            )
            print("Grid search complete")
        
        # Train final model
        model_results = self._train_final_model(
            X_train_final, y_train_winsorized.values,
            X_val_final, y_val_winsorized.values,
            X_test_final, y_test_winsorized.values
        )
        
        # Calculate metrics
        metrics = self._calculate_window_metrics(
            y_test_winsorized.values, model_results['predictions'],
            model_results['train_r2'],
            train_data, val_data, test_data, window_count
        )
        
        return {
            'predictions': model_results['predictions'],
            'actual_values': y_test_winsorized.values,
            'metrics': metrics,
            'feature_info': {
                'window': window_count + 1,
                'selected_features': self.current_selected_features.copy(),
                'n_selected': len(self.current_selected_features),
                'n_total': len(feature_columns)  # Total available features
            },
            'training_info': model_results['training_info']
        }
    
    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train the PyTorch ensemble model."""
        
        print("Training PyTorch ensemble model...")
        
        # Initialize PyTorch ensemble
        model = PyTorchEnsembleRegressor(
            input_dim=X_train.shape[1],
            n_estimators=self.best_params['n_estimators'],
            hidden_units=self.best_params['hidden_units'],
            dropout_rate=self.best_params['dropout_rate'],
            learning_rate=self.best_params['learning_rate'],
            batch_size=self.best_params['batch_size'],
            epochs=self.best_params['epochs'],
            patience=self.best_params['patience'],
            weight_decay=self.best_params['weight_decay'],
            device=self.device,
            verbose=1
        )
        
        # Train the ensemble
        model.fit(X_train, y_train, X_val, y_val)
        
        # Make predictions
        test_predictions = model.predict(X_test)
        train_predictions = model.predict(X_train)
        
        # Calculate train R2
        train_r2 = r2_score(y_train, train_predictions)
        
        # Get training history
        avg_history = model.get_average_training_history()
        
        # Create training info
        training_info = {
            'final_epoch': len(avg_history['train_losses']) if avg_history['train_losses'] else self.best_params['epochs'],
            'best_val_loss': min(avg_history['val_losses']) if avg_history['val_losses'] else 0.0,
            'train_losses': avg_history['train_losses'],
            'val_losses': avg_history['val_losses']
        }
        
        return {
            'predictions': test_predictions.tolist(),
            'train_predictions': train_predictions.tolist(),
            'train_r2': train_r2,
            'training_info': training_info
        }
    
    def _calculate_window_metrics(self, y_true: np.ndarray, y_pred: List[float],
                                train_r2: float, train_data: pd.DataFrame, 
                                val_data: pd.DataFrame, test_data: pd.DataFrame, 
                                window_count: int) -> Dict[str, Any]:
        """Calculate metrics for a single window."""
        y_pred_array = np.array(y_pred)
        
        train_start = train_data[Config.DATE_COLUMN].min()
        train_end = train_data[Config.DATE_COLUMN].max()
        val_start = val_data[Config.DATE_COLUMN].min()
        val_end = val_data[Config.DATE_COLUMN].max()
        test_start = test_data[Config.DATE_COLUMN].min()
        test_end = test_data[Config.DATE_COLUMN].max()
        
        return {
            'window': window_count + 1,
            'train_period': f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
            'val_period': f"{val_start.strftime('%Y-%m')} to {val_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
            'mse': mean_squared_error(y_true, y_pred_array),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred_array)),
            'mae': mean_absolute_error(y_true, y_pred_array),
            'r2': r2_score(y_true, y_pred_array),
            'train_r2': train_r2,
            'n_train': len(train_data),
            'n_val': len(val_data),
            'n_test': len(test_data),
            'n_features_selected': len(self.current_selected_features)
        }
    
    def _calculate_overall_metrics(self, results: Dict[str, Any], 
                                 window_count: int, total_features: int) -> Dict[str, Any]:
        """Calculate overall metrics across all windows."""
        return {
            'overall_r2': r2_score(results['actual_values'], results['predictions']),
            'overall_rmse': np.sqrt(mean_squared_error(results['actual_values'], results['predictions'])),
            'overall_mae': mean_absolute_error(results['actual_values'], results['predictions']),
            'n_windows': window_count,
            'avg_window_r2': np.mean([m['r2'] for m in results['metrics_by_window']]),
            'avg_train_r2': np.mean([m['train_r2'] for m in results['metrics_by_window']]),
            'avg_epochs': np.mean([h['final_epoch'] for h in results['training_history']]),
            'avg_val_loss': np.mean([h['best_val_loss'] for h in results['training_history'] 
                                   if h['best_val_loss'] != float('inf')]),
            'avg_features_selected': np.mean([m['n_features_selected'] for m in results['metrics_by_window']]),
            'feature_reduction_ratio': 1 - (np.mean([m['n_features_selected'] for m in results['metrics_by_window']]) / total_features),
            'train_r2_by_window': results['train_r2_by_window'],
            'test_r2_by_window': results['test_r2_by_window']
        }
    
    def _print_final_results(self, overall_metrics: Dict[str, Any], 
                           train_r2_values: List[float], test_r2_values: List[float]):
        """Print final results summary."""
        print(f"\n" + "="*60)
        print(f"FINAL RESULTS - PYTORCH ENSEMBLE")
        print(f"="*60)
        print(f"Overall Test R²:     {overall_metrics['overall_r2']:.4f}")
        print(f"Overall Train R²:    {overall_metrics['avg_train_r2']:.4f}")
        print(f"Overall RMSE:        {overall_metrics['overall_rmse']:.4f}")
        print(f"Average Window Test R²:  {overall_metrics['avg_window_r2']:.4f}")
        print(f"Average Window Train R²: {overall_metrics['avg_train_r2']:.4f}")
        print(f"Number of windows:   {overall_metrics['n_windows']}")
        print(f"Average epochs:      {overall_metrics['avg_epochs']:.1f}")
        print(f"Average features:    {overall_metrics['avg_features_selected']:.1f}")
        print(f"Feature reduction:   {overall_metrics['feature_reduction_ratio']:.1%}")
        
        print(f"\nR² BY WINDOW:")
        for i, (train_r2, test_r2) in enumerate(zip(train_r2_values, test_r2_values)):
            print(f"  Window {i+1:2d}: Train R² = {train_r2:.4f} | Test R² = {test_r2:.4f}")
        
        print(f"="*60)