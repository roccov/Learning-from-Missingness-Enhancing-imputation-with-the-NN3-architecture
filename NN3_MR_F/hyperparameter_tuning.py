# =============================================================================
# HYPERPARAMETER TUNING - GRID SEARCH
# =============================================================================

from typing import Dict, Any
import numpy as np
from itertools import product
from models import PyTorchEnsembleRegressor
from config import Config

class SimpleHyperparameterOptimizer:
    """Simple grid search hyperparameter optimizer for PyTorch ensemble only."""
    
    def __init__(self, device):
        self.device = device
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Perform grid search on 2-3 hyperparameters only."""
        
        print("Starting grid search hyperparameter optimization...")
        
        # Get parameter grid
        param_grid = Config.GRID_SEARCH_PARAMS
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        param_combinations = list(product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations:")
        for name in param_names:
            print(f"  {name}: {param_grid[name]}")
        
        best_params = None
        best_score = float('inf')
        
        for i, combination in enumerate(param_combinations):
            # Create parameter dictionary
            current_params = dict(zip(param_names, combination))
            
            print(f"\nCombination {i+1}/{len(param_combinations)}: {current_params}")
            
            try:
                # Create model with current parameters
                model = PyTorchEnsembleRegressor(
                    input_dim=X_train.shape[1],
                    n_estimators=current_params.get('n_estimators', Config.DEFAULT_N_ESTIMATORS),
                    hidden_units=Config.HIDDEN_LAYERS,  # Fixed architecture
                    dropout_rate=current_params.get('dropout_rate', Config.DEFAULT_DROPOUT_RATE),
                    learning_rate=current_params.get('learning_rate', Config.DEFAULT_LEARNING_RATE),
                    batch_size=Config.DEFAULT_BATCH_SIZE,
                    epochs=Config.DEFAULT_EPOCHS,
                    patience=Config.DEFAULT_PATIENCE,
                    weight_decay=Config.DEFAULT_WEIGHT_DECAY,
                    l1_lambda=current_params.get('l1_lambda', 1e-5),
                    device=self.device,
                    verbose=0  # Reduce output during grid search
                )
                
                # Train model
                model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_mse = np.mean((y_val - val_predictions) ** 2)
                
                print(f"  Validation MSE: {val_mse:.6f}")
                
                # Update best parameters
                if val_mse < best_score:
                    best_score = val_mse
                    best_params = current_params.copy()
                    print(f"  *** New best score: {val_mse:.6f}")
                
            except Exception as e:
                print(f"  Error with parameters {current_params}: {e}")
                continue
        
        # Add fixed parameters to best_params
        final_params = {
            'n_estimators': best_params.get('n_estimators', Config.DEFAULT_N_ESTIMATORS),
            'hidden_units': Config.HIDDEN_LAYERS,
            'dropout_rate': best_params.get('dropout_rate', Config.DEFAULT_DROPOUT_RATE),
            'learning_rate': best_params.get('learning_rate', Config.DEFAULT_LEARNING_RATE),
            'batch_size': Config.DEFAULT_BATCH_SIZE,
            'epochs': Config.DEFAULT_EPOCHS,
            'patience': Config.DEFAULT_PATIENCE,
            'weight_decay': Config.DEFAULT_WEIGHT_DECAY
        }
        
        print(f"\nGrid search completed!")
        print(f"Best validation MSE: {best_score:.6f}")
        print(f"Best parameters: {best_params}")
        
        return final_params