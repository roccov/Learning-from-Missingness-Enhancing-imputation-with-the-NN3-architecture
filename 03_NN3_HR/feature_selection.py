# =============================================================================
# FEATURE SELECTION - NO MASK/FLAG FEATURES
# =============================================================================

from typing import List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestFeatureSelector:
    """Feature selector using Random Forest importance - no mask/flag features."""
    
    def __init__(self, min_features: int = 50, max_features: Optional[int] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str]) -> List[str]:
        """Select features using Random Forest importance."""
        
        print(f"Total features available: {len(feature_names)}")
        
        # Clean the data
        print(f"Cleaning data - original shape: {X.shape}")
        
        # Check for and report problematic values
        n_inf = np.sum(np.isinf(X))
        n_nan = np.sum(np.isnan(X))
        n_large = np.sum(np.abs(X) > 1e10)
        
        if n_inf > 0 or n_nan > 0 or n_large > 0:
            print(f"Found problematic values: {n_inf} inf, {n_nan} nan, {n_large} very large")
        
        # Clean the data
        X_clean = np.nan_to_num(X, 
                               nan=0.0,           # Replace NaN with 0
                               posinf=1e6,        # Replace +inf with large but finite number
                               neginf=-1e6)       # Replace -inf with large negative number
        
        # Clip extremely large values
        X_clean = np.clip(X_clean, -1e6, 1e6)
        print(f"Data cleaned successfully")
        
        # Set RF parameters based on sample size
        n_samples = X_clean.shape[0]
        min_split = max(2, int(0.01 * n_samples))
        min_leaf = max(1, int(0.005 * n_samples))
        
        self.rf_model.set_params(
            min_samples_split=min_split,
            min_samples_leaf=min_leaf
        )
        print(f"→ Using min_samples_split={min_split}, "
              f"min_samples_leaf={min_leaf} for {n_samples} rows")
        
        # Fit Random Forest
        self.rf_model.fit(X_clean, y)
        
        # Get feature importances and rank features
        importances = self.rf_model.feature_importances_
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Determine optimal number of features through TIME-BASED validation
        print(f"Finding optimal number of features between {self.min_features} and {self.max_features}...")
        print("Using time-based validation to respect panel structure...")
        
        max_possible_features = min(self.max_features or len(feature_names), len(feature_names))
        min_required_features = min(self.min_features, len(feature_names))
        
        best_score = float('inf')
        best_n_features = min_required_features
        
        # Split data temporally: first 80% for train, last 20% for validation
        n_samples = X_clean.shape[0]
        split_idx = int(0.8 * n_samples)
        
        X_train_split = X_clean[:split_idx]
        y_train_split = y[:split_idx]
        X_val_split = X_clean[split_idx:]
        y_val_split = y[split_idx:]
        
        print(f"Time-based split: {split_idx} training samples, {n_samples - split_idx} validation samples")
        
        # Test different numbers of features from min to max
        for n_features in range(min_required_features, max_possible_features + 1, 5):  # Step by 5 for efficiency
            # Select top n features
            current_features = [pair[0] for pair in feature_importance_pairs[:n_features]]
            current_indices = [feature_names.index(f) for f in current_features]
            
            X_train_subset = X_train_split[:, current_indices]
            X_val_subset = X_val_split[:, current_indices]
            
            # Train Random Forest on training split
            temp_rf = RandomForestRegressor(
                n_estimators=50,  # Smaller for speed
                max_depth=10,
                min_samples_split=max(2, int(0.01 * len(X_train_subset))),
                min_samples_leaf=max(1, int(0.005 * len(X_train_subset))),
                random_state=42,
                n_jobs=-1
            )
            
            try:
                temp_rf.fit(X_train_subset, y_train_split)
                
                # Predict on validation split (different time period)
                val_pred = temp_rf.predict(X_val_subset)
                val_r2 = 1 - np.mean((y_val_split - val_pred) ** 2) / np.var(y_val_split)
                mse_score = 1 - val_r2  # Convert to MSE-like score (lower is better)
                
                print(f"  {n_features} features: Time-based R² = {val_r2:.4f}")
                
                if mse_score < best_score:
                    best_score = mse_score
                    best_n_features = n_features
                    
            except Exception as e:
                print(f"  {n_features} features: Failed ({e})")
                continue
        
        print(f"Optimal number of features: {best_n_features} (Time-based R² = {1-best_score:.4f})")
        
        # Select top features based on optimal number
        selected_features = [pair[0] for pair in feature_importance_pairs[:best_n_features]]
        
        print(f"Selected {len(selected_features)} optimal features from Random Forest")
        print(f"Feature reduction: {len(feature_names)} → {len(selected_features)} "
              f"({len(selected_features)/len(feature_names):.1%})")
        
        return selected_features