# =============================================================================
# DATA PREPROCESSING - EXCLUDE MASK/FLAG FROM WINSORIZATION/NORMALIZATION
# =============================================================================

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler
from config import Config

class DataPreprocessor:
    """Handles data preprocessing - excludes mask/flag features from winsorization/normalization."""
    
    def __init__(self):
        self.winsorization_limits = {}
        self.scaler = None
    
    def prepare_features(self, df: pd.DataFrame) -> List[str]:
        """Prepare feature list, excluding only target and date columns."""
        exclude_cols = [Config.TARGET_COLUMN, Config.DATE_COLUMN]
        
        # Include ALL features except target and date (including mask/flag features)
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Using {len(feature_columns)} features (down from {len(df.columns)} total columns)")
        
        return feature_columns
    
    def winsorize_data(self, X: pd.DataFrame, y: pd.Series, 
                      is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply winsorization - FIT on training data, APPLY to all splits. Exclude mask/flag features."""
        
        # Separate mask/flag features from regular features
        mask_flag_features = [col for col in X.columns if 
                             col.endswith('_mask') or col.endswith('_flag')]
        
        # Identify other protected features (don't winsorize timestamps, IDs, etc.)
        protected_features = [col for col in X.columns if 
                            any(keyword in col for keyword in 
                                ['decimal_year', 'timestamp', 'days_since', 'permno']) or
                            col in mask_flag_features]
        
        regular_features = [col for col in X.columns if col not in protected_features]
        
        X_processed = X.copy()
        
        if is_training:
            # FIT: Calculate and save limits using TRAINING data only
            if regular_features:
                self.winsorization_limits['train_percentiles_1'] = X[regular_features].quantile(0.01)
                self.winsorization_limits['train_percentiles_99'] = X[regular_features].quantile(0.99)
                self.winsorization_limits['regular_features'] = regular_features
                
                print(f"  → Fitted winsorization on {len(regular_features)} regular features (excluding {len(mask_flag_features)} mask/flag features)")
                
                # Apply winsorization to training data
                X_processed[regular_features] = pd.DataFrame(
                    mstats.winsorize(X[regular_features].values, limits=[0.01, 0.01], axis=0),
                    columns=regular_features,
                    index=X.index
                )
            
            # Winsorize target (training data)
            y_processed = pd.Series(
                mstats.winsorize(y.values, limits=[0.01, 0.01]),
                index=y.index
            )
            self.winsorization_limits['y_train_p1'] = np.percentile(y.values, 1)
            self.winsorization_limits['y_train_p99'] = np.percentile(y.values, 99)
            
        else:
            # APPLY: Use training data limits on validation/test data
            if regular_features and self.winsorization_limits.get('regular_features'):
                for col in regular_features:
                    if col in self.winsorization_limits['regular_features']:
                        X_processed[col] = np.clip(
                            X[col].values,
                            self.winsorization_limits['train_percentiles_1'][col],
                            self.winsorization_limits['train_percentiles_99'][col]
                        )
                
                print(f"  → Applied training winsorization limits to {len(regular_features)} regular features")
            
            # Apply training limits to target
            y_processed = pd.Series(
                np.clip(y.values, 
                       self.winsorization_limits['y_train_p1'], 
                       self.winsorization_limits['y_train_p99']),
                index=y.index
            )
        
        # Keep mask/flag features unchanged
        if mask_flag_features:
            print(f"  → Keeping {len(mask_flag_features)} mask/flag features unchanged")
        
        return X_processed, y_processed
    
    def normalize_data(self, X: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """Normalize features - FIT on training data, APPLY to all splits. Exclude mask/flag features."""
        
        # Separate mask/flag features from regular features
        mask_flag_features = [col for col in X.columns if 
                             col.endswith('_mask') or col.endswith('_flag')]
        
        regular_features = [col for col in X.columns if col not in mask_flag_features]
        
        if is_training:
            # FIT: Calculate normalization parameters using TRAINING data only (regular features)
            self.scaler = StandardScaler()
            self.regular_features = regular_features
            self.mask_flag_features = mask_flag_features
            
            if regular_features:
                X_regular_normalized = self.scaler.fit_transform(X[regular_features].values)
                print(f"  → Fitted normalization on {len(regular_features)} regular features (excluding {len(mask_flag_features)} mask/flag features)")
            else:
                X_regular_normalized = np.array([]).reshape(len(X), 0)
                
        else:
            # APPLY: Use training data normalization on validation/test data (regular features)
            if self.scaler is not None and regular_features:
                X_regular_normalized = self.scaler.transform(X[regular_features].values)
                print(f"  → Applied training normalization to {len(regular_features)} regular features")
            else:
                X_regular_normalized = X[regular_features].values if regular_features else np.array([]).reshape(len(X), 0)
        
        # Combine normalized regular features with unchanged mask/flag features
        if mask_flag_features:
            X_mask_flag = X[mask_flag_features].values
            X_normalized = np.hstack([X_regular_normalized, X_mask_flag])
            print(f"  → Combined {X_regular_normalized.shape[1]} normalized features with {X_mask_flag.shape[1]} unchanged mask/flag features")
        else:
            X_normalized = X_regular_normalized
        
        return X_normalized