# =============================================================================
# DATA PREPROCESSING - NO MASK/FLAG FEATURES
# =============================================================================

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler
from config import Config

class DataPreprocessor:
    """Handles data preprocessing - excludes mask/flag features entirely."""
    
    def __init__(self):
        self.winsorization_limits = {}
        self.scaler = None
    
    def prepare_features(self, df: pd.DataFrame) -> List[str]:
        """Prepare feature list, excluding target, date, and mask/flag columns."""
        exclude_cols = [Config.TARGET_COLUMN, Config.DATE_COLUMN]
        
        # Exclude mask and flag columns entirely
        feature_columns = [col for col in df.columns if 
                          col not in exclude_cols and
                          not any(keyword in col.lower() for keyword in ['_mask', '_flag'])]
        
        excluded_mask_flags = [col for col in df.columns if 
                              any(keyword in col.lower() for keyword in ['_mask', '_flag'])]
        
        print(f"Excluded {len(excluded_mask_flags)} mask/flag features")
        print(f"Using {len(feature_columns)} features (down from {len(df.columns)} total columns)")
        
        return feature_columns
    
    def winsorize_data(self, X: pd.DataFrame, y: pd.Series, 
                      is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply winsorization - FIT on training data, APPLY to all splits."""
        # Identify protected features (don't winsorize timestamps, IDs, etc.)
        protected_features = [col for col in X.columns if 
                            any(keyword in col for keyword in 
                                ['decimal_year', 'timestamp', 'days_since', 'permno'])]
        
        regular_features = [col for col in X.columns if col not in protected_features]
        
        X_processed = X.copy()
        
        if is_training:
            # FIT: Calculate and save limits using TRAINING data only
            if regular_features:
                self.winsorization_limits['train_percentiles_1'] = X[regular_features].quantile(0.01)
                self.winsorization_limits['train_percentiles_99'] = X[regular_features].quantile(0.99)
                self.winsorization_limits['regular_features'] = regular_features
                
                print(f"  → Fitted winsorization on {len(regular_features)} features using training data")
                
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
                
                print(f"  → Applied training winsorization limits to {len(regular_features)} features")
            
            # Apply training limits to target
            y_processed = pd.Series(
                np.clip(y.values, 
                       self.winsorization_limits['y_train_p1'], 
                       self.winsorization_limits['y_train_p99']),
                index=y.index
            )
        
        return X_processed, y_processed
    
    def normalize_data(self, X: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """Normalize features - FIT on training data, APPLY to all splits."""
        
        if is_training:
            # FIT: Calculate normalization parameters using TRAINING data only
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X.values)
            print(f"  → Fitted normalization on {X.shape[1]} features using training data")
        else:
            # APPLY: Use training data normalization on validation/test data
            if self.scaler is not None:
                X_normalized = self.scaler.transform(X.values)
                print(f"  → Applied training normalization to {X.shape[1]} features")
            else:
                X_normalized = X.values
        
        return X_normalized