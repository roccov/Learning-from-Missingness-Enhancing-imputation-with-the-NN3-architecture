# =============================================================================
# PYTORCH ENSEMBLE MODEL ONLY
# =============================================================================

from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PyTorchEnsembleRegressor:
    """PyTorch ensemble regressor - simplified version."""
    
    def __init__(self,
                 input_dim: int,
                 n_estimators: int = 1,
                 hidden_units: list = [32, 16, 8],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 1e-3,
                 batch_size: int = 512,
                 epochs: int = 100,
                 patience: int = 20,
                 weight_decay: float = 1e-5,
                 l1_lambda: float = 1e-5,
                 device: str = 'auto',
                 verbose: int = 1):
        
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.verbose = verbose
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.models = []
        self.training_histories = []
        
        if self.verbose:
            print(f"PyTorch Ensemble initialized - Device: {self.device}")
            print(f"Architecture: {self.hidden_units} | Estimators: {self.n_estimators}")
    
    def _build_model(self) -> nn.Module:
        """Build a single PyTorch model with fixed 3-layer architecture."""
        layers = []
        prev_size = self.input_dim
        
        # Build exactly 3 hidden layers
        for i, hidden_size in enumerate(self.hidden_units):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        return model.to(self.device)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray):
        """Train the ensemble."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        self.models = []
        self.training_histories = []
        
        for i in range(self.n_estimators):
            print(f"Training ensemble model {i+1}/{self.n_estimators}")
            
            # Build model
            model = self._build_model()
            
            # Setup optimizer and criterion
            optimizer = optim.Adam(model.parameters(), 
                                lr=self.learning_rate, 
                                weight_decay=self.weight_decay)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            
            # Training history
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Training loop
            for epoch in range(self.epochs):
                # Training phase
                model.train()
                epoch_train_loss = 0.0
                train_count = 0
                
                for batch_X, batch_y in train_loader:
                    try:
                        # Check inputs for NaN/Inf
                        if torch.any(torch.isnan(batch_X)) or torch.any(torch.isinf(batch_X)):
                            print(f"Warning: Training batch contains NaN/Inf, skipping...")
                            continue
                        if torch.any(torch.isnan(batch_y)) or torch.any(torch.isinf(batch_y)):
                            print(f"Warning: Training targets contain NaN/Inf, skipping...")
                            continue
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        
                        # Handle shape compatibility
                        if outputs.dim() == 0:
                            outputs = outputs.unsqueeze(0)
                        if batch_y.dim() == 0:
                            batch_y = batch_y.unsqueeze(0)
                        
                        # Check outputs for NaN/Inf
                        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                            print(f"Warning: Model outputs contain NaN/Inf, skipping batch...")
                            continue
                        
                        loss = criterion(outputs, batch_y)

                        # L1 penalty ONLY on first layer
                        l1_penalty = 0
                        for name, param in model.named_parameters():
                            if name == '0.weight':  # First layer weights only
                                l1_penalty += torch.sum(torch.abs(param))
                                break  # Only need the first layer
                        
                        total_loss = loss + self.l1_lambda * l1_penalty

                        # Check loss for NaN/Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: Loss is NaN/Inf, skipping batch...")
                            continue
                        
                        total_loss.backward()
                        
                        # More aggressive gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        
                        # Check gradients for NaN/Inf
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"Warning: Gradient norm is NaN/Inf, skipping update...")
                            continue
                        
                        optimizer.step()
                        
                        epoch_train_loss += loss.item()
                        train_count += 1
                        
                    except Exception as e:
                        print(f"Error in training batch: {e}")
                        continue
                
                # Validation phase
                model.eval()
                epoch_val_loss = 0.0
                val_count = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X).squeeze()
                        
                        if outputs.dim() == 0:
                            outputs = outputs.unsqueeze(0)
                        if batch_y.dim() == 0:
                            batch_y = batch_y.unsqueeze(0)
                        
                        loss = criterion(outputs, batch_y)
                        epoch_val_loss += loss.item()
                        val_count += 1
                
                # Calculate average losses
                avg_train_loss = epoch_train_loss / train_count if train_count > 0 else 0
                avg_val_loss = epoch_val_loss / val_count if val_count > 0 else float('inf')
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            self.models.append(model)
            self.training_histories.append({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            })
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble average."""
        # Clean input
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                try:
                    pred = model(X_tensor).squeeze().cpu().numpy()
                    
                    # Check for NaN in predictions
                    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if pred.ndim == 0:
                        pred = np.array([pred])
                    predictions.append(pred)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error in model prediction: {e}")
                    # Create zero prediction as fallback
                    zero_pred = np.zeros(len(X))
                    predictions.append(zero_pred)
        
        if not predictions:
            return np.zeros(len(X))
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return ensemble_pred
    
    def get_average_training_history(self) -> Dict[str, List[float]]:
        """Get averaged training history across all ensemble members."""
        if not self.training_histories:
            return {'train_losses': [], 'val_losses': []}
        
        # Get minimum epochs across all models
        min_epochs = min(len(h['train_losses']) for h in self.training_histories)
        
        # Average the losses
        avg_train_losses = []
        avg_val_losses = []
        
        for epoch in range(min_epochs):
            train_loss_epoch = np.mean([h['train_losses'][epoch] for h in self.training_histories])
            val_loss_epoch = np.mean([h['val_losses'][epoch] for h in self.training_histories])
            avg_train_losses.append(train_loss_epoch)
            avg_val_losses.append(val_loss_epoch)
        
        return {
            'train_losses': avg_train_losses,
            'val_losses': avg_val_losses
        }