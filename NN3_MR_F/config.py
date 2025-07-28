# =============================================================================
# CONFIGURATION FILE
# =============================================================================

class Config:
    """Configuration for model."""
    
    # Data parameters
    DATE_COLUMN = 'date'
    TARGET_COLUMN = 'Excess_ret_target'
    DATA_FILENAME = 'full_dataset_excess.csv'
    
    # Window parameters
    TRAIN_YEARS = 20
    VAL_YEARS = 5
    TEST_YEARS = 1
    
    # Feature selection
    MIN_FEATURES = 20
    MAX_FEATURES = 50
    FEATURE_SELECTION_FREQUENCY = 1
    
    # Hyperparameter tuning - SIMPLIFIED GRID SEARCH
    TUNE_FREQUENCY = 1
    ENABLE_HYPERPARAMETER_TUNING = True
    
    # Fixed architecture - 3 hidden layers always
    HIDDEN_LAYERS = [32, 16, 8]  # Fixed architecture
    
    # PyTorch ensemble defaults
    DEFAULT_N_ESTIMATORS = 1
    DEFAULT_DROPOUT_RATE = 0.15
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_EPOCHS = 100
    DEFAULT_PATIENCE = 20
    DEFAULT_WEIGHT_DECAY = 1e-5
    
    # Grid search parameters (only 2-3 hyperparameters)
    GRID_SEARCH_PARAMS = {
        'dropout_rate': [0.05, 0.1, 0.15]
    }
    
    # Output
    OUTPUT_FOLDER = 'results'