"""Bibliothèque Machine Learning intégrée pour ai'lang."""

from .models import (
    NeuralNetwork,
    LinearRegression,
    LogisticRegression,
    DecisionTree,
    RandomForest,
    SVM,
    KMeans,
    create_model
)

from .layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    LSTM,
    GRU,
    Embedding
)

from .optimizers import (
    SGD,
    Adam,
    RMSprop,
    AdaGrad
)

from .losses import (
    MeanSquaredError,
    CategoricalCrossentropy,
    BinaryCrossentropy,
    SparseCategoricalCrossentropy
)

from .metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MeanAbsoluteError
)

from .utils import (
    train_test_split,
    cross_validate,
    grid_search,
    save_model,
    load_model
)

__version__ = "1.0.0"
__author__ = "AI'Lang ML Team"

__all__ = [
    # Models
    'NeuralNetwork', 'LinearRegression', 'LogisticRegression',
    'DecisionTree', 'RandomForest', 'SVM', 'KMeans', 'create_model',
    
    # Layers
    'Dense', 'Conv2D', 'MaxPooling2D', 'Dropout', 'BatchNormalization',
    'LSTM', 'GRU', 'Embedding',
    
    # Optimizers
    'SGD', 'Adam', 'RMSprop', 'AdaGrad',
    
    # Losses
    'MeanSquaredError', 'CategoricalCrossentropy', 'BinaryCrossentropy',
    'SparseCategoricalCrossentropy',
    
    # Metrics
    'Accuracy', 'Precision', 'Recall', 'F1Score', 'MeanAbsoluteError',
    
    # Utils
    'train_test_split', 'cross_validate', 'grid_search', 'save_model', 'load_model'
]