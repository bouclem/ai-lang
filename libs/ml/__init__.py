# Bibliothèque de Machine Learning intégrée à ai'lang
# Fournit des outils complets pour l'apprentissage automatique

# Version de la bibliothèque
__version__ = "1.0.0"
__author__ = "ai'lang Team"
__description__ = "Bibliothèque de Machine Learning native pour ai'lang"

# ============================================================================
# Importations des modules principaux
# ============================================================================

# Utilitaires de base
from .utils import (
    # Gestion des données
    train_test_split,
    cross_validation,
    normalize_minmax,
    normalize_zscore,
    normalize_robust,
    apply_normalization,
    load_dataset,
    generate_synthetic_data,
    bootstrap_sample,
    
    # Métriques d'évaluation
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    
    # Sauvegarde et chargement
    save_model,
    load_model,
    save_model_json,
    load_model_json,
    
    # Fonctions mathématiques
    euclidean_distance,
    cosine_similarity,
    sigmoid,
    softmax,
    one_hot_encode
)

# Couches de réseaux de neurones
from .layers import (
    Layer,
    Dense,
    Dropout,
    BatchNormalization,
    create_layer
)

# Optimiseurs
from .optimizers import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    AdaGrad,
    AdaDelta,
    create_optimizer,
    get_available_optimizers,
    get_optimizer_info
)

# Fonctions de perte
from .losses import (
    Loss,
    MeanSquaredError as MSELoss,
    MeanAbsoluteError as MAELoss,
    HuberLoss,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    Hinge,
    create_loss,
    get_available_losses,
    get_loss_info
)

# Métriques d'évaluation
from .metrics import (
    Metric,
    # Métriques de régression
    MeanSquaredError as MSEMetric,
    MeanAbsoluteError as MAEMetric,
    RootMeanSquaredError,
    R2Score,
    MeanAbsolutePercentageError,
    
    # Métriques de classification
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUC,
    
    # Métriques composées
    ClassificationReport,
    
    # Fonctions utilitaires
    create_metric,
    get_available_metrics,
    evaluate_model,
    
    # Alias
    MSE,
    MAE,
    RMSE,
    R2,
    MAPE
)

# ============================================================================
# Classes et fonctions principales
# ============================================================================

class MLPipeline:
    """
    Pipeline de machine learning pour enchaîner les étapes de traitement.
    """
    
    def __init__(self):
        self.steps = []
        self.fitted = False
    
    def add_step(self, name: str, transformer):
        """
        Ajoute une étape au pipeline.
        
        Args:
            name: Nom de l'étape
            transformer: Objet avec méthodes fit() et transform()
        """
        self.steps.append((name, transformer))
        return self
    
    def fit(self, X, y=None):
        """
        Entraîne toutes les étapes du pipeline.
        """
        current_X = X
        
        for name, transformer in self.steps:
            if hasattr(transformer, 'fit'):
                transformer.fit(current_X, y)
            if hasattr(transformer, 'transform'):
                current_X = transformer.transform(current_X)
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """
        Applique toutes les transformations du pipeline.
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        current_X = X
        for name, transformer in self.steps:
            if hasattr(transformer, 'transform'):
                current_X = transformer.transform(current_X)
        
        return current_X
    
    def fit_transform(self, X, y=None):
        """
        Entraîne et applique les transformations.
        """
        return self.fit(X, y).transform(X)
    
    def predict(self, X):
        """
        Fait des prédictions (si la dernière étape est un modèle).
        """
        transformed_X = self.transform(X)
        
        # La dernière étape doit être un modèle avec predict()
        if self.steps and hasattr(self.steps[-1][1], 'predict'):
            return self.steps[-1][1].predict(transformed_X)
        else:
            raise ValueError("Last step must be a model with predict() method")

class ModelValidator:
    """
    Validateur de modèles avec validation croisée.
    """
    
    def __init__(self, model, cv_folds: int = 5):
        self.model = model
        self.cv_folds = cv_folds
    
    def validate(self, X, y, metrics=None):
        """
        Effectue une validation croisée.
        
        Args:
            X: Données d'entrée
            y: Étiquettes
            metrics: Liste des métriques à calculer
        
        Returns:
            Dictionnaire avec les résultats de validation
        """
        if metrics is None:
            metrics = ['accuracy'] if self._is_classification(y) else ['mse']
        
        results = {metric: [] for metric in metrics}
        
        # Validation croisée simple
        fold_size = len(X) // self.cv_folds
        
        for fold in range(self.cv_folds):
            # Division des données
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv_folds - 1 else len(X)
            
            # Données de test pour ce fold
            X_test_fold = X[start_idx:end_idx]
            y_test_fold = y[start_idx:end_idx]
            
            # Données d'entraînement (le reste)
            X_train_fold = X[:start_idx] + X[end_idx:]
            y_train_fold = y[:start_idx] + y[end_idx:]
            
            # Entraînement du modèle
            if hasattr(self.model, 'fit'):
                self.model.fit(X_train_fold, y_train_fold)
            
            # Prédictions
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X_test_fold)
            else:
                continue
            
            # Calcul des métriques
            for metric in metrics:
                if metric == 'accuracy':
                    score = accuracy_score(y_test_fold, y_pred)
                elif metric == 'mse':
                    score = mean_squared_error(y_test_fold, y_pred)
                elif metric == 'mae':
                    score = mean_absolute_error(y_test_fold, y_pred)
                elif metric == 'r2':
                    score = r2_score(y_test_fold, y_pred)
                else:
                    score = 0.0  # Métrique non supportée
                
                results[metric].append(score)
        
        # Calcul des statistiques
        final_results = {}
        for metric, scores in results.items():
            if scores:
                final_results[metric] = {
                    'mean': sum(scores) / len(scores),
                    'std': (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5,
                    'scores': scores
                }
        
        return final_results
    
    def _is_classification(self, y):
        """
        Détermine si c'est un problème de classification.
        """
        unique_values = len(set(y))
        return unique_values <= 10  # Seuil arbitraire

# ============================================================================
# Fonctions utilitaires globales
# ============================================================================

def create_pipeline(*steps):
    """
    Crée un pipeline ML avec les étapes spécifiées.
    
    Args:
        *steps: Tuples (nom, transformer) ou objets transformer
    
    Returns:
        Instance de MLPipeline
    """
    pipeline = MLPipeline()
    
    for i, step in enumerate(steps):
        if isinstance(step, tuple):
            name, transformer = step
        else:
            name = f"step_{i}"
            transformer = step
        
        pipeline.add_step(name, transformer)
    
    return pipeline

def quick_evaluate(model, X_train, y_train, X_test, y_test, task_type="auto"):
    """
    Évaluation rapide d'un modèle.
    
    Args:
        model: Modèle à évaluer
        X_train: Données d'entraînement
        y_train: Étiquettes d'entraînement
        X_test: Données de test
        y_test: Étiquettes de test
        task_type: Type de tâche ('classification', 'regression', 'auto')
    
    Returns:
        Dictionnaire avec les résultats d'évaluation
    """
    # Entraînement
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
    
    # Prédictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Model must have a predict() method")
    
    # Évaluation
    return evaluate_model(y_test, y_pred, task_type)

def get_ml_info():
    """
    Retourne des informations sur la bibliothèque ML.
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_layers": ["Dense", "Dropout", "BatchNormalization"],
        "available_optimizers": get_available_optimizers(),
        "available_losses": get_available_losses(),
        "available_metrics": get_available_metrics(),
        "features": [
            "Neural network layers",
            "Multiple optimizers",
            "Various loss functions",
            "Comprehensive metrics",
            "Data preprocessing utilities",
            "Model validation tools",
            "Pipeline support"
        ]
    }

# ============================================================================
# Exports principaux
# ============================================================================

__all__ = [
    # Classes principales
    "MLPipeline",
    "ModelValidator",
    
    # Couches
    "Layer", "Dense", "Dropout", "BatchNormalization",
    
    # Optimiseurs
    "Optimizer", "SGD", "Adam", "RMSprop", "AdaGrad", "AdaDelta",
    
    # Fonctions de perte
    "Loss", "MSELoss", "MAELoss", "HuberLoss", "BinaryCrossentropy",
    "CategoricalCrossentropy", "SparseCategoricalCrossentropy", "Hinge",
    
    # Métriques
    "Metric", "MSEMetric", "MAEMetric", "RootMeanSquaredError", "R2Score",
    "MeanAbsolutePercentageError", "Accuracy", "Precision", "Recall",
    "F1Score", "ConfusionMatrix", "AUC", "ClassificationReport",
    
    # Fonctions utilitaires
    "train_test_split", "cross_validation", "normalize_minmax", "normalize_zscore",
    "normalize_robust", "load_dataset", "save_model", "load_model",
    "accuracy_score", "precision_score", "recall_score", "f1_score",
    "mean_squared_error", "mean_absolute_error", "r2_score",
    "euclidean_distance", "cosine_similarity", "sigmoid", "softmax",
    
    # Fonctions de création
    "create_layer", "create_optimizer", "create_loss", "create_metric",
    "create_pipeline",
    
    # Fonctions d'information
    "get_available_optimizers", "get_available_losses", "get_available_metrics",
    "get_ml_info",
    
    # Fonctions d'évaluation
    "evaluate_model", "quick_evaluate",
    
    # Alias
    "MSE", "MAE", "RMSE", "R2", "MAPE"
]