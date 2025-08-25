# Métriques d'évaluation pour les modèles ML d'ai'lang

import math
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod

# ============================================================================
# Classe de base pour les métriques
# ============================================================================

class Metric(ABC):
    """
    Classe de base abstraite pour toutes les métriques.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute(self, y_true: List, y_pred: List) -> float:
        """
        Calcule la métrique.
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
        
        Returns:
            Valeur de la métrique
        """
        pass
    
    def __call__(self, y_true: List, y_pred: List) -> float:
        """
        Permet d'utiliser l'instance comme une fonction.
        """
        return self.compute(y_true, y_pred)

# ============================================================================
# Métriques pour la régression
# ============================================================================

class MeanSquaredError(Metric):
    """
    Erreur quadratique moyenne.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "mse")
    
    def compute(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule l'erreur quadratique moyenne.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            return 0.0
        
        return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

class MeanAbsoluteError(Metric):
    """
    Erreur absolue moyenne.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "mae")
    
    def compute(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule l'erreur absolue moyenne.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            return 0.0
        
        return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

class RootMeanSquaredError(Metric):
    """
    Racine de l'erreur quadratique moyenne.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "rmse")
    
    def compute(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule la racine de l'erreur quadratique moyenne.
        """
        mse = MeanSquaredError().compute(y_true, y_pred)
        return math.sqrt(mse)

class R2Score(Metric):
    """
    Coefficient de détermination R².
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "r2")
    
    def compute(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule le coefficient R².
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            return 0.0
        
        # Moyenne des vraies valeurs
        y_mean = sum(y_true) / len(y_true)
        
        # Somme des carrés totale
        ss_tot = sum((y - y_mean) ** 2 for y in y_true)
        
        # Somme des carrés des résidus
        ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        
        # R² = 1 - (SS_res / SS_tot)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

class MeanAbsolutePercentageError(Metric):
    """
    Erreur absolue moyenne en pourcentage.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "mape")
    
    def compute(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule l'erreur absolue moyenne en pourcentage.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            return 0.0
        
        total_error = 0.0
        valid_samples = 0
        
        for true, pred in zip(y_true, y_pred):
            if true != 0:  # Éviter la division par zéro
                total_error += abs((true - pred) / true)
                valid_samples += 1
        
        return (total_error / valid_samples * 100) if valid_samples > 0 else 0.0

# ============================================================================
# Métriques pour la classification
# ============================================================================

class Accuracy(Metric):
    """
    Précision (accuracy) pour la classification.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "accuracy")
    
    def compute(self, y_true: List, y_pred: List) -> float:
        """
        Calcule la précision.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            return 0.0
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

class Precision(Metric):
    """
    Précision pour la classification.
    """
    
    def __init__(self, average: str = "binary", pos_label: Any = 1, name: Optional[str] = None):
        super().__init__(name or "precision")
        self.average = average
        self.pos_label = pos_label
    
    def compute(self, y_true: List, y_pred: List) -> Union[float, Dict[str, float]]:
        """
        Calcule la précision.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if self.average == "binary":
            return self._binary_precision(y_true, y_pred)
        else:
            return self._multiclass_precision(y_true, y_pred)
    
    def _binary_precision(self, y_true: List, y_pred: List) -> float:
        """
        Calcule la précision binaire.
        """
        tp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == self.pos_label and pred == self.pos_label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true != self.pos_label and pred == self.pos_label)
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _multiclass_precision(self, y_true: List, y_pred: List) -> Union[float, Dict[str, float]]:
        """
        Calcule la précision multi-classe.
        """
        classes = sorted(list(set(y_true + y_pred)))
        precisions = {}
        
        for cls in classes:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
            precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        if self.average == "macro":
            return sum(precisions.values()) / len(precisions)
        elif self.average == "weighted":
            weights = {cls: y_true.count(cls) / len(y_true) for cls in classes}
            return sum(precisions[cls] * weights[cls] for cls in classes)
        else:
            return precisions

class Recall(Metric):
    """
    Rappel pour la classification.
    """
    
    def __init__(self, average: str = "binary", pos_label: Any = 1, name: Optional[str] = None):
        super().__init__(name or "recall")
        self.average = average
        self.pos_label = pos_label
    
    def compute(self, y_true: List, y_pred: List) -> Union[float, Dict[str, float]]:
        """
        Calcule le rappel.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if self.average == "binary":
            return self._binary_recall(y_true, y_pred)
        else:
            return self._multiclass_recall(y_true, y_pred)
    
    def _binary_recall(self, y_true: List, y_pred: List) -> float:
        """
        Calcule le rappel binaire.
        """
        tp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == self.pos_label and pred == self.pos_label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == self.pos_label and pred != self.pos_label)
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _multiclass_recall(self, y_true: List, y_pred: List) -> Union[float, Dict[str, float]]:
        """
        Calcule le rappel multi-classe.
        """
        classes = sorted(list(set(y_true + y_pred)))
        recalls = {}
        
        for cls in classes:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
            recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if self.average == "macro":
            return sum(recalls.values()) / len(recalls)
        elif self.average == "weighted":
            weights = {cls: y_true.count(cls) / len(y_true) for cls in classes}
            return sum(recalls[cls] * weights[cls] for cls in classes)
        else:
            return recalls

class F1Score(Metric):
    """
    F1-score pour la classification.
    """
    
    def __init__(self, average: str = "binary", pos_label: Any = 1, name: Optional[str] = None):
        super().__init__(name or "f1")
        self.average = average
        self.pos_label = pos_label
    
    def compute(self, y_true: List, y_pred: List) -> Union[float, Dict[str, float]]:
        """
        Calcule le F1-score.
        """
        precision_metric = Precision(self.average, self.pos_label)
        recall_metric = Recall(self.average, self.pos_label)
        
        precision = precision_metric.compute(y_true, y_pred)
        recall = recall_metric.compute(y_true, y_pred)
        
        if isinstance(precision, dict):
            f1_scores = {}
            for cls in precision.keys():
                p, r = precision[cls], recall[cls]
                f1_scores[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            if self.average == "macro":
                return sum(f1_scores.values()) / len(f1_scores)
            elif self.average == "weighted":
                weights = {cls: y_true.count(cls) / len(y_true) for cls in f1_scores.keys()}
                return sum(f1_scores[cls] * weights[cls] for cls in f1_scores.keys())
            else:
                return f1_scores
        else:
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

class ConfusionMatrix(Metric):
    """
    Matrice de confusion.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "confusion_matrix")
    
    def compute(self, y_true: List, y_pred: List) -> Dict[str, Dict[str, int]]:
        """
        Calcule la matrice de confusion.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        classes = sorted(list(set(y_true + y_pred)))
        matrix = {true_cls: {pred_cls: 0 for pred_cls in classes} for true_cls in classes}
        
        for true, pred in zip(y_true, y_pred):
            matrix[true][pred] += 1
        
        return matrix

class AUC(Metric):
    """
    Area Under the Curve (approximation simple).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "auc")
    
    def compute(self, y_true: List[int], y_scores: List[float]) -> float:
        """
        Calcule l'AUC (approximation).
        
        Args:
            y_true: Vraies étiquettes binaires (0 ou 1)
            y_scores: Scores de prédiction
        
        Returns:
            Valeur AUC approximée
        """
        if len(y_true) != len(y_scores):
            raise ValueError("y_true and y_scores must have the same length")
        
        # Trier par scores décroissants
        sorted_pairs = sorted(zip(y_scores, y_true), reverse=True)
        
        # Compter les positifs et négatifs
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5  # AUC indéfinie
        
        # Calcul approximatif de l'AUC
        auc = 0.0
        tp = 0
        fp = 0
        
        prev_score = None
        for score, label in sorted_pairs:
            if prev_score is not None and score != prev_score:
                # Calculer la contribution à l'AUC
                tpr = tp / n_pos
                fpr = fp / n_neg
                auc += tpr * (fpr - (fp - 1) / n_neg) if fp > 0 else 0
            
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            prev_score = score
        
        return min(1.0, max(0.0, auc))

# ============================================================================
# Métriques composées
# ============================================================================

class ClassificationReport:
    """
    Rapport de classification complet.
    """
    
    def __init__(self):
        self.accuracy = Accuracy()
        self.precision = Precision(average="weighted")
        self.recall = Recall(average="weighted")
        self.f1 = F1Score(average="weighted")
    
    def compute(self, y_true: List, y_pred: List) -> Dict[str, Any]:
        """
        Génère un rapport de classification complet.
        """
        # Métriques globales
        accuracy = self.accuracy.compute(y_true, y_pred)
        precision = self.precision.compute(y_true, y_pred)
        recall = self.recall.compute(y_true, y_pred)
        f1 = self.f1.compute(y_true, y_pred)
        
        # Métriques par classe
        classes = sorted(list(set(y_true + y_pred)))
        per_class_metrics = {}
        
        for cls in classes:
            cls_precision = Precision(average="binary", pos_label=cls).compute(y_true, y_pred)
            cls_recall = Recall(average="binary", pos_label=cls).compute(y_true, y_pred)
            cls_f1 = F1Score(average="binary", pos_label=cls).compute(y_true, y_pred)
            cls_support = y_true.count(cls)
            
            per_class_metrics[cls] = {
                "precision": cls_precision,
                "recall": cls_recall,
                "f1-score": cls_f1,
                "support": cls_support
            }
        
        return {
            "accuracy": accuracy,
            "macro_avg": {
                "precision": Precision(average="macro").compute(y_true, y_pred),
                "recall": Recall(average="macro").compute(y_true, y_pred),
                "f1-score": F1Score(average="macro").compute(y_true, y_pred),
                "support": len(y_true)
            },
            "weighted_avg": {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": len(y_true)
            },
            "per_class": per_class_metrics
        }

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def create_metric(metric_type: str, **kwargs) -> Metric:
    """
    Crée une métrique du type spécifié.
    
    Args:
        metric_type: Type de métrique
        **kwargs: Arguments pour la métrique
    
    Returns:
        Instance de la métrique
    """
    metric_type = metric_type.lower()
    
    # Métriques de régression
    if metric_type in ["mse", "mean_squared_error"]:
        return MeanSquaredError(**kwargs)
    elif metric_type in ["mae", "mean_absolute_error"]:
        return MeanAbsoluteError(**kwargs)
    elif metric_type in ["rmse", "root_mean_squared_error"]:
        return RootMeanSquaredError(**kwargs)
    elif metric_type in ["r2", "r2_score"]:
        return R2Score(**kwargs)
    elif metric_type in ["mape", "mean_absolute_percentage_error"]:
        return MeanAbsolutePercentageError(**kwargs)
    
    # Métriques de classification
    elif metric_type == "accuracy":
        return Accuracy(**kwargs)
    elif metric_type == "precision":
        return Precision(**kwargs)
    elif metric_type == "recall":
        return Recall(**kwargs)
    elif metric_type in ["f1", "f1_score"]:
        return F1Score(**kwargs)
    elif metric_type in ["confusion_matrix", "cm"]:
        return ConfusionMatrix(**kwargs)
    elif metric_type == "auc":
        return AUC(**kwargs)
    
    else:
        raise ValueError(f"Metric type '{metric_type}' not supported")

def get_available_metrics() -> Dict[str, List[str]]:
    """
    Retourne la liste des métriques disponibles par catégorie.
    """
    return {
        "regression": ["mse", "mae", "rmse", "r2", "mape"],
        "classification": ["accuracy", "precision", "recall", "f1", "confusion_matrix", "auc"]
    }

def evaluate_model(y_true: List, y_pred: List, task_type: str = "auto") -> Dict[str, Any]:
    """
    Évalue un modèle avec plusieurs métriques.
    
    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        task_type: Type de tâche ('regression', 'classification', 'auto')
    
    Returns:
        Dictionnaire avec les résultats d'évaluation
    """
    if task_type == "auto":
        # Détection automatique du type de tâche
        if all(isinstance(val, (int, float)) and val == int(val) for val in y_true + y_pred):
            unique_values = len(set(y_true + y_pred))
            if unique_values <= 10:  # Seuil arbitraire pour la classification
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            task_type = "regression"
    
    results = {"task_type": task_type}
    
    if task_type == "regression":
        results.update({
            "mse": MeanSquaredError().compute(y_true, y_pred),
            "mae": MeanAbsoluteError().compute(y_true, y_pred),
            "rmse": RootMeanSquaredError().compute(y_true, y_pred),
            "r2": R2Score().compute(y_true, y_pred),
            "mape": MeanAbsolutePercentageError().compute(y_true, y_pred)
        })
    
    elif task_type == "classification":
        report = ClassificationReport().compute(y_true, y_pred)
        results.update(report)
        results["confusion_matrix"] = ConfusionMatrix().compute(y_true, y_pred)
    
    return results

# Alias pour la compatibilité
MSE = MeanSquaredError
MAE = MeanAbsoluteError
RMSE = RootMeanSquaredError
R2 = R2Score
MAPE = MeanAbsolutePercentageError