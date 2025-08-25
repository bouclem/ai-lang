# Fonctions de perte pour l'entraînement des modèles ML d'ai'lang

import math
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ============================================================================
# Classe de base pour les fonctions de perte
# ============================================================================

class Loss(ABC):
    """
    Classe de base abstraite pour toutes les fonctions de perte.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule la perte entre les vraies valeurs et les prédictions.
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
        
        Returns:
            Valeur de la perte
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de la perte par rapport aux prédictions.
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
        
        Returns:
            Gradient de la perte
        """
        pass
    
    def __call__(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Permet d'utiliser l'instance comme une fonction.
        """
        return self.compute_loss(y_true, y_pred)

# ============================================================================
# Fonctions de perte pour la régression
# ============================================================================

class MeanSquaredError(Loss):
    """
    Erreur quadratique moyenne (MSE).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "mse")
    
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule l'erreur quadratique moyenne.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        total_samples = 0
        
        for true_row, pred_row in zip(y_true, y_pred):
            if len(true_row) != len(pred_row):
                raise ValueError("All samples must have the same dimension")
            
            for true_val, pred_val in zip(true_row, pred_row):
                total_loss += (true_val - pred_val) ** 2
                total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de MSE.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = []
            for true_val, pred_val in zip(true_row, pred_row):
                # Gradient: 2 * (pred - true) / n_samples
                grad = 2.0 * (pred_val - true_val) / n_samples
                grad_row.append(grad)
            gradient.append(grad_row)
        
        return gradient

class MeanAbsoluteError(Loss):
    """
    Erreur absolue moyenne (MAE).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "mae")
    
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule l'erreur absolue moyenne.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        total_samples = 0
        
        for true_row, pred_row in zip(y_true, y_pred):
            for true_val, pred_val in zip(true_row, pred_row):
                total_loss += abs(true_val - pred_val)
                total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de MAE.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = []
            for true_val, pred_val in zip(true_row, pred_row):
                # Gradient: sign(pred - true) / n_samples
                diff = pred_val - true_val
                if diff > 0:
                    grad = 1.0 / n_samples
                elif diff < 0:
                    grad = -1.0 / n_samples
                else:
                    grad = 0.0
                grad_row.append(grad)
            gradient.append(grad_row)
        
        return gradient

class HuberLoss(Loss):
    """
    Perte de Huber (robuste aux outliers).
    """
    
    def __init__(self, delta: float = 1.0, name: Optional[str] = None):
        super().__init__(name or "huber")
        self.delta = delta
    
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule la perte de Huber.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        total_samples = 0
        
        for true_row, pred_row in zip(y_true, y_pred):
            for true_val, pred_val in zip(true_row, pred_row):
                diff = abs(true_val - pred_val)
                if diff <= self.delta:
                    loss = 0.5 * diff ** 2
                else:
                    loss = self.delta * diff - 0.5 * self.delta ** 2
                
                total_loss += loss
                total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de la perte de Huber.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = []
            for true_val, pred_val in zip(true_row, pred_row):
                diff = pred_val - true_val
                abs_diff = abs(diff)
                
                if abs_diff <= self.delta:
                    grad = diff / n_samples
                else:
                    grad = self.delta * (1 if diff > 0 else -1) / n_samples
                
                grad_row.append(grad)
            gradient.append(grad_row)
        
        return gradient

# ============================================================================
# Fonctions de perte pour la classification
# ============================================================================

class BinaryCrossentropy(Loss):
    """
    Entropie croisée binaire pour la classification binaire.
    """
    
    def __init__(self, epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(name or "binary_crossentropy")
        self.epsilon = epsilon
    
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule l'entropie croisée binaire.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        total_samples = 0
        
        for true_row, pred_row in zip(y_true, y_pred):
            for true_val, pred_val in zip(true_row, pred_row):
                # Clipping pour éviter log(0)
                pred_clipped = max(self.epsilon, min(1 - self.epsilon, pred_val))
                
                # Entropie croisée binaire
                loss = -(true_val * math.log(pred_clipped) + (1 - true_val) * math.log(1 - pred_clipped))
                total_loss += loss
                total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de l'entropie croisée binaire.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = []
            for true_val, pred_val in zip(true_row, pred_row):
                # Clipping pour éviter la division par 0
                pred_clipped = max(self.epsilon, min(1 - self.epsilon, pred_val))
                
                # Gradient: (pred - true) / (pred * (1 - pred)) / n_samples
                grad = (pred_clipped - true_val) / (pred_clipped * (1 - pred_clipped)) / n_samples
                grad_row.append(grad)
            gradient.append(grad_row)
        
        return gradient

class CategoricalCrossentropy(Loss):
    """
    Entropie croisée catégorielle pour la classification multi-classe.
    """
    
    def __init__(self, epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(name or "categorical_crossentropy")
        self.epsilon = epsilon
    
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule l'entropie croisée catégorielle.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            sample_loss = 0.0
            for true_val, pred_val in zip(true_row, pred_row):
                # Clipping pour éviter log(0)
                pred_clipped = max(self.epsilon, min(1 - self.epsilon, pred_val))
                
                # Entropie croisée
                sample_loss += true_val * math.log(pred_clipped)
            
            total_loss -= sample_loss
        
        return total_loss / n_samples if n_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de l'entropie croisée catégorielle.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = []
            for true_val, pred_val in zip(true_row, pred_row):
                # Clipping pour éviter la division par 0
                pred_clipped = max(self.epsilon, min(1 - self.epsilon, pred_val))
                
                # Gradient: -true / pred / n_samples
                grad = -true_val / pred_clipped / n_samples
                grad_row.append(grad)
            gradient.append(grad_row)
        
        return gradient

class SparseCategoricalCrossentropy(Loss):
    """
    Entropie croisée catégorielle sparse (labels entiers).
    """
    
    def __init__(self, epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(name or "sparse_categorical_crossentropy")
        self.epsilon = epsilon
    
    def compute_loss(self, y_true: List[List[int]], y_pred: List[List[float]]) -> float:
        """
        Calcule l'entropie croisée catégorielle sparse.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            for true_class in true_row:
                if 0 <= true_class < len(pred_row):
                    # Clipping pour éviter log(0)
                    pred_val = max(self.epsilon, min(1 - self.epsilon, pred_row[true_class]))
                    total_loss -= math.log(pred_val)
        
        return total_loss / n_samples if n_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[int]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de l'entropie croisée catégorielle sparse.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = [0.0] * len(pred_row)
            
            for true_class in true_row:
                if 0 <= true_class < len(pred_row):
                    # Clipping pour éviter la division par 0
                    pred_val = max(self.epsilon, min(1 - self.epsilon, pred_row[true_class]))
                    grad_row[true_class] = -1.0 / pred_val / n_samples
            
            gradient.append(grad_row)
        
        return gradient

class Hinge(Loss):
    """
    Perte hinge pour SVM.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "hinge")
    
    def compute_loss(self, y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        """
        Calcule la perte hinge.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        total_loss = 0.0
        total_samples = 0
        
        for true_row, pred_row in zip(y_true, y_pred):
            for true_val, pred_val in zip(true_row, pred_row):
                # Perte hinge: max(0, 1 - true * pred)
                # Assume true_val is in {-1, 1}
                loss = max(0, 1 - true_val * pred_val)
                total_loss += loss
                total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def compute_gradient(self, y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de la perte hinge.
        """
        gradient = []
        n_samples = len(y_true)
        
        for true_row, pred_row in zip(y_true, y_pred):
            grad_row = []
            for true_val, pred_val in zip(true_row, pred_row):
                # Gradient: -true if true * pred < 1, else 0
                if true_val * pred_val < 1:
                    grad = -true_val / n_samples
                else:
                    grad = 0.0
                grad_row.append(grad)
            gradient.append(grad_row)
        
        return gradient

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def create_loss(loss_type: str, **kwargs) -> Loss:
    """
    Crée une fonction de perte du type spécifié.
    
    Args:
        loss_type: Type de perte
        **kwargs: Arguments pour la fonction de perte
    
    Returns:
        Instance de la fonction de perte
    """
    loss_type = loss_type.lower()
    
    if loss_type in ["mse", "mean_squared_error"]:
        return MeanSquaredError(**kwargs)
    elif loss_type in ["mae", "mean_absolute_error"]:
        return MeanAbsoluteError(**kwargs)
    elif loss_type == "huber":
        return HuberLoss(**kwargs)
    elif loss_type in ["binary_crossentropy", "bce"]:
        return BinaryCrossentropy(**kwargs)
    elif loss_type in ["categorical_crossentropy", "cce"]:
        return CategoricalCrossentropy(**kwargs)
    elif loss_type in ["sparse_categorical_crossentropy", "scce"]:
        return SparseCategoricalCrossentropy(**kwargs)
    elif loss_type == "hinge":
        return Hinge(**kwargs)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported")

def get_available_losses() -> List[str]:
    """
    Retourne la liste des fonctions de perte disponibles.
    """
    return [
        "mse", "mae", "huber", 
        "binary_crossentropy", "categorical_crossentropy", 
        "sparse_categorical_crossentropy", "hinge"
    ]

def get_loss_info(loss_type: str) -> Dict[str, Any]:
    """
    Retourne des informations sur une fonction de perte.
    
    Args:
        loss_type: Type de fonction de perte
    
    Returns:
        Dictionnaire avec les informations
    """
    info = {
        "mse": {
            "name": "Mean Squared Error",
            "description": "Erreur quadratique moyenne, pour la régression",
            "type": "regression",
            "parameters": []
        },
        "mae": {
            "name": "Mean Absolute Error",
            "description": "Erreur absolue moyenne, robuste aux outliers",
            "type": "regression",
            "parameters": []
        },
        "huber": {
            "name": "Huber Loss",
            "description": "Perte robuste combinant MSE et MAE",
            "type": "regression",
            "parameters": ["delta"]
        },
        "binary_crossentropy": {
            "name": "Binary Crossentropy",
            "description": "Entropie croisée pour classification binaire",
            "type": "classification",
            "parameters": ["epsilon"]
        },
        "categorical_crossentropy": {
            "name": "Categorical Crossentropy",
            "description": "Entropie croisée pour classification multi-classe",
            "type": "classification",
            "parameters": ["epsilon"]
        },
        "sparse_categorical_crossentropy": {
            "name": "Sparse Categorical Crossentropy",
            "description": "Entropie croisée avec labels entiers",
            "type": "classification",
            "parameters": ["epsilon"]
        },
        "hinge": {
            "name": "Hinge Loss",
            "description": "Perte hinge pour SVM",
            "type": "classification",
            "parameters": []
        }
    }
    
    return info.get(loss_type.lower(), {})

# Alias pour la compatibilité
MSE = MeanSquaredError
MAE = MeanAbsoluteError
BCE = BinaryCrossentropy
CCE = CategoricalCrossentropy
SCCE = SparseCategoricalCrossentropy