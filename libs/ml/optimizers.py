# Optimiseurs pour l'entraînement des modèles ML d'ai'lang

import math
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# ============================================================================
# Classe de base pour les optimiseurs
# ============================================================================

class Optimizer(ABC):
    """
    Classe de base abstraite pour tous les optimiseurs.
    """
    
    def __init__(self, learning_rate: float = 0.001, name: Optional[str] = None):
        self.learning_rate = learning_rate
        self.name = name or self.__class__.__name__
        self.iterations = 0
    
    @abstractmethod
    def update(self, weights: List[List[float]], gradients: List[List[float]]) -> List[List[float]]:
        """
        Met à jour les poids en utilisant les gradients.
        
        Args:
            weights: Poids actuels
            gradients: Gradients calculés
        
        Returns:
            Nouveaux poids
        """
        pass
    
    @abstractmethod
    def update_bias(self, bias: List[float], gradients: List[float]) -> List[float]:
        """
        Met à jour les biais en utilisant les gradients.
        
        Args:
            bias: Biais actuels
            gradients: Gradients calculés
        
        Returns:
            Nouveaux biais
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de l'optimiseur.
        """
        return {
            "learning_rate": self.learning_rate,
            "name": self.name,
            "iterations": self.iterations
        }
    
    def reset_state(self) -> None:
        """
        Remet à zéro l'état de l'optimiseur.
        """
        self.iterations = 0

# ============================================================================
# Gradient Descent Stochastique (SGD)
# ============================================================================

class SGD(Optimizer):
    """
    Optimiseur Stochastic Gradient Descent avec momentum optionnel.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, 
                 nesterov: bool = False, name: Optional[str] = None):
        super().__init__(learning_rate, name)
        self.momentum = momentum
        self.nesterov = nesterov
        
        # État pour le momentum
        self.velocity_weights = {}
        self.velocity_bias = {}
    
    def update(self, weights: List[List[float]], gradients: List[List[float]]) -> List[List[float]]:
        """
        Met à jour les poids avec SGD et momentum.
        """
        self.iterations += 1
        
        # Initialisation de la vélocité si nécessaire
        weights_id = id(weights)
        if weights_id not in self.velocity_weights:
            self.velocity_weights[weights_id] = [[0.0 for _ in row] for row in weights]
        
        velocity = self.velocity_weights[weights_id]
        new_weights = []
        
        for i, (weight_row, grad_row, vel_row) in enumerate(zip(weights, gradients, velocity)):
            new_weight_row = []
            for j, (w, g, v) in enumerate(zip(weight_row, grad_row, vel_row)):
                # Mise à jour de la vélocité
                new_v = self.momentum * v - self.learning_rate * g
                velocity[i][j] = new_v
                
                # Mise à jour des poids
                if self.nesterov:
                    # Nesterov momentum
                    new_w = w + self.momentum * new_v - self.learning_rate * g
                else:
                    # Momentum standard
                    new_w = w + new_v
                
                new_weight_row.append(new_w)
            new_weights.append(new_weight_row)
        
        return new_weights
    
    def update_bias(self, bias: List[float], gradients: List[float]) -> List[float]:
        """
        Met à jour les biais avec SGD et momentum.
        """
        # Initialisation de la vélocité si nécessaire
        bias_id = id(bias)
        if bias_id not in self.velocity_bias:
            self.velocity_bias[bias_id] = [0.0] * len(bias)
        
        velocity = self.velocity_bias[bias_id]
        new_bias = []
        
        for i, (b, g, v) in enumerate(zip(bias, gradients, velocity)):
            # Mise à jour de la vélocité
            new_v = self.momentum * v - self.learning_rate * g
            velocity[i] = new_v
            
            # Mise à jour du biais
            if self.nesterov:
                new_b = b + self.momentum * new_v - self.learning_rate * g
            else:
                new_b = b + new_v
            
            new_bias.append(new_b)
        
        return new_bias
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "momentum": self.momentum,
            "nesterov": self.nesterov
        })
        return config
    
    def reset_state(self) -> None:
        super().reset_state()
        self.velocity_weights.clear()
        self.velocity_bias.clear()

# ============================================================================
# Adam Optimizer
# ============================================================================

class Adam(Optimizer):
    """
    Optimiseur Adam (Adaptive Moment Estimation).
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # État pour les moments
        self.m_weights = {}  # Premier moment (momentum)
        self.v_weights = {}  # Deuxième moment (variance)
        self.m_bias = {}
        self.v_bias = {}
    
    def update(self, weights: List[List[float]], gradients: List[List[float]]) -> List[List[float]]:
        """
        Met à jour les poids avec Adam.
        """
        self.iterations += 1
        
        # Initialisation des moments si nécessaire
        weights_id = id(weights)
        if weights_id not in self.m_weights:
            self.m_weights[weights_id] = [[0.0 for _ in row] for row in weights]
            self.v_weights[weights_id] = [[0.0 for _ in row] for row in weights]
        
        m = self.m_weights[weights_id]
        v = self.v_weights[weights_id]
        new_weights = []
        
        for i, (weight_row, grad_row, m_row, v_row) in enumerate(zip(weights, gradients, m, v)):
            new_weight_row = []
            for j, (w, g, m_val, v_val) in enumerate(zip(weight_row, grad_row, m_row, v_row)):
                # Mise à jour des moments
                new_m = self.beta1 * m_val + (1 - self.beta1) * g
                new_v = self.beta2 * v_val + (1 - self.beta2) * g * g
                
                m[i][j] = new_m
                v[i][j] = new_v
                
                # Correction du biais
                m_corrected = new_m / (1 - self.beta1 ** self.iterations)
                v_corrected = new_v / (1 - self.beta2 ** self.iterations)
                
                # Mise à jour des poids
                new_w = w - self.learning_rate * m_corrected / (math.sqrt(v_corrected) + self.epsilon)
                new_weight_row.append(new_w)
            
            new_weights.append(new_weight_row)
        
        return new_weights
    
    def update_bias(self, bias: List[float], gradients: List[float]) -> List[float]:
        """
        Met à jour les biais avec Adam.
        """
        # Initialisation des moments si nécessaire
        bias_id = id(bias)
        if bias_id not in self.m_bias:
            self.m_bias[bias_id] = [0.0] * len(bias)
            self.v_bias[bias_id] = [0.0] * len(bias)
        
        m = self.m_bias[bias_id]
        v = self.v_bias[bias_id]
        new_bias = []
        
        for i, (b, g, m_val, v_val) in enumerate(zip(bias, gradients, m, v)):
            # Mise à jour des moments
            new_m = self.beta1 * m_val + (1 - self.beta1) * g
            new_v = self.beta2 * v_val + (1 - self.beta2) * g * g
            
            m[i] = new_m
            v[i] = new_v
            
            # Correction du biais
            m_corrected = new_m / (1 - self.beta1 ** self.iterations)
            v_corrected = new_v / (1 - self.beta2 ** self.iterations)
            
            # Mise à jour du biais
            new_b = b - self.learning_rate * m_corrected / (math.sqrt(v_corrected) + self.epsilon)
            new_bias.append(new_b)
        
        return new_bias
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        })
        return config
    
    def reset_state(self) -> None:
        super().reset_state()
        self.m_weights.clear()
        self.v_weights.clear()
        self.m_bias.clear()
        self.v_bias.clear()

# ============================================================================
# RMSprop Optimizer
# ============================================================================

class RMSprop(Optimizer):
    """
    Optimiseur RMSprop (Root Mean Square Propagation).
    """
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, 
                 epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(learning_rate, name)
        self.rho = rho
        self.epsilon = epsilon
        
        # État pour les moyennes mobiles des gradients au carré
        self.v_weights = {}
        self.v_bias = {}
    
    def update(self, weights: List[List[float]], gradients: List[List[float]]) -> List[List[float]]:
        """
        Met à jour les poids avec RMSprop.
        """
        self.iterations += 1
        
        # Initialisation si nécessaire
        weights_id = id(weights)
        if weights_id not in self.v_weights:
            self.v_weights[weights_id] = [[0.0 for _ in row] for row in weights]
        
        v = self.v_weights[weights_id]
        new_weights = []
        
        for i, (weight_row, grad_row, v_row) in enumerate(zip(weights, gradients, v)):
            new_weight_row = []
            for j, (w, g, v_val) in enumerate(zip(weight_row, grad_row, v_row)):
                # Mise à jour de la moyenne mobile
                new_v = self.rho * v_val + (1 - self.rho) * g * g
                v[i][j] = new_v
                
                # Mise à jour des poids
                new_w = w - self.learning_rate * g / (math.sqrt(new_v) + self.epsilon)
                new_weight_row.append(new_w)
            
            new_weights.append(new_weight_row)
        
        return new_weights
    
    def update_bias(self, bias: List[float], gradients: List[float]) -> List[float]:
        """
        Met à jour les biais avec RMSprop.
        """
        # Initialisation si nécessaire
        bias_id = id(bias)
        if bias_id not in self.v_bias:
            self.v_bias[bias_id] = [0.0] * len(bias)
        
        v = self.v_bias[bias_id]
        new_bias = []
        
        for i, (b, g, v_val) in enumerate(zip(bias, gradients, v)):
            # Mise à jour de la moyenne mobile
            new_v = self.rho * v_val + (1 - self.rho) * g * g
            v[i] = new_v
            
            # Mise à jour du biais
            new_b = b - self.learning_rate * g / (math.sqrt(new_v) + self.epsilon)
            new_bias.append(new_b)
        
        return new_bias
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "rho": self.rho,
            "epsilon": self.epsilon
        })
        return config
    
    def reset_state(self) -> None:
        super().reset_state()
        self.v_weights.clear()
        self.v_bias.clear()

# ============================================================================
# AdaGrad Optimizer
# ============================================================================

class AdaGrad(Optimizer):
    """
    Optimiseur AdaGrad (Adaptive Gradient).
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(learning_rate, name)
        self.epsilon = epsilon
        
        # Accumulation des gradients au carré
        self.accumulated_weights = {}
        self.accumulated_bias = {}
    
    def update(self, weights: List[List[float]], gradients: List[List[float]]) -> List[List[float]]:
        """
        Met à jour les poids avec AdaGrad.
        """
        self.iterations += 1
        
        # Initialisation si nécessaire
        weights_id = id(weights)
        if weights_id not in self.accumulated_weights:
            self.accumulated_weights[weights_id] = [[0.0 for _ in row] for row in weights]
        
        acc = self.accumulated_weights[weights_id]
        new_weights = []
        
        for i, (weight_row, grad_row, acc_row) in enumerate(zip(weights, gradients, acc)):
            new_weight_row = []
            for j, (w, g, acc_val) in enumerate(zip(weight_row, grad_row, acc_row)):
                # Accumulation des gradients au carré
                new_acc = acc_val + g * g
                acc[i][j] = new_acc
                
                # Mise à jour des poids
                new_w = w - self.learning_rate * g / (math.sqrt(new_acc) + self.epsilon)
                new_weight_row.append(new_w)
            
            new_weights.append(new_weight_row)
        
        return new_weights
    
    def update_bias(self, bias: List[float], gradients: List[float]) -> List[float]:
        """
        Met à jour les biais avec AdaGrad.
        """
        # Initialisation si nécessaire
        bias_id = id(bias)
        if bias_id not in self.accumulated_bias:
            self.accumulated_bias[bias_id] = [0.0] * len(bias)
        
        acc = self.accumulated_bias[bias_id]
        new_bias = []
        
        for i, (b, g, acc_val) in enumerate(zip(bias, gradients, acc)):
            # Accumulation des gradients au carré
            new_acc = acc_val + g * g
            acc[i] = new_acc
            
            # Mise à jour du biais
            new_b = b - self.learning_rate * g / (math.sqrt(new_acc) + self.epsilon)
            new_bias.append(new_b)
        
        return new_bias
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
    
    def reset_state(self) -> None:
        super().reset_state()
        self.accumulated_weights.clear()
        self.accumulated_bias.clear()

# ============================================================================
# AdaDelta Optimizer
# ============================================================================

class AdaDelta(Optimizer):
    """
    Optimiseur AdaDelta.
    """
    
    def __init__(self, learning_rate: float = 1.0, rho: float = 0.95, 
                 epsilon: float = 1e-6, name: Optional[str] = None):
        super().__init__(learning_rate, name)
        self.rho = rho
        self.epsilon = epsilon
        
        # Moyennes mobiles
        self.accumulated_grad_weights = {}
        self.accumulated_delta_weights = {}
        self.accumulated_grad_bias = {}
        self.accumulated_delta_bias = {}
    
    def update(self, weights: List[List[float]], gradients: List[List[float]]) -> List[List[float]]:
        """
        Met à jour les poids avec AdaDelta.
        """
        self.iterations += 1
        
        # Initialisation si nécessaire
        weights_id = id(weights)
        if weights_id not in self.accumulated_grad_weights:
            self.accumulated_grad_weights[weights_id] = [[0.0 for _ in row] for row in weights]
            self.accumulated_delta_weights[weights_id] = [[0.0 for _ in row] for row in weights]
        
        acc_grad = self.accumulated_grad_weights[weights_id]
        acc_delta = self.accumulated_delta_weights[weights_id]
        new_weights = []
        
        for i, (weight_row, grad_row, acc_grad_row, acc_delta_row) in enumerate(zip(weights, gradients, acc_grad, acc_delta)):
            new_weight_row = []
            for j, (w, g, acc_g, acc_d) in enumerate(zip(weight_row, grad_row, acc_grad_row, acc_delta_row)):
                # Mise à jour de l'accumulation des gradients
                new_acc_g = self.rho * acc_g + (1 - self.rho) * g * g
                acc_grad[i][j] = new_acc_g
                
                # Calcul du delta
                delta = -math.sqrt(acc_d + self.epsilon) / math.sqrt(new_acc_g + self.epsilon) * g
                
                # Mise à jour de l'accumulation des deltas
                new_acc_d = self.rho * acc_d + (1 - self.rho) * delta * delta
                acc_delta[i][j] = new_acc_d
                
                # Mise à jour des poids
                new_w = w + self.learning_rate * delta
                new_weight_row.append(new_w)
            
            new_weights.append(new_weight_row)
        
        return new_weights
    
    def update_bias(self, bias: List[float], gradients: List[float]) -> List[float]:
        """
        Met à jour les biais avec AdaDelta.
        """
        # Initialisation si nécessaire
        bias_id = id(bias)
        if bias_id not in self.accumulated_grad_bias:
            self.accumulated_grad_bias[bias_id] = [0.0] * len(bias)
            self.accumulated_delta_bias[bias_id] = [0.0] * len(bias)
        
        acc_grad = self.accumulated_grad_bias[bias_id]
        acc_delta = self.accumulated_delta_bias[bias_id]
        new_bias = []
        
        for i, (b, g, acc_g, acc_d) in enumerate(zip(bias, gradients, acc_grad, acc_delta)):
            # Mise à jour de l'accumulation des gradients
            new_acc_g = self.rho * acc_g + (1 - self.rho) * g * g
            acc_grad[i] = new_acc_g
            
            # Calcul du delta
            delta = -math.sqrt(acc_d + self.epsilon) / math.sqrt(new_acc_g + self.epsilon) * g
            
            # Mise à jour de l'accumulation des deltas
            new_acc_d = self.rho * acc_d + (1 - self.rho) * delta * delta
            acc_delta[i] = new_acc_d
            
            # Mise à jour du biais
            new_b = b + self.learning_rate * delta
            new_bias.append(new_b)
        
        return new_bias
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "rho": self.rho,
            "epsilon": self.epsilon
        })
        return config
    
    def reset_state(self) -> None:
        super().reset_state()
        self.accumulated_grad_weights.clear()
        self.accumulated_delta_weights.clear()
        self.accumulated_grad_bias.clear()
        self.accumulated_delta_bias.clear()

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def create_optimizer(optimizer_type: str, **kwargs) -> Optimizer:
    """
    Crée un optimiseur du type spécifié.
    
    Args:
        optimizer_type: Type d'optimiseur ('sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta')
        **kwargs: Arguments pour l'optimiseur
    
    Returns:
        Instance de l'optimiseur
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "sgd":
        return SGD(**kwargs)
    elif optimizer_type == "adam":
        return Adam(**kwargs)
    elif optimizer_type == "rmsprop":
        return RMSprop(**kwargs)
    elif optimizer_type == "adagrad":
        return AdaGrad(**kwargs)
    elif optimizer_type == "adadelta":
        return AdaDelta(**kwargs)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not supported")

def get_available_optimizers() -> List[str]:
    """
    Retourne la liste des optimiseurs disponibles.
    """
    return ["sgd", "adam", "rmsprop", "adagrad", "adadelta"]

def get_optimizer_info(optimizer_type: str) -> Dict[str, Any]:
    """
    Retourne des informations sur un optimiseur.
    
    Args:
        optimizer_type: Type d'optimiseur
    
    Returns:
        Dictionnaire avec les informations
    """
    info = {
        "sgd": {
            "name": "Stochastic Gradient Descent",
            "description": "Optimiseur de base avec momentum optionnel",
            "parameters": ["learning_rate", "momentum", "nesterov"]
        },
        "adam": {
            "name": "Adam",
            "description": "Adaptive Moment Estimation, combine momentum et RMSprop",
            "parameters": ["learning_rate", "beta1", "beta2", "epsilon"]
        },
        "rmsprop": {
            "name": "RMSprop",
            "description": "Root Mean Square Propagation, adapte le taux d'apprentissage",
            "parameters": ["learning_rate", "rho", "epsilon"]
        },
        "adagrad": {
            "name": "AdaGrad",
            "description": "Adaptive Gradient, accumule les gradients passés",
            "parameters": ["learning_rate", "epsilon"]
        },
        "adadelta": {
            "name": "AdaDelta",
            "description": "Extension d'AdaGrad qui limite l'accumulation",
            "parameters": ["learning_rate", "rho", "epsilon"]
        }
    }
    
    return info.get(optimizer_type.lower(), {})