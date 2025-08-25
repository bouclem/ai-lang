# Couches pour les réseaux de neurones d'ai'lang

import math
import random
from typing import List, Optional, Tuple, Any, Dict
from abc import ABC, abstractmethod

# ============================================================================
# Classe de base pour les couches
# ============================================================================

class Layer(ABC):
    """
    Classe de base abstraite pour toutes les couches de réseau de neurones.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.trainable = True
        self.built = False
        self.input_shape = None
        self.output_shape = None
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Construit la couche avec la forme d'entrée donnée.
        """
        pass
    
    @abstractmethod
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Propagation avant.
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output: List[List[float]]) -> List[List[float]]:
        """
        Propagation arrière.
        """
        pass
    
    def get_weights(self) -> Dict[str, Any]:
        """
        Retourne les poids de la couche.
        """
        return {}
    
    def set_weights(self, weights: Dict[str, Any]) -> None:
        """
        Définit les poids de la couche.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de la couche.
        """
        return {
            "name": self.name,
            "trainable": self.trainable
        }

# ============================================================================
# Couches denses (fully connected)
# ============================================================================

class Dense(Layer):
    """
    Couche dense (fully connected).
    """
    
    def __init__(self, units: int, activation: str = "linear", 
                 use_bias: bool = True, name: Optional[str] = None):
        super().__init__(name)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        
        # Poids et biais (initialisés dans build)
        self.weights = None
        self.bias = None
        
        # Pour la rétropropagation
        self.last_input = None
        self.last_output = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Construit la couche dense.
        """
        if len(input_shape) != 2:
            raise ValueError("Dense layer expects 2D input (batch_size, features)")
        
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], self.units)
        
        # Initialisation des poids (Xavier/Glorot)
        input_size = input_shape[1]
        limit = math.sqrt(6.0 / (input_size + self.units))
        
        self.weights = []
        for i in range(input_size):
            row = []
            for j in range(self.units):
                weight = random.uniform(-limit, limit)
                row.append(weight)
            self.weights.append(row)
        
        # Initialisation des biais
        if self.use_bias:
            self.bias = [0.0] * self.units
        
        self.built = True
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Propagation avant de la couche dense.
        """
        if not self.built:
            batch_size = len(inputs)
            input_size = len(inputs[0]) if inputs else 0
            self.build((batch_size, input_size))
        
        self.last_input = [row[:] for row in inputs]  # Copie pour la rétropropagation
        
        # Multiplication matricielle: inputs @ weights + bias
        outputs = []
        for input_row in inputs:
            output_row = []
            for j in range(self.units):
                # Produit scalaire
                value = sum(input_row[i] * self.weights[i][j] for i in range(len(input_row)))
                
                # Ajout du biais
                if self.use_bias:
                    value += self.bias[j]
                
                output_row.append(value)
            outputs.append(output_row)
        
        # Application de la fonction d'activation
        outputs = self._apply_activation(outputs)
        self.last_output = outputs
        
        return outputs
    
    def backward(self, grad_output: List[List[float]]) -> List[List[float]]:
        """
        Propagation arrière de la couche dense.
        """
        if self.last_input is None:
            raise ValueError("Forward pass must be called before backward pass")
        
        # Gradient par rapport à l'activation
        grad_activation = self._activation_gradient(self.last_output)
        
        # Gradient après activation
        grad_pre_activation = []
        for i in range(len(grad_output)):
            row = []
            for j in range(len(grad_output[i])):
                row.append(grad_output[i][j] * grad_activation[i][j])
            grad_pre_activation.append(row)
        
        # Gradient par rapport aux poids
        grad_weights = []
        for i in range(len(self.weights)):
            row = []
            for j in range(len(self.weights[i])):
                grad = 0.0
                for batch_idx in range(len(self.last_input)):
                    grad += self.last_input[batch_idx][i] * grad_pre_activation[batch_idx][j]
                row.append(grad / len(self.last_input))  # Moyenne sur le batch
            grad_weights.append(row)
        
        # Gradient par rapport aux biais
        grad_bias = None
        if self.use_bias:
            grad_bias = [0.0] * self.units
            for j in range(self.units):
                for batch_idx in range(len(grad_pre_activation)):
                    grad_bias[j] += grad_pre_activation[batch_idx][j]
                grad_bias[j] /= len(grad_pre_activation)  # Moyenne sur le batch
        
        # Gradient par rapport à l'entrée
        grad_input = []
        for batch_idx in range(len(self.last_input)):
            row = []
            for i in range(len(self.last_input[batch_idx])):
                grad = 0.0
                for j in range(self.units):
                    grad += self.weights[i][j] * grad_pre_activation[batch_idx][j]
                row.append(grad)
            grad_input.append(row)
        
        # Stockage des gradients pour la mise à jour des poids
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        
        return grad_input
    
    def _apply_activation(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Applique la fonction d'activation.
        """
        if self.activation == "linear":
            return inputs
        
        elif self.activation == "relu":
            return [[max(0, x) for x in row] for row in inputs]
        
        elif self.activation == "sigmoid":
            return [[self._sigmoid(x) for x in row] for row in inputs]
        
        elif self.activation == "tanh":
            return [[math.tanh(x) for x in row] for row in inputs]
        
        elif self.activation == "softmax":
            result = []
            for row in inputs:
                # Stabilité numérique
                max_val = max(row)
                exp_values = [math.exp(x - max_val) for x in row]
                sum_exp = sum(exp_values)
                softmax_row = [exp_val / sum_exp for exp_val in exp_values]
                result.append(softmax_row)
            return result
        
        else:
            raise ValueError(f"Activation function '{self.activation}' not supported")
    
    def _activation_gradient(self, outputs: List[List[float]]) -> List[List[float]]:
        """
        Calcule le gradient de la fonction d'activation.
        """
        if self.activation == "linear":
            return [[1.0 for _ in row] for row in outputs]
        
        elif self.activation == "relu":
            return [[1.0 if x > 0 else 0.0 for x in row] for row in outputs]
        
        elif self.activation == "sigmoid":
            return [[x * (1 - x) for x in row] for row in outputs]
        
        elif self.activation == "tanh":
            return [[1 - x * x for x in row] for row in outputs]
        
        elif self.activation == "softmax":
            # Pour softmax, le gradient est plus complexe
            result = []
            for row in outputs:
                grad_row = []
                for i in range(len(row)):
                    grad = row[i] * (1 - row[i])  # Approximation diagonale
                    grad_row.append(grad)
                result.append(grad_row)
            return result
        
        else:
            return [[1.0 for _ in row] for row in outputs]
    
    def _sigmoid(self, x: float) -> float:
        """
        Fonction sigmoïde avec protection contre l'overflow.
        """
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def get_weights(self) -> Dict[str, Any]:
        """
        Retourne les poids de la couche.
        """
        weights_dict = {"weights": self.weights}
        if self.use_bias:
            weights_dict["bias"] = self.bias
        return weights_dict
    
    def set_weights(self, weights: Dict[str, Any]) -> None:
        """
        Définit les poids de la couche.
        """
        if "weights" in weights:
            self.weights = weights["weights"]
        if "bias" in weights and self.use_bias:
            self.bias = weights["bias"]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de la couche.
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config

# ============================================================================
# Couches de dropout
# ============================================================================

class Dropout(Layer):
    """
    Couche de dropout pour la régularisation.
    """
    
    def __init__(self, rate: float, name: Optional[str] = None):
        super().__init__(name)
        self.rate = rate
        self.training = True
        self.mask = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Construit la couche dropout.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Propagation avant avec dropout.
        """
        if not self.built:
            batch_size = len(inputs)
            input_size = len(inputs[0]) if inputs else 0
            self.build((batch_size, input_size))
        
        if not self.training:
            return inputs
        
        # Génération du masque de dropout
        self.mask = []
        outputs = []
        
        for row in inputs:
            mask_row = []
            output_row = []
            for value in row:
                if random.random() > self.rate:
                    mask_row.append(1.0 / (1.0 - self.rate))  # Scaling
                    output_row.append(value / (1.0 - self.rate))
                else:
                    mask_row.append(0.0)
                    output_row.append(0.0)
            
            self.mask.append(mask_row)
            outputs.append(output_row)
        
        return outputs
    
    def backward(self, grad_output: List[List[float]]) -> List[List[float]]:
        """
        Propagation arrière avec dropout.
        """
        if not self.training or self.mask is None:
            return grad_output
        
        # Application du masque aux gradients
        grad_input = []
        for i, row in enumerate(grad_output):
            grad_row = []
            for j, grad in enumerate(row):
                grad_row.append(grad * self.mask[i][j])
            grad_input.append(grad_row)
        
        return grad_input
    
    def set_training(self, training: bool) -> None:
        """
        Définit le mode d'entraînement.
        """
        self.training = training
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de la couche.
        """
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

# ============================================================================
# Couches de normalisation
# ============================================================================

class BatchNormalization(Layer):
    """
    Couche de normalisation par batch.
    """
    
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-3, name: Optional[str] = None):
        super().__init__(name)
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Paramètres apprenables
        self.gamma = None  # Scale
        self.beta = None   # Shift
        
        # Statistiques mobiles
        self.moving_mean = None
        self.moving_variance = None
        
        # Pour la rétropropagation
        self.last_input = None
        self.last_normalized = None
        self.training = True
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Construit la couche de normalisation.
        """
        if len(input_shape) != 2:
            raise ValueError("BatchNormalization expects 2D input")
        
        self.input_shape = input_shape
        self.output_shape = input_shape
        
        features = input_shape[1]
        
        # Initialisation des paramètres
        self.gamma = [1.0] * features
        self.beta = [0.0] * features
        
        # Initialisation des statistiques mobiles
        self.moving_mean = [0.0] * features
        self.moving_variance = [1.0] * features
        
        self.built = True
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Propagation avant avec normalisation.
        """
        if not self.built:
            batch_size = len(inputs)
            input_size = len(inputs[0]) if inputs else 0
            self.build((batch_size, input_size))
        
        self.last_input = [row[:] for row in inputs]
        
        if self.training:
            # Calcul des statistiques du batch
            batch_mean = self._compute_batch_mean(inputs)
            batch_variance = self._compute_batch_variance(inputs, batch_mean)
            
            # Mise à jour des statistiques mobiles
            for i in range(len(batch_mean)):
                self.moving_mean[i] = self.momentum * self.moving_mean[i] + (1 - self.momentum) * batch_mean[i]
                self.moving_variance[i] = self.momentum * self.moving_variance[i] + (1 - self.momentum) * batch_variance[i]
            
            mean = batch_mean
            variance = batch_variance
        else:
            # Utilisation des statistiques mobiles
            mean = self.moving_mean
            variance = self.moving_variance
        
        # Normalisation
        normalized = []
        for row in inputs:
            norm_row = []
            for i, value in enumerate(row):
                norm_value = (value - mean[i]) / math.sqrt(variance[i] + self.epsilon)
                norm_row.append(norm_value)
            normalized.append(norm_row)
        
        self.last_normalized = normalized
        
        # Application de gamma et beta
        outputs = []
        for row in normalized:
            output_row = []
            for i, value in enumerate(row):
                output_value = self.gamma[i] * value + self.beta[i]
                output_row.append(output_value)
            outputs.append(output_row)
        
        return outputs
    
    def backward(self, grad_output: List[List[float]]) -> List[List[float]]:
        """
        Propagation arrière pour la normalisation.
        """
        if self.last_input is None or self.last_normalized is None:
            raise ValueError("Forward pass must be called before backward pass")
        
        batch_size = len(self.last_input)
        features = len(self.last_input[0])
        
        # Gradients par rapport à gamma et beta
        self.grad_gamma = [0.0] * features
        self.grad_beta = [0.0] * features
        
        for i in range(features):
            for batch_idx in range(batch_size):
                self.grad_gamma[i] += grad_output[batch_idx][i] * self.last_normalized[batch_idx][i]
                self.grad_beta[i] += grad_output[batch_idx][i]
        
        # Moyenne sur le batch
        self.grad_gamma = [g / batch_size for g in self.grad_gamma]
        self.grad_beta = [g / batch_size for g in self.grad_beta]
        
        # Gradient par rapport à l'entrée (simplifié)
        grad_input = []
        for batch_idx in range(batch_size):
            grad_row = []
            for i in range(features):
                # Approximation du gradient
                grad = grad_output[batch_idx][i] * self.gamma[i]
                grad_row.append(grad)
            grad_input.append(grad_row)
        
        return grad_input
    
    def _compute_batch_mean(self, inputs: List[List[float]]) -> List[float]:
        """
        Calcule la moyenne du batch.
        """
        batch_size = len(inputs)
        features = len(inputs[0])
        
        means = [0.0] * features
        for row in inputs:
            for i, value in enumerate(row):
                means[i] += value
        
        return [m / batch_size for m in means]
    
    def _compute_batch_variance(self, inputs: List[List[float]], means: List[float]) -> List[float]:
        """
        Calcule la variance du batch.
        """
        batch_size = len(inputs)
        features = len(inputs[0])
        
        variances = [0.0] * features
        for row in inputs:
            for i, value in enumerate(row):
                variances[i] += (value - means[i]) ** 2
        
        return [v / batch_size for v in variances]
    
    def set_training(self, training: bool) -> None:
        """
        Définit le mode d'entraînement.
        """
        self.training = training
    
    def get_weights(self) -> Dict[str, Any]:
        """
        Retourne les poids de la couche.
        """
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "moving_mean": self.moving_mean,
            "moving_variance": self.moving_variance
        }
    
    def set_weights(self, weights: Dict[str, Any]) -> None:
        """
        Définit les poids de la couche.
        """
        if "gamma" in weights:
            self.gamma = weights["gamma"]
        if "beta" in weights:
            self.beta = weights["beta"]
        if "moving_mean" in weights:
            self.moving_mean = weights["moving_mean"]
        if "moving_variance" in weights:
            self.moving_variance = weights["moving_variance"]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de la couche.
        """
        config = super().get_config()
        config.update({
            "momentum": self.momentum,
            "epsilon": self.epsilon
        })
        return config

# ============================================================================
# Fonctions utilitaires pour créer des couches
# ============================================================================

def create_layer(layer_type: str, **kwargs) -> Layer:
    """
    Crée une couche du type spécifié.
    
    Args:
        layer_type: Type de couche ('dense', 'dropout', 'batch_norm')
        **kwargs: Arguments pour la couche
    
    Returns:
        Instance de la couche
    """
    layer_type = layer_type.lower()
    
    if layer_type == "dense":
        return Dense(**kwargs)
    elif layer_type == "dropout":
        return Dropout(**kwargs)
    elif layer_type == "batch_norm" or layer_type == "batchnormalization":
        return BatchNormalization(**kwargs)
    else:
        raise ValueError(f"Layer type '{layer_type}' not supported")

# Alias pour la compatibilité
FullyConnected = Dense
BN = BatchNormalization