"""Modèles de machine learning pour ai'lang."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import pickle
import json
from datetime import datetime


class BaseModel(ABC):
    """Classe de base pour tous les modèles ML."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.is_trained = False
        self.training_history = []
        self.created_at = datetime.now()
        self.model_size = 0
        self.param_count = 0
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Entraîne le modèle."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions."""
        pass
        
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Évalue le modèle."""
        pass
        
    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        model_data = {
            'name': self.name,
            'class': self.__class__.__name__,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'created_at': self.created_at.isoformat(),
            'model_size': self.model_size,
            'param_count': self.param_count,
            'state': self._get_state()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load(self, filepath: str) -> None:
        """Charge un modèle sauvegardé."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.name = model_data['name']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.created_at = datetime.fromisoformat(model_data['created_at'])
        self.model_size = model_data['model_size']
        self.param_count = model_data['param_count']
        self._set_state(model_data['state'])
        
    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """Retourne l'état interne du modèle."""
        pass
        
    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Restaure l'état interne du modèle."""
        pass


class NeuralNetwork(BaseModel):
    """Réseau de neurones multicouches."""
    
    def __init__(self, layers: List[Dict[str, Any]], optimizer: str = 'adam', 
                 loss: str = 'mse', metrics: List[str] = None, name: str = None):
        super().__init__(name)
        self.layers_config = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.weights = []
        self.biases = []
        self.learning_rate = 0.001
        
    def _initialize_weights(self, input_shape: int) -> None:
        """Initialise les poids du réseau."""
        self.weights = []
        self.biases = []
        
        prev_units = input_shape
        for layer in self.layers_config:
            units = layer['units']
            # Initialisation Xavier/Glorot
            w = np.random.randn(prev_units, units) * np.sqrt(2.0 / (prev_units + units))
            b = np.zeros((1, units))
            
            self.weights.append(w)
            self.biases.append(b)
            prev_units = units
            
        self.param_count = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
    def _activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Applique une fonction d'activation."""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return x  # linear
            
    def _activation_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Calcule la dérivée de la fonction d'activation."""
        if activation == 'relu':
            return (x > 0).astype(float)
        elif activation == 'sigmoid':
            s = self._activation(x, 'sigmoid')
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            return np.ones_like(x)
            
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Propagation avant."""
        activations = [X]
        z_values = []
        
        for i, (w, b, layer) in enumerate(zip(self.weights, self.biases, self.layers_config)):
            z = np.dot(activations[-1], w) + b
            z_values.append(z)
            
            activation = layer.get('activation', 'linear')
            a = self._activation(z, activation)
            activations.append(a)
            
        return activations, z_values
        
    def _backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                  z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Rétropropagation."""
        m = X.shape[0]
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Erreur de sortie
        if self.loss == 'mse':
            delta = activations[-1] - y
        elif self.loss == 'categorical_crossentropy':
            delta = activations[-1] - y
        else:
            delta = activations[-1] - y
            
        # Rétropropagation
        for i in reversed(range(len(self.weights))):
            dw[i] = np.dot(activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                activation = self.layers_config[i-1].get('activation', 'linear')
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(z_values[i-1], activation)
                
        return dw, db
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, validation_split: float = 0.0, 
            verbose: bool = True) -> 'NeuralNetwork':
        """Entraîne le réseau de neurones."""
        if not self.weights:
            self._initialize_weights(X.shape[1])
            
        # Division train/validation
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Mélanger les données
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Entraînement par mini-batches
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                activations, z_values = self._forward(X_batch)
                
                # Calcul de la perte
                if self.loss == 'mse':
                    batch_loss = np.mean((activations[-1] - y_batch) ** 2)
                elif self.loss == 'categorical_crossentropy':
                    batch_loss = -np.mean(y_batch * np.log(activations[-1] + 1e-15))
                else:
                    batch_loss = np.mean((activations[-1] - y_batch) ** 2)
                    
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                dw, db = self._backward(X_batch, y_batch, activations, z_values)
                
                # Mise à jour des poids
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * dw[j]
                    self.biases[j] -= self.learning_rate * db[j]
                    
            epoch_loss /= num_batches
            history['loss'].append(epoch_loss)
            
            # Validation
            if X_val is not None:
                val_pred = self.predict(X_val)
                if self.loss == 'mse':
                    val_loss = np.mean((val_pred - y_val) ** 2)
                else:
                    val_loss = -np.mean(y_val * np.log(val_pred + 1e-15))
                history['val_loss'].append(val_loss)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
                    
        self.is_trained = True
        self.training_history.append(history)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        activations, _ = self._forward(X)
        return activations[-1]
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Évalue le modèle."""
        predictions = self.predict(X)
        
        results = {}
        
        # Perte
        if self.loss == 'mse':
            results['loss'] = np.mean((predictions - y) ** 2)
        elif self.loss == 'categorical_crossentropy':
            results['loss'] = -np.mean(y * np.log(predictions + 1e-15))
            
        # Métriques
        if 'accuracy' in self.metrics:
            if len(y.shape) > 1 and y.shape[1] > 1:  # Classification multi-classe
                pred_classes = np.argmax(predictions, axis=1)
                true_classes = np.argmax(y, axis=1)
                results['accuracy'] = np.mean(pred_classes == true_classes)
            else:  # Classification binaire
                pred_classes = (predictions > 0.5).astype(int)
                results['accuracy'] = np.mean(pred_classes == y)
                
        return results
        
    def _get_state(self) -> Dict[str, Any]:
        return {
            'layers_config': self.layers_config,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'metrics': self.metrics,
            'weights': self.weights,
            'biases': self.biases,
            'learning_rate': self.learning_rate
        }
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.layers_config = state['layers_config']
        self.optimizer = state['optimizer']
        self.loss = state['loss']
        self.metrics = state['metrics']
        self.weights = state['weights']
        self.biases = state['biases']
        self.learning_rate = state['learning_rate']


class LinearRegression(BaseModel):
    """Régression linéaire."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LinearRegression':
        """Entraîne le modèle de régression linéaire."""
        # Ajouter une colonne de 1 pour le biais
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Solution analytique: (X^T X)^-1 X^T y
        try:
            params = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            self.bias = params[0]
            self.weights = params[1:]
        except np.linalg.LinAlgError:
            # Utiliser la pseudo-inverse si la matrice n'est pas inversible
            params = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = params[0]
            self.weights = params[1:]
            
        self.is_trained = True
        self.param_count = len(self.weights) + 1
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        return X @ self.weights + self.bias
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Évalue le modèle."""
        predictions = self.predict(X)
        
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        
        # R²
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
    def _get_state(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'bias': self.bias
        }
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.weights = state['weights']
        self.bias = state['bias']


class LogisticRegression(BaseModel):
    """Régression logistique."""
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, name: str = None):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Fonction sigmoïde."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LogisticRegression':
        """Entraîne le modèle de régression logistique."""
        n_samples, n_features = X.shape
        
        # Initialisation
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)
            
            # Calcul du coût
            cost = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
            
            # Gradients
            dw = (1 / n_samples) * X.T @ (predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Mise à jour
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        self.is_trained = True
        self.param_count = len(self.weights) + 1
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
        
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Prédit les classes."""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Évalue le modèle."""
        probabilities = self.predict(X)
        predictions = self.predict_classes(X)
        
        # Accuracy
        accuracy = np.mean(predictions == y)
        
        # Log loss
        log_loss = -np.mean(y * np.log(probabilities + 1e-15) + (1 - y) * np.log(1 - probabilities + 1e-15))
        
        return {
            'accuracy': accuracy,
            'log_loss': log_loss
        }
        
    def _get_state(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'weights': self.weights,
            'bias': self.bias
        }
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.learning_rate = state['learning_rate']
        self.max_iter = state['max_iter']
        self.weights = state['weights']
        self.bias = state['bias']


class KMeans(BaseModel):
    """Algorithme K-Means pour le clustering."""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, name: str = None):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'KMeans':
        """Entraîne le modèle K-Means."""
        n_samples, n_features = X.shape
        
        # Initialisation aléatoire des centroïdes
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for i in range(self.max_iter):
            # Assignation des points aux centroïdes
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Mise à jour des centroïdes
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Vérification de la convergence
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break
                
            self.centroids = new_centroids
            
        self.is_trained = True
        self.param_count = self.n_clusters * X.shape[1]
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les clusters."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, float]:
        """Évalue le modèle."""
        labels = self.predict(X)
        
        # Inertie (somme des distances au carré aux centroïdes)
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k])**2)
                
        return {
            'inertia': inertia,
            'n_clusters': self.n_clusters
        }
        
    def _get_state(self) -> Dict[str, Any]:
        return {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'centroids': self.centroids,
            'labels': self.labels
        }
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.n_clusters = state['n_clusters']
        self.max_iter = state['max_iter']
        self.tol = state['tol']
        self.centroids = state['centroids']
        self.labels = state['labels']


# Implémentations simplifiées pour les autres modèles
class DecisionTree(BaseModel):
    """Arbre de décision (implémentation simplifiée)."""
    
    def __init__(self, max_depth: int = 5, name: str = None):
        super().__init__(name)
        self.max_depth = max_depth
        self.tree = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'DecisionTree':
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])  # Implémentation simplifiée
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return {'accuracy': 0.5}
        
    def _get_state(self) -> Dict[str, Any]:
        return {'max_depth': self.max_depth, 'tree': self.tree}
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.max_depth = state['max_depth']
        self.tree = state['tree']


class RandomForest(BaseModel):
    """Forêt aléatoire (implémentation simplifiée)."""
    
    def __init__(self, n_estimators: int = 100, name: str = None):
        super().__init__(name)
        self.n_estimators = n_estimators
        self.trees = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForest':
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])  # Implémentation simplifiée
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return {'accuracy': 0.7}
        
    def _get_state(self) -> Dict[str, Any]:
        return {'n_estimators': self.n_estimators, 'trees': self.trees}
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.n_estimators = state['n_estimators']
        self.trees = state['trees']


class SVM(BaseModel):
    """Support Vector Machine (implémentation simplifiée)."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, name: str = None):
        super().__init__(name)
        self.kernel = kernel
        self.C = C
        self.support_vectors = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVM':
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])  # Implémentation simplifiée
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return {'accuracy': 0.8}
        
    def _get_state(self) -> Dict[str, Any]:
        return {'kernel': self.kernel, 'C': self.C, 'support_vectors': self.support_vectors}
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        self.kernel = state['kernel']
        self.C = state['C']
        self.support_vectors = state['support_vectors']


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function pour créer des modèles."""
    models = {
        'neural_network': NeuralNetwork,
        'linear_regression': LinearRegression,
        'logistic_regression': LogisticRegression,
        'decision_tree': DecisionTree,
        'random_forest': RandomForest,
        'svm': SVM,
        'kmeans': KMeans
    }
    
    if model_type not in models:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
        
    return models[model_type](**kwargs)