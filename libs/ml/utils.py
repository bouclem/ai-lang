# Utilitaires pour la bibliothèque ML d'ai'lang

import json
import pickle
import random
import math
from typing import List, Tuple, Dict, Any, Optional, Union

# ============================================================================
# Gestion des données
# ============================================================================

def train_test_split(X: List, y: List, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[List, List, List, List]:
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Args:
        X: Données d'entrée
        y: Labels/cibles
        test_size: Proportion des données pour le test (0.0 à 1.0)
        random_state: Graine pour la reproductibilité
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Créer les indices et les mélanger
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculer la taille de l'ensemble de test
    test_count = int(len(X) * test_size)
    
    # Diviser les indices
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Créer les ensembles
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def cross_validation_split(X: List, y: List, k_folds: int = 5, random_state: Optional[int] = None) -> List[Tuple[List, List, List, List]]:
    """
    Crée des plis pour la validation croisée k-fold.
    
    Args:
        X: Données d'entrée
        y: Labels/cibles
        k_folds: Nombre de plis
        random_state: Graine pour la reproductibilité
    
    Returns:
        Liste de tuples (X_train, X_val, y_train, y_val) pour chaque pli
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Créer les indices et les mélanger
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculer la taille de chaque pli
    fold_size = len(X) // k_folds
    folds = []
    
    for i in range(k_folds):
        # Indices de validation pour ce pli
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k_folds - 1 else len(X)
        val_indices = indices[start_idx:end_idx]
        
        # Indices d'entraînement (tous les autres)
        train_indices = indices[:start_idx] + indices[end_idx:]
        
        # Créer les ensembles pour ce pli
        X_train = [X[idx] for idx in train_indices]
        X_val = [X[idx] for idx in val_indices]
        y_train = [y[idx] for idx in train_indices]
        y_val = [y[idx] for idx in val_indices]
        
        folds.append((X_train, X_val, y_train, y_val))
    
    return folds

def normalize_data(data: List[List[float]], method: str = "minmax") -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Normalise les données numériques.
    
    Args:
        data: Données à normaliser (liste de listes)
        method: Méthode de normalisation ('minmax', 'zscore', 'robust')
    
    Returns:
        Données normalisées et paramètres de normalisation
    """
    if not data or not data[0]:
        return data, {}
    
    n_features = len(data[0])
    normalized_data = []
    params = {"method": method, "features": []}
    
    # Calculer les statistiques par feature
    for feature_idx in range(n_features):
        feature_values = [row[feature_idx] for row in data]
        
        if method == "minmax":
            min_val = min(feature_values)
            max_val = max(feature_values)
            params["features"].append({"min": min_val, "max": max_val})
        
        elif method == "zscore":
            mean_val = sum(feature_values) / len(feature_values)
            variance = sum((x - mean_val) ** 2 for x in feature_values) / len(feature_values)
            std_val = math.sqrt(variance)
            params["features"].append({"mean": mean_val, "std": std_val})
        
        elif method == "robust":
            sorted_values = sorted(feature_values)
            n = len(sorted_values)
            median = sorted_values[n // 2]
            q1 = sorted_values[n // 4]
            q3 = sorted_values[3 * n // 4]
            iqr = q3 - q1
            params["features"].append({"median": median, "iqr": iqr})
    
    # Normaliser les données
    for row in data:
        normalized_row = []
        for feature_idx, value in enumerate(row):
            feature_params = params["features"][feature_idx]
            
            if method == "minmax":
                min_val, max_val = feature_params["min"], feature_params["max"]
                if max_val > min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.0
            
            elif method == "zscore":
                mean_val, std_val = feature_params["mean"], feature_params["std"]
                if std_val > 0:
                    normalized_value = (value - mean_val) / std_val
                else:
                    normalized_value = 0.0
            
            elif method == "robust":
                median, iqr = feature_params["median"], feature_params["iqr"]
                if iqr > 0:
                    normalized_value = (value - median) / iqr
                else:
                    normalized_value = 0.0
            
            normalized_row.append(normalized_value)
        
        normalized_data.append(normalized_row)
    
    return normalized_data, params

def apply_normalization(data: List[List[float]], params: Dict[str, Any]) -> List[List[float]]:
    """
    Applique une normalisation existante à de nouvelles données.
    
    Args:
        data: Nouvelles données à normaliser
        params: Paramètres de normalisation obtenus de normalize_data
    
    Returns:
        Données normalisées
    """
    if not data or not params:
        return data
    
    method = params["method"]
    normalized_data = []
    
    for row in data:
        normalized_row = []
        for feature_idx, value in enumerate(row):
            if feature_idx < len(params["features"]):
                feature_params = params["features"][feature_idx]
                
                if method == "minmax":
                    min_val, max_val = feature_params["min"], feature_params["max"]
                    if max_val > min_val:
                        normalized_value = (value - min_val) / (max_val - min_val)
                    else:
                        normalized_value = 0.0
                
                elif method == "zscore":
                    mean_val, std_val = feature_params["mean"], feature_params["std"]
                    if std_val > 0:
                        normalized_value = (value - mean_val) / std_val
                    else:
                        normalized_value = 0.0
                
                elif method == "robust":
                    median, iqr = feature_params["median"], feature_params["iqr"]
                    if iqr > 0:
                        normalized_value = (value - median) / iqr
                    else:
                        normalized_value = 0.0
                
                normalized_row.append(normalized_value)
            else:
                normalized_row.append(value)
        
        normalized_data.append(normalized_row)
    
    return normalized_data

# ============================================================================
# Métriques d'évaluation
# ============================================================================

def accuracy_score(y_true: List, y_pred: List) -> float:
    """
    Calcule la précision (accuracy).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes doivent avoir la même longueur")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_score(y_true: List, y_pred: List, average: str = "binary") -> Union[float, Dict[str, float]]:
    """
    Calcule la précision.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        average: 'binary', 'macro', 'micro', 'weighted'
    
    Returns:
        Score de précision
    """
    if average == "binary":
        # Pour classification binaire
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    else:
        # Pour classification multi-classe
        classes = list(set(y_true + y_pred))
        precisions = {}
        
        for cls in classes:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
            precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        if average == "macro":
            return sum(precisions.values()) / len(precisions)
        elif average == "weighted":
            weights = {cls: y_true.count(cls) / len(y_true) for cls in classes}
            return sum(precisions[cls] * weights[cls] for cls in classes)
        else:
            return precisions

def recall_score(y_true: List, y_pred: List, average: str = "binary") -> Union[float, Dict[str, float]]:
    """
    Calcule le rappel.
    """
    if average == "binary":
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    else:
        classes = list(set(y_true + y_pred))
        recalls = {}
        
        for cls in classes:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
            recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if average == "macro":
            return sum(recalls.values()) / len(recalls)
        elif average == "weighted":
            weights = {cls: y_true.count(cls) / len(y_true) for cls in classes}
            return sum(recalls[cls] * weights[cls] for cls in classes)
        else:
            return recalls

def f1_score(y_true: List, y_pred: List, average: str = "binary") -> Union[float, Dict[str, float]]:
    """
    Calcule le F1-score.
    """
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    
    if isinstance(precision, dict):
        f1_scores = {}
        for cls in precision.keys():
            p, r = precision[cls], recall[cls]
            f1_scores[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        if average == "macro":
            return sum(f1_scores.values()) / len(f1_scores)
        elif average == "weighted":
            weights = {cls: y_true.count(cls) / len(y_true) for cls in f1_scores.keys()}
            return sum(f1_scores[cls] * weights[cls] for cls in f1_scores.keys())
        else:
            return f1_scores
    else:
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calcule l'erreur quadratique moyenne.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes doivent avoir la même longueur")
    
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calcule l'erreur absolue moyenne.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes doivent avoir la même longueur")
    
    return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calcule le coefficient de détermination R².
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes doivent avoir la même longueur")
    
    y_mean = sum(y_true) / len(y_true)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def confusion_matrix(y_true: List, y_pred: List) -> Dict[str, Dict[str, int]]:
    """
    Calcule la matrice de confusion.
    
    Returns:
        Dictionnaire de dictionnaires représentant la matrice de confusion
    """
    classes = sorted(list(set(y_true + y_pred)))
    matrix = {true_cls: {pred_cls: 0 for pred_cls in classes} for true_cls in classes}
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix

# ============================================================================
# Sauvegarde et chargement de modèles
# ============================================================================

def save_model(model: Any, filepath: str, format: str = "pickle") -> bool:
    """
    Sauvegarde un modèle.
    
    Args:
        model: Modèle à sauvegarder
        filepath: Chemin du fichier
        format: Format de sauvegarde ('pickle', 'json')
    
    Returns:
        True si succès, False sinon
    """
    try:
        if format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        elif format == "json":
            # Pour les modèles sérialisables en JSON
            if hasattr(model, 'to_dict'):
                model_dict = model.to_dict()
            else:
                model_dict = model.__dict__
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_dict, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Format non supporté: {format}")
        
        return True
    
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return False

def load_model(filepath: str, format: str = "pickle") -> Any:
    """
    Charge un modèle.
    
    Args:
        filepath: Chemin du fichier
        format: Format de chargement ('pickle', 'json')
    
    Returns:
        Modèle chargé ou None si erreur
    """
    try:
        if format == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return None

# ============================================================================
# Utilitaires de données
# ============================================================================

def load_dataset(filepath: str, format: str = "auto") -> Tuple[List, List]:
    """
    Charge un dataset depuis un fichier.
    
    Args:
        filepath: Chemin du fichier
        format: Format du fichier ('auto', 'csv', 'json', 'txt')
    
    Returns:
        Tuple (données, labels) ou (None, None) si erreur
    """
    try:
        if format == "auto":
            if filepath.endswith('.csv'):
                format = "csv"
            elif filepath.endswith('.json'):
                format = "json"
            elif filepath.endswith('.txt'):
                format = "txt"
            else:
                raise ValueError("Format de fichier non reconnu")
        
        if format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'X' in data and 'y' in data:
                    return data['X'], data['y']
                else:
                    return data, []
        
        elif format == "csv":
            # Lecture CSV simple
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(',')
                    data.append(parts)
            
            if data:
                X = [row[:-1] for row in data]  # Toutes les colonnes sauf la dernière
                y = [row[-1] for row in data]   # Dernière colonne
                return X, y
            else:
                return [], []
        
        elif format == "txt":
            # Lecture de fichier texte simple
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
                return lines, []
        
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {e}")
        return None, None

def generate_synthetic_data(n_samples: int, n_features: int, n_classes: int = 2, 
                          noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[List[List[float]], List[int]]:
    """
    Génère des données synthétiques pour les tests.
    
    Args:
        n_samples: Nombre d'échantillons
        n_features: Nombre de caractéristiques
        n_classes: Nombre de classes
        noise: Niveau de bruit
        random_state: Graine pour la reproductibilité
    
    Returns:
        Tuple (X, y) avec les données et labels
    """
    if random_state is not None:
        random.seed(random_state)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Générer un échantillon
        sample = []
        for j in range(n_features):
            # Valeur de base + bruit
            base_value = (i % n_classes) * 2.0 + j * 0.5
            noisy_value = base_value + random.gauss(0, noise)
            sample.append(noisy_value)
        
        X.append(sample)
        y.append(i % n_classes)
    
    return X, y

# ============================================================================
# Fonctions utilitaires mathématiques
# ============================================================================

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calcule la distance euclidienne entre deux points.
    """
    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir la même dimension")
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Les vecteurs doivent avoir la même dimension")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def sigmoid(x: float) -> float:
    """
    Fonction sigmoïde.
    """
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

def softmax(values: List[float]) -> List[float]:
    """
    Fonction softmax.
    """
    # Stabilité numérique
    max_val = max(values)
    exp_values = [math.exp(x - max_val) for x in values]
    sum_exp = sum(exp_values)
    
    return [exp_val / sum_exp for exp_val in exp_values]

def one_hot_encode(labels: List, num_classes: Optional[int] = None) -> List[List[int]]:
    """
    Encode les labels en one-hot.
    
    Args:
        labels: Liste des labels
        num_classes: Nombre de classes (auto-détecté si None)
    
    Returns:
        Matrice one-hot encodée
    """
    unique_labels = sorted(list(set(labels)))
    if num_classes is None:
        num_classes = len(unique_labels)
    
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    encoded = []
    for label in labels:
        one_hot = [0] * num_classes
        if label in label_to_index:
            one_hot[label_to_index[label]] = 1
        encoded.append(one_hot)
    
    return encoded

def bootstrap_sample(X: List, y: List, random_state: Optional[int] = None) -> Tuple[List, List]:
    """
    Crée un échantillon bootstrap.
    
    Args:
        X: Données d'entrée
        y: Labels
        random_state: Graine pour la reproductibilité
    
    Returns:
        Échantillon bootstrap (X_bootstrap, y_bootstrap)
    """
    if random_state is not None:
        random.seed(random_state)
    
    n_samples = len(X)
    indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
    
    X_bootstrap = [X[i] for i in indices]
    y_bootstrap = [y[i] for i in indices]
    
    return X_bootstrap, y_bootstrap