"""Module de métriques pour l'évaluation des modèles NLP."""

import numpy as np
from typing import List, Dict, Tuple, Union, Set
from collections import Counter
import re
import math


def bleu_score(reference: str, candidate: str, n: int = 4) -> float:
    """Calcule le score BLEU entre une référence et un candidat."""
    
    def tokenize(text: str) -> List[str]:
        """Tokenise le texte."""
        return text.lower().split()
    
    def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Génère les n-grammes."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Calcul des précisions pour chaque n-gramme
    precisions = []
    
    for i in range(1, n + 1):
        ref_ngrams = Counter(get_ngrams(ref_tokens, i))
        cand_ngrams = Counter(get_ngrams(cand_tokens, i))
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Comptage des n-grammes correspondants
        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        precision = matches / sum(cand_ngrams.values())
        precisions.append(precision)
    
    # Pénalité de brièveté
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Score BLEU final
    if all(p > 0 for p in precisions):
        log_precisions = [math.log(p) for p in precisions]
        bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
    else:
        bleu = 0.0
    
    return bleu


def rouge_score(reference: str, candidate: str, rouge_type: str = "rouge-1") -> Dict[str, float]:
    """Calcule le score ROUGE."""
    
    def tokenize(text: str) -> List[str]:
        """Tokenise le texte."""
        return text.lower().split()
    
    def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
        """Génère les n-grammes uniques."""
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if rouge_type == "rouge-1":
        n = 1
    elif rouge_type == "rouge-2":
        n = 2
    elif rouge_type == "rouge-l":
        # ROUGE-L utilise la plus longue sous-séquence commune
        return _rouge_l(ref_tokens, cand_tokens)
    else:
        raise ValueError(f"Type ROUGE non supporté: {rouge_type}")
    
    ref_ngrams = get_ngrams(ref_tokens, n)
    cand_ngrams = get_ngrams(cand_tokens, n)
    
    if len(ref_ngrams) == 0 and len(cand_ngrams) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(ref_ngrams) == 0 or len(cand_ngrams) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Intersection des n-grammes
    overlap = ref_ngrams.intersection(cand_ngrams)
    
    precision = len(overlap) / len(cand_ngrams)
    recall = len(overlap) / len(ref_ngrams)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def _rouge_l(ref_tokens: List[str], cand_tokens: List[str]) -> Dict[str, float]:
    """Calcule ROUGE-L basé sur la plus longue sous-séquence commune."""
    
    def lcs_length(x: List[str], y: List[str]) -> int:
        """Calcule la longueur de la plus longue sous-séquence commune."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    if len(ref_tokens) == 0 and len(cand_tokens) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs_len = lcs_length(ref_tokens, cand_tokens)
    
    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def perplexity(probabilities: List[float]) -> float:
    """Calcule la perplexité d'un modèle de langue."""
    if not probabilities or any(p <= 0 for p in probabilities):
        return float('inf')
    
    log_probs = [math.log(p) for p in probabilities]
    avg_log_prob = sum(log_probs) / len(log_probs)
    
    return math.exp(-avg_log_prob)


def f1_score_nlp(y_true: List[str], y_pred: List[str], average: str = "macro") -> Union[float, Dict[str, float]]:
    """Calcule le F1-score pour la classification de texte."""
    
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes y_true et y_pred doivent avoir la même longueur")
    
    # Obtenir toutes les classes uniques
    classes = set(y_true + y_pred)
    
    class_metrics = {}
    
    for cls in classes:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    if average == "macro":
        # Moyenne non pondérée
        avg_f1 = sum(metrics["f1"] for metrics in class_metrics.values()) / len(class_metrics)
        return avg_f1
    elif average == "micro":
        # Calcul global
        total_tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total_samples = len(y_true)
        return total_tp / total_samples if total_samples > 0 else 0.0
    elif average == "weighted":
        # Moyenne pondérée par le support
        class_counts = Counter(y_true)
        total_samples = len(y_true)
        
        weighted_f1 = 0.0
        for cls, metrics in class_metrics.items():
            weight = class_counts[cls] / total_samples
            weighted_f1 += weight * metrics["f1"]
        
        return weighted_f1
    elif average is None:
        # Retourner les métriques pour chaque classe
        return {cls: metrics["f1"] for cls, metrics in class_metrics.items()}
    else:
        raise ValueError(f"Paramètre 'average' non supporté: {average}")


def accuracy_nlp(y_true: List[str], y_pred: List[str]) -> float:
    """Calcule l'exactitude pour la classification de texte."""
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes y_true et y_pred doivent avoir la même longueur")
    
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def precision_recall_nlp(y_true: List[str], y_pred: List[str], average: str = "macro") -> Tuple[float, float]:
    """Calcule la précision et le rappel pour la classification de texte."""
    
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes y_true et y_pred doivent avoir la même longueur")
    
    classes = set(y_true + y_pred)
    
    class_metrics = {}
    
    for cls in classes:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall
        }
    
    if average == "macro":
        avg_precision = sum(metrics["precision"] for metrics in class_metrics.values()) / len(class_metrics)
        avg_recall = sum(metrics["recall"] for metrics in class_metrics.values()) / len(class_metrics)
        return avg_precision, avg_recall
    elif average == "micro":
        # Calcul global
        total_tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total_samples = len(y_true)
        micro_score = total_tp / total_samples if total_samples > 0 else 0.0
        return micro_score, micro_score
    elif average == "weighted":
        class_counts = Counter(y_true)
        total_samples = len(y_true)
        
        weighted_precision = 0.0
        weighted_recall = 0.0
        
        for cls, metrics in class_metrics.items():
            weight = class_counts[cls] / total_samples
            weighted_precision += weight * metrics["precision"]
            weighted_recall += weight * metrics["recall"]
        
        return weighted_precision, weighted_recall
    else:
        raise ValueError(f"Paramètre 'average' non supporté: {average}")


def confusion_matrix_nlp(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    """Calcule la matrice de confusion pour la classification de texte."""
    
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes y_true et y_pred doivent avoir la même longueur")
    
    classes = sorted(set(y_true + y_pred))
    
    # Initialisation de la matrice
    matrix = {true_cls: {pred_cls: 0 for pred_cls in classes} for true_cls in classes}
    
    # Remplissage de la matrice
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix


def classification_report_nlp(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    """Génère un rapport de classification complet."""
    
    classes = sorted(set(y_true + y_pred))
    report = {}
    
    # Métriques par classe
    for cls in classes:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for true in y_true if true == cls)
        
        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
    
    # Moyennes
    macro_precision = sum(report[cls]["precision"] for cls in classes) / len(classes)
    macro_recall = sum(report[cls]["recall"] for cls in classes) / len(classes)
    macro_f1 = sum(report[cls]["f1-score"] for cls in classes) / len(classes)
    
    total_support = len(y_true)
    weighted_precision = sum(report[cls]["precision"] * report[cls]["support"] for cls in classes) / total_support
    weighted_recall = sum(report[cls]["recall"] * report[cls]["support"] for cls in classes) / total_support
    weighted_f1 = sum(report[cls]["f1-score"] * report[cls]["support"] for cls in classes) / total_support
    
    accuracy = accuracy_nlp(y_true, y_pred)
    
    report["accuracy"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1-score": accuracy,
        "support": total_support
    }
    
    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": total_support
    }
    
    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": total_support
    }
    
    return report


def semantic_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """Calcule la similarité sémantique entre deux textes."""
    
    def tokenize(text: str) -> Set[str]:
        """Tokenise et normalise le texte."""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return set(words)
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if method == "jaccard":
        # Indice de Jaccard
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if len(union) == 0:
            return 1.0 if len(tokens1) == 0 and len(tokens2) == 0 else 0.0
        
        return len(intersection) / len(union)
    
    elif method == "cosine":
        # Similarité cosinus basée sur les mots
        all_words = tokens1.union(tokens2)
        
        if len(all_words) == 0:
            return 1.0
        
        # Vecteurs de fréquence
        vec1 = [1 if word in tokens1 else 0 for word in all_words]
        vec2 = [1 if word in tokens2 else 0 for word in all_words]
        
        # Produit scalaire
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Normes
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    else:
        raise ValueError(f"Méthode non supportée: {method}")