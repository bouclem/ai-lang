"""Module d'utilitaires pour ai'lang NLP."""

import json
import pickle
import csv
import re
import math
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import Counter
import numpy as np
from pathlib import Path


def load_dataset(filepath: str, format: str = "auto") -> Dict[str, List[str]]:
    """Charge un dataset depuis un fichier."""
    
    filepath = Path(filepath)
    
    if format == "auto":
        format = filepath.suffix.lower()
    
    if format in [".json", "json"]:
        return _load_json_dataset(filepath)
    elif format in [".csv", "csv"]:
        return _load_csv_dataset(filepath)
    elif format in [".txt", "txt"]:
        return _load_text_dataset(filepath)
    else:
        raise ValueError(f"Format de fichier non supporté: {format}")


def _load_json_dataset(filepath: Path) -> Dict[str, List[str]]:
    """Charge un dataset JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Liste de textes
        return {"texts": data}
    elif isinstance(data, dict):
        # Dictionnaire avec clés
        return data
    else:
        raise ValueError("Format JSON non supporté")


def _load_csv_dataset(filepath: Path) -> Dict[str, List[str]]:
    """Charge un dataset CSV."""
    data = {}
    
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(value)
    
    return data


def _load_text_dataset(filepath: Path) -> Dict[str, List[str]]:
    """Charge un dataset texte simple."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    return {"texts": lines}


def save_model(model: Any, filepath: str, format: str = "pickle"):
    """Sauvegarde un modèle."""
    
    if format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    elif format == "json":
        # Pour les modèles qui supportent la sérialisation JSON
        if hasattr(model, 'to_dict'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model.to_dict(), f, indent=2)
        else:
            raise ValueError("Le modèle ne supporte pas la sérialisation JSON")
    else:
        raise ValueError(f"Format de sauvegarde non supporté: {format}")


def load_model(filepath: str, format: str = "pickle") -> Any:
    """Charge un modèle."""
    
    if format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == "json":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Note: Il faudrait une méthode from_dict dans le modèle
        return data
    else:
        raise ValueError(f"Format de chargement non supporté: {format}")


def preprocess_dataset(texts: List[str], 
                      lowercase: bool = True,
                      remove_punctuation: bool = True,
                      remove_numbers: bool = False,
                      remove_extra_whitespace: bool = True,
                      min_length: int = 0,
                      max_length: Optional[int] = None) -> List[str]:
    """Préprocesse un dataset de textes."""
    
    processed_texts = []
    
    for text in texts:
        # Conversion en minuscules
        if lowercase:
            text = text.lower()
        
        # Suppression de la ponctuation
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Suppression des nombres
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Suppression des espaces multiples
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Filtrage par longueur
        if len(text) >= min_length:
            if max_length is None or len(text) <= max_length:
                processed_texts.append(text)
    
    return processed_texts


def split_sentences(text: str) -> List[str]:
    """Divise un texte en phrases."""
    
    # Patterns pour détecter la fin de phrase
    sentence_endings = r'[.!?]+'
    
    # Division en phrases
    sentences = re.split(sentence_endings, text)
    
    # Nettoyage et filtrage
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def count_words(text: str, unique: bool = False) -> Union[int, Dict[str, int]]:
    """Compte les mots dans un texte."""
    
    # Tokenisation simple
    words = re.findall(r'\w+', text.lower())
    
    if unique:
        return dict(Counter(words))
    else:
        return len(words)


def calculate_readability(text: str) -> Dict[str, float]:
    """Calcule des métriques de lisibilité du texte."""
    
    # Comptage de base
    sentences = split_sentences(text)
    words = re.findall(r'\w+', text)
    syllables = sum(_count_syllables(word) for word in words)
    
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = syllables
    
    if num_sentences == 0 or num_words == 0:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "avg_sentence_length": 0.0,
            "avg_syllables_per_word": 0.0
        }
    
    # Longueur moyenne des phrases
    avg_sentence_length = num_words / num_sentences
    
    # Nombre moyen de syllabes par mot
    avg_syllables_per_word = num_syllables / num_words
    
    # Score de facilité de lecture Flesch
    flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Niveau scolaire Flesch-Kincaid
    flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    return {
        "flesch_reading_ease": max(0, min(100, flesch_reading_ease)),
        "flesch_kincaid_grade": max(0, flesch_kincaid_grade),
        "avg_sentence_length": avg_sentence_length,
        "avg_syllables_per_word": avg_syllables_per_word
    }


def _count_syllables(word: str) -> int:
    """Compte approximativement les syllabes dans un mot."""
    
    word = word.lower()
    
    # Suppression des caractères non alphabétiques
    word = re.sub(r'[^a-z]', '', word)
    
    if len(word) == 0:
        return 0
    
    # Comptage des voyelles
    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    # Règles spéciales
    if word.endswith('e'):
        syllable_count -= 1
    
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1
    
    # Au moins une syllabe par mot
    return max(1, syllable_count)


def extract_keywords(text: str, num_keywords: int = 10, method: str = "tfidf") -> List[Tuple[str, float]]:
    """Extrait les mots-clés d'un texte."""
    
    # Tokenisation et nettoyage
    words = re.findall(r'\w+', text.lower())
    
    # Suppression des mots vides
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
        'its', 'our', 'their'
    }
    
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    if method == "frequency":
        # Fréquence simple
        word_counts = Counter(filtered_words)
        total_words = len(filtered_words)
        
        keywords = [(word, count / total_words) for word, count in word_counts.most_common(num_keywords)]
    
    elif method == "tfidf":
        # TF-IDF simplifié (un seul document)
        word_counts = Counter(filtered_words)
        total_words = len(filtered_words)
        unique_words = len(set(filtered_words))
        
        tfidf_scores = []
        for word, count in word_counts.items():
            tf = count / total_words
            # IDF simplifié (log du nombre total de mots uniques)
            idf = math.log(unique_words / (1 + count))
            tfidf = tf * idf
            tfidf_scores.append((word, tfidf))
        
        tfidf_scores.sort(key=lambda x: x[1], reverse=True)
        keywords = tfidf_scores[:num_keywords]
    
    else:
        raise ValueError(f"Méthode non supportée: {method}")
    
    return keywords


def text_statistics(text: str) -> Dict[str, Union[int, float]]:
    """Calcule des statistiques détaillées sur un texte."""
    
    # Comptages de base
    characters = len(text)
    characters_no_spaces = len(text.replace(' ', ''))
    
    sentences = split_sentences(text)
    num_sentences = len(sentences)
    
    words = re.findall(r'\w+', text)
    num_words = len(words)
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    num_paragraphs = len(paragraphs)
    
    # Mots uniques
    unique_words = set(word.lower() for word in words)
    num_unique_words = len(unique_words)
    
    # Longueurs
    word_lengths = [len(word) for word in words]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    sentence_lengths = [len(re.findall(r'\w+', sentence)) for sentence in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    
    # Diversité lexicale (Type-Token Ratio)
    lexical_diversity = num_unique_words / num_words if num_words > 0 else 0
    
    return {
        "characters": characters,
        "characters_no_spaces": characters_no_spaces,
        "words": num_words,
        "unique_words": num_unique_words,
        "sentences": num_sentences,
        "paragraphs": num_paragraphs,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity
    }


def clean_html(text: str) -> str:
    """Supprime les balises HTML d'un texte."""
    
    # Suppression des balises HTML
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Décodage des entités HTML communes
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' '
    }
    
    for entity, replacement in html_entities.items():
        clean_text = clean_text.replace(entity, replacement)
    
    return clean_text


def normalize_whitespace(text: str) -> str:
    """Normalise les espaces dans un texte."""
    
    # Remplacement des différents types d'espaces par des espaces normaux
    text = re.sub(r'[\t\r\n\f\v]', ' ', text)
    
    # Suppression des espaces multiples
    text = re.sub(r' +', ' ', text)
    
    # Suppression des espaces en début/fin
    text = text.strip()
    
    return text


def split_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Divise un texte en chunks avec chevauchement."""
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Si ce n'est pas le dernier chunk, essayer de couper à un espace
        if end < len(text):
            # Chercher le dernier espace avant la fin
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Déplacement avec chevauchement
        start = end - overlap
        
        # Éviter les boucles infinies
        if start <= chunks[-1] if chunks else 0:
            start = end
    
    return chunks


def detect_encoding(filepath: str) -> str:
    """Détecte l'encodage d'un fichier texte."""
    
    # Essai de différents encodages
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    
    # Par défaut
    return 'utf-8'