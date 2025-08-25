"""Module d'embeddings pour ai'lang NLP."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import json
import pickle
from collections import defaultdict, Counter
import math


class WordEmbedding(ABC):
    """Classe de base pour les embeddings de mots."""
    
    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self.word_vectors = {}
        self.vocab = set()
        
    @abstractmethod
    def train(self, corpus: List[List[str]], **kwargs):
        """Entraîne le modèle d'embedding."""
        pass
    
    @abstractmethod
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Retourne le vecteur d'un mot."""
        pass
    
    def similarity(self, word1: str, word2: str) -> float:
        """Calcule la similarité cosinus entre deux mots."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Similarité cosinus
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Trouve les mots les plus similaires."""
        if word not in self.vocab:
            return []
        
        word_vec = self.get_vector(word)
        if word_vec is None:
            return []
        
        similarities = []
        for other_word in self.vocab:
            if other_word != word:
                sim = self.similarity(word, other_word)
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        data = {
            'vector_size': self.vector_size,
            'word_vectors': {word: vec.tolist() for word, vec in self.word_vectors.items()},
            'vocab': list(self.vocab)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Charge le modèle."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vector_size = data['vector_size']
        self.word_vectors = {word: np.array(vec) for word, vec in data['word_vectors'].items()}
        self.vocab = set(data['vocab'])


class Word2Vec(WordEmbedding):
    """Implémentation simplifiée de Word2Vec."""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, 
                 sg: int = 0, epochs: int = 5, learning_rate: float = 0.025):
        super().__init__(vector_size)
        self.window = window
        self.min_count = min_count
        self.sg = sg  # 0 for CBOW, 1 for Skip-gram
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def train(self, corpus: List[List[str]], **kwargs):
        """Entraîne le modèle Word2Vec."""
        # Construction du vocabulaire
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)
        
        # Filtrage par fréquence minimale
        self.vocab = {word for word, count in word_counts.items() if count >= self.min_count}
        
        # Initialisation des vecteurs
        for word in self.vocab:
            self.word_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size) / self.vector_size
        
        # Entraînement simplifié
        for epoch in range(self.epochs):
            for sentence in corpus:
                for i, target_word in enumerate(sentence):
                    if target_word not in self.vocab:
                        continue
                    
                    # Définition de la fenêtre de contexte
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)
                    
                    context_words = [sentence[j] for j in range(start, end) if j != i and sentence[j] in self.vocab]
                    
                    if self.sg == 0:  # CBOW
                        self._train_cbow(target_word, context_words)
                    else:  # Skip-gram
                        self._train_skipgram(target_word, context_words)
    
    def _train_cbow(self, target_word: str, context_words: List[str]):
        """Entraînement CBOW simplifié."""
        if not context_words:
            return
        
        # Moyenne des vecteurs de contexte
        context_vector = np.mean([self.word_vectors[word] for word in context_words], axis=0)
        target_vector = self.word_vectors[target_word]
        
        # Mise à jour simple (gradient descent approximation)
        error = target_vector - context_vector
        self.word_vectors[target_word] += self.learning_rate * error
        
        for context_word in context_words:
            self.word_vectors[context_word] += self.learning_rate * error / len(context_words)
    
    def _train_skipgram(self, target_word: str, context_words: List[str]):
        """Entraînement Skip-gram simplifié."""
        target_vector = self.word_vectors[target_word]
        
        for context_word in context_words:
            context_vector = self.word_vectors[context_word]
            
            # Mise à jour simple
            error = context_vector - target_vector
            self.word_vectors[target_word] += self.learning_rate * error
            self.word_vectors[context_word] += self.learning_rate * error
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Retourne le vecteur d'un mot."""
        return self.word_vectors.get(word)


class GloVe(WordEmbedding):
    """Implémentation simplifiée de GloVe."""
    
    def __init__(self, vector_size: int = 100, window: int = 15, epochs: int = 50, 
                 learning_rate: float = 0.05, x_max: int = 100, alpha: float = 0.75):
        super().__init__(vector_size)
        self.window = window
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(float))
        
    def train(self, corpus: List[List[str]], **kwargs):
        """Entraîne le modèle GloVe."""
        # Construction du vocabulaire
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)
        
        self.vocab = set(word_counts.keys())
        
        # Construction de la matrice de cooccurrence
        self._build_cooccurrence_matrix(corpus)
        
        # Initialisation des vecteurs
        for word in self.vocab:
            self.word_vectors[word] = np.random.normal(0, 0.1, self.vector_size)
        
        # Entraînement
        for epoch in range(self.epochs):
            self._train_epoch()
    
    def _build_cooccurrence_matrix(self, corpus: List[List[str]]):
        """Construit la matrice de cooccurrence."""
        for sentence in corpus:
            for i, word1 in enumerate(sentence):
                if word1 not in self.vocab:
                    continue
                
                for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1)):
                    if i != j:
                        word2 = sentence[j]
                        if word2 in self.vocab:
                            distance = abs(i - j)
                            self.cooccurrence_matrix[word1][word2] += 1.0 / distance
    
    def _train_epoch(self):
        """Entraîne une époque."""
        for word1 in self.cooccurrence_matrix:
            for word2, count in self.cooccurrence_matrix[word1].items():
                if count > 0:
                    self._update_vectors(word1, word2, count)
    
    def _update_vectors(self, word1: str, word2: str, count: float):
        """Met à jour les vecteurs pour une paire de mots."""
        vec1 = self.word_vectors[word1]
        vec2 = self.word_vectors[word2]
        
        # Fonction de pondération
        weight = min(1.0, (count / self.x_max) ** self.alpha)
        
        # Calcul de l'erreur
        dot_product = np.dot(vec1, vec2)
        error = weight * (dot_product - math.log(count))
        
        # Mise à jour des gradients
        gradient = error * vec2
        self.word_vectors[word1] -= self.learning_rate * gradient
        
        gradient = error * vec1
        self.word_vectors[word2] -= self.learning_rate * gradient
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Retourne le vecteur d'un mot."""
        return self.word_vectors.get(word)


class FastText(WordEmbedding):
    """Implémentation simplifiée de FastText."""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1,
                 epochs: int = 5, learning_rate: float = 0.025, min_n: int = 3, max_n: int = 6):
        super().__init__(vector_size)
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_n = min_n
        self.max_n = max_n
        self.subword_vectors = {}
        
    def _get_subwords(self, word: str) -> List[str]:
        """Génère les sous-mots d'un mot."""
        word = f"<{word}>"
        subwords = []
        
        for i in range(len(word)):
            for j in range(i + self.min_n, min(len(word) + 1, i + self.max_n + 1)):
                subwords.append(word[i:j])
        
        return subwords
    
    def train(self, corpus: List[List[str]], **kwargs):
        """Entraîne le modèle FastText."""
        # Construction du vocabulaire
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)
        
        self.vocab = {word for word, count in word_counts.items() if count >= self.min_count}
        
        # Initialisation des vecteurs de mots et sous-mots
        all_subwords = set()
        for word in self.vocab:
            self.word_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size) / self.vector_size
            subwords = self._get_subwords(word)
            all_subwords.update(subwords)
        
        for subword in all_subwords:
            self.subword_vectors[subword] = np.random.uniform(-0.5, 0.5, self.vector_size) / self.vector_size
        
        # Entraînement
        for epoch in range(self.epochs):
            for sentence in corpus:
                for i, target_word in enumerate(sentence):
                    if target_word not in self.vocab:
                        continue
                    
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)
                    
                    context_words = [sentence[j] for j in range(start, end) if j != i and sentence[j] in self.vocab]
                    
                    self._train_word(target_word, context_words)
    
    def _train_word(self, target_word: str, context_words: List[str]):
        """Entraîne un mot avec son contexte."""
        if not context_words:
            return
        
        target_vector = self.get_vector(target_word)
        context_vector = np.mean([self.get_vector(word) for word in context_words], axis=0)
        
        error = target_vector - context_vector
        
        # Mise à jour du vecteur principal
        self.word_vectors[target_word] += self.learning_rate * error
        
        # Mise à jour des sous-mots
        subwords = self._get_subwords(target_word)
        for subword in subwords:
            if subword in self.subword_vectors:
                self.subword_vectors[subword] += self.learning_rate * error / len(subwords)
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Retourne le vecteur d'un mot (combinaison mot + sous-mots)."""
        if word not in self.vocab:
            # Pour les mots hors vocabulaire, utiliser seulement les sous-mots
            subwords = self._get_subwords(word)
            subword_vecs = [self.subword_vectors[sw] for sw in subwords if sw in self.subword_vectors]
            
            if subword_vecs:
                return np.mean(subword_vecs, axis=0)
            else:
                return None
        
        # Pour les mots du vocabulaire, combiner le vecteur du mot et des sous-mots
        word_vec = self.word_vectors[word]
        subwords = self._get_subwords(word)
        subword_vecs = [self.subword_vectors[sw] for sw in subwords if sw in self.subword_vectors]
        
        if subword_vecs:
            subword_mean = np.mean(subword_vecs, axis=0)
            return (word_vec + subword_mean) / 2
        
        return word_vec


class BERTEmbedding(WordEmbedding):
    """Embedding BERT simplifié (simulation)."""
    
    def __init__(self, vector_size: int = 768, max_length: int = 512):
        super().__init__(vector_size)
        self.max_length = max_length
        self.tokenizer_vocab = {}
        
    def train(self, corpus: List[List[str]], **kwargs):
        """Simule l'entraînement BERT."""
        # Construction du vocabulaire
        all_words = set()
        for sentence in corpus:
            all_words.update(sentence)
        
        self.vocab = all_words
        
        # Simulation de vecteurs BERT (normalement pré-entraînés)
        for word in self.vocab:
            self.word_vectors[word] = np.random.normal(0, 0.1, self.vector_size)
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Retourne le vecteur BERT d'un mot."""
        return self.word_vectors.get(word)
    
    def encode_sentence(self, sentence: List[str]) -> np.ndarray:
        """Encode une phrase complète."""
        vectors = []
        for word in sentence[:self.max_length]:
            vec = self.get_vector(word)
            if vec is not None:
                vectors.append(vec)
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)


def create_embedding(embedding_type: str, **kwargs) -> WordEmbedding:
    """Factory pour créer des embeddings."""
    if embedding_type.lower() == "word2vec":
        return Word2Vec(**kwargs)
    elif embedding_type.lower() == "glove":
        return GloVe(**kwargs)
    elif embedding_type.lower() == "fasttext":
        return FastText(**kwargs)
    elif embedding_type.lower() == "bert":
        return BERTEmbedding(**kwargs)
    else:
        raise ValueError(f"Type d'embedding non supporté: {embedding_type}")