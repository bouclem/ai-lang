"""Module de modèles NLP pour ai'lang."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import json
import pickle
from collections import Counter, defaultdict
import re


class BaseNLPModel(ABC):
    """Classe de base pour les modèles NLP."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def train(self, X: List[str], y: Optional[List] = None, **kwargs):
        """Entraîne le modèle."""
        pass
    
    @abstractmethod
    def predict(self, X: Union[str, List[str]]) -> Union[Any, List[Any]]:
        """Fait des prédictions."""
        pass
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Charge le modèle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class TextClassifier(BaseNLPModel):
    """Classificateur de texte simple."""
    
    def __init__(self, algorithm: str = "naive_bayes"):
        super().__init__("TextClassifier")
        self.algorithm = algorithm
        self.vocabulary = {}
        self.class_probs = {}
        self.word_probs = {}
        self.classes = set()
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenise le texte."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _build_vocabulary(self, texts: List[str]):
        """Construit le vocabulaire."""
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Garder seulement les mots fréquents
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.most_common(5000))}
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """Convertit le texte en vecteur de caractéristiques."""
        features = np.zeros(len(self.vocabulary))
        tokens = self._tokenize(text)
        
        for token in tokens:
            if token in self.vocabulary:
                features[self.vocabulary[token]] += 1
        
        return features
    
    def train(self, X: List[str], y: List[str], **kwargs):
        """Entraîne le classificateur."""
        self.classes = set(y)
        self._build_vocabulary(X)
        
        if self.algorithm == "naive_bayes":
            self._train_naive_bayes(X, y)
        elif self.algorithm == "logistic_regression":
            self._train_logistic_regression(X, y)
        
        self.is_trained = True
    
    def _train_naive_bayes(self, X: List[str], y: List[str]):
        """Entraîne un classificateur Naive Bayes."""
        # Calcul des probabilités de classes
        class_counts = Counter(y)
        total_docs = len(y)
        
        for class_name, count in class_counts.items():
            self.class_probs[class_name] = count / total_docs
        
        # Calcul des probabilités de mots par classe
        class_word_counts = defaultdict(lambda: defaultdict(int))
        class_total_words = defaultdict(int)
        
        for text, label in zip(X, y):
            tokens = self._tokenize(text)
            for token in tokens:
                if token in self.vocabulary:
                    class_word_counts[label][token] += 1
                    class_total_words[label] += 1
        
        # Lissage de Laplace
        vocab_size = len(self.vocabulary)
        for class_name in self.classes:
            self.word_probs[class_name] = {}
            for word in self.vocabulary:
                count = class_word_counts[class_name][word]
                total = class_total_words[class_name]
                self.word_probs[class_name][word] = (count + 1) / (total + vocab_size)
    
    def _train_logistic_regression(self, X: List[str], y: List[str]):
        """Entraîne une régression logistique simple."""
        # Conversion en matrices
        X_features = np.array([self._text_to_features(text) for text in X])
        
        # Encodage des labels (binaire pour simplifier)
        if len(self.classes) == 2:
            classes_list = list(self.classes)
            y_encoded = np.array([1 if label == classes_list[0] else 0 for label in y])
            
            # Régression logistique simple avec gradient descent
            self.weights = np.random.normal(0, 0.01, X_features.shape[1])
            self.bias = 0
            
            learning_rate = 0.01
            epochs = 100
            
            for epoch in range(epochs):
                # Forward pass
                z = np.dot(X_features, self.weights) + self.bias
                predictions = 1 / (1 + np.exp(-z))
                
                # Calcul de la perte
                loss = -np.mean(y_encoded * np.log(predictions + 1e-15) + 
                               (1 - y_encoded) * np.log(1 - predictions + 1e-15))
                
                # Backward pass
                dw = np.dot(X_features.T, (predictions - y_encoded)) / len(y_encoded)
                db = np.mean(predictions - y_encoded)
                
                # Mise à jour des poids
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
    
    def predict(self, X: Union[str, List[str]]) -> Union[str, List[str]]:
        """Prédit la classe des textes."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        if isinstance(X, str):
            return self._predict_single(X)
        else:
            return [self._predict_single(text) for text in X]
    
    def _predict_single(self, text: str) -> str:
        """Prédit la classe d'un seul texte."""
        if self.algorithm == "naive_bayes":
            return self._predict_naive_bayes(text)
        elif self.algorithm == "logistic_regression":
            return self._predict_logistic_regression(text)
    
    def _predict_naive_bayes(self, text: str) -> str:
        """Prédiction Naive Bayes."""
        tokens = self._tokenize(text)
        class_scores = {}
        
        for class_name in self.classes:
            score = np.log(self.class_probs[class_name])
            
            for token in tokens:
                if token in self.vocabulary:
                    score += np.log(self.word_probs[class_name][token])
            
            class_scores[class_name] = score
        
        return max(class_scores, key=class_scores.get)
    
    def _predict_logistic_regression(self, text: str) -> str:
        """Prédiction régression logistique."""
        features = self._text_to_features(text)
        z = np.dot(features, self.weights) + self.bias
        probability = 1 / (1 + np.exp(-z))
        
        classes_list = list(self.classes)
        return classes_list[0] if probability > 0.5 else classes_list[1]


class SentimentAnalyzer(TextClassifier):
    """Analyseur de sentiment."""
    
    def __init__(self):
        super().__init__("naive_bayes")
        self.name = "SentimentAnalyzer"
        
        # Dictionnaire de mots de sentiment
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
            'sad', 'disappointed', 'frustrated', 'annoying', 'worst', 'disgusting',
            'pathetic', 'useless', 'boring', 'stupid', 'ridiculous'
        }
    
    def analyze_sentiment_simple(self, text: str) -> Dict[str, float]:
        """Analyse de sentiment basée sur un dictionnaire."""
        tokens = self._tokenize(text)
        
        positive_score = sum(1 for token in tokens if token in self.positive_words)
        negative_score = sum(1 for token in tokens if token in self.negative_words)
        
        total_score = positive_score + negative_score
        
        if total_score == 0:
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 1.0}
        
        pos_prob = positive_score / total_score
        neg_prob = negative_score / total_score
        
        if pos_prob > neg_prob:
            return {'positive': pos_prob, 'negative': neg_prob, 'neutral': 0.0}
        elif neg_prob > pos_prob:
            return {'positive': pos_prob, 'negative': neg_prob, 'neutral': 0.0}
        else:
            return {'positive': pos_prob, 'negative': neg_prob, 'neutral': 0.5}


class LanguageModel(BaseNLPModel):
    """Modèle de langue simple (n-grammes)."""
    
    def __init__(self, n: int = 2):
        super().__init__("LanguageModel")
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocabulary = set()
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenise le texte."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = ['<START>'] * (self.n - 1) + text.split() + ['<END>']
        return tokens
    
    def train(self, X: List[str], y: Optional[List] = None, **kwargs):
        """Entraîne le modèle de langue."""
        for text in X:
            tokens = self._tokenize(text)
            self.vocabulary.update(tokens)
            
            # Génération des n-grammes
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.ngrams[context][next_word] += 1
        
        self.is_trained = True
    
    def predict_next_word(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Prédit le mot suivant."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        context_tuple = tuple(context[-(self.n-1):])
        
        if context_tuple not in self.ngrams:
            return []
        
        word_counts = self.ngrams[context_tuple]
        total_count = sum(word_counts.values())
        
        probabilities = [(word, count / total_count) for word, count in word_counts.items()]
        probabilities.sort(key=lambda x: x[1], reverse=True)
        
        return probabilities[:top_k]
    
    def generate_text(self, seed: List[str], max_length: int = 50) -> str:
        """Génère du texte."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        generated = seed.copy()
        
        for _ in range(max_length):
            predictions = self.predict_next_word(generated)
            
            if not predictions or predictions[0][0] == '<END>':
                break
            
            # Sélection probabiliste
            words, probs = zip(*predictions)
            probs = np.array(probs)
            probs = probs / probs.sum()  # Normalisation
            
            next_word = np.random.choice(words, p=probs)
            generated.append(next_word)
        
        # Suppression des tokens spéciaux
        result = [word for word in generated if word not in ['<START>', '<END>']]
        return ' '.join(result)
    
    def predict(self, X: Union[str, List[str]]) -> Union[str, List[str]]:
        """Interface de prédiction générale."""
        if isinstance(X, str):
            tokens = self._tokenize(X)
            return self.generate_text(tokens[:self.n-1])
        else:
            return [self.generate_text(self._tokenize(text)[:self.n-1]) for text in X]


class NamedEntityRecognizer(BaseNLPModel):
    """Reconnaissance d'entités nommées simple."""
    
    def __init__(self):
        super().__init__("NamedEntityRecognizer")
        self.entity_patterns = {
            'PERSON': [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'],
            'EMAIL': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'PHONE': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'],
            'DATE': [r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'],
            'MONEY': [r'\$\d+(?:,\d{3})*(?:\.\d{2})?'],
            'URL': [r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+']
        }
        self.is_trained = True  # Basé sur des règles, pas d'entraînement nécessaire
    
    def train(self, X: List[str], y: Optional[List] = None, **kwargs):
        """Pas d'entraînement nécessaire pour ce modèle basé sur des règles."""
        pass
    
    def predict(self, X: Union[str, List[str]]) -> Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]]:
        """Extrait les entités nommées."""
        if isinstance(X, str):
            return self._extract_entities(X)
        else:
            return [self._extract_entities(text) for text in X]
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extrait les entités d'un texte."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append((match.group(), entity_type))
        
        return entities


class TextSummarizer(BaseNLPModel):
    """Résumeur de texte extractif simple."""
    
    def __init__(self, num_sentences: int = 3):
        super().__init__("TextSummarizer")
        self.num_sentences = num_sentences
        self.is_trained = True  # Pas d'entraînement nécessaire
    
    def train(self, X: List[str], y: Optional[List] = None, **kwargs):
        """Pas d'entraînement nécessaire."""
        pass
    
    def predict(self, X: Union[str, List[str]]) -> Union[str, List[str]]:
        """Génère un résumé."""
        if isinstance(X, str):
            return self._summarize_text(X)
        else:
            return [self._summarize_text(text) for text in X]
    
    def _summarize_text(self, text: str) -> str:
        """Résume un texte."""
        # Découpage en phrases
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= self.num_sentences:
            return text
        
        # Calcul des scores de phrases (fréquence des mots)
        word_freq = Counter()
        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())
            word_freq.update(words)
        
        # Score de chaque phrase
        sentence_scores = []
        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())
            score = sum(word_freq[word] for word in words) / len(words) if words else 0
            sentence_scores.append((sentence, score))
        
        # Sélection des meilleures phrases
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:self.num_sentences]
        
        # Remise dans l'ordre original
        original_order = []
        for sentence, _ in top_sentences:
            original_order.append((sentences.index(sentence), sentence))
        
        original_order.sort(key=lambda x: x[0])
        summary_sentences = [sentence for _, sentence in original_order]
        
        return '. '.join(summary_sentences) + '.'


class QuestionAnswering(BaseNLPModel):
    """Système de questions-réponses simple."""
    
    def __init__(self):
        super().__init__("QuestionAnswering")
        self.knowledge_base = {}
        
    def train(self, X: List[str], y: Optional[List[str]] = None, **kwargs):
        """Entraîne avec des paires question-réponse."""
        if y is None:
            raise ValueError("Les réponses (y) sont requises pour l'entraînement")
        
        for question, answer in zip(X, y):
            # Extraction de mots-clés de la question
            keywords = self._extract_keywords(question)
            for keyword in keywords:
                if keyword not in self.knowledge_base:
                    self.knowledge_base[keyword] = []
                self.knowledge_base[keyword].append((question, answer))
        
        self.is_trained = True
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrait les mots-clés d'un texte."""
        # Mots vides à ignorer
        stopwords = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
        
        words = re.findall(r'\w+', text.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def predict(self, X: Union[str, List[str]]) -> Union[str, List[str]]:
        """Répond aux questions."""
        if isinstance(X, str):
            return self._answer_question(X)
        else:
            return [self._answer_question(question) for question in X]
    
    def _answer_question(self, question: str) -> str:
        """Répond à une question."""
        if not self.is_trained:
            return "Le modèle n'est pas entraîné."
        
        keywords = self._extract_keywords(question)
        
        # Recherche de réponses basée sur les mots-clés
        candidate_answers = []
        for keyword in keywords:
            if keyword in self.knowledge_base:
                candidate_answers.extend(self.knowledge_base[keyword])
        
        if not candidate_answers:
            return "Je ne connais pas la réponse à cette question."
        
        # Sélection de la meilleure réponse (la plus fréquente)
        answer_counts = Counter(answer for _, answer in candidate_answers)
        best_answer = answer_counts.most_common(1)[0][0]
        
        return best_answer


def create_nlp_model(model_type: str, **kwargs) -> BaseNLPModel:
    """Factory pour créer des modèles NLP."""
    if model_type.lower() == "text_classifier":
        return TextClassifier(**kwargs)
    elif model_type.lower() == "sentiment_analyzer":
        return SentimentAnalyzer()
    elif model_type.lower() == "language_model":
        return LanguageModel(**kwargs)
    elif model_type.lower() == "ner":
        return NamedEntityRecognizer()
    elif model_type.lower() == "summarizer":
        return TextSummarizer(**kwargs)
    elif model_type.lower() == "qa":
        return QuestionAnswering()
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")