"""Module de traitement de texte pour ai'lang NLP."""

import re
import string
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


class TextProcessor:
    """Classe principale pour le traitement de texte."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.stopwords = self._load_stopwords(language)
        
    def _load_stopwords(self, language: str) -> Set[str]:
        """Charge les mots vides pour une langue donnée."""
        # Mots vides de base pour l'anglais
        english_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # Mots vides de base pour le français
        french_stopwords = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
            'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne',
            'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'le', 'de',
            'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas',
            'tout', 'plus', 'par', 'grand', 'en', 'me', 'même', 'y', 'ces',
            'là', 'sans', 'peut', 'lui', 'nous', 'comme', 'mais', 'ou', 'si',
            'leur', 'bien', 'encore', 'après', 'ici', 'cela', 'notre'
        }
        
        if language == "fr":
            return french_stopwords
        return english_stopwords
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenise le texte en mots."""
        # Nettoyage de base
        text = text.lower()
        # Suppression de la ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokenisation
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte en supprimant les caractères indésirables."""
        # Suppression des caractères spéciaux
        text = re.sub(r'[^\w\s]', ' ', text)
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        # Suppression des espaces en début/fin
        text = text.strip()
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Supprime les mots vides de la liste de tokens."""
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def stem_words(self, tokens: List[str]) -> List[str]:
        """Applique un stemming simple aux tokens."""
        stemmed = []
        for token in tokens:
            # Stemming simple pour l'anglais
            if token.endswith('ing'):
                stemmed.append(token[:-3])
            elif token.endswith('ed'):
                stemmed.append(token[:-2])
            elif token.endswith('er'):
                stemmed.append(token[:-2])
            elif token.endswith('est'):
                stemmed.append(token[:-3])
            elif token.endswith('ly'):
                stemmed.append(token[:-2])
            elif token.endswith('s') and len(token) > 3:
                stemmed.append(token[:-1])
            else:
                stemmed.append(token)
        return stemmed
    
    def lemmatize_words(self, tokens: List[str]) -> List[str]:
        """Applique une lemmatisation simple aux tokens."""
        # Dictionnaire simple de lemmatisation
        lemma_dict = {
            'running': 'run', 'ran': 'run', 'runs': 'run',
            'better': 'good', 'best': 'good',
            'children': 'child', 'mice': 'mouse', 'feet': 'foot',
            'geese': 'goose', 'teeth': 'tooth', 'people': 'person',
            'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
            'had': 'have', 'has': 'have', 'having': 'have'
        }
        
        return [lemma_dict.get(token.lower(), token) for token in tokens]
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extrait des entités nommées simples du texte."""
        entities = []
        
        # Patterns simples pour différents types d'entités
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Nom simple
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append((match.group(), entity_type))
        
        return entities
    
    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Étiquetage morpho-syntaxique simple."""
        tagged = []
        
        # Dictionnaire simple de tags POS
        pos_dict = {
            # Pronoms
            'i': 'PRP', 'you': 'PRP', 'he': 'PRP', 'she': 'PRP', 'it': 'PRP',
            'we': 'PRP', 'they': 'PRP', 'me': 'PRP', 'him': 'PRP', 'her': 'PRP',
            'us': 'PRP', 'them': 'PRP',
            
            # Déterminants
            'the': 'DT', 'a': 'DT', 'an': 'DT', 'this': 'DT', 'that': 'DT',
            'these': 'DT', 'those': 'DT',
            
            # Prépositions
            'in': 'IN', 'on': 'IN', 'at': 'IN', 'by': 'IN', 'for': 'IN',
            'with': 'IN', 'to': 'IN', 'from': 'IN', 'of': 'IN', 'about': 'IN',
            
            # Conjonctions
            'and': 'CC', 'or': 'CC', 'but': 'CC', 'so': 'CC',
            
            # Verbes auxiliaires
            'is': 'VBZ', 'are': 'VBP', 'was': 'VBD', 'were': 'VBD',
            'have': 'VBP', 'has': 'VBZ', 'had': 'VBD',
            'will': 'MD', 'would': 'MD', 'can': 'MD', 'could': 'MD',
            'should': 'MD', 'may': 'MD', 'might': 'MD'
        }
        
        for token in tokens:
            token_lower = token.lower()
            
            if token_lower in pos_dict:
                tagged.append((token, pos_dict[token_lower]))
            elif token.isdigit():
                tagged.append((token, 'CD'))  # Cardinal number
            elif token.endswith('ing'):
                tagged.append((token, 'VBG'))  # Gerund
            elif token.endswith('ed'):
                tagged.append((token, 'VBD'))  # Past tense verb
            elif token.endswith('ly'):
                tagged.append((token, 'RB'))   # Adverb
            elif token.endswith('s') and len(token) > 3:
                tagged.append((token, 'NNS'))  # Plural noun
            elif token[0].isupper():
                tagged.append((token, 'NNP'))  # Proper noun
            else:
                tagged.append((token, 'NN'))   # Noun
        
        return tagged


# Fonctions utilitaires globales
def tokenize(text: str, language: str = "en") -> List[str]:
    """Tokenise le texte."""
    processor = TextProcessor(language)
    return processor.tokenize(text)


def clean_text(text: str) -> str:
    """Nettoie le texte."""
    processor = TextProcessor()
    return processor.clean_text(text)


def remove_stopwords(tokens: List[str], language: str = "en") -> List[str]:
    """Supprime les mots vides."""
    processor = TextProcessor(language)
    return processor.remove_stopwords(tokens)


def stem_words(tokens: List[str]) -> List[str]:
    """Applique un stemming."""
    processor = TextProcessor()
    return processor.stem_words(tokens)


def lemmatize_words(tokens: List[str]) -> List[str]:
    """Applique une lemmatisation."""
    processor = TextProcessor()
    return processor.lemmatize_words(tokens)


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extrait les entités nommées."""
    processor = TextProcessor()
    return processor.extract_entities(text)


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """Étiquetage morpho-syntaxique."""
    processor = TextProcessor()
    return processor.pos_tag(tokens)