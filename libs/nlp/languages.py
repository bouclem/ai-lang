"""Module de support multilingue pour ai'lang NLP."""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter


# Dictionnaires de mots caractéristiques par langue
LANGUAGE_PATTERNS = {
    'en': {
        'common_words': {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
        },
        'patterns': [r'\bthe\b', r'\band\b', r'\bof\b', r'ing\b', r'ed\b'],
        'name': 'English'
    },
    'fr': {
        'common_words': {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
            'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne',
            'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'me', 'même',
            'y', 'ces', 'là', 'sans', 'peut', 'lui', 'nous', 'comme', 'mais',
            'ou', 'si', 'leur', 'bien', 'encore', 'après', 'ici', 'cela', 'notre'
        },
        'patterns': [r'\ble\b', r'\bde\b', r'\bet\b', r'\bque\b', r'tion\b'],
        'name': 'Français'
    },
    'es': {
        'common_words': {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
            'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para',
            'al', 'del', 'los', 'se', 'las', 'me', 'una', 'todo', 'pero',
            'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro',
            'ese', 'la', 'si', 'ya', 'porque', 'cuando', 'muy', 'sin', 'sobre'
        },
        'patterns': [r'\bel\b', r'\bla\b', r'\bde\b', r'\bque\b', r'ción\b'],
        'name': 'Español'
    },
    'de': {
        'common_words': {
            'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit',
            'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein',
            'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat',
            'dass', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind',
            'noch', 'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben'
        },
        'patterns': [r'\bder\b', r'\bdie\b', r'\bund\b', r'\bdas\b', r'ung\b'],
        'name': 'Deutsch'
    },
    'it': {
        'common_words': {
            'il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'del',
            'da', 'a', 'al', 'le', 'si', 'dei', 'come', 'io', 'questo',
            'ma', 'tutto', 'te', 'della', 'uno', 'volta', 'molto', 'quando',
            'essere', 'dove', 'quella', 'bene', 'può', 'cosa', 'tanto', 'cui',
            'sua', 'mio', 'fare', 'era', 'loro', 'quella', 'grande', 'così'
        },
        'patterns': [r'\bil\b', r'\bdi\b', r'\bche\b', r'\bla\b', r'zione\b'],
        'name': 'Italiano'
    },
    'pt': {
        'common_words': {
            'o', 'de', 'a', 'e', 'do', 'da', 'em', 'um', 'para', 'é',
            'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
            'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem',
            'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos',
            'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso'
        },
        'patterns': [r'\bo\b', r'\bde\b', r'\ba\b', r'\be\b', r'ção\b'],
        'name': 'Português'
    },
    'ru': {
        'common_words': {
            'в', 'и', 'не', 'на', 'я', 'быть', 'то', 'он', 'с', 'а',
            'как', 'по', 'это', 'она', 'этот', 'к', 'но', 'они', 'мы',
            'что', 'за', 'из', 'у', 'который', 'о', 'от', 'до', 'вы',
            'все', 'так', 'его', 'для', 'со', 'если', 'уже', 'или', 'ни',
            'бы', 'то', 'только', 'её', 'мне', 'было', 'вот', 'от', 'меня'
        },
        'patterns': [r'\bв\b', r'\bи\b', r'\bне\b', r'\bна\b', r'ть\b'],
        'name': 'Русский'
    }
}


# Dictionnaires de traduction simple
TRANSLATION_DICT = {
    ('en', 'fr'): {
        'hello': 'bonjour',
        'goodbye': 'au revoir',
        'yes': 'oui',
        'no': 'non',
        'please': 's\'il vous plaît',
        'thank you': 'merci',
        'good': 'bon',
        'bad': 'mauvais',
        'big': 'grand',
        'small': 'petit',
        'water': 'eau',
        'food': 'nourriture',
        'house': 'maison',
        'car': 'voiture',
        'book': 'livre',
        'time': 'temps',
        'day': 'jour',
        'night': 'nuit',
        'love': 'amour',
        'friend': 'ami'
    },
    ('en', 'es'): {
        'hello': 'hola',
        'goodbye': 'adiós',
        'yes': 'sí',
        'no': 'no',
        'please': 'por favor',
        'thank you': 'gracias',
        'good': 'bueno',
        'bad': 'malo',
        'big': 'grande',
        'small': 'pequeño',
        'water': 'agua',
        'food': 'comida',
        'house': 'casa',
        'car': 'coche',
        'book': 'libro',
        'time': 'tiempo',
        'day': 'día',
        'night': 'noche',
        'love': 'amor',
        'friend': 'amigo'
    },
    ('en', 'de'): {
        'hello': 'hallo',
        'goodbye': 'auf wiedersehen',
        'yes': 'ja',
        'no': 'nein',
        'please': 'bitte',
        'thank you': 'danke',
        'good': 'gut',
        'bad': 'schlecht',
        'big': 'groß',
        'small': 'klein',
        'water': 'wasser',
        'food': 'essen',
        'house': 'haus',
        'car': 'auto',
        'book': 'buch',
        'time': 'zeit',
        'day': 'tag',
        'night': 'nacht',
        'love': 'liebe',
        'friend': 'freund'
    }
}


def detect_language(text: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """Détecte la langue d'un texte."""
    
    if not text.strip():
        return []
    
    # Nettoyage du texte
    text = text.lower()
    words = re.findall(r'\w+', text)
    
    if not words:
        return []
    
    language_scores = {}
    
    for lang_code, lang_data in LANGUAGE_PATTERNS.items():
        score = 0
        
        # Score basé sur les mots communs
        common_words = lang_data['common_words']
        word_matches = sum(1 for word in words if word in common_words)
        word_score = word_matches / len(words) if words else 0
        
        # Score basé sur les patterns
        pattern_score = 0
        for pattern in lang_data['patterns']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            pattern_score += matches
        
        pattern_score = pattern_score / len(text) if text else 0
        
        # Score combiné
        language_scores[lang_code] = (word_score * 0.7) + (pattern_score * 0.3)
    
    # Tri par score décroissant
    sorted_scores = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_scores[:top_n]


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Traduit un texte simple entre deux langues."""
    
    # Vérification de la disponibilité de la traduction
    translation_key = (source_lang, target_lang)
    reverse_key = (target_lang, source_lang)
    
    if translation_key in TRANSLATION_DICT:
        translation_dict = TRANSLATION_DICT[translation_key]
    elif reverse_key in TRANSLATION_DICT:
        # Utilisation du dictionnaire inverse
        original_dict = TRANSLATION_DICT[reverse_key]
        translation_dict = {v: k for k, v in original_dict.items()}
    else:
        return f"Traduction non disponible de {source_lang} vers {target_lang}"
    
    # Tokenisation
    words = re.findall(r'\w+|[^\w\s]', text.lower())
    
    translated_words = []
    i = 0
    
    while i < len(words):
        # Essai de traduction de phrases de 2 mots
        if i < len(words) - 1:
            two_word_phrase = f"{words[i]} {words[i+1]}"
            if two_word_phrase in translation_dict:
                translated_words.append(translation_dict[two_word_phrase])
                i += 2
                continue
        
        # Traduction de mots simples
        word = words[i]
        if word in translation_dict:
            translated_words.append(translation_dict[word])
        else:
            translated_words.append(word)  # Garder le mot original si pas de traduction
        
        i += 1
    
    return ' '.join(translated_words)


def get_supported_languages() -> Dict[str, str]:
    """Retourne la liste des langues supportées."""
    return {code: data['name'] for code, data in LANGUAGE_PATTERNS.items()}


def get_language_info(lang_code: str) -> Optional[Dict[str, any]]:
    """Retourne les informations sur une langue."""
    if lang_code in LANGUAGE_PATTERNS:
        return {
            'code': lang_code,
            'name': LANGUAGE_PATTERNS[lang_code]['name'],
            'common_words_count': len(LANGUAGE_PATTERNS[lang_code]['common_words']),
            'patterns_count': len(LANGUAGE_PATTERNS[lang_code]['patterns'])
        }
    return None


def is_language_supported(lang_code: str) -> bool:
    """Vérifie si une langue est supportée."""
    return lang_code in LANGUAGE_PATTERNS


def get_translation_pairs() -> List[Tuple[str, str]]:
    """Retourne la liste des paires de langues supportées pour la traduction."""
    pairs = list(TRANSLATION_DICT.keys())
    
    # Ajouter les paires inverses
    reverse_pairs = [(target, source) for source, target in pairs]
    
    return pairs + reverse_pairs


def analyze_text_language(text: str) -> Dict[str, any]:
    """Analyse complète de la langue d'un texte."""
    
    # Détection de langue
    detected_languages = detect_language(text, top_n=3)
    
    # Statistiques de base
    words = re.findall(r'\w+', text.lower())
    unique_words = set(words)
    
    # Analyse des caractères
    char_analysis = {
        'latin': len(re.findall(r'[a-zA-Z]', text)),
        'cyrillic': len(re.findall(r'[а-яё]', text, re.IGNORECASE)),
        'digits': len(re.findall(r'\d', text)),
        'punctuation': len(re.findall(r'[^\w\s]', text))
    }
    
    # Langue la plus probable
    most_likely_lang = detected_languages[0] if detected_languages else ('unknown', 0.0)
    
    return {
        'detected_languages': detected_languages,
        'most_likely': {
            'code': most_likely_lang[0],
            'confidence': most_likely_lang[1],
            'name': LANGUAGE_PATTERNS.get(most_likely_lang[0], {}).get('name', 'Unknown')
        },
        'text_stats': {
            'total_words': len(words),
            'unique_words': len(unique_words),
            'characters': len(text),
            'character_analysis': char_analysis
        },
        'supported_translations': [
            target for source, target in get_translation_pairs() 
            if source == most_likely_lang[0]
        ]
    }


def multilingual_tokenize(text: str, language: Optional[str] = None) -> List[str]:
    """Tokenisation adaptée à la langue."""
    
    if language is None:
        # Détection automatique
        detected = detect_language(text, top_n=1)
        language = detected[0][0] if detected else 'en'
    
    # Tokenisation de base
    if language in ['zh', 'ja']:  # Chinois, Japonais
        # Pour les langues sans espaces, tokenisation caractère par caractère
        return list(re.sub(r'\s+', '', text))
    else:
        # Tokenisation par mots pour les autres langues
        return re.findall(r'\w+', text.lower())


def get_stopwords(language: str) -> set:
    """Retourne les mots vides pour une langue."""
    if language in LANGUAGE_PATTERNS:
        return LANGUAGE_PATTERNS[language]['common_words']
    else:
        # Retourner les mots vides anglais par défaut
        return LANGUAGE_PATTERNS['en']['common_words']


def normalize_text_for_language(text: str, language: str) -> str:
    """Normalise le texte selon les conventions de la langue."""
    
    # Normalisation de base
    text = text.strip()
    
    if language == 'de':  # Allemand
        # Conversion des caractères spéciaux allemands
        replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
        for old, new in replacements.items():
            text = text.replace(old, new)
            text = text.replace(old.upper(), new.upper())
    
    elif language == 'fr':  # Français
        # Normalisation des accents (optionnel)
        replacements = {
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        # Optionnel: décommenter pour supprimer les accents
        # for old, new in replacements.items():
        #     text = text.replace(old, new)
        #     text = text.replace(old.upper(), new.upper())
    
    return text