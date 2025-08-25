"""Bibliothèque Natural Language Processing intégrée pour ai'lang."""

from .text_processing import (
    tokenize,
    clean_text,
    remove_stopwords,
    stem_words,
    lemmatize_words,
    extract_entities,
    pos_tag,
    TextProcessor
)

from .embeddings import (
    WordEmbedding,
    Word2Vec,
    GloVe,
    FastText,
    BERTEmbedding,
    create_embedding
)

from .models import (
    TextClassifier,
    SentimentAnalyzer,
    LanguageModel,
    NamedEntityRecognizer,
    TextSummarizer,
    QuestionAnswering,
    create_nlp_model
)

from .metrics import (
    bleu_score,
    rouge_score,
    perplexity,
    f1_score_nlp,
    accuracy_nlp
)

from .utils import (
    load_dataset,
    save_model,
    load_model,
    preprocess_dataset,
    split_sentences,
    count_words,
    calculate_readability
)

from .languages import (
    detect_language,
    translate_text,
    get_supported_languages
)

__version__ = "1.0.0"
__author__ = "AI'Lang NLP Team"

__all__ = [
    # Text Processing
    'tokenize', 'clean_text', 'remove_stopwords', 'stem_words', 'lemmatize_words',
    'extract_entities', 'pos_tag', 'TextProcessor',
    
    # Embeddings
    'WordEmbedding', 'Word2Vec', 'GloVe', 'FastText', 'BERTEmbedding', 'create_embedding',
    
    # Models
    'TextClassifier', 'SentimentAnalyzer', 'LanguageModel', 'NamedEntityRecognizer',
    'TextSummarizer', 'QuestionAnswering', 'create_nlp_model',
    
    # Metrics
    'bleu_score', 'rouge_score', 'perplexity', 'f1_score_nlp', 'accuracy_nlp',
    
    # Utils
    'load_dataset', 'save_model', 'load_model', 'preprocess_dataset',
    'split_sentences', 'count_words', 'calculate_readability',
    
    # Languages
    'detect_language', 'translate_text', 'get_supported_languages'
]