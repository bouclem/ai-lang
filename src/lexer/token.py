"""Définition des tokens pour le langage ai'lang."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional


class TokenType(Enum):
    """Types de tokens supportés par ai'lang."""
    
    # Littéraux
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    NONE = auto()
    
    # Identifiants
    IDENTIFIER = auto()
    
    # Mots-clés
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    DEF = auto()
    CLASS = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    TRY = auto()
    EXCEPT = auto()
    FINALLY = auto()
    RAISE = auto()
    WITH = auto()
    ASYNC = auto()
    AWAIT = auto()
    LAMBDA = auto()
    GLOBAL = auto()
    NONLOCAL = auto()
    
    # Mots-clés spécifiques à ai'lang
    MODEL = auto()
    TRAIN = auto()
    PREDICT = auto()
    DATASET = auto()
    TENSOR = auto()
    NEURAL_NETWORK = auto()
    OPTIMIZER = auto()
    LOSS = auto()
    METRIC = auto()
    
    # Opérateurs arithmétiques
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    FLOOR_DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    
    # Opérateurs de comparaison
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_THAN = auto()
    GREATER_EQUAL = auto()
    
    # Opérateurs logiques
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Opérateurs d'assignation
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    MULTIPLY_ASSIGN = auto()
    DIVIDE_ASSIGN = auto()
    
    # Opérateurs bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    
    # Délimiteurs
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()
    
    # Indentation et nouvelles lignes
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    
    # Spéciaux
    EOF = auto()
    COMMENT = auto()
    

@dataclass
class Token:
    """Représente un token dans le code source."""
    
    type: TokenType
    value: Any
    line: int
    column: int
    filename: Optional[str] = None
    
    def __str__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Mapping des mots-clés
KEYWORDS = {
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'elif': TokenType.ELIF,
    'while': TokenType.WHILE,
    'for': TokenType.FOR,
    'in': TokenType.IN,
    'def': TokenType.DEF,
    'class': TokenType.CLASS,
    'return': TokenType.RETURN,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'pass': TokenType.PASS,
    'import': TokenType.IMPORT,
    'from': TokenType.FROM,
    'as': TokenType.AS,
    'try': TokenType.TRY,
    'except': TokenType.EXCEPT,
    'finally': TokenType.FINALLY,
    'raise': TokenType.RAISE,
    'with': TokenType.WITH,
    'async': TokenType.ASYNC,
    'await': TokenType.AWAIT,
    'lambda': TokenType.LAMBDA,
    'global': TokenType.GLOBAL,
    'nonlocal': TokenType.NONLOCAL,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'True': TokenType.BOOLEAN,
    'False': TokenType.BOOLEAN,
    'None': TokenType.NONE,
    
    # Mots-clés spécifiques à ai'lang
    'model': TokenType.MODEL,
    'train': TokenType.TRAIN,
    'predict': TokenType.PREDICT,
    'dataset': TokenType.DATASET,
    'tensor': TokenType.TENSOR,
    'NeuralNetwork': TokenType.NEURAL_NETWORK,
    'optimizer': TokenType.OPTIMIZER,
    'loss': TokenType.LOSS,
    'metric': TokenType.METRIC,
}

# Mapping des opérateurs
OPERATORS = {
    '+': TokenType.PLUS,
    '-': TokenType.MINUS,
    '*': TokenType.MULTIPLY,
    '/': TokenType.DIVIDE,
    '//': TokenType.FLOOR_DIVIDE,
    '%': TokenType.MODULO,
    '**': TokenType.POWER,
    '=': TokenType.ASSIGN,
    '+=': TokenType.PLUS_ASSIGN,
    '-=': TokenType.MINUS_ASSIGN,
    '*=': TokenType.MULTIPLY_ASSIGN,
    '/=': TokenType.DIVIDE_ASSIGN,
    '==': TokenType.EQUAL,
    '!=': TokenType.NOT_EQUAL,
    '<': TokenType.LESS_THAN,
    '<=': TokenType.LESS_EQUAL,
    '>': TokenType.GREATER_THAN,
    '>=': TokenType.GREATER_EQUAL,
    '&': TokenType.BIT_AND,
    '|': TokenType.BIT_OR,
    '^': TokenType.BIT_XOR,
    '~': TokenType.BIT_NOT,
    '<<': TokenType.LEFT_SHIFT,
    '>>': TokenType.RIGHT_SHIFT,
    '->': TokenType.ARROW,
}

# Mapping des délimiteurs
DELIMITERS = {
    '(': TokenType.LEFT_PAREN,
    ')': TokenType.RIGHT_PAREN,
    '[': TokenType.LEFT_BRACKET,
    ']': TokenType.RIGHT_BRACKET,
    '{': TokenType.LEFT_BRACE,
    '}': TokenType.RIGHT_BRACE,
    ',': TokenType.COMMA,
    '.': TokenType.DOT,
    ':': TokenType.COLON,
    ';': TokenType.SEMICOLON,
}