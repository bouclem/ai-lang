"""Module principal ai'lang - Langage de programmation optimisé pour l'IA."""

# Lexer
from .lexer import Lexer, LexerError, Token, TokenType

# Parser
from .parser import Parser, ParseError, parse
from .parser.ast_nodes import *

# Interpreter
from .interpreter import (
    Interpreter, 
    InterpreterError, 
    interpret_code,
    AILangValue,
    AILangNumber,
    AILangString,
    AILangBoolean,
    AILangNone,
    AILangList,
    AILangDict,
    AILangTensor,
    AILangFunction,
    AILangClass,
    python_to_ailang,
    ailang_to_python
)

# Compiler
from .compiler import Compiler, CompilerError, compile_code

# Version
__version__ = "0.1.0"
__author__ = "AI'Lang Development Team"
__description__ = "Langage de programmation inspiré de Python, optimisé pour l'IA"


def run_code(source_code: str, mode: str = 'interpret') -> None:
    """Exécute du code ai'lang.
    
    Args:
        source_code: Le code source ai'lang à exécuter
        mode: 'interpret' pour l'interprétation directe, 'compile' pour la compilation
    """
    if mode == 'interpret':
        interpret_code(source_code)
    elif mode == 'compile':
        python_code = compile_code(source_code)
        exec(python_code)
    else:
        raise ValueError(f"Mode non supporté: {mode}. Utilisez 'interpret' ou 'compile'.")


def parse_code(source_code: str) -> List[Statement]:
    """Parse du code ai'lang et retourne l'AST.
    
    Args:
        source_code: Le code source ai'lang à parser
        
    Returns:
        Liste des déclarations de l'AST
    """
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    return parser.parse()


def tokenize_code(source_code: str) -> List[Token]:
    """Tokenise du code ai'lang.
    
    Args:
        source_code: Le code source ai'lang à tokeniser
        
    Returns:
        Liste des tokens
    """
    lexer = Lexer(source_code)
    return lexer.tokenize()


# Fonctions utilitaires pour l'REPL
def create_interpreter() -> Interpreter:
    """Crée une nouvelle instance d'interpréteur."""
    return Interpreter()


def create_compiler() -> Compiler:
    """Crée une nouvelle instance de compilateur."""
    return Compiler()


__all__ = [
    # Core components
    'Lexer', 'LexerError', 'Token', 'TokenType',
    'Parser', 'ParseError', 'parse',
    'Interpreter', 'InterpreterError', 'interpret_code',
    'Compiler', 'CompilerError', 'compile_code',
    
    # AST nodes
    'ASTNode', 'Expression', 'Statement',
    'LiteralExpr', 'IdentifierExpr', 'BinaryExpr', 'UnaryExpr',
    'CallExpr', 'AttributeExpr', 'IndexExpr', 'ListExpr', 'DictExpr',
    'LambdaExpr', 'TensorExpr',
    'ExpressionStmt', 'AssignmentStmt', 'IfStmt', 'WhileStmt', 'ForStmt',
    'FunctionDef', 'ClassDef', 'ReturnStmt', 'BreakStmt', 'ContinueStmt',
    'ImportStmt', 'TryStmt', 'WithStmt',
    'ModelDef', 'TrainStmt', 'PredictStmt', 'DatasetDef',
    
    # Values
    'AILangValue', 'AILangNumber', 'AILangString', 'AILangBoolean',
    'AILangNone', 'AILangList', 'AILangDict', 'AILangTensor',
    'AILangFunction', 'AILangClass',
    'python_to_ailang', 'ailang_to_python',
    
    # Utility functions
    'run_code', 'parse_code', 'tokenize_code',
    'create_interpreter', 'create_compiler',
    
    # Metadata
    '__version__', '__author__', '__description__'
]