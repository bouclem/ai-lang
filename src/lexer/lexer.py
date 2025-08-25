"""Analyseur lexical pour le langage ai'lang."""

import re
import string
from typing import List, Optional, Iterator
from .token import Token, TokenType, KEYWORDS, OPERATORS, DELIMITERS


class LexerError(Exception):
    """Exception levée lors d'erreurs de lexing."""
    
    def __init__(self, message: str, line: int, column: int, filename: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename
        super().__init__(f"{filename or '<unknown>'}:{line}:{column}: {message}")


class Lexer:
    """Analyseur lexical pour ai'lang."""
    
    def __init__(self, source: str, filename: Optional[str] = None):
        self.source = source
        self.filename = filename
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.indent_stack = [0]  # Stack pour gérer l'indentation
        
    def error(self, message: str) -> None:
        """Lève une erreur de lexing."""
        raise LexerError(message, self.line, self.column, self.filename)
        
    def peek(self, offset: int = 0) -> Optional[str]:
        """Regarde le caractère à la position actuelle + offset."""
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
        
    def advance(self) -> Optional[str]:
        """Avance d'un caractère et retourne le caractère actuel."""
        if self.position >= len(self.source):
            return None
            
        char = self.source[self.position]
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
            
        return char
        
    def skip_whitespace(self) -> None:
        """Ignore les espaces et tabulations (mais pas les nouvelles lignes)."""
        while self.peek() and self.peek() in ' \t':
            self.advance()
            
    def read_string(self, quote_char: str) -> str:
        """Lit une chaîne de caractères."""
        value = ''
        self.advance()  # Skip opening quote
        
        while True:
            char = self.peek()
            if char is None:
                self.error(f"Unterminated string literal")
            elif char == quote_char:
                self.advance()  # Skip closing quote
                break
            elif char == '\\':
                self.advance()  # Skip backslash
                escaped = self.advance()
                if escaped is None:
                    self.error("Unexpected end of file in string literal")
                # Handle escape sequences
                escape_map = {
                    'n': '\n', 't': '\t', 'r': '\r', '\\': '\\',
                    '\'': '\'', '"': '"', '0': '\0'
                }
                value += escape_map.get(escaped, escaped)
            else:
                value += self.advance()
                
        return value
        
    def read_number(self) -> tuple[str, TokenType]:
        """Lit un nombre (entier ou flottant)."""
        value = ''
        has_dot = False
        
        while True:
            char = self.peek()
            if char and char.isdigit():
                value += self.advance()
            elif char == '.' and not has_dot:
                has_dot = True
                value += self.advance()
            elif char == '_':  # Support pour les séparateurs de milliers
                self.advance()
            else:
                break
                
        # Support pour la notation scientifique
        if self.peek() and self.peek().lower() == 'e':
            value += self.advance()
            if self.peek() and self.peek() in '+-':
                value += self.advance()
            while self.peek() and self.peek().isdigit():
                value += self.advance()
                
        return value, TokenType.NUMBER
        
    def read_identifier(self) -> tuple[str, TokenType]:
        """Lit un identifiant ou mot-clé."""
        value = ''
        
        while True:
            char = self.peek()
            if char and (char.isalnum() or char == '_'):
                value += self.advance()
            else:
                break
                
        # Vérifier si c'est un mot-clé
        token_type = KEYWORDS.get(value, TokenType.IDENTIFIER)
        return value, token_type
        
    def read_comment(self) -> str:
        """Lit un commentaire."""
        value = ''
        self.advance()  # Skip '#'
        
        while self.peek() and self.peek() != '\n':
            value += self.advance()
            
        return value
        
    def handle_indentation(self, line_start: bool = False) -> List[Token]:
        """Gère l'indentation en début de ligne."""
        if not line_start:
            return []
            
        indent_level = 0
        while self.peek() and self.peek() in ' \t':
            if self.peek() == ' ':
                indent_level += 1
            else:  # tab
                indent_level += 8  # Tab = 8 espaces
            self.advance()
            
        tokens = []
        current_indent = self.indent_stack[-1]
        
        if indent_level > current_indent:
            self.indent_stack.append(indent_level)
            tokens.append(Token(TokenType.INDENT, indent_level, self.line, self.column, self.filename))
        elif indent_level < current_indent:
            while self.indent_stack and self.indent_stack[-1] > indent_level:
                self.indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, indent_level, self.line, self.column, self.filename))
            if not self.indent_stack or self.indent_stack[-1] != indent_level:
                self.error("Indentation error")
                
        return tokens
        
    def tokenize(self) -> List[Token]:
        """Tokenise le code source."""
        self.tokens = []
        line_start = True
        
        while self.position < len(self.source):
            # Gérer l'indentation en début de ligne
            if line_start:
                indent_tokens = self.handle_indentation(True)
                self.tokens.extend(indent_tokens)
                line_start = False
                
            char = self.peek()
            
            if char is None:
                break
                
            # Ignorer les espaces et tabulations
            if char in ' \t':
                self.skip_whitespace()
                continue
                
            # Nouvelle ligne
            if char == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line - 1, self.column, self.filename))
                line_start = True
                continue
                
            # Commentaires
            if char == '#':
                comment = self.read_comment()
                self.tokens.append(Token(TokenType.COMMENT, comment, self.line, self.column, self.filename))
                continue
                
            # Chaînes de caractères
            if char in '"\'':
                string_value = self.read_string(char)
                self.tokens.append(Token(TokenType.STRING, string_value, self.line, self.column, self.filename))
                continue
                
            # Nombres
            if char.isdigit():
                number_value, token_type = self.read_number()
                self.tokens.append(Token(token_type, number_value, self.line, self.column, self.filename))
                continue
                
            # Identifiants et mots-clés
            if char.isalpha() or char == '_':
                identifier_value, token_type = self.read_identifier()
                # Traitement spécial pour les booléens
                if token_type == TokenType.BOOLEAN:
                    value = identifier_value == 'True'
                else:
                    value = identifier_value
                self.tokens.append(Token(token_type, value, self.line, self.column, self.filename))
                continue
                
            # Opérateurs multi-caractères
            two_char = char + (self.peek(1) or '')
            if two_char in OPERATORS:
                self.advance()
                self.advance()
                self.tokens.append(Token(OPERATORS[two_char], two_char, self.line, self.column - 2, self.filename))
                continue
                
            # Opérateurs et délimiteurs simple caractère
            if char in OPERATORS:
                self.advance()
                self.tokens.append(Token(OPERATORS[char], char, self.line, self.column - 1, self.filename))
                continue
                
            if char in DELIMITERS:
                self.advance()
                self.tokens.append(Token(DELIMITERS[char], char, self.line, self.column - 1, self.filename))
                continue
                
            # Caractère non reconnu
            self.error(f"Unexpected character: {char!r}")
            
        # Ajouter les DEDENT nécessaires à la fin
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, 0, self.line, self.column, self.filename))
            
        # Ajouter EOF
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column, self.filename))
        
        return self.tokens
        
    def __iter__(self) -> Iterator[Token]:
        """Permet d'itérer sur les tokens."""
        if not self.tokens:
            self.tokenize()
        return iter(self.tokens)