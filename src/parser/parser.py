"""Analyseur syntaxique pour le langage ai'lang."""

from typing import List, Optional, Union
from ..lexer.token import Token, TokenType
from ..lexer.lexer import Lexer
from .ast_nodes import *


class ParseError(Exception):
    """Exception levée lors d'erreurs de parsing."""
    
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{token.filename or '<unknown>'}:{token.line}:{token.column}: {message}")


class Parser:
    """Analyseur syntaxique pour ai'lang."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        
    def error(self, message: str) -> None:
        """Lève une erreur de parsing."""
        token = self.peek()
        raise ParseError(message, token)
        
    def peek(self, offset: int = 0) -> Token:
        """Regarde le token à la position actuelle + offset."""
        pos = self.current + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[pos]
        
    def advance(self) -> Token:
        """Avance au token suivant et retourne le token actuel."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
        
    def is_at_end(self) -> bool:
        """Vérifie si on est à la fin des tokens."""
        return self.peek().type == TokenType.EOF
        
    def previous(self) -> Token:
        """Retourne le token précédent."""
        return self.tokens[self.current - 1]
        
    def check(self, token_type: TokenType) -> bool:
        """Vérifie si le token actuel est du type donné."""
        if self.is_at_end():
            return False
        return self.peek().type == token_type
        
    def match(self, *types: TokenType) -> bool:
        """Vérifie si le token actuel correspond à un des types donnés."""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
        
    def consume(self, token_type: TokenType, message: str) -> Token:
        """Consomme un token du type donné ou lève une erreur."""
        if self.check(token_type):
            return self.advance()
        self.error(message)
        
    def synchronize(self) -> None:
        """Synchronise après une erreur de parsing."""
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.NEWLINE:
                return
                
            if self.peek().type in [
                TokenType.CLASS, TokenType.DEF, TokenType.IF,
                TokenType.WHILE, TokenType.FOR, TokenType.RETURN,
                TokenType.MODEL, TokenType.TRAIN, TokenType.PREDICT
            ]:
                return
                
            self.advance()
            
    def skip_newlines(self) -> None:
        """Ignore les tokens NEWLINE."""
        while self.match(TokenType.NEWLINE):
            pass
            
    # ========================================================================
    # PARSING DES EXPRESSIONS
    # ========================================================================
    
    def expression(self) -> Expression:
        """Parse une expression."""
        return self.logical_or()
        
    def logical_or(self) -> Expression:
        """Parse une expression OR logique."""
        expr = self.logical_and()
        
        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.logical_and()
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def logical_and(self) -> Expression:
        """Parse une expression AND logique."""
        expr = self.equality()
        
        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def equality(self) -> Expression:
        """Parse une expression d'égalité."""
        expr = self.comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.previous()
            right = self.comparison()
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def comparison(self) -> Expression:
        """Parse une expression de comparaison."""
        expr = self.term()
        
        while self.match(TokenType.GREATER_THAN, TokenType.GREATER_EQUAL,
                         TokenType.LESS_THAN, TokenType.LESS_EQUAL):
            operator = self.previous()
            right = self.term()
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def term(self) -> Expression:
        """Parse une expression de terme (+ -)."""
        expr = self.factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.factor()
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def factor(self) -> Expression:
        """Parse une expression de facteur (* / % //)."""
        expr = self.power()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE,
                         TokenType.MODULO, TokenType.FLOOR_DIVIDE):
            operator = self.previous()
            right = self.power()
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def power(self) -> Expression:
        """Parse une expression de puissance (**)."""
        expr = self.unary()
        
        if self.match(TokenType.POWER):
            operator = self.previous()
            right = self.power()  # Associativité à droite
            expr = BinaryExpression(expr, operator, right)
            
        return expr
        
    def unary(self) -> Expression:
        """Parse une expression unaire."""
        if self.match(TokenType.NOT, TokenType.MINUS, TokenType.PLUS):
            operator = self.previous()
            right = self.unary()
            return UnaryExpression(operator, right)
            
        return self.call()
        
    def call(self) -> Expression:
        """Parse un appel de fonction ou accès d'attribut/index."""
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LEFT_PAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expected property name after '.'")
                expr = AttributeExpression(expr, name.value, name)
            elif self.match(TokenType.LEFT_BRACKET):
                index = self.expression()
                self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after index")
                expr = IndexExpression(expr, index, self.previous())
            else:
                break
                
        return expr
        
    def finish_call(self, callee: Expression) -> CallExpression:
        """Termine le parsing d'un appel de fonction."""
        arguments = []
        
        if not self.check(TokenType.RIGHT_PAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
                
        paren = self.consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments")
        return CallExpression(callee, arguments, paren)
        
    def primary(self) -> Expression:
        """Parse une expression primaire."""
        if self.match(TokenType.BOOLEAN):
            return LiteralExpression(self.previous().value, self.previous())
            
        if self.match(TokenType.NONE):
            return LiteralExpression(None, self.previous())
            
        if self.match(TokenType.NUMBER):
            value = self.previous().value
            # Convertir en int ou float
            if '.' in value or 'e' in value.lower():
                value = float(value)
            else:
                value = int(value)
            return LiteralExpression(value, self.previous())
            
        if self.match(TokenType.STRING):
            return LiteralExpression(self.previous().value, self.previous())
            
        if self.match(TokenType.IDENTIFIER):
            return IdentifierExpression(self.previous().value, self.previous())
            
        if self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
            
        if self.match(TokenType.LEFT_BRACKET):
            return self.list_expression()
            
        if self.match(TokenType.LEFT_BRACE):
            return self.dict_expression()
            
        if self.match(TokenType.LAMBDA):
            return self.lambda_expression()
            
        if self.match(TokenType.TENSOR):
            return self.tensor_expression()
            
        self.error(f"Unexpected token: {self.peek().value}")
        
    def list_expression(self) -> ListExpression:
        """Parse une expression de liste."""
        elements = []
        token = self.previous()
        
        if not self.check(TokenType.RIGHT_BRACKET):
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                elements.append(self.expression())
                
        self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after list elements")
        return ListExpression(elements, token)
        
    def dict_expression(self) -> DictExpression:
        """Parse une expression de dictionnaire."""
        pairs = []
        token = self.previous()
        
        if not self.check(TokenType.RIGHT_BRACE):
            key = self.expression()
            self.consume(TokenType.COLON, "Expected ':' after dictionary key")
            value = self.expression()
            pairs.append((key, value))
            
            while self.match(TokenType.COMMA):
                key = self.expression()
                self.consume(TokenType.COLON, "Expected ':' after dictionary key")
                value = self.expression()
                pairs.append((key, value))
                
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after dictionary pairs")
        return DictExpression(pairs, token)
        
    def lambda_expression(self) -> LambdaExpression:
        """Parse une expression lambda."""
        token = self.previous()
        parameters = []
        
        if self.check(TokenType.IDENTIFIER):
            parameters.append(self.advance().value)
            while self.match(TokenType.COMMA):
                parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
                
        self.consume(TokenType.COLON, "Expected ':' after lambda parameters")
        body = self.expression()
        
        return LambdaExpression(parameters, body, token)
        
    def tensor_expression(self) -> TensorExpression:
        """Parse une expression de tenseur."""
        token = self.previous()
        
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'tensor'")
        
        # Parse shape
        shape = []
        if self.match(TokenType.LEFT_BRACKET):
            if not self.check(TokenType.RIGHT_BRACKET):
                shape.append(self.expression())
                while self.match(TokenType.COMMA):
                    shape.append(self.expression())
            self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after tensor shape")
        
        dtype = None
        data = None
        
        # Parse optional dtype and data
        while self.match(TokenType.COMMA):
            if self.check(TokenType.IDENTIFIER):
                param_name = self.advance().value
                self.consume(TokenType.ASSIGN, f"Expected '=' after '{param_name}'")
                
                if param_name == 'dtype':
                    dtype_token = self.consume(TokenType.STRING, "Expected string for dtype")
                    dtype = dtype_token.value
                elif param_name == 'data':
                    data = self.expression()
                    
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after tensor parameters")
        return TensorExpression(shape, dtype, data, token)
        
    # ========================================================================
    # PARSING DES STATEMENTS
    # ========================================================================
    
    def statement(self) -> Statement:
        """Parse une déclaration."""
        try:
            if self.match(TokenType.IF):
                return self.if_statement()
            if self.match(TokenType.WHILE):
                return self.while_statement()
            if self.match(TokenType.FOR):
                return self.for_statement()
            if self.match(TokenType.DEF):
                return self.function_definition()
            if self.match(TokenType.CLASS):
                return self.class_definition()
            if self.match(TokenType.RETURN):
                return self.return_statement()
            if self.match(TokenType.BREAK):
                return self.break_statement()
            if self.match(TokenType.CONTINUE):
                return self.continue_statement()
            if self.match(TokenType.IMPORT):
                return self.import_statement()
            if self.match(TokenType.FROM):
                return self.from_import_statement()
            if self.match(TokenType.TRY):
                return self.try_statement()
            if self.match(TokenType.WITH):
                return self.with_statement()
            if self.match(TokenType.MODEL):
                return self.model_definition()
            if self.match(TokenType.TRAIN):
                return self.train_statement()
            if self.match(TokenType.PREDICT):
                return self.predict_statement()
            if self.match(TokenType.DATASET):
                return self.dataset_definition()
                
            return self.expression_statement()
            
        except ParseError:
            self.synchronize()
            raise
            
    def if_statement(self) -> IfStatement:
        """Parse une déclaration if."""
        token = self.previous()
        condition = self.expression()
        self.consume(TokenType.COLON, "Expected ':' after if condition")
        
        then_branch = self.block()
        elif_branches = []
        else_branch = None
        
        while self.match(TokenType.ELIF):
            elif_condition = self.expression()
            self.consume(TokenType.COLON, "Expected ':' after elif condition")
            elif_body = self.block()
            elif_branches.append((elif_condition, elif_body))
            
        if self.match(TokenType.ELSE):
            self.consume(TokenType.COLON, "Expected ':' after else")
            else_branch = self.block()
            
        return IfStatement(condition, then_branch, elif_branches, else_branch, token)
        
    def while_statement(self) -> WhileStatement:
        """Parse une déclaration while."""
        token = self.previous()
        condition = self.expression()
        self.consume(TokenType.COLON, "Expected ':' after while condition")
        body = self.block()
        
        return WhileStatement(condition, body, token)
        
    def for_statement(self) -> ForStatement:
        """Parse une déclaration for."""
        token = self.previous()
        target = self.consume(TokenType.IDENTIFIER, "Expected variable name after 'for'").value
        self.consume(TokenType.IN, "Expected 'in' after for variable")
        iterable = self.expression()
        self.consume(TokenType.COLON, "Expected ':' after for clause")
        body = self.block()
        
        return ForStatement(target, iterable, body, token)
        
    def function_definition(self) -> FunctionDefinition:
        """Parse une définition de fonction."""
        token = self.previous()
        name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        parameters = []
        
        if not self.check(TokenType.RIGHT_PAREN):
            parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
            while self.match(TokenType.COMMA):
                parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
                
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.consume(TokenType.IDENTIFIER, "Expected return type").value
            
        self.consume(TokenType.COLON, "Expected ':' after function signature")
        body = self.block()
        
        return FunctionDefinition(name, parameters, body, return_type, False, token)
        
    def class_definition(self) -> ClassDefinition:
        """Parse une définition de classe."""
        token = self.previous()
        name = self.consume(TokenType.IDENTIFIER, "Expected class name").value
        
        superclass = None
        if self.match(TokenType.LEFT_PAREN):
            superclass = self.consume(TokenType.IDENTIFIER, "Expected superclass name").value
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after superclass")
            
        self.consume(TokenType.COLON, "Expected ':' after class name")
        
        # Parse class body
        self.consume(TokenType.NEWLINE, "Expected newline after ':'")
        self.consume(TokenType.INDENT, "Expected indentation in class body")
        
        methods = []
        attributes = []
        
        while not self.check(TokenType.DEDENT) and not self.is_at_end():
            self.skip_newlines()
            if self.match(TokenType.DEF):
                methods.append(self.function_definition())
            else:
                stmt = self.statement()
                if isinstance(stmt, AssignmentStatement):
                    attributes.append(stmt)
                    
        self.consume(TokenType.DEDENT, "Expected dedent after class body")
        
        return ClassDefinition(name, superclass, methods, attributes, token)
        
    def return_statement(self) -> ReturnStatement:
        """Parse une déclaration return."""
        token = self.previous()
        value = None
        
        if not self.check(TokenType.NEWLINE):
            value = self.expression()
            
        return ReturnStatement(value, token)
        
    def break_statement(self) -> BreakStatement:
        """Parse une déclaration break."""
        return BreakStatement(self.previous())
        
    def continue_statement(self) -> ContinueStatement:
        """Parse une déclaration continue."""
        return ContinueStatement(self.previous())
        
    def import_statement(self) -> ImportStatement:
        """Parse une déclaration import."""
        token = self.previous()
        module = self.consume(TokenType.IDENTIFIER, "Expected module name").value
        
        alias = None
        if self.match(TokenType.AS):
            alias = self.consume(TokenType.IDENTIFIER, "Expected alias name").value
            
        return ImportStatement(module, alias, None, token)
        
    def from_import_statement(self) -> ImportStatement:
        """Parse une déclaration from...import."""
        token = self.previous()
        module = self.consume(TokenType.IDENTIFIER, "Expected module name").value
        self.consume(TokenType.IMPORT, "Expected 'import' after module name")
        
        items = []
        items.append(self.consume(TokenType.IDENTIFIER, "Expected import item").value)
        
        while self.match(TokenType.COMMA):
            items.append(self.consume(TokenType.IDENTIFIER, "Expected import item").value)
            
        return ImportStatement(module, None, items, token)
        
    def expression_statement(self) -> Statement:
        """Parse une déclaration d'expression ou d'assignation."""
        expr = self.expression()
        
        if self.match(TokenType.ASSIGN):
            value = self.expression()
            return AssignmentStatement(expr, value, self.previous())
            
        return ExpressionStatement(expr)
        
    def model_definition(self) -> ModelDefinition:
        """Parse une définition de modèle."""
        token = self.previous()
        name = self.consume(TokenType.IDENTIFIER, "Expected model name").value
        self.consume(TokenType.ASSIGN, "Expected '=' after model name")
        
        model_type = self.consume(TokenType.IDENTIFIER, "Expected model type").value
        self.consume(TokenType.LEFT_BRACE, "Expected '{' after model type")
        
        layers = []
        optimizer = None
        loss = None
        metrics = None
        
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            if self.match(TokenType.IDENTIFIER):
                param_name = self.previous().value
                self.consume(TokenType.COLON, f"Expected ':' after '{param_name}'")
                
                if param_name == 'layers':
                    self.consume(TokenType.LEFT_BRACKET, "Expected '[' after 'layers:'")
                    if not self.check(TokenType.RIGHT_BRACKET):
                        layers.append(self.expression())
                        while self.match(TokenType.COMMA):
                            layers.append(self.expression())
                    self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after layers")
                elif param_name == 'optimizer':
                    optimizer = self.expression()
                elif param_name == 'loss':
                    loss = self.expression()
                elif param_name == 'metrics':
                    metrics = [self.expression()]
                    
            if self.match(TokenType.COMMA):
                continue
                
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after model definition")
        
        return ModelDefinition(name, model_type, layers, optimizer, loss, metrics, token)
        
    def train_statement(self) -> TrainStatement:
        """Parse une déclaration train."""
        token = self.previous()
        model = self.expression()
        self.consume(TokenType.DOT, "Expected '.' after model")
        self.consume(TokenType.IDENTIFIER, "Expected 'train' method")
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'train'")
        
        dataset = self.expression()
        epochs = None
        batch_size = None
        validation_data = None
        callbacks = None
        
        while self.match(TokenType.COMMA):
            param = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
            self.consume(TokenType.ASSIGN, f"Expected '=' after '{param}'")
            
            if param == 'epochs':
                epochs = self.expression()
            elif param == 'batch_size':
                batch_size = self.expression()
            elif param == 'validation_data':
                validation_data = self.expression()
            elif param == 'callbacks':
                callbacks = [self.expression()]
                
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after train parameters")
        
        return TrainStatement(model, dataset, epochs, batch_size, validation_data, callbacks, token)
        
    def predict_statement(self) -> PredictStatement:
        """Parse une déclaration predict."""
        token = self.previous()
        model = self.expression()
        self.consume(TokenType.DOT, "Expected '.' after model")
        self.consume(TokenType.IDENTIFIER, "Expected 'predict' method")
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'predict'")
        
        data = self.expression()
        target = None
        
        if self.match(TokenType.COMMA):
            target = self.expression()
            
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after predict parameters")
        
        return PredictStatement(model, data, target, token)
        
    def dataset_definition(self) -> DatasetDefinition:
        """Parse une définition de dataset."""
        token = self.previous()
        name = self.consume(TokenType.IDENTIFIER, "Expected dataset name").value
        self.consume(TokenType.ASSIGN, "Expected '=' after dataset name")
        
        source = self.expression()
        preprocessing = None
        split_ratio = None
        
        while self.match(TokenType.COMMA):
            param = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
            self.consume(TokenType.ASSIGN, f"Expected '=' after '{param}'")
            
            if param == 'preprocessing':
                preprocessing = [self.expression()]
            elif param == 'split_ratio':
                split_ratio = self.expression()
                
        return DatasetDefinition(name, source, preprocessing, split_ratio, token)
        
    def try_statement(self) -> TryStatement:
        """Parse une déclaration try."""
        token = self.previous()
        self.consume(TokenType.COLON, "Expected ':' after 'try'")
        try_body = self.block()
        
        except_clauses = []
        finally_body = None
        
        while self.match(TokenType.EXCEPT):
            exception_type = None
            exception_name = None
            
            if self.check(TokenType.IDENTIFIER):
                exception_type = self.advance().value
                if self.match(TokenType.AS):
                    exception_name = self.consume(TokenType.IDENTIFIER, "Expected exception variable name").value
                    
            self.consume(TokenType.COLON, "Expected ':' after except clause")
            except_body = self.block()
            except_clauses.append((exception_type, exception_name, except_body))
            
        if self.match(TokenType.FINALLY):
            self.consume(TokenType.COLON, "Expected ':' after 'finally'")
            finally_body = self.block()
            
        return TryStatement(try_body, except_clauses, finally_body, token)
        
    def with_statement(self) -> WithStatement:
        """Parse une déclaration with."""
        token = self.previous()
        context_expr = self.expression()
        
        target = None
        if self.match(TokenType.AS):
            target = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
            
        self.consume(TokenType.COLON, "Expected ':' after with clause")
        body = self.block()
        
        return WithStatement(context_expr, target, body, token)
        
    def block(self) -> List[Statement]:
        """Parse un bloc de code indenté."""
        statements = []
        
        self.consume(TokenType.NEWLINE, "Expected newline after ':'")
        self.consume(TokenType.INDENT, "Expected indentation")
        
        while not self.check(TokenType.DEDENT) and not self.is_at_end():
            self.skip_newlines()
            if not self.check(TokenType.DEDENT):
                statements.append(self.statement())
                
        self.consume(TokenType.DEDENT, "Expected dedent")
        
        return statements
        
    def parse(self) -> Program:
        """Parse le programme complet."""
        statements = []
        
        while not self.is_at_end():
            self.skip_newlines()
            if not self.is_at_end():
                statements.append(self.statement())
                
        return Program(statements)


def parse(source: str, filename: Optional[str] = None) -> Program:
    """Parse le code source ai'lang et retourne l'AST."""
    lexer = Lexer(source, filename)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()