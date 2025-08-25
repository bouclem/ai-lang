"""Interpréteur principal pour ai'lang."""

from typing import Any, List, Dict, Optional, Union
from ..parser.ast_nodes import *
from .environment import Environment, GlobalEnvironment, FunctionEnvironment, ClassEnvironment
from .values import *
import sys
import traceback


class InterpreterError(Exception):
    """Erreur d'exécution de l'interpréteur."""
    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        super().__init__(self.format_message())
        
    def format_message(self) -> str:
        if self.line:
            return f"Runtime Error (line {self.line}): {self.message}"
        return f"Runtime Error: {self.message}"


class ReturnException(Exception):
    """Exception pour gérer les instructions return."""
    def __init__(self, value: AILangValue):
        self.value = value


class BreakException(Exception):
    """Exception pour gérer les instructions break."""
    pass


class ContinueException(Exception):
    """Exception pour gérer les instructions continue."""
    pass


class Interpreter(ASTVisitor):
    """Interpréteur pour ai'lang."""
    
    def __init__(self):
        self.globals = GlobalEnvironment()
        self.environment = self.globals
        self.locals = {}
        
    def interpret(self, statements: List[Statement]) -> None:
        """Interprète une liste de déclarations."""
        try:
            for statement in statements:
                self.execute(statement)
        except InterpreterError:
            raise
        except Exception as e:
            raise InterpreterError(f"Unexpected error: {str(e)}")
            
    def execute(self, stmt: Statement) -> None:
        """Exécute une déclaration."""
        try:
            stmt.accept(self)
        except (ReturnException, BreakException, ContinueException):
            raise
        except InterpreterError:
            raise
        except Exception as e:
            line = getattr(stmt, 'line', None)
            raise InterpreterError(f"Error executing statement: {str(e)}", line)
            
    def evaluate(self, expr: Expression) -> AILangValue:
        """Évalue une expression."""
        try:
            return expr.accept(self)
        except InterpreterError:
            raise
        except Exception as e:
            line = getattr(expr, 'line', None)
            raise InterpreterError(f"Error evaluating expression: {str(e)}", line)
            
    def execute_block(self, statements: List[Statement], environment: Environment) -> None:
        """Exécute un bloc de déclarations dans un environnement donné."""
        previous = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous
            
    def execute_function_body(self, body: List[Statement], environment: Environment) -> AILangValue:
        """Exécute le corps d'une fonction et retourne la valeur de retour."""
        try:
            self.execute_block(body, environment)
        except ReturnException as ret:
            return ret.value
        return AILangNone()
        
    # Visiteurs pour les expressions
    
    def visit_literal_expr(self, expr: LiteralExpr) -> AILangValue:
        """Visite une expression littérale."""
        return python_to_ailang(expr.value)
        
    def visit_identifier_expr(self, expr: IdentifierExpr) -> AILangValue:
        """Visite un identifiant."""
        return self.environment.get(expr.name)
        
    def visit_binary_expr(self, expr: BinaryExpr) -> AILangValue:
        """Visite une expression binaire."""
        left = self.evaluate(expr.left)
        
        # Court-circuit pour les opérateurs logiques
        if expr.operator == 'and':
            if not left.is_truthy():
                return left
            return self.evaluate(expr.right)
        elif expr.operator == 'or':
            if left.is_truthy():
                return left
            return self.evaluate(expr.right)
            
        right = self.evaluate(expr.right)
        
        # Opérateurs arithmétiques
        if expr.operator == '+':
            if isinstance(left, (AILangNumber, AILangString, AILangList, AILangTensor)):
                return left + right
        elif expr.operator == '-':
            if isinstance(left, (AILangNumber, AILangTensor)):
                return left - right
        elif expr.operator == '*':
            if isinstance(left, (AILangNumber, AILangString, AILangTensor)):
                return left * right
        elif expr.operator == '/':
            if isinstance(left, (AILangNumber, AILangTensor)):
                return left / right
        elif expr.operator == '//':
            if isinstance(left, AILangNumber):
                return left // right
        elif expr.operator == '%':
            if isinstance(left, AILangNumber):
                return left % right
        elif expr.operator == '**':
            if isinstance(left, AILangNumber):
                return left ** right
        elif expr.operator == '@':
            if isinstance(left, AILangTensor):
                return left @ right
                
        # Opérateurs de comparaison
        elif expr.operator == '==':
            return AILangBoolean(left == right)
        elif expr.operator == '!=':
            return AILangBoolean(left != right)
        elif expr.operator == '<':
            if isinstance(left, AILangNumber):
                return AILangBoolean(left < right)
        elif expr.operator == '<=':
            if isinstance(left, AILangNumber):
                return AILangBoolean(left <= right)
        elif expr.operator == '>':
            if isinstance(left, AILangNumber):
                return AILangBoolean(left > right)
        elif expr.operator == '>=':
            if isinstance(left, AILangNumber):
                return AILangBoolean(left >= right)
                
        raise InterpreterError(f"Unsupported binary operator '{expr.operator}' for types {type(left).__name__} and {type(right).__name__}")
        
    def visit_unary_expr(self, expr: UnaryExpr) -> AILangValue:
        """Visite une expression unaire."""
        operand = self.evaluate(expr.operand)
        
        if expr.operator == '-':
            if isinstance(operand, AILangNumber):
                return -operand
        elif expr.operator == 'not':
            return AILangBoolean(not operand.is_truthy())
            
        raise InterpreterError(f"Unsupported unary operator '{expr.operator}' for type {type(operand).__name__}")
        
    def visit_call_expr(self, expr: CallExpr) -> AILangValue:
        """Visite un appel de fonction."""
        callee = self.evaluate(expr.callee)
        
        arguments = []
        for arg in expr.arguments:
            arguments.append(self.evaluate(arg))
            
        if isinstance(callee, AILangFunction):
            if len(arguments) != len(callee.parameters):
                raise InterpreterError(f"Function '{callee.name}' expects {len(callee.parameters)} arguments but got {len(arguments)}")
            return callee.call(self, arguments)
        elif isinstance(callee, AILangClass):
            return callee.instantiate(self, arguments)
        elif isinstance(callee, AILangBoundMethod):
            return callee.call(self, arguments)
        else:
            raise InterpreterError(f"'{type(callee).__name__}' object is not callable")
            
    def visit_attribute_expr(self, expr: AttributeExpr) -> AILangValue:
        """Visite un accès d'attribut."""
        obj = self.evaluate(expr.object)
        
        if isinstance(obj, AILangInstance):
            return obj.get(expr.name)
        elif isinstance(obj, AILangClass):
            if expr.name in obj.methods:
                return obj.methods[expr.name]
            elif expr.name in obj.attributes:
                return obj.attributes[expr.name]
                
        raise InterpreterError(f"'{type(obj).__name__}' object has no attribute '{expr.name}'")
        
    def visit_index_expr(self, expr: IndexExpr) -> AILangValue:
        """Visite un accès par index."""
        obj = self.evaluate(expr.object)
        index = self.evaluate(expr.index)
        
        if isinstance(obj, (AILangList, AILangString)):
            if isinstance(index, AILangNumber) and isinstance(index.value, int):
                try:
                    return obj[index.value]
                except IndexError:
                    raise InterpreterError(f"Index {index.value} out of range")
        elif isinstance(obj, AILangDict):
            if isinstance(index, AILangString):
                try:
                    return obj[index.value]
                except KeyError:
                    raise InterpreterError(f"Key '{index.value}' not found")
                    
        raise InterpreterError(f"'{type(obj).__name__}' object is not subscriptable or invalid index type")
        
    def visit_list_expr(self, expr: ListExpr) -> AILangValue:
        """Visite une expression de liste."""
        elements = []
        for element in expr.elements:
            elements.append(self.evaluate(element))
        return AILangList(elements)
        
    def visit_dict_expr(self, expr: DictExpr) -> AILangValue:
        """Visite une expression de dictionnaire."""
        pairs = {}
        for key_expr, value_expr in expr.pairs:
            key = self.evaluate(key_expr)
            value = self.evaluate(value_expr)
            if isinstance(key, AILangString):
                pairs[key.value] = value
            else:
                pairs[str(key.to_string())] = value
        return AILangDict(pairs)
        
    def visit_lambda_expr(self, expr: LambdaExpr) -> AILangValue:
        """Visite une expression lambda."""
        return AILangFunction(
            name="<lambda>",
            parameters=expr.parameters,
            body=[ReturnStmt(expr.body)],  # Convertir l'expression en return
            closure=self.environment
        )
        
    def visit_tensor_expr(self, expr: TensorExpr) -> AILangValue:
        """Visite une expression de tenseur."""
        import numpy as np
        
        # Évaluer les éléments
        elements = []
        for element in expr.elements:
            value = self.evaluate(element)
            if isinstance(value, AILangNumber):
                elements.append(value.value)
            else:
                raise InterpreterError(f"Tensor elements must be numbers, got {type(value).__name__}")
                
        # Créer le tenseur
        data = np.array(elements, dtype=expr.dtype)
        if expr.shape:
            data = data.reshape(expr.shape)
            
        tensor = AILangTensor(list(data.shape), expr.dtype)
        tensor.data = data
        return tensor
        
    # Visiteurs pour les déclarations
    
    def visit_expression_stmt(self, stmt: ExpressionStmt) -> None:
        """Visite une déclaration d'expression."""
        self.evaluate(stmt.expression)
        
    def visit_assignment_stmt(self, stmt: AssignmentStmt) -> None:
        """Visite une assignation."""
        value = self.evaluate(stmt.value)
        
        if isinstance(stmt.target, IdentifierExpr):
            self.environment.define(stmt.target.name, value)
        elif isinstance(stmt.target, AttributeExpr):
            obj = self.evaluate(stmt.target.object)
            if isinstance(obj, AILangInstance):
                obj.set(stmt.target.name, value)
            else:
                raise InterpreterError(f"Cannot set attribute on {type(obj).__name__}")
        elif isinstance(stmt.target, IndexExpr):
            obj = self.evaluate(stmt.target.object)
            index = self.evaluate(stmt.target.index)
            
            if isinstance(obj, AILangList) and isinstance(index, AILangNumber):
                obj[int(index.value)] = value
            elif isinstance(obj, AILangDict) and isinstance(index, AILangString):
                obj[index.value] = value
            else:
                raise InterpreterError(f"Invalid assignment target")
        else:
            raise InterpreterError(f"Invalid assignment target")
            
    def visit_if_stmt(self, stmt: IfStmt) -> None:
        """Visite une déclaration if."""
        condition = self.evaluate(stmt.condition)
        if condition.is_truthy():
            for s in stmt.then_branch:
                self.execute(s)
        elif stmt.else_branch:
            for s in stmt.else_branch:
                self.execute(s)
                
    def visit_while_stmt(self, stmt: WhileStmt) -> None:
        """Visite une boucle while."""
        try:
            while True:
                condition = self.evaluate(stmt.condition)
                if not condition.is_truthy():
                    break
                    
                try:
                    for s in stmt.body:
                        self.execute(s)
                except BreakException:
                    break
                except ContinueException:
                    continue
        except (BreakException, ContinueException):
            pass
            
    def visit_for_stmt(self, stmt: ForStmt) -> None:
        """Visite une boucle for."""
        iterable = self.evaluate(stmt.iterable)
        
        if isinstance(iterable, AILangList):
            items = iterable.elements
        elif isinstance(iterable, AILangString):
            items = [AILangString(char) for char in iterable.value]
        elif isinstance(iterable, AILangDict):
            items = [AILangString(key) for key in iterable.keys()]
        else:
            raise InterpreterError(f"'{type(iterable).__name__}' object is not iterable")
            
        try:
            for item in items:
                self.environment.define(stmt.variable, item)
                try:
                    for s in stmt.body:
                        self.execute(s)
                except BreakException:
                    break
                except ContinueException:
                    continue
        except (BreakException, ContinueException):
            pass
            
    def visit_function_def(self, stmt: FunctionDef) -> None:
        """Visite une définition de fonction."""
        function = AILangFunction(
            name=stmt.name,
            parameters=stmt.parameters,
            body=stmt.body,
            closure=self.environment
        )
        self.environment.define(stmt.name, function)
        
    def visit_class_def(self, stmt: ClassDef) -> None:
        """Visite une définition de classe."""
        superclass = None
        if stmt.superclass:
            superclass_value = self.evaluate(stmt.superclass)
            if not isinstance(superclass_value, AILangClass):
                raise InterpreterError("Superclass must be a class")
            superclass = superclass_value
            
        # Créer l'environnement de classe
        class_env = ClassEnvironment(self.environment)
        
        # Exécuter le corps de la classe
        previous = self.environment
        self.environment = class_env
        
        try:
            for method in stmt.methods:
                self.execute(method)
        finally:
            self.environment = previous
            
        # Récupérer les méthodes définies
        methods = {}
        for name, value in class_env.values.items():
            if isinstance(value, AILangFunction):
                methods[name] = value
                
        # Créer la classe
        klass = AILangClass(
            name=stmt.name,
            methods=methods,
            attributes={},
            superclass=superclass
        )
        
        self.environment.define(stmt.name, klass)
        
    def visit_return_stmt(self, stmt: ReturnStmt) -> None:
        """Visite une déclaration return."""
        value = AILangNone()
        if stmt.value:
            value = self.evaluate(stmt.value)
        raise ReturnException(value)
        
    def visit_break_stmt(self, stmt: BreakStmt) -> None:
        """Visite une déclaration break."""
        raise BreakException()
        
    def visit_continue_stmt(self, stmt: ContinueStmt) -> None:
        """Visite une déclaration continue."""
        raise ContinueException()
        
    def visit_import_stmt(self, stmt: ImportStmt) -> None:
        """Visite une déclaration import."""
        # Pour l'instant, on simule l'import
        if stmt.alias:
            self.environment.define(stmt.alias, AILangString(f"module {stmt.module}"))
        else:
            self.environment.define(stmt.module, AILangString(f"module {stmt.module}"))
            
    def visit_try_stmt(self, stmt: TryStmt) -> None:
        """Visite une déclaration try/except."""
        try:
            for s in stmt.try_block:
                self.execute(s)
        except InterpreterError as e:
            # Chercher un handler approprié
            for except_clause in stmt.except_clauses:
                if except_clause.exception_type is None:  # catch all
                    if except_clause.variable:
                        self.environment.define(except_clause.variable, AILangString(str(e)))
                    for s in except_clause.body:
                        self.execute(s)
                    break
            else:
                raise  # Re-raise si aucun handler trouvé
        finally:
            if stmt.finally_block:
                for s in stmt.finally_block:
                    self.execute(s)
                    
    def visit_with_stmt(self, stmt: WithStmt) -> None:
        """Visite une déclaration with."""
        # Pour l'instant, on exécute simplement le corps
        context = self.evaluate(stmt.context_expr)
        if stmt.variable:
            self.environment.define(stmt.variable, context)
            
        for s in stmt.body:
            self.execute(s)
            
    # Visiteurs pour les déclarations spécifiques à l'IA
    
    def visit_model_def(self, stmt: ModelDef) -> None:
        """Visite une définition de modèle IA."""
        # Créer un objet modèle spécial
        model = AILangDict({
            'name': AILangString(stmt.name),
            'type': AILangString('model'),
            'layers': AILangList([]),
            'compiled': AILangBoolean(False)
        })
        self.environment.define(stmt.name, model)
        
    def visit_train_stmt(self, stmt: TrainStmt) -> None:
        """Visite une déclaration d'entraînement."""
        model = self.evaluate(stmt.model)
        data = self.evaluate(stmt.data)
        
        # Simuler l'entraînement
        if isinstance(model, AILangDict):
            model.pairs['trained'] = AILangBoolean(True)
            
    def visit_predict_stmt(self, stmt: PredictStmt) -> None:
        """Visite une déclaration de prédiction."""
        model = self.evaluate(stmt.model)
        data = self.evaluate(stmt.data)
        
        # Simuler la prédiction
        result = AILangList([AILangNumber(0.5)])  # Résultat fictif
        if stmt.target:
            self.environment.define(stmt.target, result)
            
    def visit_dataset_def(self, stmt: DatasetDef) -> None:
        """Visite une définition de dataset."""
        source = self.evaluate(stmt.source)
        
        # Créer un objet dataset
        dataset = AILangDict({
            'name': AILangString(stmt.name),
            'type': AILangString('dataset'),
            'source': source,
            'size': AILangNumber(1000)  # Taille fictive
        })
        self.environment.define(stmt.name, dataset)


def interpret_code(source_code: str) -> None:
    """Fonction utilitaire pour interpréter du code ai'lang."""
    from ..lexer import Lexer
    from ..parser import Parser
    
    try:
        # Lexer
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parser
        parser = Parser(tokens)
        statements = parser.parse()
        
        # Interpreter
        interpreter = Interpreter()
        interpreter.interpret(statements)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()