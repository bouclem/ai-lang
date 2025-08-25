"""Nœuds de l'arbre syntaxique abstrait (AST) pour ai'lang."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union
from dataclasses import dataclass
from ..lexer.token import Token


class ASTNode(ABC):
    """Classe de base pour tous les nœuds de l'AST."""
    
    @abstractmethod
    def accept(self, visitor):
        """Méthode pour le pattern Visitor."""
        pass


# ============================================================================
# EXPRESSIONS
# ============================================================================

class Expression(ASTNode):
    """Classe de base pour toutes les expressions."""
    pass


@dataclass
class LiteralExpression(Expression):
    """Expression littérale (nombre, chaîne, booléen, None)."""
    value: Any
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_literal_expression(self)


@dataclass
class IdentifierExpression(Expression):
    """Expression d'identifiant (nom de variable)."""
    name: str
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_identifier_expression(self)


@dataclass
class BinaryExpression(Expression):
    """Expression binaire (a + b, a == b, etc.)."""
    left: Expression
    operator: Token
    right: Expression
    
    def accept(self, visitor):
        return visitor.visit_binary_expression(self)


@dataclass
class UnaryExpression(Expression):
    """Expression unaire (-a, not a, etc.)."""
    operator: Token
    operand: Expression
    
    def accept(self, visitor):
        return visitor.visit_unary_expression(self)


@dataclass
class CallExpression(Expression):
    """Appel de fonction ou méthode."""
    callee: Expression
    arguments: List[Expression]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_call_expression(self)


@dataclass
class AttributeExpression(Expression):
    """Accès à un attribut (obj.attr)."""
    object: Expression
    name: str
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_attribute_expression(self)


@dataclass
class IndexExpression(Expression):
    """Accès par index (arr[0], dict['key'])."""
    object: Expression
    index: Expression
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_index_expression(self)


@dataclass
class ListExpression(Expression):
    """Expression de liste [1, 2, 3]."""
    elements: List[Expression]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_list_expression(self)


@dataclass
class DictExpression(Expression):
    """Expression de dictionnaire {'key': value}."""
    pairs: List[tuple[Expression, Expression]]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_dict_expression(self)


@dataclass
class LambdaExpression(Expression):
    """Expression lambda."""
    parameters: List[str]
    body: Expression
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_lambda_expression(self)


@dataclass
class TensorExpression(Expression):
    """Expression de tenseur spécifique à ai'lang."""
    shape: List[Expression]
    dtype: Optional[str]
    data: Optional[Expression]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_tensor_expression(self)


# ============================================================================
# STATEMENTS
# ============================================================================

class Statement(ASTNode):
    """Classe de base pour toutes les déclarations."""
    pass


@dataclass
class ExpressionStatement(Statement):
    """Déclaration d'expression."""
    expression: Expression
    
    def accept(self, visitor):
        return visitor.visit_expression_statement(self)


@dataclass
class AssignmentStatement(Statement):
    """Déclaration d'assignation."""
    target: Expression
    value: Expression
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_assignment_statement(self)


@dataclass
class IfStatement(Statement):
    """Déclaration if/elif/else."""
    condition: Expression
    then_branch: List[Statement]
    elif_branches: List[tuple[Expression, List[Statement]]]
    else_branch: Optional[List[Statement]]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_if_statement(self)


@dataclass
class WhileStatement(Statement):
    """Déclaration while."""
    condition: Expression
    body: List[Statement]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_while_statement(self)


@dataclass
class ForStatement(Statement):
    """Déclaration for."""
    target: str
    iterable: Expression
    body: List[Statement]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_for_statement(self)


@dataclass
class FunctionDefinition(Statement):
    """Définition de fonction."""
    name: str
    parameters: List[str]
    body: List[Statement]
    return_type: Optional[str]
    is_async: bool
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_function_definition(self)


@dataclass
class ClassDefinition(Statement):
    """Définition de classe."""
    name: str
    superclass: Optional[str]
    methods: List[FunctionDefinition]
    attributes: List[AssignmentStatement]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_class_definition(self)


@dataclass
class ReturnStatement(Statement):
    """Déclaration return."""
    value: Optional[Expression]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_return_statement(self)


@dataclass
class BreakStatement(Statement):
    """Déclaration break."""
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_break_statement(self)


@dataclass
class ContinueStatement(Statement):
    """Déclaration continue."""
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_continue_statement(self)


@dataclass
class ImportStatement(Statement):
    """Déclaration import."""
    module: str
    alias: Optional[str]
    items: Optional[List[str]]  # Pour from module import item1, item2
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_import_statement(self)


@dataclass
class TryStatement(Statement):
    """Déclaration try/except/finally."""
    try_body: List[Statement]
    except_clauses: List[tuple[Optional[str], Optional[str], List[Statement]]]
    finally_body: Optional[List[Statement]]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_try_statement(self)


@dataclass
class WithStatement(Statement):
    """Déclaration with."""
    context_expr: Expression
    target: Optional[str]
    body: List[Statement]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_with_statement(self)


# ============================================================================
# NŒUDS SPÉCIFIQUES À AI'LANG
# ============================================================================

@dataclass
class ModelDefinition(Statement):
    """Définition de modèle IA."""
    name: str
    model_type: str
    layers: List[Expression]
    optimizer: Optional[Expression]
    loss: Optional[Expression]
    metrics: Optional[List[Expression]]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_model_definition(self)


@dataclass
class TrainStatement(Statement):
    """Déclaration d'entraînement de modèle."""
    model: Expression
    dataset: Expression
    epochs: Optional[Expression]
    batch_size: Optional[Expression]
    validation_data: Optional[Expression]
    callbacks: Optional[List[Expression]]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_train_statement(self)


@dataclass
class PredictStatement(Statement):
    """Déclaration de prédiction."""
    model: Expression
    data: Expression
    target: Optional[Expression]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_predict_statement(self)


@dataclass
class DatasetDefinition(Statement):
    """Définition de dataset."""
    name: str
    source: Expression
    preprocessing: Optional[List[Expression]]
    split_ratio: Optional[Expression]
    token: Token
    
    def accept(self, visitor):
        return visitor.visit_dataset_definition(self)


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

@dataclass
class Program(ASTNode):
    """Nœud racine représentant un programme complet."""
    statements: List[Statement]
    
    def accept(self, visitor):
        return visitor.visit_program(self)


# ============================================================================
# VISITOR INTERFACE
# ============================================================================

class ASTVisitor(ABC):
    """Interface pour le pattern Visitor sur l'AST."""
    
    @abstractmethod
    def visit_literal_expression(self, node: LiteralExpression): pass
    
    @abstractmethod
    def visit_identifier_expression(self, node: IdentifierExpression): pass
    
    @abstractmethod
    def visit_binary_expression(self, node: BinaryExpression): pass
    
    @abstractmethod
    def visit_unary_expression(self, node: UnaryExpression): pass
    
    @abstractmethod
    def visit_call_expression(self, node: CallExpression): pass
    
    @abstractmethod
    def visit_attribute_expression(self, node: AttributeExpression): pass
    
    @abstractmethod
    def visit_index_expression(self, node: IndexExpression): pass
    
    @abstractmethod
    def visit_list_expression(self, node: ListExpression): pass
    
    @abstractmethod
    def visit_dict_expression(self, node: DictExpression): pass
    
    @abstractmethod
    def visit_lambda_expression(self, node: LambdaExpression): pass
    
    @abstractmethod
    def visit_tensor_expression(self, node: TensorExpression): pass
    
    @abstractmethod
    def visit_expression_statement(self, node: ExpressionStatement): pass
    
    @abstractmethod
    def visit_assignment_statement(self, node: AssignmentStatement): pass
    
    @abstractmethod
    def visit_if_statement(self, node: IfStatement): pass
    
    @abstractmethod
    def visit_while_statement(self, node: WhileStatement): pass
    
    @abstractmethod
    def visit_for_statement(self, node: ForStatement): pass
    
    @abstractmethod
    def visit_function_definition(self, node: FunctionDefinition): pass
    
    @abstractmethod
    def visit_class_definition(self, node: ClassDefinition): pass
    
    @abstractmethod
    def visit_return_statement(self, node: ReturnStatement): pass
    
    @abstractmethod
    def visit_break_statement(self, node: BreakStatement): pass
    
    @abstractmethod
    def visit_continue_statement(self, node: ContinueStatement): pass
    
    @abstractmethod
    def visit_import_statement(self, node: ImportStatement): pass
    
    @abstractmethod
    def visit_try_statement(self, node: TryStatement): pass
    
    @abstractmethod
    def visit_with_statement(self, node: WithStatement): pass
    
    @abstractmethod
    def visit_model_definition(self, node: ModelDefinition): pass
    
    @abstractmethod
    def visit_train_statement(self, node: TrainStatement): pass
    
    @abstractmethod
    def visit_predict_statement(self, node: PredictStatement): pass
    
    @abstractmethod
    def visit_dataset_definition(self, node: DatasetDefinition): pass
    
    @abstractmethod
    def visit_program(self, node: Program): pass