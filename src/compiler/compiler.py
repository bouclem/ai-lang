"""Compilateur pour ai'lang avec optimisations de performance."""

from typing import List, Dict, Any, Optional, Set
from ..parser.ast_nodes import *
from ..interpreter.values import *
import ast
import types
import dis


class CompilerError(Exception):
    """Erreur de compilation."""
    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        super().__init__(self.format_message())
        
    def format_message(self) -> str:
        if self.line:
            return f"Compilation Error (line {self.line}): {self.message}"
        return f"Compilation Error: {self.message}"


class OptimizationPass:
    """Classe de base pour les passes d'optimisation."""
    
    def optimize(self, node: ASTNode) -> ASTNode:
        """Applique l'optimisation au nœud AST."""
        return node


class ConstantFolding(OptimizationPass):
    """Optimisation de pliage de constantes."""
    
    def optimize(self, node: ASTNode) -> ASTNode:
        """Plie les expressions constantes."""
        if isinstance(node, BinaryExpr):
            left = self.optimize(node.left)
            right = self.optimize(node.right)
            
            # Si les deux opérandes sont des littéraux, calculer le résultat
            if isinstance(left, LiteralExpr) and isinstance(right, LiteralExpr):
                try:
                    result = self._evaluate_binary_op(left.value, node.operator, right.value)
                    return LiteralExpr(result)
                except:
                    pass  # Garder l'expression originale si l'évaluation échoue
                    
            return BinaryExpr(left, node.operator, right)
            
        elif isinstance(node, UnaryExpr):
            operand = self.optimize(node.operand)
            
            if isinstance(operand, LiteralExpr):
                try:
                    result = self._evaluate_unary_op(node.operator, operand.value)
                    return LiteralExpr(result)
                except:
                    pass
                    
            return UnaryExpr(node.operator, operand)
            
        return node
        
    def _evaluate_binary_op(self, left: Any, op: str, right: Any) -> Any:
        """Évalue une opération binaire sur des constantes."""
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right
        elif op == '//':
            return left // right
        elif op == '%':
            return left % right
        elif op == '**':
            return left ** right
        elif op == '==':
            return left == right
        elif op == '!=':
            return left != right
        elif op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        else:
            raise ValueError(f"Unknown operator: {op}")
            
    def _evaluate_unary_op(self, op: str, operand: Any) -> Any:
        """Évalue une opération unaire sur une constante."""
        if op == '-':
            return -operand
        elif op == 'not':
            return not operand
        else:
            raise ValueError(f"Unknown unary operator: {op}")


class DeadCodeElimination(OptimizationPass):
    """Élimination du code mort."""
    
    def optimize(self, node: ASTNode) -> ASTNode:
        """Élimine le code inaccessible."""
        if isinstance(node, IfStmt):
            condition = node.condition
            
            # Si la condition est une constante
            if isinstance(condition, LiteralExpr):
                if condition.value:  # Condition toujours vraie
                    return node.then_branch[0] if node.then_branch else None
                else:  # Condition toujours fausse
                    return node.else_branch[0] if node.else_branch else None
                    
        return node


class LoopOptimization(OptimizationPass):
    """Optimisations de boucles."""
    
    def optimize(self, node: ASTNode) -> ASTNode:
        """Optimise les boucles."""
        if isinstance(node, WhileStmt):
            # Détection de boucles infinies évidentes
            if isinstance(node.condition, LiteralExpr) and node.condition.value is True:
                # Marquer comme boucle infinie pour optimisation ultérieure
                pass
                
        elif isinstance(node, ForStmt):
            # Optimisation des boucles for avec ranges
            if isinstance(node.iterable, CallExpr):
                if isinstance(node.iterable.callee, IdentifierExpr) and node.iterable.callee.name == 'range':
                    # Optimiser les boucles range
                    pass
                    
        return node


class InlineOptimization(OptimizationPass):
    """Optimisation d'inlining de fonctions."""
    
    def __init__(self):
        self.inline_candidates: Set[str] = set()
        
    def optimize(self, node: ASTNode) -> ASTNode:
        """Inline les petites fonctions."""
        if isinstance(node, FunctionDef):
            # Marquer les petites fonctions comme candidates à l'inlining
            if len(node.body) <= 3:  # Fonctions de 3 lignes ou moins
                self.inline_candidates.add(node.name)
                
        elif isinstance(node, CallExpr):
            if isinstance(node.callee, IdentifierExpr):
                if node.callee.name in self.inline_candidates:
                    # Ici on pourrait inliner la fonction
                    pass
                    
        return node


class PythonCodeGenerator:
    """Générateur de code Python optimisé."""
    
    def __init__(self):
        self.code_lines: List[str] = []
        self.indent_level = 0
        self.imports: Set[str] = set()
        
    def generate(self, statements: List[Statement]) -> str:
        """Génère du code Python à partir de l'AST ai'lang."""
        self.code_lines = []
        self.indent_level = 0
        self.imports = set()
        
        # Ajouter les imports nécessaires
        self.imports.add("import numpy as np")
        self.imports.add("from typing import Any, List, Dict, Optional")
        
        # Générer le code pour chaque déclaration
        for stmt in statements:
            self._generate_statement(stmt)
            
        # Construire le code final
        result = "\n".join(self.imports) + "\n\n" + "\n".join(self.code_lines)
        return result
        
    def _emit(self, code: str) -> None:
        """Émet une ligne de code avec l'indentation appropriée."""
        indent = "    " * self.indent_level
        self.code_lines.append(indent + code)
        
    def _generate_statement(self, stmt: Statement) -> None:
        """Génère le code pour une déclaration."""
        if isinstance(stmt, ExpressionStmt):
            expr_code = self._generate_expression(stmt.expression)
            self._emit(expr_code)
            
        elif isinstance(stmt, AssignmentStmt):
            target_code = self._generate_expression(stmt.target)
            value_code = self._generate_expression(stmt.value)
            self._emit(f"{target_code} = {value_code}")
            
        elif isinstance(stmt, IfStmt):
            condition_code = self._generate_expression(stmt.condition)
            self._emit(f"if {condition_code}:")
            self.indent_level += 1
            for s in stmt.then_branch:
                self._generate_statement(s)
            self.indent_level -= 1
            
            if stmt.else_branch:
                self._emit("else:")
                self.indent_level += 1
                for s in stmt.else_branch:
                    self._generate_statement(s)
                self.indent_level -= 1
                
        elif isinstance(stmt, WhileStmt):
            condition_code = self._generate_expression(stmt.condition)
            self._emit(f"while {condition_code}:")
            self.indent_level += 1
            for s in stmt.body:
                self._generate_statement(s)
            self.indent_level -= 1
            
        elif isinstance(stmt, ForStmt):
            iterable_code = self._generate_expression(stmt.iterable)
            self._emit(f"for {stmt.variable} in {iterable_code}:")
            self.indent_level += 1
            for s in stmt.body:
                self._generate_statement(s)
            self.indent_level -= 1
            
        elif isinstance(stmt, FunctionDef):
            params = ", ".join(stmt.parameters)
            self._emit(f"def {stmt.name}({params}):")
            self.indent_level += 1
            for s in stmt.body:
                self._generate_statement(s)
            self.indent_level -= 1
            
        elif isinstance(stmt, ClassDef):
            superclass = ""
            if stmt.superclass:
                superclass = f"({self._generate_expression(stmt.superclass)})"
            self._emit(f"class {stmt.name}{superclass}:")
            self.indent_level += 1
            for method in stmt.methods:
                self._generate_statement(method)
            self.indent_level -= 1
            
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                value_code = self._generate_expression(stmt.value)
                self._emit(f"return {value_code}")
            else:
                self._emit("return")
                
        elif isinstance(stmt, BreakStmt):
            self._emit("break")
            
        elif isinstance(stmt, ContinueStmt):
            self._emit("continue")
            
        elif isinstance(stmt, ImportStmt):
            if stmt.alias:
                self._emit(f"import {stmt.module} as {stmt.alias}")
            else:
                self._emit(f"import {stmt.module}")
                
        # Déclarations spécifiques à l'IA
        elif isinstance(stmt, ModelDef):
            self._emit(f"# Model definition: {stmt.name}")
            self._emit(f"{stmt.name} = {{'type': 'model', 'name': '{stmt.name}'}}")
            
        elif isinstance(stmt, TrainStmt):
            model_code = self._generate_expression(stmt.model)
            data_code = self._generate_expression(stmt.data)
            self._emit(f"# Training: {model_code} with {data_code}")
            
        elif isinstance(stmt, PredictStmt):
            model_code = self._generate_expression(stmt.model)
            data_code = self._generate_expression(stmt.data)
            if stmt.target:
                self._emit(f"{stmt.target} = {model_code}.predict({data_code})")
            else:
                self._emit(f"{model_code}.predict({data_code})")
                
        elif isinstance(stmt, DatasetDef):
            source_code = self._generate_expression(stmt.source)
            self._emit(f"{stmt.name} = {{'type': 'dataset', 'source': {source_code}}}")
            
    def _generate_expression(self, expr: Expression) -> str:
        """Génère le code pour une expression."""
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, str):
                return f"'{expr.value}'"
            else:
                return str(expr.value)
                
        elif isinstance(expr, IdentifierExpr):
            return expr.name
            
        elif isinstance(expr, BinaryExpr):
            left_code = self._generate_expression(expr.left)
            right_code = self._generate_expression(expr.right)
            return f"({left_code} {expr.operator} {right_code})"
            
        elif isinstance(expr, UnaryExpr):
            operand_code = self._generate_expression(expr.operand)
            return f"({expr.operator} {operand_code})"
            
        elif isinstance(expr, CallExpr):
            callee_code = self._generate_expression(expr.callee)
            args_code = ", ".join(self._generate_expression(arg) for arg in expr.arguments)
            return f"{callee_code}({args_code})"
            
        elif isinstance(expr, AttributeExpr):
            obj_code = self._generate_expression(expr.object)
            return f"{obj_code}.{expr.name}"
            
        elif isinstance(expr, IndexExpr):
            obj_code = self._generate_expression(expr.object)
            index_code = self._generate_expression(expr.index)
            return f"{obj_code}[{index_code}]"
            
        elif isinstance(expr, ListExpr):
            elements_code = ", ".join(self._generate_expression(elem) for elem in expr.elements)
            return f"[{elements_code}]"
            
        elif isinstance(expr, DictExpr):
            pairs_code = ", ".join(
                f"{self._generate_expression(key)}: {self._generate_expression(value)}"
                for key, value in expr.pairs
            )
            return f"{{{pairs_code}}}"
            
        elif isinstance(expr, LambdaExpr):
            params = ", ".join(expr.parameters)
            body_code = self._generate_expression(expr.body)
            return f"lambda {params}: {body_code}"
            
        elif isinstance(expr, TensorExpr):
            elements_code = ", ".join(self._generate_expression(elem) for elem in expr.elements)
            shape_code = str(expr.shape) if expr.shape else "None"
            return f"np.array([{elements_code}], dtype='{expr.dtype}').reshape({shape_code})"
            
        else:
            return "# Unknown expression"


class Compiler:
    """Compilateur principal pour ai'lang."""
    
    def __init__(self):
        self.optimizations = [
            ConstantFolding(),
            DeadCodeElimination(),
            LoopOptimization(),
            InlineOptimization()
        ]
        self.code_generator = PythonCodeGenerator()
        
    def compile(self, statements: List[Statement], optimize: bool = True) -> str:
        """Compile l'AST ai'lang en code Python optimisé."""
        try:
            # Appliquer les optimisations
            if optimize:
                statements = self._optimize(statements)
                
            # Générer le code Python
            python_code = self.code_generator.generate(statements)
            
            return python_code
            
        except Exception as e:
            raise CompilerError(f"Compilation failed: {str(e)}")
            
    def _optimize(self, statements: List[Statement]) -> List[Statement]:
        """Applique les passes d'optimisation."""
        optimized = statements
        
        for optimization in self.optimizations:
            optimized = [optimization.optimize(stmt) for stmt in optimized]
            # Filtrer les None (code éliminé)
            optimized = [stmt for stmt in optimized if stmt is not None]
            
        return optimized
        
    def compile_to_bytecode(self, statements: List[Statement]) -> types.CodeType:
        """Compile en bytecode Python pour une exécution plus rapide."""
        python_code = self.compile(statements)
        
        # Compiler en bytecode Python
        try:
            compiled = compile(python_code, '<ai\'lang>', 'exec')
            return compiled
        except SyntaxError as e:
            raise CompilerError(f"Generated Python code has syntax error: {str(e)}")
            
    def analyze_performance(self, statements: List[Statement]) -> Dict[str, Any]:
        """Analyse les performances potentielles du code."""
        analysis = {
            'complexity': 'O(1)',
            'loops': 0,
            'function_calls': 0,
            'memory_usage': 'low',
            'optimizations_applied': len(self.optimizations)
        }
        
        def analyze_node(node: ASTNode) -> None:
            if isinstance(node, (WhileStmt, ForStmt)):
                analysis['loops'] += 1
                analysis['complexity'] = 'O(n)'
            elif isinstance(node, CallExpr):
                analysis['function_calls'] += 1
            elif isinstance(node, TensorExpr):
                analysis['memory_usage'] = 'high'
                
        for stmt in statements:
            analyze_node(stmt)
            
        return analysis


def compile_code(source_code: str, optimize: bool = True) -> str:
    """Fonction utilitaire pour compiler du code ai'lang."""
    from ..lexer import Lexer
    from ..parser import Parser
    
    try:
        # Lexer
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parser
        parser = Parser(tokens)
        statements = parser.parse()
        
        # Compiler
        compiler = Compiler()
        python_code = compiler.compile(statements, optimize)
        
        return python_code
        
    except Exception as e:
        raise CompilerError(f"Compilation failed: {str(e)}")