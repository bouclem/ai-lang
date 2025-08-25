"""Module compilateur pour ai'lang."""

from .compiler import (
    Compiler,
    CompilerError,
    compile_code,
    OptimizationPass,
    ConstantFolding,
    DeadCodeElimination,
    LoopOptimization,
    InlineOptimization,
    PythonCodeGenerator
)

__all__ = [
    'Compiler',
    'CompilerError',
    'compile_code',
    'OptimizationPass',
    'ConstantFolding',
    'DeadCodeElimination',
    'LoopOptimization',
    'InlineOptimization',
    'PythonCodeGenerator'
]