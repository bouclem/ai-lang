"""Environnement d'exécution pour ai'lang."""

from typing import Any, Dict, Optional
from collections import ChainMap


class EnvironmentError(Exception):
    """Exception levée lors d'erreurs d'environnement."""
    pass


class Environment:
    """Environnement d'exécution gérant les variables et les scopes."""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.variables: Dict[str, Any] = {}
        
    def define(self, name: str, value: Any) -> None:
        """Définit une variable dans l'environnement actuel."""
        self.variables[name] = value
        
    def get(self, name: str) -> Any:
        """Récupère la valeur d'une variable."""
        if name in self.variables:
            return self.variables[name]
        
        if self.parent is not None:
            return self.parent.get(name)
            
        raise EnvironmentError(f"Undefined variable '{name}'")
        
    def assign(self, name: str, value: Any) -> None:
        """Assigne une nouvelle valeur à une variable existante."""
        if name in self.variables:
            self.variables[name] = value
            return
            
        if self.parent is not None:
            try:
                self.parent.assign(name, value)
                return
            except EnvironmentError:
                pass
                
        # Si la variable n'existe pas, on la crée dans l'environnement actuel
        self.variables[name] = value
        
    def contains(self, name: str) -> bool:
        """Vérifie si une variable existe dans l'environnement."""
        if name in self.variables:
            return True
        
        if self.parent is not None:
            return self.parent.contains(name)
            
        return False
        
    def get_all_variables(self) -> Dict[str, Any]:
        """Retourne toutes les variables accessibles."""
        if self.parent is not None:
            result = self.parent.get_all_variables()
            result.update(self.variables)
            return result
        else:
            return self.variables.copy()
            
    def create_child(self) -> 'Environment':
        """Crée un environnement enfant."""
        return Environment(self)
        
    def __str__(self) -> str:
        return f"Environment({list(self.variables.keys())})"
        
    def __repr__(self) -> str:
        return self.__str__()


class GlobalEnvironment(Environment):
    """Environnement global avec les fonctions et types built-in."""
    
    def __init__(self):
        super().__init__()
        self._setup_builtins()
        
    def _setup_builtins(self) -> None:
        """Configure les fonctions et types built-in."""
        
        # Types de base
        self.define('int', int)
        self.define('float', float)
        self.define('str', str)
        self.define('bool', bool)
        self.define('list', list)
        self.define('dict', dict)
        self.define('tuple', tuple)
        self.define('set', set)
        
        # Fonctions built-in
        self.define('print', self._builtin_print)
        self.define('len', len)
        self.define('range', range)
        self.define('enumerate', enumerate)
        self.define('zip', zip)
        self.define('map', map)
        self.define('filter', filter)
        self.define('sum', sum)
        self.define('min', min)
        self.define('max', max)
        self.define('abs', abs)
        self.define('round', round)
        self.define('sorted', sorted)
        self.define('reversed', reversed)
        self.define('any', any)
        self.define('all', all)
        
        # Fonctions d'entrée/sortie
        self.define('input', input)
        self.define('open', open)
        
        # Fonctions de conversion
        self.define('ord', ord)
        self.define('chr', chr)
        self.define('hex', hex)
        self.define('oct', oct)
        self.define('bin', bin)
        
        # Fonctions mathématiques de base
        import math
        self.define('math', math)
        
        # Constantes
        self.define('True', True)
        self.define('False', False)
        self.define('None', None)
        
    def _builtin_print(self, *args, sep=' ', end='\n', file=None, flush=False):
        """Fonction print built-in avec support des paramètres."""
        import sys
        if file is None:
            file = sys.stdout
            
        output = sep.join(str(arg) for arg in args)
        file.write(output + end)
        
        if flush:
            file.flush()


class FunctionEnvironment(Environment):
    """Environnement spécialisé pour les fonctions."""
    
    def __init__(self, parent: Environment, parameters: list[str], arguments: list[Any]):
        super().__init__(parent)
        
        if len(parameters) != len(arguments):
            raise EnvironmentError(
                f"Function expects {len(parameters)} arguments, got {len(arguments)}"
            )
            
        # Lier les paramètres aux arguments
        for param, arg in zip(parameters, arguments):
            self.define(param, arg)


class ClassEnvironment(Environment):
    """Environnement spécialisé pour les classes."""
    
    def __init__(self, parent: Environment, class_name: str):
        super().__init__(parent)
        self.class_name = class_name
        
    def get_method(self, name: str) -> Any:
        """Récupère une méthode de la classe."""
        if name in self.variables:
            method = self.variables[name]
            if callable(method):
                return method
        
        if self.parent is not None:
            return self.parent.get(name)
            
        raise EnvironmentError(f"Method '{name}' not found in class '{self.class_name}'")


class ModuleEnvironment(Environment):
    """Environnement spécialisé pour les modules."""
    
    def __init__(self, module_name: str, parent: Optional[Environment] = None):
        super().__init__(parent)
        self.module_name = module_name
        
    def export(self, name: str, value: Any) -> None:
        """Exporte une variable du module."""
        self.define(name, value)
        
    def get_exports(self) -> Dict[str, Any]:
        """Retourne toutes les exportations du module."""
        return self.variables.copy()