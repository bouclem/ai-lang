"""Types de valeurs et objets pour l'interpréteur ai'lang."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np


class AILangValue(ABC):
    """Classe de base pour toutes les valeurs ai'lang."""
    
    @abstractmethod
    def to_python(self) -> Any:
        """Convertit la valeur en objet Python natif."""
        pass
        
    @abstractmethod
    def to_string(self) -> str:
        """Convertit la valeur en chaîne de caractères."""
        pass
        
    @abstractmethod
    def is_truthy(self) -> bool:
        """Détermine si la valeur est considérée comme vraie."""
        pass
        
    def __str__(self) -> str:
        return self.to_string()
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_string()})"


@dataclass
class AILangNumber(AILangValue):
    """Valeur numérique (int ou float)."""
    value: Union[int, float]
    
    def to_python(self) -> Union[int, float]:
        return self.value
        
    def to_string(self) -> str:
        return str(self.value)
        
    def is_truthy(self) -> bool:
        return self.value != 0
        
    def __add__(self, other: 'AILangNumber') -> 'AILangNumber':
        return AILangNumber(self.value + other.value)
        
    def __sub__(self, other: 'AILangNumber') -> 'AILangNumber':
        return AILangNumber(self.value - other.value)
        
    def __mul__(self, other: 'AILangNumber') -> 'AILangNumber':
        return AILangNumber(self.value * other.value)
        
    def __truediv__(self, other: 'AILangNumber') -> 'AILangNumber':
        if other.value == 0:
            raise ZeroDivisionError("Division by zero")
        return AILangNumber(self.value / other.value)
        
    def __floordiv__(self, other: 'AILangNumber') -> 'AILangNumber':
        if other.value == 0:
            raise ZeroDivisionError("Division by zero")
        return AILangNumber(self.value // other.value)
        
    def __mod__(self, other: 'AILangNumber') -> 'AILangNumber':
        return AILangNumber(self.value % other.value)
        
    def __pow__(self, other: 'AILangNumber') -> 'AILangNumber':
        return AILangNumber(self.value ** other.value)
        
    def __neg__(self) -> 'AILangNumber':
        return AILangNumber(-self.value)
        
    def __eq__(self, other) -> bool:
        if isinstance(other, AILangNumber):
            return self.value == other.value
        return False
        
    def __lt__(self, other: 'AILangNumber') -> bool:
        return self.value < other.value
        
    def __le__(self, other: 'AILangNumber') -> bool:
        return self.value <= other.value
        
    def __gt__(self, other: 'AILangNumber') -> bool:
        return self.value > other.value
        
    def __ge__(self, other: 'AILangNumber') -> bool:
        return self.value >= other.value


@dataclass
class AILangString(AILangValue):
    """Valeur chaîne de caractères."""
    value: str
    
    def to_python(self) -> str:
        return self.value
        
    def to_string(self) -> str:
        return self.value
        
    def is_truthy(self) -> bool:
        return len(self.value) > 0
        
    def __add__(self, other: 'AILangString') -> 'AILangString':
        return AILangString(self.value + other.value)
        
    def __mul__(self, other: AILangNumber) -> 'AILangString':
        return AILangString(self.value * int(other.value))
        
    def __eq__(self, other) -> bool:
        if isinstance(other, AILangString):
            return self.value == other.value
        return False
        
    def __len__(self) -> int:
        return len(self.value)
        
    def __getitem__(self, index: int) -> 'AILangString':
        return AILangString(self.value[index])


@dataclass
class AILangBoolean(AILangValue):
    """Valeur booléenne."""
    value: bool
    
    def to_python(self) -> bool:
        return self.value
        
    def to_string(self) -> str:
        return "True" if self.value else "False"
        
    def is_truthy(self) -> bool:
        return self.value
        
    def __eq__(self, other) -> bool:
        if isinstance(other, AILangBoolean):
            return self.value == other.value
        return False


class AILangNone(AILangValue):
    """Valeur None."""
    
    def to_python(self) -> None:
        return None
        
    def to_string(self) -> str:
        return "None"
        
    def is_truthy(self) -> bool:
        return False
        
    def __eq__(self, other) -> bool:
        return isinstance(other, AILangNone)


@dataclass
class AILangList(AILangValue):
    """Valeur liste."""
    elements: List[AILangValue]
    
    def to_python(self) -> List[Any]:
        return [elem.to_python() for elem in self.elements]
        
    def to_string(self) -> str:
        elements_str = ", ".join(elem.to_string() for elem in self.elements)
        return f"[{elements_str}]"
        
    def is_truthy(self) -> bool:
        return len(self.elements) > 0
        
    def __len__(self) -> int:
        return len(self.elements)
        
    def __getitem__(self, index: int) -> AILangValue:
        return self.elements[index]
        
    def __setitem__(self, index: int, value: AILangValue) -> None:
        self.elements[index] = value
        
    def append(self, value: AILangValue) -> None:
        self.elements.append(value)
        
    def extend(self, other: 'AILangList') -> None:
        self.elements.extend(other.elements)
        
    def __add__(self, other: 'AILangList') -> 'AILangList':
        return AILangList(self.elements + other.elements)


@dataclass
class AILangDict(AILangValue):
    """Valeur dictionnaire."""
    pairs: Dict[str, AILangValue]
    
    def to_python(self) -> Dict[str, Any]:
        return {key: value.to_python() for key, value in self.pairs.items()}
        
    def to_string(self) -> str:
        pairs_str = ", ".join(f"'{key}': {value.to_string()}" for key, value in self.pairs.items())
        return f"{{{pairs_str}}}"
        
    def is_truthy(self) -> bool:
        return len(self.pairs) > 0
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, key: str) -> AILangValue:
        if key not in self.pairs:
            raise KeyError(f"Key '{key}' not found")
        return self.pairs[key]
        
    def __setitem__(self, key: str, value: AILangValue) -> None:
        self.pairs[key] = value
        
    def __contains__(self, key: str) -> bool:
        return key in self.pairs
        
    def keys(self) -> List[str]:
        return list(self.pairs.keys())
        
    def values(self) -> List[AILangValue]:
        return list(self.pairs.values())
        
    def items(self) -> List[tuple[str, AILangValue]]:
        return list(self.pairs.items())


@dataclass
class AILangTensor(AILangValue):
    """Valeur tenseur pour les calculs IA."""
    data: np.ndarray
    dtype: str
    
    def __init__(self, shape: List[int], dtype: str = 'float32', data: Optional[np.ndarray] = None):
        self.dtype = dtype
        
        if data is not None:
            self.data = data.astype(dtype)
        else:
            self.data = np.zeros(shape, dtype=dtype)
            
    def to_python(self) -> np.ndarray:
        return self.data
        
    def to_string(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype})"
        
    def is_truthy(self) -> bool:
        return self.data.size > 0
        
    @property
    def shape(self) -> tuple:
        return self.data.shape
        
    @property
    def size(self) -> int:
        return self.data.size
        
    def __add__(self, other: 'AILangTensor') -> 'AILangTensor':
        result = AILangTensor(list(self.data.shape), self.dtype)
        result.data = self.data + other.data
        return result
        
    def __sub__(self, other: 'AILangTensor') -> 'AILangTensor':
        result = AILangTensor(list(self.data.shape), self.dtype)
        result.data = self.data - other.data
        return result
        
    def __mul__(self, other: 'AILangTensor') -> 'AILangTensor':
        result = AILangTensor(list(self.data.shape), self.dtype)
        result.data = self.data * other.data
        return result
        
    def __truediv__(self, other: 'AILangTensor') -> 'AILangTensor':
        result = AILangTensor(list(self.data.shape), self.dtype)
        result.data = self.data / other.data
        return result
        
    def __matmul__(self, other: 'AILangTensor') -> 'AILangTensor':
        """Multiplication matricielle."""
        result_data = np.matmul(self.data, other.data)
        result = AILangTensor(list(result_data.shape), self.dtype)
        result.data = result_data
        return result
        
    def reshape(self, new_shape: List[int]) -> 'AILangTensor':
        result = AILangTensor(new_shape, self.dtype)
        result.data = self.data.reshape(new_shape)
        return result
        
    def transpose(self) -> 'AILangTensor':
        result = AILangTensor(list(self.data.T.shape), self.dtype)
        result.data = self.data.T
        return result


@dataclass
class AILangFunction(AILangValue):
    """Valeur fonction."""
    name: str
    parameters: List[str]
    body: Any  # AST nodes
    closure: Any  # Environment
    is_native: bool = False
    native_func: Optional[Callable] = None
    
    def to_python(self) -> Callable:
        if self.is_native and self.native_func:
            return self.native_func
        # Pour les fonctions ai'lang, on ne peut pas les convertir directement
        return lambda *args: f"<ai'lang function {self.name}>"
        
    def to_string(self) -> str:
        return f"<function {self.name}>"
        
    def is_truthy(self) -> bool:
        return True
        
    def call(self, interpreter, arguments: List[AILangValue]) -> AILangValue:
        """Appelle la fonction avec les arguments donnés."""
        if self.is_native and self.native_func:
            # Convertir les arguments en valeurs Python
            python_args = [arg.to_python() for arg in arguments]
            result = self.native_func(*python_args)
            return python_to_ailang(result)
        else:
            # Exécuter le corps de la fonction dans un nouvel environnement
            from .environment import FunctionEnvironment
            func_env = FunctionEnvironment(self.closure, self.parameters, arguments)
            return interpreter.execute_function_body(self.body, func_env)


@dataclass
class AILangClass(AILangValue):
    """Valeur classe."""
    name: str
    methods: Dict[str, AILangFunction]
    attributes: Dict[str, AILangValue]
    superclass: Optional['AILangClass'] = None
    
    def to_python(self) -> type:
        # Créer une classe Python dynamique
        def __init__(self, *args, **kwargs):
            pass
            
        attrs = {'__init__': __init__}
        for name, method in self.methods.items():
            attrs[name] = method.to_python()
            
        return type(self.name, (), attrs)
        
    def to_string(self) -> str:
        return f"<class {self.name}>"
        
    def is_truthy(self) -> bool:
        return True
        
    def get_method(self, name: str) -> Optional[AILangFunction]:
        """Récupère une méthode de la classe."""
        if name in self.methods:
            return self.methods[name]
        
        if self.superclass:
            return self.superclass.get_method(name)
            
        return None
        
    def instantiate(self, interpreter, arguments: List[AILangValue]) -> 'AILangInstance':
        """Crée une instance de la classe."""
        instance = AILangInstance(self)
        
        # Appeler le constructeur s'il existe
        init_method = self.get_method('__init__')
        if init_method:
            init_method.call(interpreter, [instance] + arguments)
            
        return instance


@dataclass
class AILangInstance(AILangValue):
    """Instance d'une classe."""
    class_def: AILangClass
    fields: Dict[str, AILangValue] = None
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = {}
            
    def to_python(self) -> object:
        # Créer un objet Python avec les champs
        class DynamicObject:
            pass
            
        obj = DynamicObject()
        for name, value in self.fields.items():
            setattr(obj, name, value.to_python())
            
        return obj
        
    def to_string(self) -> str:
        return f"<{self.class_def.name} instance>"
        
    def is_truthy(self) -> bool:
        return True
        
    def get(self, name: str) -> AILangValue:
        """Récupère un attribut ou une méthode."""
        if name in self.fields:
            return self.fields[name]
            
        method = self.class_def.get_method(name)
        if method:
            # Lier la méthode à l'instance
            return AILangBoundMethod(self, method)
            
        raise AttributeError(f"'{self.class_def.name}' object has no attribute '{name}'")
        
    def set(self, name: str, value: AILangValue) -> None:
        """Définit un attribut."""
        self.fields[name] = value


@dataclass
class AILangBoundMethod(AILangValue):
    """Méthode liée à une instance."""
    instance: AILangInstance
    method: AILangFunction
    
    def to_python(self) -> Callable:
        return lambda *args: f"<bound method {self.method.name}>"
        
    def to_string(self) -> str:
        return f"<bound method {self.method.name}>"
        
    def is_truthy(self) -> bool:
        return True
        
    def call(self, interpreter, arguments: List[AILangValue]) -> AILangValue:
        """Appelle la méthode avec l'instance comme premier argument."""
        return self.method.call(interpreter, [self.instance] + arguments)


def python_to_ailang(value: Any) -> AILangValue:
    """Convertit une valeur Python en valeur ai'lang."""
    if value is None:
        return AILangNone()
    elif isinstance(value, bool):
        return AILangBoolean(value)
    elif isinstance(value, (int, float)):
        return AILangNumber(value)
    elif isinstance(value, str):
        return AILangString(value)
    elif isinstance(value, list):
        return AILangList([python_to_ailang(item) for item in value])
    elif isinstance(value, dict):
        return AILangDict({str(k): python_to_ailang(v) for k, v in value.items()})
    elif isinstance(value, np.ndarray):
        tensor = AILangTensor(list(value.shape))
        tensor.data = value
        return tensor
    elif callable(value):
        return AILangFunction(getattr(value, '__name__', 'anonymous'), [], None, None, True, value)
    else:
        # Pour les autres types, on les encapsule dans une chaîne
        return AILangString(str(value))


def ailang_to_python(value: AILangValue) -> Any:
    """Convertit une valeur ai'lang en valeur Python."""
    return value.to_python()