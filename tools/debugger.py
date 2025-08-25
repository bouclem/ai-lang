# Débogueur intégré pour ai'lang
# Fournit des fonctionnalités de débogage avancées pour le langage ai'lang

import sys
import traceback
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Types et structures de données
# ============================================================================

class BreakpointType(Enum):
    """Types de points d'arrêt."""
    LINE = "line"
    FUNCTION = "function"
    CONDITIONAL = "conditional"
    EXCEPTION = "exception"

class DebuggerState(Enum):
    """États du débogueur."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    FINISHED = "finished"

@dataclass
class Breakpoint:
    """Représente un point d'arrêt."""
    id: int
    file_path: str
    line_number: int
    breakpoint_type: BreakpointType
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    ignore_count: int = 0

@dataclass
class StackFrame:
    """Représente une frame de la pile d'exécution."""
    function_name: str
    file_path: str
    line_number: int
    local_variables: Dict[str, Any]
    arguments: Dict[str, Any]

@dataclass
class DebugEvent:
    """Événement de débogage."""
    event_type: str
    timestamp: float
    file_path: str
    line_number: int
    function_name: str
    variables: Dict[str, Any]
    message: Optional[str] = None

# ============================================================================
# Débogueur principal
# ============================================================================

class AILangDebugger:
    """
    Débogueur intégré pour ai'lang avec fonctionnalités avancées.
    """
    
    def __init__(self):
        self.state = DebuggerState.STOPPED
        self.breakpoints: Dict[int, Breakpoint] = {}
        self.next_breakpoint_id = 1
        self.call_stack: List[StackFrame] = []
        self.current_frame = 0
        self.step_mode = False
        self.step_into = False
        self.step_over = False
        self.step_out = False
        self.debug_events: List[DebugEvent] = []
        self.variable_watchers: Dict[str, Any] = {}
        self.exception_breakpoints = True
        self.output_handlers: List[Callable] = []
        self.current_file = ""
        self.current_line = 0
        self.execution_context = {}
    
    def add_breakpoint(self, file_path: str, line_number: int, 
                      breakpoint_type: BreakpointType = BreakpointType.LINE,
                      condition: Optional[str] = None) -> int:
        """
        Ajoute un point d'arrêt.
        
        Args:
            file_path: Chemin du fichier
            line_number: Numéro de ligne
            breakpoint_type: Type de point d'arrêt
            condition: Condition pour les points d'arrêt conditionnels
        
        Returns:
            ID du point d'arrêt créé
        """
        breakpoint_id = self.next_breakpoint_id
        self.next_breakpoint_id += 1
        
        breakpoint = Breakpoint(
            id=breakpoint_id,
            file_path=file_path,
            line_number=line_number,
            breakpoint_type=breakpoint_type,
            condition=condition
        )
        
        self.breakpoints[breakpoint_id] = breakpoint
        self._log_event("breakpoint_added", f"Breakpoint {breakpoint_id} added at {file_path}:{line_number}")
        
        return breakpoint_id
    
    def remove_breakpoint(self, breakpoint_id: int) -> bool:
        """
        Supprime un point d'arrêt.
        
        Args:
            breakpoint_id: ID du point d'arrêt
        
        Returns:
            True si supprimé avec succès
        """
        if breakpoint_id in self.breakpoints:
            del self.breakpoints[breakpoint_id]
            self._log_event("breakpoint_removed", f"Breakpoint {breakpoint_id} removed")
            return True
        return False
    
    def toggle_breakpoint(self, breakpoint_id: int) -> bool:
        """
        Active/désactive un point d'arrêt.
        
        Args:
            breakpoint_id: ID du point d'arrêt
        
        Returns:
            Nouvel état du point d'arrêt
        """
        if breakpoint_id in self.breakpoints:
            self.breakpoints[breakpoint_id].enabled = not self.breakpoints[breakpoint_id].enabled
            return self.breakpoints[breakpoint_id].enabled
        return False
    
    def list_breakpoints(self) -> List[Breakpoint]:
        """
        Liste tous les points d'arrêt.
        
        Returns:
            Liste des points d'arrêt
        """
        return list(self.breakpoints.values())
    
    def should_break(self, file_path: str, line_number: int, 
                    local_vars: Dict[str, Any] = None) -> bool:
        """
        Détermine si l'exécution doit s'arrêter à cette ligne.
        
        Args:
            file_path: Chemin du fichier actuel
            line_number: Ligne actuelle
            local_vars: Variables locales
        
        Returns:
            True si l'exécution doit s'arrêter
        """
        # Vérification des points d'arrêt de ligne
        for bp in self.breakpoints.values():
            if (bp.enabled and 
                bp.file_path == file_path and 
                bp.line_number == line_number and
                bp.breakpoint_type == BreakpointType.LINE):
                
                # Vérification de la condition si présente
                if bp.condition and local_vars:
                    try:
                        if not self._evaluate_condition(bp.condition, local_vars):
                            continue
                    except Exception:
                        continue
                
                # Vérification du compteur d'ignore
                if bp.ignore_count > 0:
                    bp.ignore_count -= 1
                    continue
                
                bp.hit_count += 1
                return True
        
        # Mode pas à pas
        if self.step_mode:
            return True
        
        return False
    
    def step_into(self):
        """
        Active le mode "step into" (entrer dans les fonctions).
        """
        self.step_mode = True
        self.step_into = True
        self.step_over = False
        self.step_out = False
        self.state = DebuggerState.STEPPING
        self._log_event("step_into", "Step into mode activated")
    
    def step_over(self):
        """
        Active le mode "step over" (passer par-dessus les fonctions).
        """
        self.step_mode = True
        self.step_into = False
        self.step_over = True
        self.step_out = False
        self.state = DebuggerState.STEPPING
        self._log_event("step_over", "Step over mode activated")
    
    def step_out(self):
        """
        Active le mode "step out" (sortir de la fonction actuelle).
        """
        self.step_mode = True
        self.step_into = False
        self.step_over = False
        self.step_out = True
        self.state = DebuggerState.STEPPING
        self._log_event("step_out", "Step out mode activated")
    
    def continue_execution(self):
        """
        Continue l'exécution normale.
        """
        self.step_mode = False
        self.step_into = False
        self.step_over = False
        self.step_out = False
        self.state = DebuggerState.RUNNING
        self._log_event("continue", "Execution continued")
    
    def pause_execution(self):
        """
        Met en pause l'exécution.
        """
        self.state = DebuggerState.PAUSED
        self._log_event("pause", "Execution paused")
    
    def stop_debugging(self):
        """
        Arrête le débogage.
        """
        self.state = DebuggerState.STOPPED
        self.call_stack.clear()
        self.debug_events.clear()
        self._log_event("stop", "Debugging stopped")
    
    def enter_function(self, function_name: str, file_path: str, 
                      line_number: int, arguments: Dict[str, Any],
                      local_vars: Dict[str, Any] = None):
        """
        Enregistre l'entrée dans une fonction.
        
        Args:
            function_name: Nom de la fonction
            file_path: Chemin du fichier
            line_number: Ligne de la fonction
            arguments: Arguments de la fonction
            local_vars: Variables locales
        """
        frame = StackFrame(
            function_name=function_name,
            file_path=file_path,
            line_number=line_number,
            local_variables=local_vars or {},
            arguments=arguments
        )
        
        self.call_stack.append(frame)
        self.current_frame = len(self.call_stack) - 1
        
        self._log_event("function_enter", f"Entered function {function_name}")
    
    def exit_function(self, function_name: str, return_value: Any = None):
        """
        Enregistre la sortie d'une fonction.
        
        Args:
            function_name: Nom de la fonction
            return_value: Valeur de retour
        """
        if self.call_stack:
            frame = self.call_stack.pop()
            self.current_frame = len(self.call_stack) - 1
            
            self._log_event("function_exit", 
                          f"Exited function {function_name} with return value: {return_value}")
    
    def update_variables(self, variables: Dict[str, Any]):
        """
        Met à jour les variables du contexte actuel.
        
        Args:
            variables: Dictionnaire des variables
        """
        if self.call_stack:
            self.call_stack[self.current_frame].local_variables.update(variables)
        
        # Vérification des watchers
        for var_name, old_value in self.variable_watchers.items():
            if var_name in variables and variables[var_name] != old_value:
                self._log_event("variable_changed", 
                              f"Variable {var_name} changed from {old_value} to {variables[var_name]}")
                self.variable_watchers[var_name] = variables[var_name]
    
    def add_variable_watcher(self, variable_name: str, initial_value: Any = None):
        """
        Ajoute un observateur de variable.
        
        Args:
            variable_name: Nom de la variable à observer
            initial_value: Valeur initiale
        """
        self.variable_watchers[variable_name] = initial_value
        self._log_event("watcher_added", f"Added watcher for variable {variable_name}")
    
    def remove_variable_watcher(self, variable_name: str):
        """
        Supprime un observateur de variable.
        
        Args:
            variable_name: Nom de la variable
        """
        if variable_name in self.variable_watchers:
            del self.variable_watchers[variable_name]
            self._log_event("watcher_removed", f"Removed watcher for variable {variable_name}")
    
    def get_call_stack(self) -> List[StackFrame]:
        """
        Retourne la pile d'appels actuelle.
        
        Returns:
            Liste des frames de la pile
        """
        return self.call_stack.copy()
    
    def get_current_variables(self) -> Dict[str, Any]:
        """
        Retourne les variables du contexte actuel.
        
        Returns:
            Dictionnaire des variables
        """
        if self.call_stack and self.current_frame < len(self.call_stack):
            return self.call_stack[self.current_frame].local_variables.copy()
        return {}
    
    def evaluate_expression(self, expression: str, context: Dict[str, Any] = None) -> Any:
        """
        Évalue une expression dans le contexte actuel.
        
        Args:
            expression: Expression à évaluer
            context: Contexte d'évaluation
        
        Returns:
            Résultat de l'évaluation
        """
        if context is None:
            context = self.get_current_variables()
        
        try:
            # Évaluation sécurisée (simplifiée)
            # Dans une implémentation réelle, il faudrait un évaluateur plus sûr
            result = eval(expression, {"__builtins__": {}}, context)
            self._log_event("expression_evaluated", f"Evaluated '{expression}' = {result}")
            return result
        except Exception as e:
            self._log_event("expression_error", f"Error evaluating '{expression}': {str(e)}")
            raise
    
    def handle_exception(self, exception: Exception, file_path: str, line_number: int):
        """
        Gère une exception pendant le débogage.
        
        Args:
            exception: Exception levée
            file_path: Fichier où l'exception s'est produite
            line_number: Ligne où l'exception s'est produite
        """
        if self.exception_breakpoints:
            self.state = DebuggerState.PAUSED
            self._log_event("exception", f"Exception {type(exception).__name__}: {str(exception)}")
            
            # Ajouter l'exception au contexte
            self.execution_context["last_exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "file": file_path,
                "line": line_number,
                "traceback": traceback.format_exc()
            }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Retourne les informations de débogage actuelles.
        
        Returns:
            Dictionnaire avec les informations de débogage
        """
        return {
            "state": self.state.value,
            "current_file": self.current_file,
            "current_line": self.current_line,
            "breakpoints": [{
                "id": bp.id,
                "file": bp.file_path,
                "line": bp.line_number,
                "type": bp.breakpoint_type.value,
                "enabled": bp.enabled,
                "hit_count": bp.hit_count,
                "condition": bp.condition
            } for bp in self.breakpoints.values()],
            "call_stack": [{
                "function": frame.function_name,
                "file": frame.file_path,
                "line": frame.line_number,
                "variables": frame.local_variables
            } for frame in self.call_stack],
            "watchers": self.variable_watchers,
            "step_mode": self.step_mode,
            "exception_breakpoints": self.exception_breakpoints
        }
    
    def export_debug_session(self, file_path: str):
        """
        Exporte la session de débogage vers un fichier.
        
        Args:
            file_path: Chemin du fichier d'export
        """
        session_data = {
            "debug_info": self.get_debug_info(),
            "events": [{
                "type": event.event_type,
                "timestamp": event.timestamp,
                "file": event.file_path,
                "line": event.line_number,
                "function": event.function_name,
                "variables": event.variables,
                "message": event.message
            } for event in self.debug_events],
            "execution_context": self.execution_context
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            self._log_event("session_exported", f"Debug session exported to {file_path}")
        except Exception as e:
            self._log_event("export_error", f"Error exporting session: {str(e)}")
    
    def add_output_handler(self, handler: Callable[[str], None]):
        """
        Ajoute un gestionnaire de sortie pour les messages de débogage.
        
        Args:
            handler: Fonction de gestion des messages
        """
        self.output_handlers.append(handler)
    
    def _log_event(self, event_type: str, message: str):
        """
        Enregistre un événement de débogage.
        
        Args:
            event_type: Type d'événement
            message: Message de l'événement
        """
        event = DebugEvent(
            event_type=event_type,
            timestamp=time.time(),
            file_path=self.current_file,
            line_number=self.current_line,
            function_name=self.call_stack[-1].function_name if self.call_stack else "<global>",
            variables=self.get_current_variables(),
            message=message
        )
        
        self.debug_events.append(event)
        
        # Notifier les gestionnaires de sortie
        for handler in self.output_handlers:
            try:
                handler(f"[{event_type.upper()}] {message}")
            except Exception:
                pass  # Ignorer les erreurs des handlers
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Évalue une condition de point d'arrêt.
        
        Args:
            condition: Condition à évaluer
            context: Contexte d'évaluation
        
        Returns:
            Résultat de l'évaluation
        """
        try:
            return bool(eval(condition, {"__builtins__": {}}, context))
        except Exception:
            return False

# ============================================================================
# Interface de débogage interactive
# ============================================================================

class DebuggerInterface:
    """
    Interface interactive pour le débogueur.
    """
    
    def __init__(self, debugger: AILangDebugger):
        self.debugger = debugger
        self.commands = {
            'help': self._help,
            'h': self._help,
            'break': self._add_breakpoint,
            'b': self._add_breakpoint,
            'delete': self._delete_breakpoint,
            'd': self._delete_breakpoint,
            'list': self._list_breakpoints,
            'l': self._list_breakpoints,
            'continue': self._continue,
            'c': self._continue,
            'step': self._step_into,
            's': self._step_into,
            'next': self._step_over,
            'n': self._step_over,
            'finish': self._step_out,
            'f': self._step_out,
            'print': self._print_variable,
            'p': self._print_variable,
            'eval': self._evaluate,
            'e': self._evaluate,
            'stack': self._show_stack,
            'vars': self._show_variables,
            'watch': self._add_watcher,
            'w': self._add_watcher,
            'unwatch': self._remove_watcher,
            'info': self._show_info,
            'export': self._export_session,
            'quit': self._quit,
            'q': self._quit
        }
    
    def start_interactive_session(self):
        """
        Démarre une session interactive de débogage.
        """
        print("ai'lang Debugger - Interactive Session")
        print("Type 'help' for available commands")
        print("-" * 40)
        
        while self.debugger.state != DebuggerState.STOPPED:
            try:
                command_line = input("(ailang-db) ").strip()
                if not command_line:
                    continue
                
                parts = command_line.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if command in self.commands:
                    self.commands[command](args)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit the debugger.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _help(self, args):
        """Affiche l'aide."""
        help_text = """
Available commands:
  help, h                 - Show this help
  break <file:line>, b    - Set breakpoint
  delete <id>, d          - Delete breakpoint
  list, l                 - List breakpoints
  continue, c             - Continue execution
  step, s                 - Step into
  next, n                 - Step over
  finish, f               - Step out
  print <var>, p          - Print variable
  eval <expr>, e          - Evaluate expression
  stack                   - Show call stack
  vars                    - Show variables
  watch <var>, w          - Watch variable
  unwatch <var>           - Remove variable watcher
  info                    - Show debug info
  export <file>           - Export debug session
  quit, q                 - Quit debugger
"""
        print(help_text)
    
    def _add_breakpoint(self, args):
        """Ajoute un point d'arrêt."""
        if not args:
            print("Usage: break <file:line>")
            return
        
        try:
            if ':' in args[0]:
                file_path, line_str = args[0].rsplit(':', 1)
                line_number = int(line_str)
            else:
                print("Usage: break <file:line>")
                return
            
            bp_id = self.debugger.add_breakpoint(file_path, line_number)
            print(f"Breakpoint {bp_id} set at {file_path}:{line_number}")
        
        except ValueError:
            print("Invalid line number")
        except Exception as e:
            print(f"Error setting breakpoint: {str(e)}")
    
    def _delete_breakpoint(self, args):
        """Supprime un point d'arrêt."""
        if not args:
            print("Usage: delete <breakpoint_id>")
            return
        
        try:
            bp_id = int(args[0])
            if self.debugger.remove_breakpoint(bp_id):
                print(f"Breakpoint {bp_id} deleted")
            else:
                print(f"Breakpoint {bp_id} not found")
        except ValueError:
            print("Invalid breakpoint ID")
    
    def _list_breakpoints(self, args):
        """Liste les points d'arrêt."""
        breakpoints = self.debugger.list_breakpoints()
        if not breakpoints:
            print("No breakpoints set")
            return
        
        print("Breakpoints:")
        for bp in breakpoints:
            status = "enabled" if bp.enabled else "disabled"
            print(f"  {bp.id}: {bp.file_path}:{bp.line_number} ({status}) hits: {bp.hit_count}")
            if bp.condition:
                print(f"      condition: {bp.condition}")
    
    def _continue(self, args):
        """Continue l'exécution."""
        self.debugger.continue_execution()
        print("Continuing execution...")
    
    def _step_into(self, args):
        """Step into."""
        self.debugger.step_into()
        print("Stepping into...")
    
    def _step_over(self, args):
        """Step over."""
        self.debugger.step_over()
        print("Stepping over...")
    
    def _step_out(self, args):
        """Step out."""
        self.debugger.step_out()
        print("Stepping out...")
    
    def _print_variable(self, args):
        """Affiche une variable."""
        if not args:
            print("Usage: print <variable_name>")
            return
        
        var_name = args[0]
        variables = self.debugger.get_current_variables()
        
        if var_name in variables:
            print(f"{var_name} = {variables[var_name]}")
        else:
            print(f"Variable '{var_name}' not found")
    
    def _evaluate(self, args):
        """Évalue une expression."""
        if not args:
            print("Usage: eval <expression>")
            return
        
        expression = ' '.join(args)
        try:
            result = self.debugger.evaluate_expression(expression)
            print(f"{expression} = {result}")
        except Exception as e:
            print(f"Error evaluating expression: {str(e)}")
    
    def _show_stack(self, args):
        """Affiche la pile d'appels."""
        stack = self.debugger.get_call_stack()
        if not stack:
            print("No call stack available")
            return
        
        print("Call stack:")
        for i, frame in enumerate(reversed(stack)):
            marker = "->" if i == 0 else "  "
            print(f"{marker} {frame.function_name} at {frame.file_path}:{frame.line_number}")
    
    def _show_variables(self, args):
        """Affiche les variables."""
        variables = self.debugger.get_current_variables()
        if not variables:
            print("No variables in current scope")
            return
        
        print("Variables:")
        for name, value in variables.items():
            print(f"  {name} = {value}")
    
    def _add_watcher(self, args):
        """Ajoute un observateur de variable."""
        if not args:
            print("Usage: watch <variable_name>")
            return
        
        var_name = args[0]
        variables = self.debugger.get_current_variables()
        initial_value = variables.get(var_name)
        
        self.debugger.add_variable_watcher(var_name, initial_value)
        print(f"Added watcher for variable '{var_name}'")
    
    def _remove_watcher(self, args):
        """Supprime un observateur de variable."""
        if not args:
            print("Usage: unwatch <variable_name>")
            return
        
        var_name = args[0]
        self.debugger.remove_variable_watcher(var_name)
        print(f"Removed watcher for variable '{var_name}'")
    
    def _show_info(self, args):
        """Affiche les informations de débogage."""
        info = self.debugger.get_debug_info()
        print(f"Debugger state: {info['state']}")
        print(f"Current file: {info['current_file']}")
        print(f"Current line: {info['current_line']}")
        print(f"Step mode: {info['step_mode']}")
        print(f"Exception breakpoints: {info['exception_breakpoints']}")
        print(f"Active breakpoints: {len([bp for bp in info['breakpoints'] if bp['enabled']])}")
        print(f"Call stack depth: {len(info['call_stack'])}")
        print(f"Watched variables: {len(info['watchers'])}")
    
    def _export_session(self, args):
        """Exporte la session de débogage."""
        if not args:
            print("Usage: export <filename>")
            return
        
        filename = args[0]
        try:
            self.debugger.export_debug_session(filename)
            print(f"Debug session exported to {filename}")
        except Exception as e:
            print(f"Error exporting session: {str(e)}")
    
    def _quit(self, args):
        """Quitte le débogueur."""
        self.debugger.stop_debugging()
        print("Debugger stopped")

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def create_debugger() -> AILangDebugger:
    """
    Crée une nouvelle instance du débogueur.
    
    Returns:
        Instance du débogueur
    """
    return AILangDebugger()

def start_debug_session(debugger: AILangDebugger = None) -> DebuggerInterface:
    """
    Démarre une session de débogage interactive.
    
    Args:
        debugger: Instance du débogueur (optionnel)
    
    Returns:
        Interface du débogueur
    """
    if debugger is None:
        debugger = create_debugger()
    
    interface = DebuggerInterface(debugger)
    return interface

# Instance globale du débogueur
_global_debugger = None

def get_global_debugger() -> AILangDebugger:
    """
    Retourne l'instance globale du débogueur.
    
    Returns:
        Instance globale du débogueur
    """
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = create_debugger()
    return _global_debugger

def debug_trace(func):
    """
    Décorateur pour tracer l'exécution d'une fonction.
    
    Args:
        func: Fonction à tracer
    
    Returns:
        Fonction décorée
    """
    def wrapper(*args, **kwargs):
        debugger = get_global_debugger()
        
        # Entrée dans la fonction
        debugger.enter_function(
            func.__name__,
            func.__code__.co_filename if hasattr(func, '__code__') else "<unknown>",
            func.__code__.co_firstlineno if hasattr(func, '__code__') else 0,
            {f"arg_{i}": arg for i, arg in enumerate(args)},
            kwargs
        )
        
        try:
            result = func(*args, **kwargs)
            debugger.exit_function(func.__name__, result)
            return result
        except Exception as e:
            debugger.handle_exception(e, 
                                    func.__code__.co_filename if hasattr(func, '__code__') else "<unknown>",
                                    func.__code__.co_firstlineno if hasattr(func, '__code__') else 0)
            raise
    
    return wrapper