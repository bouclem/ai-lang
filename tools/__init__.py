# Module d'outils de développement pour ai'lang
# Fournit des outils de débogage et de profilage intégrés

# Importations du débogueur
from .debugger import (
    AILangDebugger,
    DebuggerInterface,
    BreakpointType,
    DebuggerState,
    Breakpoint,
    StackFrame,
    DebugEvent,
    create_debugger,
    start_debug_session,
    get_global_debugger,
    debug_trace
)

# Importations du profiler
from .profiler import (
    AILangProfiler,
    FunctionProfile,
    MemorySnapshot,
    CPUSnapshot,
    PerformanceMetrics,
    create_profiler,
    profile_function,
    profile_block,
    start_profiling,
    stop_profiling,
    get_profiler,
    generate_report,
    benchmark
)

# Version du module
__version__ = "1.0.0"

# Métadonnées
__author__ = "AI'Lang Development Team"
__description__ = "Outils de développement intégrés pour ai'lang"

# Exports principaux
__all__ = [
    # Débogueur
    "AILangDebugger",
    "DebuggerInterface",
    "BreakpointType",
    "DebuggerState",
    "Breakpoint",
    "StackFrame",
    "DebugEvent",
    "create_debugger",
    "start_debug_session",
    "get_global_debugger",
    "debug_trace",
    
    # Profiler
    "AILangProfiler",
    "FunctionProfile",
    "MemorySnapshot",
    "CPUSnapshot",
    "PerformanceMetrics",
    "create_profiler",
    "profile_function",
    "profile_block",
    "start_profiling",
    "stop_profiling",
    "get_profiler",
    "generate_report",
    "benchmark",
    
    # Fonctions utilitaires
    "get_tools_info",
    "quick_debug",
    "quick_profile",
    "performance_test"
]

# ============================================================================
# Fonctions utilitaires globales
# ============================================================================

def get_tools_info() -> dict:
    """
    Retourne les informations sur les outils disponibles.
    
    Returns:
        Dictionnaire avec les informations des outils
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "tools": {
            "debugger": {
                "description": "Débogueur intégré avec points d'arrêt, inspection de variables et contrôle d'exécution",
                "features": [
                    "Points d'arrêt conditionnels",
                    "Inspection de la pile d'appels",
                    "Observation de variables",
                    "Mode pas à pas",
                    "Interface interactive",
                    "Export de sessions"
                ]
            },
            "profiler": {
                "description": "Profiler de performance avec analyse mémoire et CPU",
                "features": [
                    "Profilage de fonctions",
                    "Monitoring mémoire en temps réel",
                    "Monitoring CPU",
                    "Détection de points chauds",
                    "Suggestions d'optimisation",
                    "Rapports détaillés (text, JSON, HTML)",
                    "Benchmarking"
                ]
            }
        }
    }

def quick_debug(func=None, *, breakpoints=None, interactive=True):
    """
    Décorateur pour un débogage rapide d'une fonction.
    
    Args:
        func: Fonction à déboguer
        breakpoints: Liste de lignes pour les points d'arrêt
        interactive: Démarrer en mode interactif
    
    Returns:
        Fonction décorée ou décorateur
    """
    def decorator(f):
        @debug_trace
        def wrapper(*args, **kwargs):
            debugger = get_global_debugger()
            
            # Ajout des points d'arrêt si spécifiés
            if breakpoints:
                file_path = getattr(f, '__code__', {}).get('co_filename', '')
                for line in breakpoints:
                    debugger.add_breakpoint(file_path, line)
            
            # Démarrage de l'interface interactive si demandé
            if interactive:
                interface = start_debug_session(debugger)
                print(f"Debugging function: {f.__name__}")
                print("Type 'continue' to start execution")
                interface.start_interactive_session()
            
            return f(*args, **kwargs)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

def quick_profile(func=None, *, report_format="text", detailed=True):
    """
    Décorateur pour un profilage rapide d'une fonction.
    
    Args:
        func: Fonction à profiler
        report_format: Format du rapport (text, json, html)
        detailed: Profilage détaillé avec cProfile
    
    Returns:
        Fonction décorée ou décorateur
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            
            print(f"Starting profiling for function: {f.__name__}")
            profiler.start_profiling(detailed=detailed)
            
            try:
                with profiler.profile_block(f"function_{f.__name__}"):
                    result = f(*args, **kwargs)
                return result
            finally:
                metrics = profiler.stop_profiling()
                print(f"\nProfiling completed in {metrics.execution_time:.3f}s")
                
                if report_format:
                    report = profiler.generate_report(report_format)
                    if report_format == "html":
                        # Sauvegarde du rapport HTML
                        filename = f"profile_report_{f.__name__}.html"
                        with open(filename, 'w', encoding='utf-8') as file:
                            file.write(report)
                        print(f"HTML report saved to: {filename}")
                    else:
                        print("\n" + report)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

def performance_test(func, *args, iterations=1000, warmup=10, **kwargs):
    """
    Effectue un test de performance complet d'une fonction.
    
    Args:
        func: Fonction à tester
        *args: Arguments de la fonction
        iterations: Nombre d'itérations pour le test
        warmup: Nombre d'itérations de préchauffage
        **kwargs: Arguments nommés de la fonction
    
    Returns:
        Dictionnaire avec les résultats détaillés
    """
    print(f"Performance test for function: {func.__name__}")
    print(f"Warmup: {warmup} iterations")
    print(f"Test: {iterations} iterations")
    print("-" * 50)
    
    # Préchauffage
    print("Warming up...")
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Test avec profiler
    print("Starting profiled test...")
    profiler = create_profiler()
    profiler.start_profiling()
    
    with profiler.profile_block(f"performance_test_{func.__name__}"):
        benchmark_results = benchmark(func, *args, iterations=iterations, **kwargs)
    
    metrics = profiler.stop_profiling()
    
    # Compilation des résultats
    results = {
        "function_name": func.__name__,
        "iterations": iterations,
        "benchmark": benchmark_results,
        "profiling": {
            "total_execution_time": metrics.execution_time,
            "memory_peak_mb": metrics.memory_peak / 1024 / 1024,
            "memory_average_mb": metrics.memory_average / 1024 / 1024,
            "cpu_peak_percent": metrics.cpu_peak,
            "cpu_average_percent": metrics.cpu_average,
            "function_calls": metrics.function_calls
        },
        "performance_score": _calculate_performance_score(benchmark_results, metrics)
    }
    
    # Affichage des résultats
    print("\nPerformance Test Results:")
    print(f"Average time per call: {benchmark_results['average_time']:.6f}s")
    print(f"Min time: {benchmark_results['min_time']:.6f}s")
    print(f"Max time: {benchmark_results['max_time']:.6f}s")
    print(f"Standard deviation: {benchmark_results['std_dev']:.6f}s")
    print(f"Memory peak: {results['profiling']['memory_peak_mb']:.1f} MB")
    print(f"CPU average: {results['profiling']['cpu_average_percent']:.1f}%")
    print(f"Performance score: {results['performance_score']:.2f}/100")
    
    # Suggestions d'optimisation
    suggestions = profiler.suggest_optimizations()
    if suggestions:
        print("\nOptimization suggestions:")
        for suggestion in suggestions:
            print(f"• {suggestion}")
    
    return results

def _calculate_performance_score(benchmark_results, metrics):
    """
    Calcule un score de performance basé sur les métriques.
    
    Args:
        benchmark_results: Résultats du benchmark
        metrics: Métriques du profiler
    
    Returns:
        Score de performance (0-100)
    """
    # Score basé sur la vitesse (plus c'est rapide, mieux c'est)
    speed_score = max(0, 100 - (benchmark_results['average_time'] * 1000))  # Pénalité par ms
    
    # Score basé sur la consistance (moins de variation, mieux c'est)
    consistency_score = max(0, 100 - (benchmark_results['std_dev'] * 10000))  # Pénalité par variation
    
    # Score basé sur l'utilisation mémoire (moins c'est mieux)
    memory_score = max(0, 100 - (metrics.memory_peak / 1024 / 1024))  # Pénalité par MB
    
    # Score basé sur l'utilisation CPU (moins c'est mieux pour l'efficacité)
    cpu_score = max(0, 100 - metrics.cpu_average)
    
    # Score global pondéré
    total_score = (
        speed_score * 0.4 +
        consistency_score * 0.3 +
        memory_score * 0.2 +
        cpu_score * 0.1
    )
    
    return min(100, max(0, total_score))

# ============================================================================
# Configuration par défaut
# ============================================================================

# Configuration du débogueur global
_debugger_config = {
    "auto_start_interactive": False,
    "exception_breakpoints": True,
    "output_format": "text"
}

# Configuration du profiler global
_profiler_config = {
    "sampling_interval": 0.1,
    "track_memory": True,
    "track_cpu": True,
    "detailed_profiling": True
}

def configure_debugger(**kwargs):
    """
    Configure le débogueur global.
    
    Args:
        **kwargs: Options de configuration
    """
    _debugger_config.update(kwargs)
    debugger = get_global_debugger()
    
    if "exception_breakpoints" in kwargs:
        debugger.exception_breakpoints = kwargs["exception_breakpoints"]

def configure_profiler(**kwargs):
    """
    Configure le profiler global.
    
    Args:
        **kwargs: Options de configuration
    """
    _profiler_config.update(kwargs)
    # Note: La configuration sera appliquée au prochain démarrage du profiler

def get_debugger_config():
    """
    Retourne la configuration actuelle du débogueur.
    
    Returns:
        Dictionnaire de configuration
    """
    return _debugger_config.copy()

def get_profiler_config():
    """
    Retourne la configuration actuelle du profiler.
    
    Returns:
        Dictionnaire de configuration
    """
    return _profiler_config.copy()

# Message d'initialisation
print(f"AI'Lang Development Tools v{__version__} loaded")
print("Available tools: debugger, profiler")
print("Use get_tools_info() for more information")