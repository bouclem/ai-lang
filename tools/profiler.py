# Profiler de performance pour ai'lang
# Fournit des outils d'analyse de performance et d'optimisation

import time
import sys
import gc
import threading
import psutil
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
from contextlib import contextmanager
import cProfile
import pstats
import io
from memory_profiler import profile as memory_profile

# ============================================================================
# Types et structures de données
# ============================================================================

@dataclass
class FunctionProfile:
    """Profil d'une fonction."""
    name: str
    file_path: str
    line_number: int
    call_count: int = 0
    total_time: float = 0.0
    cumulative_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    last_called: Optional[float] = None

@dataclass
class MemorySnapshot:
    """Instantané de l'utilisation mémoire."""
    timestamp: float
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    process_memory: float
    gc_objects: int

@dataclass
class CPUSnapshot:
    """Instantané de l'utilisation CPU."""
    timestamp: float
    cpu_percent: float
    cpu_count: int
    load_average: Optional[Tuple[float, float, float]]
    process_cpu_percent: float

@dataclass
class PerformanceMetrics:
    """Métriques de performance globales."""
    execution_time: float
    memory_peak: float
    memory_average: float
    cpu_peak: float
    cpu_average: float
    function_calls: int
    gc_collections: int
    io_operations: int

# ============================================================================
# Profiler principal
# ============================================================================

class AILangProfiler:
    """
    Profiler de performance pour ai'lang avec analyse détaillée.
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.is_profiling = False
        self.start_time = 0.0
        self.end_time = 0.0
        
        # Profils des fonctions
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.call_stack: List[Tuple[str, float]] = []
        
        # Snapshots système
        self.memory_snapshots: List[MemorySnapshot] = []
        self.cpu_snapshots: List[CPUSnapshot] = []
        
        # Métriques en temps réel
        self.current_memory = 0.0
        self.peak_memory = 0.0
        self.current_cpu = 0.0
        self.peak_cpu = 0.0
        
        # Thread de monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Profiler cProfile intégré
        self.cprofile_profiler: Optional[cProfile.Profile] = None
        
        # Compteurs
        self.function_call_count = 0
        self.gc_collection_count = 0
        self.io_operation_count = 0
        
        # Configuration
        self.track_memory = True
        self.track_cpu = True
        self.track_io = True
        self.detailed_profiling = True
    
    def start_profiling(self, track_memory: bool = True, track_cpu: bool = True, 
                       track_io: bool = True, detailed: bool = True):
        """
        Démarre le profilage.
        
        Args:
            track_memory: Suivre l'utilisation mémoire
            track_cpu: Suivre l'utilisation CPU
            track_io: Suivre les opérations I/O
            detailed: Profilage détaillé avec cProfile
        """
        if self.is_profiling:
            raise RuntimeError("Profiling is already active")
        
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self.track_io = track_io
        self.detailed_profiling = detailed
        
        # Réinitialisation
        self.function_profiles.clear()
        self.call_stack.clear()
        self.memory_snapshots.clear()
        self.cpu_snapshots.clear()
        
        self.current_memory = 0.0
        self.peak_memory = 0.0
        self.current_cpu = 0.0
        self.peak_cpu = 0.0
        
        self.function_call_count = 0
        self.gc_collection_count = 0
        self.io_operation_count = 0
        
        # Démarrage du profilage
        self.start_time = time.time()
        self.is_profiling = True
        
        # Démarrage du profiler cProfile si demandé
        if self.detailed_profiling:
            self.cprofile_profiler = cProfile.Profile()
            self.cprofile_profiler.enable()
        
        # Démarrage du thread de monitoring
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        print(f"Profiling started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def stop_profiling(self) -> PerformanceMetrics:
        """
        Arrête le profilage et retourne les métriques.
        
        Returns:
            Métriques de performance
        """
        if not self.is_profiling:
            raise RuntimeError("Profiling is not active")
        
        self.end_time = time.time()
        self.is_profiling = False
        
        # Arrêt du profiler cProfile
        if self.cprofile_profiler:
            self.cprofile_profiler.disable()
        
        # Arrêt du monitoring
        self.stop_monitoring.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # Calcul des métriques finales
        metrics = self._calculate_metrics()
        
        print(f"Profiling stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {metrics.execution_time:.3f}s")
        
        return metrics
    
    def profile_function(self, func_name: str, file_path: str = "", line_number: int = 0):
        """
        Décorateur pour profiler une fonction spécifique.
        
        Args:
            func_name: Nom de la fonction
            file_path: Chemin du fichier
            line_number: Numéro de ligne
        
        Returns:
            Décorateur
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_profiling:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = self._get_process_memory()
                
                # Entrée dans la fonction
                self._enter_function(func_name, file_path, line_number)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Sortie de la fonction
                    end_time = time.time()
                    end_memory = self._get_process_memory()
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._exit_function(func_name, execution_time, memory_delta)
            
            return wrapper
        return decorator
    
    @contextmanager
    def profile_block(self, block_name: str):
        """
        Context manager pour profiler un bloc de code.
        
        Args:
            block_name: Nom du bloc
        """
        if not self.is_profiling:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_process_memory()
        
        self._enter_function(block_name, "<block>", 0)
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_process_memory()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self._exit_function(block_name, execution_time, memory_delta)
    
    def add_custom_metric(self, name: str, value: float, timestamp: float = None):
        """
        Ajoute une métrique personnalisée.
        
        Args:
            name: Nom de la métrique
            value: Valeur de la métrique
            timestamp: Timestamp (optionnel)
        """
        if not self.is_profiling:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        # Stockage dans les profils de fonction comme métrique spéciale
        if name not in self.function_profiles:
            self.function_profiles[name] = FunctionProfile(
                name=name,
                file_path="<custom_metric>",
                line_number=0
            )
        
        profile = self.function_profiles[name]
        profile.call_count += 1
        profile.total_time += value
        profile.last_called = timestamp
        
        if value < profile.min_time:
            profile.min_time = value
        if value > profile.max_time:
            profile.max_time = value
        
        profile.avg_time = profile.total_time / profile.call_count
    
    def get_function_profiles(self) -> Dict[str, FunctionProfile]:
        """
        Retourne les profils des fonctions.
        
        Returns:
            Dictionnaire des profils
        """
        return self.function_profiles.copy()
    
    def get_top_functions(self, limit: int = 10, sort_by: str = "total_time") -> List[FunctionProfile]:
        """
        Retourne les fonctions les plus coûteuses.
        
        Args:
            limit: Nombre de fonctions à retourner
            sort_by: Critère de tri (total_time, call_count, avg_time)
        
        Returns:
            Liste des profils triés
        """
        profiles = list(self.function_profiles.values())
        
        if sort_by == "total_time":
            profiles.sort(key=lambda p: p.total_time, reverse=True)
        elif sort_by == "call_count":
            profiles.sort(key=lambda p: p.call_count, reverse=True)
        elif sort_by == "avg_time":
            profiles.sort(key=lambda p: p.avg_time, reverse=True)
        elif sort_by == "max_time":
            profiles.sort(key=lambda p: p.max_time, reverse=True)
        
        return profiles[:limit]
    
    def get_memory_usage_over_time(self) -> List[Tuple[float, float]]:
        """
        Retourne l'évolution de l'utilisation mémoire.
        
        Returns:
            Liste de tuples (timestamp, memory_usage)
        """
        return [(snap.timestamp, snap.process_memory) for snap in self.memory_snapshots]
    
    def get_cpu_usage_over_time(self) -> List[Tuple[float, float]]:
        """
        Retourne l'évolution de l'utilisation CPU.
        
        Returns:
            Liste de tuples (timestamp, cpu_usage)
        """
        return [(snap.timestamp, snap.process_cpu_percent) for snap in self.cpu_snapshots]
    
    def generate_report(self, output_format: str = "text") -> str:
        """
        Génère un rapport de performance.
        
        Args:
            output_format: Format de sortie (text, json, html)
        
        Returns:
            Rapport formaté
        """
        if output_format == "text":
            return self._generate_text_report()
        elif output_format == "json":
            return self._generate_json_report()
        elif output_format == "html":
            return self._generate_html_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def export_cprofile_stats(self, output_file: str):
        """
        Exporte les statistiques cProfile vers un fichier.
        
        Args:
            output_file: Fichier de sortie
        """
        if not self.cprofile_profiler:
            raise RuntimeError("cProfile profiling was not enabled")
        
        # Création d'un buffer pour capturer les stats
        s = io.StringIO()
        ps = pstats.Stats(self.cprofile_profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        # Écriture dans le fichier
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(s.getvalue())
    
    def get_hotspots(self, threshold: float = 0.01) -> List[FunctionProfile]:
        """
        Identifie les points chauds de performance.
        
        Args:
            threshold: Seuil de temps minimum (en secondes)
        
        Returns:
            Liste des points chauds
        """
        hotspots = []
        total_execution_time = self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
        
        for profile in self.function_profiles.values():
            # Fonction considérée comme point chaud si elle prend plus de X% du temps total
            if profile.total_time > threshold and profile.total_time > total_execution_time * 0.05:
                hotspots.append(profile)
        
        # Tri par temps total décroissant
        hotspots.sort(key=lambda p: p.total_time, reverse=True)
        return hotspots
    
    def suggest_optimizations(self) -> List[str]:
        """
        Suggère des optimisations basées sur l'analyse.
        
        Returns:
            Liste de suggestions
        """
        suggestions = []
        
        # Analyse des fonctions
        top_functions = self.get_top_functions(5, "total_time")
        hotspots = self.get_hotspots()
        
        if hotspots:
            suggestions.append(f"Optimiser les {len(hotspots)} points chauds identifiés")
            for hotspot in hotspots[:3]:
                suggestions.append(f"  - {hotspot.name}: {hotspot.total_time:.3f}s total, {hotspot.call_count} appels")
        
        # Analyse mémoire
        if self.memory_snapshots:
            memory_growth = self.memory_snapshots[-1].process_memory - self.memory_snapshots[0].process_memory
            if memory_growth > 100 * 1024 * 1024:  # 100MB
                suggestions.append(f"Croissance mémoire importante détectée: {memory_growth / 1024 / 1024:.1f}MB")
                suggestions.append("  - Vérifier les fuites mémoire")
                suggestions.append("  - Optimiser la gestion des objets")
        
        # Analyse CPU
        if self.cpu_snapshots:
            avg_cpu = statistics.mean([snap.process_cpu_percent for snap in self.cpu_snapshots])
            if avg_cpu > 80:
                suggestions.append(f"Utilisation CPU élevée: {avg_cpu:.1f}%")
                suggestions.append("  - Considérer la parallélisation")
                suggestions.append("  - Optimiser les algorithmes coûteux")
        
        # Analyse des appels de fonction
        high_call_count = [p for p in self.function_profiles.values() if p.call_count > 1000]
        if high_call_count:
            suggestions.append(f"{len(high_call_count)} fonctions appelées très fréquemment")
            for func in sorted(high_call_count, key=lambda p: p.call_count, reverse=True)[:3]:
                suggestions.append(f"  - {func.name}: {func.call_count} appels")
            suggestions.append("  - Considérer la mise en cache ou l'optimisation")
        
        if not suggestions:
            suggestions.append("Aucune optimisation majeure détectée")
        
        return suggestions
    
    def _enter_function(self, func_name: str, file_path: str, line_number: int):
        """
        Enregistre l'entrée dans une fonction.
        """
        current_time = time.time()
        self.call_stack.append((func_name, current_time))
        self.function_call_count += 1
        
        # Initialisation du profil si nécessaire
        if func_name not in self.function_profiles:
            self.function_profiles[func_name] = FunctionProfile(
                name=func_name,
                file_path=file_path,
                line_number=line_number
            )
    
    def _exit_function(self, func_name: str, execution_time: float, memory_delta: float):
        """
        Enregistre la sortie d'une fonction.
        """
        if not self.call_stack or self.call_stack[-1][0] != func_name:
            return  # Pile d'appels incohérente
        
        start_time = self.call_stack.pop()[1]
        current_time = time.time()
        actual_execution_time = current_time - start_time
        
        # Mise à jour du profil
        profile = self.function_profiles[func_name]
        profile.call_count += 1
        profile.total_time += actual_execution_time
        profile.last_called = current_time
        
        if actual_execution_time < profile.min_time:
            profile.min_time = actual_execution_time
        if actual_execution_time > profile.max_time:
            profile.max_time = actual_execution_time
        
        profile.avg_time = profile.total_time / profile.call_count
        
        # Enregistrement de l'utilisation mémoire
        if self.track_memory:
            profile.memory_usage.append(memory_delta)
    
    def _monitor_system(self):
        """
        Thread de monitoring système.
        """
        while not self.stop_monitoring.is_set():
            try:
                current_time = time.time()
                
                # Snapshot mémoire
                if self.track_memory:
                    memory_info = psutil.virtual_memory()
                    process = psutil.Process()
                    process_memory = process.memory_info().rss
                    
                    snapshot = MemorySnapshot(
                        timestamp=current_time,
                        total_memory=memory_info.total,
                        available_memory=memory_info.available,
                        used_memory=memory_info.used,
                        memory_percent=memory_info.percent,
                        process_memory=process_memory,
                        gc_objects=len(gc.get_objects())
                    )
                    
                    self.memory_snapshots.append(snapshot)
                    self.current_memory = process_memory
                    
                    if process_memory > self.peak_memory:
                        self.peak_memory = process_memory
                
                # Snapshot CPU
                if self.track_cpu:
                    cpu_percent = psutil.cpu_percent()
                    process = psutil.Process()
                    process_cpu = process.cpu_percent()
                    
                    load_avg = None
                    if hasattr(psutil, 'getloadavg'):
                        try:
                            load_avg = psutil.getloadavg()
                        except AttributeError:
                            pass
                    
                    snapshot = CPUSnapshot(
                        timestamp=current_time,
                        cpu_percent=cpu_percent,
                        cpu_count=psutil.cpu_count(),
                        load_average=load_avg,
                        process_cpu_percent=process_cpu
                    )
                    
                    self.cpu_snapshots.append(snapshot)
                    self.current_cpu = process_cpu
                    
                    if process_cpu > self.peak_cpu:
                        self.peak_cpu = process_cpu
                
                # Attente avant le prochain échantillon
                self.stop_monitoring.wait(self.sampling_interval)
                
            except Exception:
                # Ignorer les erreurs de monitoring pour ne pas interrompre le profilage
                pass
    
    def _get_process_memory(self) -> float:
        """
        Retourne l'utilisation mémoire du processus actuel.
        
        Returns:
            Mémoire utilisée en bytes
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0.0
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """
        Calcule les métriques de performance finales.
        
        Returns:
            Métriques de performance
        """
        execution_time = self.end_time - self.start_time
        
        # Métriques mémoire
        memory_peak = self.peak_memory
        memory_average = 0.0
        if self.memory_snapshots:
            memory_values = [snap.process_memory for snap in self.memory_snapshots]
            memory_average = statistics.mean(memory_values)
        
        # Métriques CPU
        cpu_peak = self.peak_cpu
        cpu_average = 0.0
        if self.cpu_snapshots:
            cpu_values = [snap.process_cpu_percent for snap in self.cpu_snapshots]
            cpu_average = statistics.mean(cpu_values)
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_peak=memory_peak,
            memory_average=memory_average,
            cpu_peak=cpu_peak,
            cpu_average=cpu_average,
            function_calls=self.function_call_count,
            gc_collections=self.gc_collection_count,
            io_operations=self.io_operation_count
        )
    
    def _generate_text_report(self) -> str:
        """
        Génère un rapport texte.
        
        Returns:
            Rapport formaté en texte
        """
        lines = []
        lines.append("=" * 60)
        lines.append("AI'LANG PERFORMANCE REPORT")
        lines.append("=" * 60)
        
        # Métriques générales
        metrics = self._calculate_metrics()
        lines.append(f"\nExecution Time: {metrics.execution_time:.3f}s")
        lines.append(f"Function Calls: {metrics.function_calls:,}")
        lines.append(f"Memory Peak: {metrics.memory_peak / 1024 / 1024:.1f} MB")
        lines.append(f"Memory Average: {metrics.memory_average / 1024 / 1024:.1f} MB")
        lines.append(f"CPU Peak: {metrics.cpu_peak:.1f}%")
        lines.append(f"CPU Average: {metrics.cpu_average:.1f}%")
        
        # Top fonctions
        lines.append("\n" + "-" * 60)
        lines.append("TOP FUNCTIONS BY TOTAL TIME")
        lines.append("-" * 60)
        
        top_functions = self.get_top_functions(10, "total_time")
        for i, func in enumerate(top_functions, 1):
            lines.append(f"{i:2d}. {func.name}")
            lines.append(f"    Total: {func.total_time:.3f}s | Calls: {func.call_count:,} | Avg: {func.avg_time:.6f}s")
            lines.append(f"    Min: {func.min_time:.6f}s | Max: {func.max_time:.6f}s")
            if func.file_path and func.file_path != "<block>":
                lines.append(f"    File: {func.file_path}:{func.line_number}")
            lines.append("")
        
        # Points chauds
        hotspots = self.get_hotspots()
        if hotspots:
            lines.append("-" * 60)
            lines.append("PERFORMANCE HOTSPOTS")
            lines.append("-" * 60)
            
            for hotspot in hotspots:
                percentage = (hotspot.total_time / metrics.execution_time) * 100
                lines.append(f"• {hotspot.name}: {hotspot.total_time:.3f}s ({percentage:.1f}% of total time)")
        
        # Suggestions d'optimisation
        suggestions = self.suggest_optimizations()
        if suggestions:
            lines.append("\n" + "-" * 60)
            lines.append("OPTIMIZATION SUGGESTIONS")
            lines.append("-" * 60)
            
            for suggestion in suggestions:
                lines.append(f"• {suggestion}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def _generate_json_report(self) -> str:
        """
        Génère un rapport JSON.
        
        Returns:
            Rapport formaté en JSON
        """
        metrics = self._calculate_metrics()
        
        report_data = {
            "summary": {
                "execution_time": metrics.execution_time,
                "function_calls": metrics.function_calls,
                "memory_peak_mb": metrics.memory_peak / 1024 / 1024,
                "memory_average_mb": metrics.memory_average / 1024 / 1024,
                "cpu_peak_percent": metrics.cpu_peak,
                "cpu_average_percent": metrics.cpu_average
            },
            "functions": [
                {
                    "name": func.name,
                    "file_path": func.file_path,
                    "line_number": func.line_number,
                    "call_count": func.call_count,
                    "total_time": func.total_time,
                    "average_time": func.avg_time,
                    "min_time": func.min_time,
                    "max_time": func.max_time
                }
                for func in self.function_profiles.values()
            ],
            "hotspots": [
                {
                    "name": hotspot.name,
                    "total_time": hotspot.total_time,
                    "percentage": (hotspot.total_time / metrics.execution_time) * 100
                }
                for hotspot in self.get_hotspots()
            ],
            "suggestions": self.suggest_optimizations(),
            "memory_timeline": [
                {
                    "timestamp": snap.timestamp,
                    "memory_mb": snap.process_memory / 1024 / 1024
                }
                for snap in self.memory_snapshots
            ],
            "cpu_timeline": [
                {
                    "timestamp": snap.timestamp,
                    "cpu_percent": snap.process_cpu_percent
                }
                for snap in self.cpu_snapshots
            ]
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self) -> str:
        """
        Génère un rapport HTML avec graphiques.
        
        Returns:
            Rapport formaté en HTML
        """
        metrics = self._calculate_metrics()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI'Lang Performance Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
        .section {{ margin: 20px 0; }}
        .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AI'Lang Performance Report</h1>
        <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Execution Time: {metrics.execution_time:.3f}s</div>
        <div class="metric">Function Calls: {metrics.function_calls:,}</div>
        <div class="metric">Memory Peak: {metrics.memory_peak / 1024 / 1024:.1f} MB</div>
        <div class="metric">CPU Peak: {metrics.cpu_peak:.1f}%</div>
    </div>
    
    <div class="section">
        <h2>Top Functions</h2>
        <table>
            <tr>
                <th>Function</th>
                <th>Total Time (s)</th>
                <th>Calls</th>
                <th>Avg Time (s)</th>
                <th>File</th>
            </tr>
"""
        
        top_functions = self.get_top_functions(10, "total_time")
        for func in top_functions:
            html += f"""
            <tr>
                <td>{func.name}</td>
                <td>{func.total_time:.3f}</td>
                <td>{func.call_count:,}</td>
                <td>{func.avg_time:.6f}</td>
                <td>{func.file_path}:{func.line_number}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Memory Usage Over Time</h2>
        <div id="memoryChart" class="chart"></div>
    </div>
    
    <div class="section">
        <h2>CPU Usage Over Time</h2>
        <div id="cpuChart" class="chart"></div>
    </div>
    
    <script>
        // Memory chart
        var memoryData = [
"""
        
        memory_times = [snap.timestamp for snap in self.memory_snapshots]
        memory_values = [snap.process_memory / 1024 / 1024 for snap in self.memory_snapshots]
        
        html += f"""
            {{
                x: {memory_times},
                y: {memory_values},
                type: 'scatter',
                mode: 'lines',
                name: 'Memory Usage (MB)'
            }}
        ];
        
        Plotly.newPlot('memoryChart', memoryData, {{
            title: 'Memory Usage Over Time',
            xaxis: {{ title: 'Time' }},
            yaxis: {{ title: 'Memory (MB)' }}
        }});
        
        // CPU chart
        var cpuData = [
"""
        
        cpu_times = [snap.timestamp for snap in self.cpu_snapshots]
        cpu_values = [snap.process_cpu_percent for snap in self.cpu_snapshots]
        
        html += f"""
            {{
                x: {cpu_times},
                y: {cpu_values},
                type: 'scatter',
                mode: 'lines',
                name: 'CPU Usage (%)'
            }}
        ];
        
        Plotly.newPlot('cpuChart', cpuData, {{
            title: 'CPU Usage Over Time',
            xaxis: {{ title: 'Time' }},
            yaxis: {{ title: 'CPU (%)' }}
        }});
    </script>
</body>
</html>
"""
        
        return html

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def create_profiler(sampling_interval: float = 0.1) -> AILangProfiler:
    """
    Crée une nouvelle instance du profiler.
    
    Args:
        sampling_interval: Intervalle d'échantillonnage en secondes
    
    Returns:
        Instance du profiler
    """
    return AILangProfiler(sampling_interval)

def profile_function(func_name: str = None, file_path: str = "", line_number: int = 0):
    """
    Décorateur pour profiler une fonction avec le profiler global.
    
    Args:
        func_name: Nom de la fonction (optionnel)
        file_path: Chemin du fichier
        line_number: Numéro de ligne
    
    Returns:
        Décorateur
    """
    def decorator(func):
        name = func_name or func.__name__
        path = file_path or getattr(func, '__code__', {}).get('co_filename', '')
        line = line_number or getattr(func, '__code__', {}).get('co_firstlineno', 0)
        
        return _global_profiler.profile_function(name, path, line)(func)
    
    return decorator

@contextmanager
def profile_block(block_name: str):
    """
    Context manager pour profiler un bloc de code avec le profiler global.
    
    Args:
        block_name: Nom du bloc
    """
    with _global_profiler.profile_block(block_name):
        yield

def start_profiling(**kwargs):
    """
    Démarre le profilage global.
    
    Args:
        **kwargs: Arguments pour start_profiling
    """
    _global_profiler.start_profiling(**kwargs)

def stop_profiling() -> PerformanceMetrics:
    """
    Arrête le profilage global.
    
    Returns:
        Métriques de performance
    """
    return _global_profiler.stop_profiling()

def get_profiler() -> AILangProfiler:
    """
    Retourne l'instance globale du profiler.
    
    Returns:
        Instance du profiler
    """
    return _global_profiler

def generate_report(output_format: str = "text") -> str:
    """
    Génère un rapport de performance avec le profiler global.
    
    Args:
        output_format: Format de sortie
    
    Returns:
        Rapport formaté
    """
    return _global_profiler.generate_report(output_format)

def benchmark(func, *args, iterations: int = 1000, **kwargs):
    """
    Effectue un benchmark d'une fonction.
    
    Args:
        func: Fonction à benchmarker
        *args: Arguments de la fonction
        iterations: Nombre d'itérations
        **kwargs: Arguments nommés de la fonction
    
    Returns:
        Dictionnaire avec les résultats du benchmark
    """
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'iterations': iterations,
        'total_time': sum(times),
        'average_time': statistics.mean(times),
        'min_time': min(times),
        'max_time': max(times),
        'median_time': statistics.median(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0
    }

# Instance globale du profiler
_global_profiler = AILangProfiler()