"""
Performance benchmarking framework for liquid-audio-nets.
"""
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import platform
import subprocess


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    name: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    power_estimate_mw: Optional[float] = None
    accuracy: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    latency_percentiles: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class PerformanceBenchmark:
    """Framework for running performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmarking context."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "architecture": platform.architecture()[0]
        }
    
    def benchmark_function(
        self, 
        func: Callable,
        name: str,
        args: tuple = (),
        kwargs: dict = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
        measure_accuracy: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        Benchmark a function's performance.
        
        Args:
            func: Function to benchmark
            name: Name for this benchmark
            args: Arguments to pass to function
            kwargs: Keyword arguments to pass to function
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            measure_accuracy: Optional function to measure accuracy
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if kwargs is None:
            kwargs = {}
            
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Warmup failed for {name}: {e}")
                raise
        
        # Benchmark runs
        durations = []
        memory_usage = []
        cpu_usage = []
        accuracies = []
        
        process = psutil.Process()
        
        for _ in range(benchmark_runs):
            # Measure memory before
            mem_before = process.memory_info().rss / (1024**2)  # MB
            
            # Measure CPU and time
            cpu_before = process.cpu_percent()
            start_time = time.perf_counter()
            
            # Run the function
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            cpu_after = process.cpu_percent()
            
            # Measure memory after
            mem_after = process.memory_info().rss / (1024**2)  # MB
            
            duration_ms = (end_time - start_time) * 1000
            memory_mb = max(mem_after - mem_before, 0)  # Memory increase
            cpu_percent = max(cpu_after - cpu_before, 0)
            
            durations.append(duration_ms)
            memory_usage.append(memory_mb)
            cpu_usage.append(cpu_percent)
            
            # Measure accuracy if provided
            if measure_accuracy:
                accuracy = measure_accuracy(result)
                accuracies.append(accuracy)
        
        # Calculate statistics
        avg_duration = np.mean(durations)
        avg_memory = np.mean(memory_usage)
        avg_cpu = np.mean(cpu_usage)
        avg_accuracy = np.mean(accuracies) if accuracies else None
        
        # Calculate latency percentiles
        latency_percentiles = {
            "p50": np.percentile(durations, 50),
            "p90": np.percentile(durations, 90),
            "p95": np.percentile(durations, 95),
            "p99": np.percentile(durations, 99),
            "min": np.min(durations),
            "max": np.max(durations),
            "std": np.std(durations)
        }
        
        # Estimate power consumption (rough approximation)
        power_estimate = self._estimate_power_consumption(avg_cpu, avg_memory)
        
        benchmark_result = BenchmarkResult(
            name=name,
            duration_ms=avg_duration,
            memory_mb=avg_memory,
            cpu_percent=avg_cpu,
            power_estimate_mw=power_estimate,
            accuracy=avg_accuracy,
            latency_percentiles=latency_percentiles,
            metadata={
                "warmup_runs": warmup_runs,
                "benchmark_runs": benchmark_runs,
                "system_info": self.system_info
            }
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def _estimate_power_consumption(self, cpu_percent: float, memory_mb: float) -> float:
        """
        Rough estimate of power consumption based on CPU and memory usage.
        This is a very approximate estimation for comparison purposes.
        """
        # Base power consumption (idle)
        base_power_mw = 100  # ~0.1W baseline
        
        # CPU power scaling (very rough approximation)
        cpu_power_mw = (cpu_percent / 100) * 2000  # Up to 2W for CPU
        
        # Memory power scaling
        memory_power_mw = memory_mb * 0.1  # ~0.1mW per MB of additional memory
        
        return base_power_mw + cpu_power_mw + memory_power_mw
    
    def benchmark_throughput(
        self,
        func: Callable,
        name: str,
        test_data: List[Any],
        batch_size: int = 1,
        duration_seconds: float = 5.0
    ) -> BenchmarkResult:
        """
        Benchmark throughput (samples per second).
        
        Args:
            func: Function to benchmark
            name: Name for this benchmark
            test_data: List of test samples
            batch_size: Batch size for processing
            duration_seconds: How long to run the throughput test
            
        Returns:
            BenchmarkResult with throughput metrics
        """
        samples_processed = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        process = psutil.Process()
        mem_start = process.memory_info().rss / (1024**2)
        
        while time.perf_counter() < end_time:
            # Get batch of data
            batch_start = samples_processed % len(test_data)
            batch_end = min(batch_start + batch_size, len(test_data))
            batch = test_data[batch_start:batch_end]
            
            # Process batch
            for sample in batch:
                func(sample)
                samples_processed += 1
        
        actual_duration = time.perf_counter() - start_time
        mem_end = process.memory_info().rss / (1024**2)
        
        throughput = samples_processed / actual_duration
        
        return BenchmarkResult(
            name=name,
            duration_ms=actual_duration * 1000,
            memory_mb=mem_end - mem_start,
            cpu_percent=0,  # Not measured in throughput test
            throughput_samples_per_sec=throughput,
            metadata={
                "samples_processed": samples_processed,
                "batch_size": batch_size,
                "test_duration_s": actual_duration
            }
        )
    
    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        test_args: tuple = (),
        test_kwargs: dict = None,
        accuracy_func: Optional[Callable] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple implementations of the same functionality.
        
        Args:
            implementations: Dict mapping names to functions
            test_args: Arguments to pass to all functions
            test_kwargs: Keyword arguments to pass to all functions
            accuracy_func: Function to measure accuracy
            
        Returns:
            Dict mapping implementation names to BenchmarkResults
        """
        if test_kwargs is None:
            test_kwargs = {}
            
        results = {}
        
        for name, func in implementations.items():
            result = self.benchmark_function(
                func=func,
                name=f"{self.name}_{name}",
                args=test_args,
                kwargs=test_kwargs,
                measure_accuracy=accuracy_func
            )
            results[name] = result
            
        return results
    
    def save_results(self, filepath: Path) -> None:
        """Save benchmark results to JSON file."""
        data = {
            "benchmark_name": self.name,
            "system_info": self.system_info,
            "timestamp": time.time(),
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print(f"\n=== Benchmark Results: {self.name} ===")
        print(f"System: {self.system_info['platform']}")
        print(f"CPU: {self.system_info['processor']}")
        print(f"Memory: {self.system_info['memory_gb']:.1f} GB")
        print()
        
        for result in self.results:
            print(f"Test: {result.name}")
            print(f"  Duration: {result.duration_ms:.2f} ms")
            print(f"  Memory: {result.memory_mb:.2f} MB")
            print(f"  CPU: {result.cpu_percent:.1f}%")
            
            if result.power_estimate_mw:
                print(f"  Power Est: {result.power_estimate_mw:.1f} mW")
            
            if result.accuracy:
                print(f"  Accuracy: {result.accuracy:.3f}")
                
            if result.throughput_samples_per_sec:
                print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
                
            if result.latency_percentiles:
                p95 = result.latency_percentiles["p95"]
                print(f"  Latency P95: {p95:.2f} ms")
            
            print()


class HardwareBenchmark(PerformanceBenchmark):
    """Extended benchmark for hardware-specific testing."""
    
    def __init__(self, name: str, hardware_info: Dict[str, Any] = None):
        super().__init__(name)
        self.hardware_info = hardware_info or {}
    
    def benchmark_embedded_constraints(
        self,
        func: Callable,
        name: str,
        memory_limit_kb: int,
        power_budget_mw: float,
        latency_limit_ms: float,
        test_args: tuple = (),
        test_kwargs: dict = None
    ) -> BenchmarkResult:
        """
        Benchmark with embedded system constraints.
        
        Args:
            func: Function to benchmark
            name: Name for this benchmark
            memory_limit_kb: Memory limit in KB
            power_budget_mw: Power budget in mW
            latency_limit_ms: Latency limit in ms
            test_args: Arguments to pass to function
            test_kwargs: Keyword arguments to pass to function
            
        Returns:
            BenchmarkResult with constraint violation flags
        """
        if test_kwargs is None:
            test_kwargs = {}
            
        result = self.benchmark_function(func, name, test_args, test_kwargs)
        
        # Check constraint violations
        constraints_met = {
            "memory": result.memory_mb * 1024 <= memory_limit_kb,
            "power": result.power_estimate_mw <= power_budget_mw,
            "latency": result.duration_ms <= latency_limit_ms
        }
        
        result.metadata = result.metadata or {}
        result.metadata.update({
            "constraints": {
                "memory_limit_kb": memory_limit_kb,
                "power_budget_mw": power_budget_mw,
                "latency_limit_ms": latency_limit_ms
            },
            "constraints_met": constraints_met,
            "all_constraints_met": all(constraints_met.values())
        })
        
        return result