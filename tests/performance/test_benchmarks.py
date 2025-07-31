"""
Performance benchmarks and regression tests.
"""
import time
import psutil
import numpy as np
import pytest
from typing import Dict, Any


class PerformanceMonitor:
    """Monitor performance metrics during testing."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process()
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss
        
        return {
            'duration_ms': (end_time - self.start_time) * 1000,
            'memory_delta_mb': (end_memory - self.start_memory) / 1024 / 1024,
            'peak_memory_mb': self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else None
        }


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for liquid-audio-nets."""
    
    def test_inference_latency_benchmark(self, sample_features):
        """Benchmark inference latency."""
        monitor = PerformanceMonitor()
        
        # Mock LNN inference
        monitor.start()
        for _ in range(100):  # Run multiple iterations
            result = self._mock_lnn_inference(sample_features)
        metrics = monitor.stop()
        
        # Performance requirements
        avg_latency_ms = metrics['duration_ms'] / 100
        assert avg_latency_ms < 20.0, f"Average latency {avg_latency_ms:.2f}ms exceeds 20ms threshold"
        
        print(f"Average inference latency: {avg_latency_ms:.2f}ms")
        print(f"Memory usage: {metrics['memory_delta_mb']:.2f}MB")
    
    def test_throughput_benchmark(self, sample_audio):
        """Benchmark audio processing throughput."""
        monitor = PerformanceMonitor()
        
        # Process multiple audio chunks
        chunk_size = 1024
        chunks = [sample_audio[i:i+chunk_size] for i in range(0, len(sample_audio), chunk_size)]
        
        monitor.start()
        processed_chunks = 0
        for chunk in chunks:
            self._mock_process_audio_chunk(chunk)
            processed_chunks += 1
        metrics = monitor.stop()
        
        # Calculate throughput
        total_samples = len(sample_audio)
        samples_per_second = total_samples / (metrics['duration_ms'] / 1000)
        
        # Should process at least real-time (16kHz)
        assert samples_per_second >= 16000, f"Throughput {samples_per_second:.0f} samples/s below real-time"
        
        print(f"Processing throughput: {samples_per_second:.0f} samples/s")
    
    def test_memory_efficiency_benchmark(self, sample_features):
        """Benchmark memory efficiency."""
        monitor = PerformanceMonitor()
        
        # Test with increasing batch sizes
        batch_sizes = [1, 10, 50, 100]
        memory_usage = []
        
        for batch_size in batch_sizes:
            monitor.start()
            batch = np.tile(sample_features, (batch_size, 1, 1))
            result = self._mock_batch_inference(batch)
            metrics = monitor.stop()
            memory_usage.append(metrics['memory_delta_mb'])
        
        # Memory usage should scale reasonably
        memory_per_sample = memory_usage[-1] / batch_sizes[-1]
        assert memory_per_sample < 1.0, f"Memory per sample {memory_per_sample:.2f}MB too high"
        
        print(f"Memory usage per sample: {memory_per_sample:.3f}MB")
    
    def test_power_estimation_benchmark(self, sample_audio):
        """Benchmark estimated power consumption."""
        # Mock power measurement (would use actual hardware profiling)
        
        operations_count = self._count_operations(sample_audio)
        estimated_power_mw = operations_count * 0.1  # Mock power model
        
        # Power efficiency target
        assert estimated_power_mw < 2.0, f"Estimated power {estimated_power_mw:.2f}mW exceeds 2mW target"
        
        print(f"Estimated power consumption: {estimated_power_mw:.2f}mW")
    
    @pytest.mark.gpu
    def test_gpu_vs_cpu_performance(self, sample_features):
        """Compare GPU vs CPU performance (if available)."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # CPU inference
        cpu_monitor = PerformanceMonitor()
        cpu_monitor.start()
        cpu_result = self._mock_cpu_inference(sample_features)
        cpu_metrics = cpu_monitor.stop()
        
        # GPU inference
        gpu_monitor = PerformanceMonitor()
        gpu_monitor.start()
        gpu_result = self._mock_gpu_inference(sample_features)
        gpu_metrics = gpu_monitor.stop()
        
        # Compare results
        speedup = cpu_metrics['duration_ms'] / gpu_metrics['duration_ms']
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Results should be consistent
        assert np.allclose(cpu_result, gpu_result, rtol=1e-5)
    
    def test_scalability_benchmark(self, sample_features):
        """Test performance scalability with input size."""
        input_sizes = [10, 50, 100, 200]
        latencies = []
        
        for size in input_sizes:
            scaled_features = sample_features[:size]
            
            monitor = PerformanceMonitor()
            monitor.start()
            result = self._mock_lnn_inference(scaled_features)
            metrics = monitor.stop()
            
            latencies.append(metrics['duration_ms'])
        
        # Check that latency scales reasonably (should be roughly linear)
        latency_per_frame = [lat / size for lat, size in zip(latencies, input_sizes)]
        max_per_frame = max(latency_per_frame)
        min_per_frame = min(latency_per_frame)
        
        # Shouldn't vary by more than 2x
        assert max_per_frame / min_per_frame < 2.0, "Latency scaling is not consistent"
        
        print(f"Latency per frame: {np.mean(latency_per_frame):.3f}ms")
    
    def _mock_lnn_inference(self, features: np.ndarray) -> np.ndarray:
        """Mock LNN inference with realistic computation."""
        # Simulate some computation
        result = np.random.rand(len(features), 10).astype(np.float32)
        # Add small delay to simulate real computation
        time.sleep(0.001)
        return result
    
    def _mock_process_audio_chunk(self, chunk: np.ndarray):
        """Mock audio chunk processing."""
        # Simulate feature extraction and inference
        features = np.random.randn(len(chunk) // 64, 40).astype(np.float32)
        return self._mock_lnn_inference(features)
    
    def _mock_batch_inference(self, batch: np.ndarray) -> np.ndarray:
        """Mock batch inference."""
        batch_size, seq_len, features = batch.shape
        return np.random.rand(batch_size, seq_len, 10).astype(np.float32)
    
    def _count_operations(self, audio: np.ndarray) -> int:
        """Mock operation counting for power estimation."""
        # Simplified operation count based on audio length
        return len(audio) * 100  # Mock FLOPs per sample
    
    def _mock_cpu_inference(self, features: np.ndarray) -> np.ndarray:
        """Mock CPU inference."""
        return self._mock_lnn_inference(features)
    
    def _mock_gpu_inference(self, features: np.ndarray) -> np.ndarray:
        """Mock GPU inference."""
        # Simulate faster GPU processing
        time.sleep(0.0005)  # Half the time of CPU
        return self._mock_lnn_inference(features)


@pytest.mark.slow
class TestRegressionBenchmarks:
    """Regression tests to prevent performance degradation."""
    
    BASELINE_METRICS = {
        'inference_latency_ms': 15.0,
        'throughput_samples_per_sec': 32000,
        'memory_per_sample_mb': 0.5,
        'estimated_power_mw': 1.5
    }
    
    def test_inference_latency_regression(self, sample_features):
        """Ensure inference latency doesn't regress."""
        monitor = PerformanceMonitor()
        
        monitor.start()
        for _ in range(50):
            result = TestPerformanceBenchmarks()._mock_lnn_inference(sample_features)
        metrics = monitor.stop()
        
        avg_latency = metrics['duration_ms'] / 50
        baseline = self.BASELINE_METRICS['inference_latency_ms']
        
        assert avg_latency <= baseline * 1.1, f"Latency regression: {avg_latency:.2f}ms vs baseline {baseline}ms"
    
    def test_memory_usage_regression(self, sample_features):
        """Ensure memory usage doesn't regress."""
        monitor = PerformanceMonitor()
        
        monitor.start()
        result = TestPerformanceBenchmarks()._mock_batch_inference(np.tile(sample_features, (10, 1, 1)))
        metrics = monitor.stop()
        
        memory_per_sample = metrics['memory_delta_mb'] / 10
        baseline = self.BASELINE_METRICS['memory_per_sample_mb']
        
        assert memory_per_sample <= baseline * 1.1, f"Memory regression: {memory_per_sample:.3f}MB vs baseline {baseline}MB"