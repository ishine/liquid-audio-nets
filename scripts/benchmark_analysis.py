#!/usr/bin/env python3
"""
Performance benchmark analysis and reporting for liquid-audio-nets.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    latency_ms: float
    throughput_samples_per_sec: float
    memory_mb: float
    power_mw: float
    accuracy: float
    timestamp: str
    metadata: Dict[str, Any] = None


class BenchmarkAnalyzer:
    """Analyze and compare benchmark results."""
    
    def __init__(self, baseline_path: Path = None):
        self.baseline_path = baseline_path
        self.baseline_results = self._load_baseline() if baseline_path else None
    
    def _load_baseline(self) -> Dict[str, BenchmarkResult]:
        """Load baseline benchmark results."""
        if not self.baseline_path.exists():
            logger.warning(f"Baseline file {self.baseline_path} not found")
            return {}
        
        with open(self.baseline_path) as f:
            data = json.load(f)
        
        results = {}
        for name, metrics in data.items():
            results[name] = BenchmarkResult(
                name=name,
                latency_ms=metrics['latency_ms'],
                throughput_samples_per_sec=metrics['throughput_samples_per_sec'],
                memory_mb=metrics['memory_mb'],
                power_mw=metrics['power_mw'],
                accuracy=metrics['accuracy'],
                timestamp=metrics['timestamp'],
                metadata=metrics.get('metadata', {})
            )
        
        return results
    
    def load_current_results(self, results_path: Path) -> Dict[str, BenchmarkResult]:
        """Load current benchmark results."""
        with open(results_path) as f:
            data = json.load(f)
        
        results = {}
        for name, metrics in data.items():
            results[name] = BenchmarkResult(
                name=name,
                latency_ms=metrics['latency_ms'],
                throughput_samples_per_sec=metrics['throughput_samples_per_sec'],
                memory_mb=metrics['memory_mb'],
                power_mw=metrics['power_mw'],
                accuracy=metrics['accuracy'],
                timestamp=metrics['timestamp'],
                metadata=metrics.get('metadata', {})
            )
        
        return results
    
    def compare_results(self, current_results: Dict[str, BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """Compare current results against baseline."""
        if not self.baseline_results:
            logger.warning("No baseline results available for comparison")
            return {}
        
        comparisons = {}
        
        for name, current in current_results.items():
            if name not in self.baseline_results:
                logger.warning(f"No baseline found for benchmark {name}")
                continue
            
            baseline = self.baseline_results[name]
            
            # Calculate percentage changes
            comparisons[name] = {
                'latency_change_percent': ((current.latency_ms - baseline.latency_ms) / baseline.latency_ms) * 100,
                'throughput_change_percent': ((current.throughput_samples_per_sec - baseline.throughput_samples_per_sec) / baseline.throughput_samples_per_sec) * 100,
                'memory_change_percent': ((current.memory_mb - baseline.memory_mb) / baseline.memory_mb) * 100,
                'power_change_percent': ((current.power_mw - baseline.power_mw) / baseline.power_mw) * 100,
                'accuracy_change_percent': ((current.accuracy - baseline.accuracy) / baseline.accuracy) * 100,
            }
        
        return comparisons
    
    def check_regressions(self, current_results: Dict[str, BenchmarkResult], 
                         thresholds: Dict[str, float] = None) -> List[str]:
        """Check for performance regressions."""
        if thresholds is None:
            thresholds = {
                'latency_threshold_percent': 10.0,
                'throughput_threshold_percent': -5.0,
                'memory_threshold_percent': 10.0,
                'power_threshold_percent': 10.0,
                'accuracy_threshold_percent': -2.0,
            }
        
        comparisons = self.compare_results(current_results)
        regressions = []
        
        for name, changes in comparisons.items():
            # Check for regressions (negative changes are improvements for latency, memory, power)
            if changes['latency_change_percent'] > thresholds['latency_threshold_percent']:
                regressions.append(f"{name}: Latency regression +{changes['latency_change_percent']:.1f}%")
            
            if changes['throughput_change_percent'] < thresholds['throughput_threshold_percent']:
                regressions.append(f"{name}: Throughput regression {changes['throughput_change_percent']:.1f}%")
            
            if changes['memory_change_percent'] > thresholds['memory_threshold_percent']:
                regressions.append(f"{name}: Memory regression +{changes['memory_change_percent']:.1f}%")
            
            if changes['power_change_percent'] > thresholds['power_threshold_percent']:
                regressions.append(f"{name}: Power regression +{changes['power_change_percent']:.1f}%")
            
            if changes['accuracy_change_percent'] < thresholds['accuracy_threshold_percent']:
                regressions.append(f"{name}: Accuracy regression {changes['accuracy_change_percent']:.1f}%")
        
        return regressions
    
    def generate_report(self, current_results: Dict[str, BenchmarkResult], 
                       output_path: Path = None) -> str:
        """Generate a comprehensive benchmark report."""
        report_lines = [
            "# Liquid Audio Nets - Benchmark Report",
            "",
            f"Generated at: {current_results[list(current_results.keys())[0]].timestamp}",
            "",
            "## Current Results",
            "",
            "| Benchmark | Latency (ms) | Throughput (samples/s) | Memory (MB) | Power (mW) | Accuracy |",
            "|-----------|--------------|------------------------|-------------|------------|----------|"
        ]
        
        for name, result in current_results.items():
            report_lines.append(
                f"| {name} | {result.latency_ms:.2f} | {result.throughput_samples_per_sec:.0f} | "
                f"{result.memory_mb:.2f} | {result.power_mw:.2f} | {result.accuracy:.3f} |"
            )
        
        if self.baseline_results:
            report_lines.extend([
                "",
                "## Comparison with Baseline",
                ""
            ])
            
            comparisons = self.compare_results(current_results)
            regressions = self.check_regressions(current_results)
            
            if regressions:
                report_lines.extend([
                    "### ⚠️ Regressions Detected",
                    ""
                ])
                for regression in regressions:
                    report_lines.append(f"- {regression}")
                report_lines.append("")
            else:
                report_lines.extend([
                    "### ✅ No Regressions Detected",
                    ""
                ])
            
            report_lines.extend([
                "### Detailed Changes",
                "",
                "| Benchmark | Latency | Throughput | Memory | Power | Accuracy |",
                "|-----------|---------|------------|--------|-------|----------|"
            ])
            
            for name, changes in comparisons.items():
                def format_change(value):
                    sign = "+" if value >= 0 else ""
                    return f"{sign}{value:.1f}%"
                
                report_lines.append(
                    f"| {name} | {format_change(changes['latency_change_percent'])} | "
                    f"{format_change(changes['throughput_change_percent'])} | "
                    f"{format_change(changes['memory_change_percent'])} | "
                    f"{format_change(changes['power_change_percent'])} | "
                    f"{format_change(changes['accuracy_change_percent'])} |"
                )
        
        # Add target compliance
        report_lines.extend([
            "",
            "## Target Compliance",
            "",
            "| Benchmark | Latency Target | Throughput Target | Memory Target | Power Target | Accuracy Target |",
            "|-----------|----------------|-------------------|---------------|--------------|-----------------|"
        ])
        
        targets = {
            'STM32F4': {'latency': 15.0, 'throughput': 16000, 'memory': 0.064, 'power': 1.2, 'accuracy': 0.938},
            'nRF52840': {'latency': 12.0, 'throughput': 16000, 'memory': 0.032, 'power': 0.9, 'accuracy': 0.935},
        }
        
        for name, result in current_results.items():
            # Assume target based on benchmark name
            target_key = 'STM32F4' if 'stm32' in name.lower() else 'nRF52840'
            target = targets.get(target_key, targets['STM32F4'])
            
            def compliance_status(value, target, lower_is_better=True):
                if lower_is_better:
                    return "✅" if value <= target else "❌"
                else:
                    return "✅" if value >= target else "❌"
            
            report_lines.append(
                f"| {name} | {compliance_status(result.latency_ms, target['latency'])} {result.latency_ms:.1f}/{target['latency']}ms | "
                f"{compliance_status(result.throughput_samples_per_sec, target['throughput'], False)} {result.throughput_samples_per_sec:.0f}/{target['throughput']} | "
                f"{compliance_status(result.memory_mb, target['memory'])} {result.memory_mb:.3f}/{target['memory']}MB | "
                f"{compliance_status(result.power_mw, target['power'])} {result.power_mw:.2f}/{target['power']}mW | "
                f"{compliance_status(result.accuracy, target['accuracy'], False)} {result.accuracy:.3f}/{target['accuracy']} |"
            )
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def generate_plots(self, current_results: Dict[str, BenchmarkResult], 
                      output_dir: Path = None):
        """Generate performance plots."""
        if not output_dir:
            output_dir = Path("benchmark_plots")
        
        output_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        names = list(current_results.keys())
        latencies = [r.latency_ms for r in current_results.values()]
        throughputs = [r.throughput_samples_per_sec for r in current_results.values()]
        memory_usage = [r.memory_mb for r in current_results.values()]
        power_consumption = [r.power_mw for r in current_results.values()]
        accuracies = [r.accuracy for r in current_results.values()]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Liquid Audio Nets - Performance Benchmarks', fontsize=16)
        
        # Latency plot
        axes[0, 0].bar(names, latencies, color='skyblue')
        axes[0, 0].set_title('Inference Latency')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput plot
        axes[0, 1].bar(names, throughputs, color='lightgreen')
        axes[0, 1].set_title('Throughput')
        axes[0, 1].set_ylabel('Samples/sec')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory plot
        axes[0, 2].bar(names, memory_usage, color='orange')
        axes[0, 2].set_title('Memory Usage')
        axes[0, 2].set_ylabel('Memory (MB)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Power plot
        axes[1, 0].bar(names, power_consumption, color='red')
        axes[1, 0].set_title('Power Consumption')
        axes[1, 0].set_ylabel('Power (mW)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy plot
        axes[1, 1].bar(names, accuracies, color='purple')
        axes[1, 1].set_title('Model Accuracy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Power vs Accuracy scatter
        axes[1, 2].scatter(power_consumption, accuracies, c='darkblue', s=100)
        axes[1, 2].set_title('Power vs Accuracy Trade-off')
        axes[1, 2].set_xlabel('Power (mW)')
        axes[1, 2].set_ylabel('Accuracy')
        for i, name in enumerate(names):
            axes[1, 2].annotate(name, (power_consumption[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plot_path = output_dir / "benchmark_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--current", type=Path, required=True, help="Current benchmark results JSON")
    parser.add_argument("--baseline", type=Path, help="Baseline benchmark results JSON")
    parser.add_argument("--output", type=Path, help="Output directory for reports and plots")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with error if regressions detected")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = args.output or Path("benchmark_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(args.baseline)
    
    # Load current results
    current_results = analyzer.load_current_results(args.current)
    
    # Generate report
    report = analyzer.generate_report(current_results, output_dir / "benchmark_report.md")
    print(report)
    
    # Generate plots
    analyzer.generate_plots(current_results, output_dir)
    
    # Check for regressions
    regressions = analyzer.check_regressions(current_results)
    
    if regressions:
        logger.error("Performance regressions detected:")
        for regression in regressions:
            logger.error(f"  - {regression}")
        
        if args.fail_on_regression:
            exit(1)
    else:
        logger.info("No performance regressions detected ✅")


if __name__ == "__main__":
    main()