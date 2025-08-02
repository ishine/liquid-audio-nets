"""
Configuration generators for testing different scenarios and hardware targets.
"""
from typing import Dict, Any, List
import json
from pathlib import Path


def get_test_config(config_name: str) -> Dict[str, Any]:
    """
    Get test configuration by name.
    
    Args:
        config_name: Name of configuration ('minimal', 'standard', 'embedded_stm32', etc.)
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "minimal": _get_minimal_config(),
        "standard": _get_standard_config(),
        "embedded_stm32": _get_embedded_stm32_config(),
        "embedded_nrf52": _get_embedded_nrf52_config(),
        "high_performance": _get_high_performance_config(),
        "ultra_low_power": _get_ultra_low_power_config(),
        "testing": _get_testing_config(),
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


def _get_minimal_config() -> Dict[str, Any]:
    """Minimal configuration for basic testing."""
    return {
        "model": {
            "input_dim": 13,
            "hidden_dim": 16,
            "output_dim": 2,
            "ode_solver": "euler",
            "timestep_range": (0.01, 0.05),
            "complexity_penalty": 0.0
        },
        "audio": {
            "sample_rate": 16000,
            "frame_size": 256,
            "hop_length": 128,
            "n_mfcc": 13,
            "n_fft": 512
        },
        "power": {
            "budget_mw": 10.0,
            "measurement_enabled": False
        },
        "performance": {
            "max_latency_ms": 100,
            "min_accuracy": 0.8
        }
    }


def _get_standard_config() -> Dict[str, Any]:
    """Standard configuration for development testing."""
    return {
        "model": {
            "input_dim": 40,
            "hidden_dim": 64,
            "output_dim": 10,
            "ode_solver": "adaptive_heun",
            "timestep_range": (0.001, 0.050),
            "complexity_penalty": 0.01,
            "sparsity": 0.3
        },
        "audio": {
            "sample_rate": 16000,
            "frame_size": 512,
            "hop_length": 256,
            "n_mfcc": 40,
            "n_fft": 1024,
            "window": "hann",
            "pre_emphasis": 0.97
        },
        "power": {
            "budget_mw": 5.0,
            "measurement_enabled": True,
            "adaptive_scaling": True
        },
        "performance": {
            "max_latency_ms": 50,
            "min_accuracy": 0.9,
            "batch_size": 1
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 100,
            "early_stopping": True,
            "patience": 10
        }
    }


def _get_embedded_stm32_config() -> Dict[str, Any]:
    """Configuration optimized for STM32F4 embedded deployment."""
    return {
        "model": {
            "input_dim": 13,
            "hidden_dim": 32,
            "output_dim": 5,
            "ode_solver": "euler",  # Simpler solver for embedded
            "timestep_range": (0.005, 0.020),
            "complexity_penalty": 0.05,
            "quantization": "int8",
            "sparsity": 0.5
        },
        "audio": {
            "sample_rate": 16000,
            "frame_size": 256,
            "hop_length": 128,
            "n_mfcc": 13,
            "n_fft": 256,  # Smaller FFT for memory efficiency
            "window": "hann"
        },
        "power": {
            "budget_mw": 2.0,
            "measurement_enabled": True,
            "adaptive_scaling": True,
            "sleep_mode": "deep"
        },
        "memory": {
            "flash_kb": 1024,
            "ram_kb": 192,
            "stack_kb": 32,
            "heap_kb": 16,
            "model_max_kb": 128,
            "buffer_kb": 32
        },
        "hardware": {
            "platform": "stm32f407",
            "cpu_mhz": 168,
            "fpu_enabled": True,
            "cmsis_dsp": True,
            "dma_enabled": True
        },
        "performance": {
            "max_latency_ms": 20,
            "min_accuracy": 0.85,
            "power_efficiency_priority": True
        }
    }


def _get_embedded_nrf52_config() -> Dict[str, Any]:
    """Configuration optimized for nRF52840 embedded deployment."""
    return {
        "model": {
            "input_dim": 13,
            "hidden_dim": 24,
            "output_dim": 3,
            "ode_solver": "euler",
            "timestep_range": (0.010, 0.050),
            "complexity_penalty": 0.1,
            "quantization": "int8",
            "sparsity": 0.6
        },
        "audio": {
            "sample_rate": 16000,
            "frame_size": 256,
            "hop_length": 128,
            "n_mfcc": 13,
            "n_fft": 256,
            "window": "hann"
        },
        "power": {
            "budget_mw": 1.0,  # Even more aggressive for nRF52
            "measurement_enabled": True,
            "adaptive_scaling": True,
            "sleep_mode": "system_off"
        },
        "memory": {
            "flash_kb": 1024,
            "ram_kb": 256,
            "stack_kb": 16,
            "heap_kb": 8,
            "model_max_kb": 64,
            "buffer_kb": 16
        },
        "hardware": {
            "platform": "nrf52840",
            "cpu_mhz": 64,
            "fpu_enabled": True,
            "radio_enabled": True,
            "ble_integration": True
        },
        "performance": {
            "max_latency_ms": 30,
            "min_accuracy": 0.8,
            "ultra_low_power": True
        }
    }


def _get_high_performance_config() -> Dict[str, Any]:
    """Configuration for high-performance testing (accuracy over power)."""
    return {
        "model": {
            "input_dim": 80,
            "hidden_dim": 128,
            "output_dim": 20,
            "ode_solver": "rk4",
            "timestep_range": (0.0005, 0.010),
            "complexity_penalty": 0.001,
            "sparsity": 0.1
        },
        "audio": {
            "sample_rate": 22050,
            "frame_size": 1024,
            "hop_length": 512,
            "n_mfcc": 80,
            "n_fft": 2048,
            "window": "hann",
            "pre_emphasis": 0.97
        },
        "power": {
            "budget_mw": 50.0,  # Higher power budget
            "measurement_enabled": True,
            "adaptive_scaling": False  # Consistent performance
        },
        "performance": {
            "max_latency_ms": 10,
            "min_accuracy": 0.95,
            "batch_size": 8
        }
    }


def _get_ultra_low_power_config() -> Dict[str, Any]:
    """Configuration for ultra-low power testing."""
    return {
        "model": {
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 2,
            "ode_solver": "euler",
            "timestep_range": (0.020, 0.100),  # Large timesteps
            "complexity_penalty": 0.2,
            "quantization": "int4",  # Aggressive quantization
            "sparsity": 0.8
        },
        "audio": {
            "sample_rate": 8000,  # Lower sample rate
            "frame_size": 128,
            "hop_length": 64,
            "n_mfcc": 8,
            "n_fft": 128,
            "window": "hann"
        },
        "power": {
            "budget_mw": 0.5,  # Extreme power constraint
            "measurement_enabled": True,
            "adaptive_scaling": True,
            "sleep_mode": "deep",
            "duty_cycle": 0.1  # Only active 10% of time
        },
        "performance": {
            "max_latency_ms": 200,  # Relaxed latency
            "min_accuracy": 0.7,    # Relaxed accuracy
            "power_efficiency_priority": True
        }
    }


def _get_testing_config() -> Dict[str, Any]:
    """Configuration optimized for fast testing."""
    return {
        "model": {
            "input_dim": 8,
            "hidden_dim": 8,
            "output_dim": 2,
            "ode_solver": "euler",
            "timestep_range": (0.01, 0.02),
            "complexity_penalty": 0.0
        },
        "audio": {
            "sample_rate": 8000,
            "frame_size": 128,
            "hop_length": 64,
            "n_mfcc": 8,
            "n_fft": 128
        },
        "power": {
            "budget_mw": 100.0,  # No power constraints for testing
            "measurement_enabled": False
        },
        "performance": {
            "max_latency_ms": 1000,  # No latency constraints
            "min_accuracy": 0.5,     # Minimal accuracy requirement
            "fast_testing": True
        },
        "training": {
            "learning_rate": 0.01,
            "batch_size": 4,
            "max_epochs": 5,  # Minimal training
            "early_stopping": False
        }
    }


def generate_config_variants(base_config: str, variants: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Generate configuration variants for testing robustness.
    
    Args:
        base_config: Base configuration name
        variants: List of variant types ('noisy', 'quantized', 'sparse', etc.)
        
    Returns:
        Dictionary mapping variant names to configurations
    """
    base = get_test_config(base_config)
    configs = {"base": base}
    
    for variant in variants:
        config = base.copy()
        
        if variant == "noisy":
            # Add noise to training
            config["training"] = config.get("training", {})
            config["training"]["noise_injection"] = True
            config["training"]["noise_std"] = 0.01
            
        elif variant == "quantized":
            # More aggressive quantization
            config["model"]["quantization"] = "int4"
            config["model"]["quantization_aware"] = True
            
        elif variant == "sparse":
            # Increase sparsity
            config["model"]["sparsity"] = min(0.9, config["model"].get("sparsity", 0.3) + 0.3)
            
        elif variant == "low_power":
            # Reduce power budget
            config["power"]["budget_mw"] *= 0.5
            config["model"]["timestep_range"] = (
                config["model"]["timestep_range"][0] * 2,
                config["model"]["timestep_range"][1] * 2
            )
            
        elif variant == "high_accuracy":
            # Optimize for accuracy
            config["model"]["hidden_dim"] = int(config["model"]["hidden_dim"] * 1.5)
            config["model"]["complexity_penalty"] *= 0.5
            config["performance"]["min_accuracy"] += 0.05
            
        configs[variant] = config
    
    return configs


def save_config_to_file(config: Dict[str, Any], filepath: Path) -> None:
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config_from_file(filepath: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)