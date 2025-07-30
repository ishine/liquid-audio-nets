# Architecture Overview

This document describes the architecture and design decisions for liquid-audio-nets, an edge-efficient Liquid Neural Network implementation for audio processing.

## System Architecture

```
┌───────────────────────────────────┐
│          Python API Layer          │
│  ┌─────────────────────────────┐  │
│  │     Training Framework     │  │
│  │  (PyTorch + Lightning)   │  │
│  └─────────────────────────────┘  │
├───────────────────────────────────┤
│         Rust Core Library         │
│  ┌───────────┐ ┌─────────────────┐  │
│  │ LNN Core │ │ Adaptive Solver │  │
│  └───────────┘ └─────────────────┘  │
│  ┌───────────┐ ┌─────────────────┐  │
│  │ Audio DSP │ │ Power Manager  │  │
│  └───────────┘ └─────────────────┘  │
├───────────────────────────────────┤
│       C/C++ Platform Layer        │
│  ┌───────────┐ ┌─────────────────┐  │
│  │ ARM CMSIS │ │ Hardware Impl  │  │
│  └───────────┘ └─────────────────┘  │
└───────────────────────────────────┘
```

## Core Components

### 1. Liquid Neural Network Core

The heart of the system implementing continuous-time neural dynamics:

```rust
pub struct LNN {
    pub layers: Vec<LiquidLayer>,
    pub ode_solver: AdaptiveODESolver,
    pub power_manager: PowerManager,
    pub config: LNNConfig,
}

pub struct LiquidLayer {
    pub weights: Matrix<f32>,
    pub tau: Vector<f32>,        // Time constants
    pub activation: ActivationFn,
    pub connectivity: SparseMatrix<f32>,
}
```

**Key Features:**
- Continuous-time dynamics using ODEs
- Adaptive timestep control
- Sparse connectivity patterns
- Power-aware computation

### 2. Adaptive ODE Solver

Handles the numerical integration of neural dynamics:

```rust
pub struct AdaptiveODESolver {
    pub method: ODEMethod,           // Heun, RK4, etc.
    pub timestep_controller: TimestepController,
    pub error_tolerance: f32,
    pub min_timestep: f32,
    pub max_timestep: f32,
}

pub struct TimestepController {
    pub complexity_estimator: ComplexityMetric,
    pub power_budget: f32,
    pub adaptive_policy: AdaptivePolicy,
}
```

**Adaptive Strategy:**
1. Estimate signal complexity (spectral flux, energy)
2. Adjust timestep based on complexity and power budget
3. Larger timesteps for simple signals = less computation
4. Smaller timesteps for complex signals = better accuracy

### 3. Audio Processing Pipeline

```
Audio Input → Feature Extraction → LNN Processing → Decision Output
    │               │                    │              │
   ADC         FFT/MFCC/Mel      Liquid States    Classification
```

**Feature Extraction:**
- **MFCC**: Mel-frequency cepstral coefficients
- **Log Mel**: Log mel-scale spectrograms  
- **Raw Audio**: Direct waveform processing
- **Spectral Features**: Flux, rolloff, centroid

**Processing Modes:**
- **Keyword Spotting**: Wake word detection
- **Voice Activity Detection**: Speech/non-speech
- **Audio Event Classification**: General audio events
- **Anomaly Detection**: Unusual audio patterns

### 4. Power Management

```rust
pub struct PowerManager {
    pub current_power_mw: f32,
    pub power_budget_mw: f32,
    pub power_states: Vec<PowerState>,
    pub adaptive_controller: AdaptivePowerController,
}

pub enum PowerState {
    Active,      // Full processing
    Reduced,     // Lower sample rate/features
    Minimal,     // Basic detection only
    Sleep,       // Minimal power consumption
}
```

**Power Optimization Strategies:**
- Dynamic voltage/frequency scaling
- Adaptive feature extraction depth
- Conditional layer activation
- Sleep mode between audio events

## Multi-Language Architecture

### Language Separation of Concerns

**Python Layer:**
- High-level API and configuration
- Training framework integration
- Data preprocessing and augmentation
- Model analysis and visualization
- Deployment tools and utilities

**Rust Core:**
- Numerical computation kernels
- Memory-safe systems programming
- Cross-compilation for embedded targets
- Zero-cost abstractions
- Parallel processing with Rayon

**C/C++ Platform:**
- Hardware-specific optimizations
- ARM CMSIS-DSP integration
- Real-time embedded implementations
- Legacy system compatibility
- Direct hardware access

### Inter-Language Communication

```
Python ←─── PyO3 ───→ Rust ←─── FFI ───→ C/C++
```

**Python ↔ Rust (PyO3):**
```rust
#[pyclass]
pub struct LNN {
    inner: RustLNN,
}

#[pymethods]
impl LNN {
    #[new]
    pub fn new() -> Self {
        LNN { inner: RustLNN::new() }
    }
    
    pub fn process(&mut self, audio: Vec<f32>) -> PyResult<ProcessingResult> {
        let result = self.inner.process(&audio)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(result.into())
    }
}
```

**Rust ↔ C (FFI):**
```rust
#[no_mangle]
pub extern "C" fn lnn_create() -> *mut LNN {
    Box::into_raw(Box::new(LNN::new()))
}

#[no_mangle]
pub extern "C" fn lnn_process(
    lnn: *mut LNN,
    audio: *const f32,
    len: usize,
    result: *mut ProcessingResult
) -> i32 {
    // Safe wrapper around Rust implementation
}
```

## Embedded Architecture

### Target Platforms

**STM32F4 (Primary):**
- ARM Cortex-M4F @ 168MHz
- 1MB Flash, 192KB RAM
- Hardware FPU
- CMSIS-DSP optimization

**nRF52840 (Secondary):**
- ARM Cortex-M4F @ 64MHz
- 1MB Flash, 256KB RAM
- Low-power radio
- Bluetooth integration

**ESP32-S3 (Experimental):**
- Xtensa LX7 @ 240MHz
- 8MB Flash, 512KB RAM
- WiFi connectivity
- Hardware acceleration

### Memory Architecture

```
Flash Memory Layout:
┌────────────────────┐
│    Bootloader (16KB)    │
├────────────────────┤
│  Application (512KB)   │
├────────────────────┤
│  LNN Models (256KB)    │
├────────────────────┤
│ Configuration (16KB)  │
├────────────────────┤
│    Reserved (200KB)    │
└────────────────────┘

RAM Memory Layout:
┌────────────────────┐
│     Stack (32KB)      │
├────────────────────┤
│  Audio Buffer (32KB)  │
├────────────────────┤
│ Neural State (64KB)   │
├────────────────────┤
│ Working Memory (48KB) │
├────────────────────┤
│     Heap (16KB)       │
└────────────────────┘
```

### Real-time Processing

```c
// Interrupt-driven audio processing
void AUDIO_DMA_IRQHandler(void) {
    if (audio_buffer_ready) {
        // Process in background task
        osSignalSet(audio_task_id, AUDIO_READY_SIGNAL);
    }
}

void audio_processing_task(void const *argument) {
    while (1) {
        osSignalWait(AUDIO_READY_SIGNAL, osWaitForever);
        
        // Process audio with LNN
        lnn_result_t result;
        lnn_process_audio(audio_buffer, BUFFER_SIZE, &result);
        
        if (result.detection_confidence > threshold) {
            // Trigger action
            trigger_wake_sequence();
        }
        
        // Enter low-power mode
        enter_sleep_mode();
    }
}
```

## Performance Characteristics

### Computational Complexity

**Traditional RNN/LSTM:**
- O(n²) matrix operations per timestep
- Fixed computation regardless of input
- High memory bandwidth requirements

**Liquid Neural Network:**
- O(n·s) where s is sparsity (s << n)
- Adaptive computation based on signal complexity
- Reduced memory access through state persistence

### Power Scaling

```
Power(t) = P_base + P_dynamic(complexity(t), timestep(t))

where:
- P_base: Base power consumption (~0.1 mW)
- P_dynamic: Varies from 0.5-2.5 mW based on signal
- complexity(t): Real-time signal complexity measure
- timestep(t): Adaptive timestep (1-50ms)
```

### Memory Usage

**Model Storage:**
- Weights: 32-128KB (quantized to int8/int16)
- Architecture: 1-4KB (sparse connectivity)
- Configuration: < 1KB

**Runtime Memory:**
- Neural states: 8-32KB
- Audio buffers: 16-64KB  
- Working memory: 8-16KB

## Security Architecture

### Model Protection

```rust
pub struct SecureModel {
    encrypted_weights: Vec<u8>,
    signature: [u8; 32],
    key_derivation: KeyDerivation,
}

impl SecureModel {
    pub fn load_and_verify(&self, device_key: &[u8]) -> Result<LNN, SecurityError> {
        // Verify signature
        if !verify_signature(&self.encrypted_weights, &self.signature) {
            return Err(SecurityError::InvalidSignature);
        }
        
        // Decrypt with device-specific key
        let weights = decrypt_weights(&self.encrypted_weights, device_key)?;
        
        Ok(LNN::from_weights(weights)?)
    }
}
```

### Input Validation

```rust
pub fn validate_audio_input(audio: &[f32]) -> Result<(), ValidationError> {
    // Check bounds
    if audio.len() > MAX_AUDIO_LEN {
        return Err(ValidationError::TooLong);
    }
    
    // Check for anomalous values
    for &sample in audio {
        if !sample.is_finite() || sample.abs() > MAX_AMPLITUDE {
            return Err(ValidationError::InvalidSample);
        }
    }
    
    // Check for potential adversarial patterns
    if detect_adversarial_pattern(audio) {
        return Err(ValidationError::AdversarialInput);
    }
    
    Ok(())
}
```

## Testing Architecture

### Multi-level Testing

```
Unit Tests → Integration Tests → Hardware Tests → System Tests
    │               │                  │               │
 Individual     Cross-language    Real hardware   End-to-end
 functions      interactions      validation      scenarios
```

### Continuous Integration

```yaml
# Testing matrix
Matrix:
  - Python: [3.8, 3.9, 3.10, 3.11]
  - Rust: [1.70, stable, beta]
  - Platform: [x86_64, aarch64, thumbv7em]
  - Features: [std, no_std, embedded]
```

## Future Architecture Considerations

### Scalability
- **Distributed Processing**: Multi-core and cluster deployment
- **Federated Learning**: Distributed model training
- **Edge-Cloud Hybrid**: Selective cloud offloading

### Hardware Acceleration
- **NPU Integration**: Neural Processing Unit support
- **Custom Silicon**: ASIC/FPGA implementations
- **GPU Acceleration**: CUDA/OpenCL backends

### Advanced Features
- **Self-Learning**: Online adaptation and learning
- **Meta-Learning**: Few-shot learning capabilities
- **Neuromorphic Computing**: Spiking neural network integration

This architecture enables efficient, secure, and scalable audio processing across diverse deployment scenarios while maintaining the flexibility for future enhancements.