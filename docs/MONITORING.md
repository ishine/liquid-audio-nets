# Monitoring and Observability

This document describes the monitoring and observability setup for liquid-audio-nets.

## Overview

The monitoring system tracks:
- **Performance**: Inference latency, throughput, memory usage
- **Power Consumption**: Real-time power monitoring for embedded devices
- **Model Quality**: Accuracy, loss metrics, drift detection
- **Hardware Health**: Temperature, CPU/memory usage, device status
- **Training Progress**: Loss convergence, validation metrics

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Application   │───▶│  Prometheus  │───▶│   Grafana   │
│    Metrics      │    │   (Storage)  │    │ (Dashboard) │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Alertmanager │
                       │  (Alerts)    │
                       └──────────────┘
```

## Metrics Categories

### Performance Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| `inference_latency_ms` | Time to process one audio frame | < 20ms |
| `throughput_samples_per_sec` | Audio samples processed per second | > 16kHz |
| `memory_usage_mb` | RAM usage during inference | < 10MB |
| `cpu_utilization_percent` | CPU usage percentage | < 50% |

### Power Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| `power_consumption_mw` | Real-time power draw | < 2mW |
| `energy_per_inference_uj` | Energy per inference operation | < 100μJ |
| `battery_level_percent` | Remaining battery charge | Monitor |
| `estimated_battery_hours` | Remaining battery life | > 48h |

### Model Quality Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| `model_accuracy` | Classification accuracy | > 90% |
| `false_positive_rate` | Rate of false classifications | < 5% |
| `confidence_score` | Average prediction confidence | > 0.8 |
| `drift_score` | Model drift detection | < 0.1 |

### Hardware Metrics

| Metric | Description | Alerts |
|--------|-------------|---------|
| `device_temperature_celsius` | MCU temperature | > 70°C |
| `memory_available_kb` | Available RAM | < 1KB |
| `flash_usage_percent` | Flash memory usage | > 90% |
| `device_uptime_seconds` | Device uptime | Monitor |

## Setup Instructions

### 1. Local Development

```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Access dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### 2. Production Deployment

```bash
# Deploy monitoring to Kubernetes
kubectl apply -f monitoring/k8s/

# Or use Helm
helm install monitoring monitoring/helm-chart/
```

### 3. Embedded Device Setup

```c
// Include metrics library
#include "liquid_audio_metrics.h"

// Initialize metrics
lnn_metrics_init();

// Record metrics during inference
void process_audio_frame(float* audio_data) {
    uint32_t start_time = HAL_GetTick();
    
    // Process audio
    lnn_result_t result = lnn_inference(audio_data);
    
    uint32_t end_time = HAL_GetTick();
    
    // Record metrics
    lnn_metrics_record_latency(end_time - start_time);
    lnn_metrics_record_confidence(result.confidence);
    lnn_metrics_record_power(get_power_consumption());
}
```

## Dashboard Configuration

### Grafana Dashboards

1. **Real-time Performance Dashboard**
   - Inference latency over time
   - Throughput vs. real-time requirements
   - Memory usage trends
   - CPU utilization

2. **Power Efficiency Dashboard**
   - Power consumption over time
   - Energy per operation
   - Battery life estimation
   - Power breakdown by component

3. **Model Quality Dashboard**
   - Accuracy trends
   - Confidence score distribution
   - Error rate analysis
   - Model drift detection

4. **Hardware Health Dashboard**
   - Temperature monitoring
   - Resource utilization
   - Device status
   - Network connectivity

### Key Panels

```json
{
  "dashboard": {
    "title": "Liquid Audio Nets - Performance",
    "panels": [
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, inference_latency_ms)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, inference_latency_ms)",
            "legendFormat": "Median"
          }
        ]
      }
    ]
  }
}
```

## Alerting Rules

### Critical Alerts

- **DeviceOffline**: Embedded device unreachable
- **LowThroughput**: Processing below real-time
- **LowModelAccuracy**: Model performance degraded

### Warning Alerts

- **HighInferenceLatency**: Latency above threshold
- **HighPowerConsumption**: Power usage above target
- **HighMemoryUsage**: Memory usage high
- **DeviceTemperatureHigh**: Hardware overheating

### Alert Routing

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'critical-alerts'
  slack_configs:
  - api_url: '$SLACK_WEBHOOK_URL'
    channel: '#alerts-critical'
- name: 'warning-alerts'
  email_configs:
  - to: 'team@company.com'
    subject: 'Warning: {{ .GroupLabels.alertname }}'
```

## Custom Metrics

### Adding Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Time spent on inference')
POWER_CONSUMPTION = Gauge('power_consumption_watts', 'Current power consumption')
ACCURACY_SCORE = Gauge('model_accuracy', 'Current model accuracy')

# Record metrics
with INFERENCE_LATENCY.time():
    result = model.inference(audio_data)

POWER_CONSUMPTION.set(get_power_reading())
ACCURACY_SCORE.set(current_accuracy)
```

### Rust Metrics

```rust
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};

lazy_static! {
    static ref INFERENCE_COUNTER: Counter = register_counter!(
        "inferences_total", "Total number of inferences"
    ).unwrap();
    
    static ref LATENCY_HISTOGRAM: Histogram = register_histogram!(
        "inference_latency_seconds", "Inference latency in seconds"
    ).unwrap();
}

// Record metrics
INFERENCE_COUNTER.inc();
let timer = LATENCY_HISTOGRAM.start_timer();
let result = model.inference(&audio_data);
timer.observe_duration();
```

## Performance Baselines

### Target Metrics

| Metric | STM32F4 | nRF52840 | Desktop |
|--------|---------|----------|---------|
| Latency | < 15ms | < 12ms | < 5ms |
| Power | < 1.2mW | < 0.9mW | N/A |
| Memory | < 64KB | < 32KB | < 100MB |
| Accuracy | > 93.8% | > 93.5% | > 95% |

### Regression Detection

Automated alerts trigger when metrics deviate >10% from baselines:

```yaml
- alert: PerformanceRegression
  expr: |
    (
      inference_latency_ms > 16.5 and on() label_replace(kube_node_labels{label_node_type="stm32f4"}, "device", "$1", "node", ".*")
    ) or (
      inference_latency_ms > 13.2 and on() label_replace(kube_node_labels{label_node_type="nrf52840"}, "device", "$1", "node", ".*")
    )
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Performance regression detected"
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check CPU utilization
   - Verify model quantization
   - Monitor thermal throttling

2. **High Power Consumption**
   - Check adaptive timestep configuration
   - Verify clock settings
   - Monitor peripheral usage

3. **Memory Leaks**
   - Monitor heap usage over time
   - Check for proper cleanup
   - Verify buffer management

### Debug Commands

```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# Query Prometheus
curl "http://localhost:9090/api/v1/query?query=inference_latency_ms"

# Export metrics for analysis
prometheus-cli query --output csv 'inference_latency_ms[1h]' > latency.csv
```

## Integration with CI/CD

### Performance Tests in CI

```yaml
# .github/workflows/performance.yml
- name: Run performance benchmarks
  run: |
    make bench
    python scripts/analyze_performance.py --baseline baseline.json --current results.json
    
- name: Check performance regression
  run: |
    if python scripts/check_regression.py; then
      echo "Performance within acceptable range"
    else
      echo "Performance regression detected"
      exit 1
    fi
```

### Automatic Baseline Updates

```python
# scripts/update_baseline.py
def update_baseline_if_improved():
    current_metrics = load_current_metrics()
    baseline_metrics = load_baseline_metrics()
    
    if all_metrics_improved(current_metrics, baseline_metrics):
        save_baseline_metrics(current_metrics)
        print("Baseline updated with improved performance")
```

This monitoring setup ensures comprehensive observability of liquid-audio-nets performance, power efficiency, and model quality across development, testing, and production environments.