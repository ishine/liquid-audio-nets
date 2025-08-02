# Test Fixtures

This directory contains test data fixtures and sample files for testing liquid-audio-nets.

## Structure

```
fixtures/
├── README.md              # This file
├── audio/                 # Audio test samples
│   ├── samples.py        # Generate audio samples
│   ├── keyword_clips/    # Wake word samples
│   └── noise_clips/      # Background noise samples
├── models/               # Pre-trained test models
│   ├── tiny_model.lnn    # Minimal test model
│   └── reference.lnn     # Reference accuracy model
├── configs/              # Test configurations
│   ├── test_configs.py   # Configuration generators
│   └── hardware_profiles.json
└── datasets/             # Small test datasets
    ├── speech_commands/  # Google Speech Commands subset
    └── environmental/    # Environmental audio samples
```

## Usage

Import fixtures in tests:

```python
from tests.fixtures.audio.samples import generate_test_audio
from tests.fixtures.models import load_test_model
from tests.fixtures.configs import get_test_config

def test_audio_processing():
    audio = generate_test_audio("sine_wave", duration=1.0)
    model = load_test_model("tiny_model")
    config = get_test_config("embedded_stm32")
    
    result = model.process(audio, config)
    assert result.is_valid()
```

## Adding New Fixtures

1. Keep fixtures small (<1MB) to avoid repository bloat
2. Use synthetic data when possible
3. Document data generation methods
4. Include both positive and negative test cases
5. Consider different hardware constraints

## Test Data Sources

- **Synthetic Audio**: Generated sine waves, noise, chirps
- **Speech Commands**: Subset of Google Speech Commands dataset
- **Environmental**: Urban, nature, industrial soundscapes  
- **Edge Cases**: Silence, clipping, unusual frequencies