"""Basic tests for liquid-audio-nets package."""

import pytest
import numpy as np
from liquid_audio_nets import LNN, AdaptiveConfig


class TestLNN:
    """Test cases for LNN class."""
    
    def test_lnn_initialization(self):
        """Test LNN can be initialized."""
        lnn = LNN()
        assert lnn is not None
        assert lnn.current_power_mw() == 0.0
    
    def test_adaptive_config(self):
        """Test adaptive configuration."""
        config = AdaptiveConfig(
            min_timestep=0.001,
            max_timestep=0.050,
            energy_threshold=0.1
        )
        
        lnn = LNN()
        lnn.set_adaptive_config(config)
        
        assert lnn._config == config
    
    def test_audio_processing(self):
        """Test basic audio processing."""
        lnn = LNN()
        audio_buffer = np.random.randn(256).astype(np.float32)
        
        result = lnn.process(audio_buffer)
        
        assert isinstance(result, dict)
        assert "keyword_detected" in result
        assert "confidence" in result
        assert "power_mw" in result
        assert result["power_mw"] > 0
    
    def test_activity_detection(self):
        """Test voice activity detection."""
        lnn = LNN()
        audio_frame = np.random.randn(160).astype(np.float32)
        
        result = lnn.detect_activity(audio_frame)
        
        assert isinstance(result, dict)
        assert "is_speech" in result
        assert "energy" in result
        assert "confidence" in result
    
    def test_power_estimation(self):
        """Test power estimation scales with signal energy."""
        lnn = LNN()
        
        # Low energy signal
        low_energy = np.random.randn(256) * 0.01
        result_low = lnn.process(low_energy)
        
        # High energy signal  
        high_energy = np.random.randn(256) * 1.0
        result_high = lnn.process(high_energy)
        
        assert result_high["power_mw"] > result_low["power_mw"]


class TestAdaptiveConfig:
    """Test cases for AdaptiveConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveConfig()
        
        assert config.min_timestep == 0.001
        assert config.max_timestep == 0.050
        assert config.energy_threshold == 0.1
        assert config.complexity_metric == "spectral_flux"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdaptiveConfig(
            min_timestep=0.002,
            max_timestep=0.100,
            energy_threshold=0.2,
            complexity_metric="mfcc_variance"
        )
        
        assert config.min_timestep == 0.002
        assert config.max_timestep == 0.100
        assert config.energy_threshold == 0.2
        assert config.complexity_metric == "mfcc_variance"