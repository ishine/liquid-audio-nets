"""
Audio sample generation utilities for testing.
"""
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import json


def generate_test_audio(
    signal_type: str,
    duration: float = 1.0,
    sample_rate: int = 16000,
    **kwargs
) -> np.ndarray:
    """
    Generate test audio signals for various testing scenarios.
    
    Args:
        signal_type: Type of signal ('sine', 'noise', 'chirp', 'silence', 'speech_like')
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        **kwargs: Additional parameters specific to signal type
        
    Returns:
        Audio signal as float32 numpy array
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    if signal_type == "sine":
        frequency = kwargs.get("frequency", 440.0)
        amplitude = kwargs.get("amplitude", 0.5)
        return amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
    elif signal_type == "noise":
        noise_type = kwargs.get("noise_type", "white")
        amplitude = kwargs.get("amplitude", 0.1)
        
        if noise_type == "white":
            return amplitude * np.random.normal(0, 1, n_samples).astype(np.float32)
        elif noise_type == "pink":
            # Simple pink noise approximation
            white = np.random.normal(0, 1, n_samples)
            # Apply simple filter to approximate pink noise
            filtered = np.convolve(white, [1, -0.5], mode='same')
            return amplitude * filtered.astype(np.float32)
            
    elif signal_type == "chirp":
        f_start = kwargs.get("f_start", 100.0)
        f_end = kwargs.get("f_end", 2000.0)
        amplitude = kwargs.get("amplitude", 0.5)
        
        # Linear chirp
        instantaneous_freq = f_start + (f_end - f_start) * t / duration
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        return amplitude * np.sin(phase).astype(np.float32)
        
    elif signal_type == "silence":
        return np.zeros(n_samples, dtype=np.float32)
        
    elif signal_type == "speech_like":
        # Generate speech-like signal with formants
        amplitude = kwargs.get("amplitude", 0.3)
        
        # Fundamental frequency (pitch)
        f0 = kwargs.get("f0", 120.0)
        
        # Formant frequencies for vowel-like sound
        formants = kwargs.get("formants", [800, 1200, 2500])
        
        signal = np.zeros(n_samples)
        
        # Generate harmonics with formant emphasis
        for harmonic in range(1, 20):
            freq = f0 * harmonic
            if freq > sample_rate / 2:
                break
                
            # Emphasize frequencies near formants
            gain = 1.0
            for formant in formants:
                if abs(freq - formant) < 200:
                    gain *= 3.0
                    
            gain /= harmonic  # Natural harmonic rolloff
            signal += gain * np.sin(2 * np.pi * freq * t)
            
        # Add some noise for realism
        signal += 0.05 * np.random.normal(0, 1, n_samples)
        
        return amplitude * signal.astype(np.float32)
        
    elif signal_type == "impulse":
        signal = np.zeros(n_samples, dtype=np.float32)
        impulse_pos = kwargs.get("position", n_samples // 2)
        amplitude = kwargs.get("amplitude", 1.0)
        signal[impulse_pos] = amplitude
        return signal
        
    elif signal_type == "step":
        amplitude = kwargs.get("amplitude", 0.5)
        step_pos = kwargs.get("position", n_samples // 2)
        signal = np.zeros(n_samples, dtype=np.float32)
        signal[step_pos:] = amplitude
        return signal
        
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def generate_keyword_audio(
    keyword: str,
    duration: float = 1.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Generate synthetic keyword-like audio for testing wake word detection."""
    
    # Simplified phoneme-based synthesis for common keywords
    phoneme_patterns = {
        "hey": [(400, 0.3), (800, 0.4), (1200, 0.3)],  # Rough approximation
        "okay": [(600, 0.2), (400, 0.3), (800, 0.3), (600, 0.2)],
        "hello": [(500, 0.2), (300, 0.2), (800, 0.3), (600, 0.3)],
        "wake": [(600, 0.3), (400, 0.4), (1000, 0.3)],
        "stop": [(800, 0.3), (600, 0.4), (400, 0.3)]
    }
    
    if keyword not in phoneme_patterns:
        # Generate generic speech-like pattern
        return generate_test_audio("speech_like", duration=duration, sample_rate=sample_rate)
    
    pattern = phoneme_patterns[keyword]
    n_samples = int(duration * sample_rate)
    signal = np.zeros(n_samples, dtype=np.float32)
    
    # Generate phonemes
    samples_per_phoneme = n_samples // len(pattern)
    
    for i, (freq, amplitude) in enumerate(pattern):
        start_idx = i * samples_per_phoneme
        end_idx = min((i + 1) * samples_per_phoneme, n_samples)
        phoneme_duration = (end_idx - start_idx) / sample_rate
        
        phoneme_signal = generate_test_audio(
            "speech_like",
            duration=phoneme_duration,
            sample_rate=sample_rate,
            f0=freq/6,  # Fundamental frequency
            amplitude=amplitude,
            formants=[freq, freq*1.5, freq*2.5]
        )
        
        signal[start_idx:end_idx] = phoneme_signal[:end_idx-start_idx]
    
    return signal


def generate_background_noise(
    noise_type: str,
    duration: float = 1.0,
    sample_rate: int = 16000,
    **kwargs
) -> np.ndarray:
    """Generate various types of background noise for testing robustness."""
    
    if noise_type == "cafe":
        # Simulate cafe background noise
        base_noise = generate_test_audio("noise", duration=duration, 
                                       sample_rate=sample_rate, 
                                       noise_type="pink", amplitude=0.1)
        
        # Add occasional "conversations" (burst of energy)
        n_samples = len(base_noise)
        for _ in range(np.random.randint(3, 8)):
            start = np.random.randint(0, n_samples - sample_rate//2)
            length = np.random.randint(sample_rate//4, sample_rate//2)
            end = min(start + length, n_samples)
            
            conversation = generate_test_audio("speech_like", 
                                             duration=(end-start)/sample_rate,
                                             sample_rate=sample_rate,
                                             amplitude=0.15)
            base_noise[start:end] += conversation[:end-start]
            
        return base_noise.astype(np.float32)
        
    elif noise_type == "traffic":
        # Low-frequency rumble with occasional higher frequency content
        base_rumble = generate_test_audio("noise", duration=duration,
                                        sample_rate=sample_rate,
                                        noise_type="pink", amplitude=0.2)
        
        # Apply low-pass filter for rumble effect
        from scipy import signal as scipy_signal
        sos = scipy_signal.butter(4, 200, btype='low', fs=sample_rate, output='sos')
        filtered = scipy_signal.sosfilt(sos, base_rumble)
        
        return filtered.astype(np.float32)
        
    elif noise_type == "office":
        # Mix of air conditioning hum and occasional keyboard/mouse clicks
        hum = generate_test_audio("sine", duration=duration, sample_rate=sample_rate,
                                frequency=60, amplitude=0.05)
        
        base_noise = generate_test_audio("noise", duration=duration,
                                       sample_rate=sample_rate,
                                       noise_type="white", amplitude=0.03)
        
        # Add occasional clicks
        n_samples = len(base_noise)
        for _ in range(np.random.randint(10, 20)):
            click_pos = np.random.randint(0, n_samples - 100)
            click = generate_test_audio("impulse", duration=0.01, 
                                      sample_rate=sample_rate,
                                      amplitude=0.1, position=50)
            base_noise[click_pos:click_pos+len(click)] += click
            
        return (hum + base_noise).astype(np.float32)
        
    else:
        return generate_test_audio("noise", duration=duration, 
                                 sample_rate=sample_rate, **kwargs)


class AudioTestData:
    """Container for various audio test data scenarios."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def get_clean_speech(self, duration: float = 1.0) -> np.ndarray:
        """Clean speech-like signal."""
        return generate_test_audio("speech_like", duration=duration, 
                                 sample_rate=self.sample_rate)
        
    def get_noisy_speech(self, duration: float = 1.0, snr_db: float = 10.0) -> np.ndarray:
        """Speech with additive noise at specified SNR."""
        speech = self.get_clean_speech(duration)
        noise = generate_test_audio("noise", duration=duration, 
                                  sample_rate=self.sample_rate,
                                  noise_type="white")
        
        # Calculate noise scaling for desired SNR
        speech_power = np.mean(speech**2)
        noise_power = np.mean(noise**2)
        snr_linear = 10**(snr_db/10)
        noise_scale = np.sqrt(speech_power / (snr_linear * noise_power))
        
        return speech + noise_scale * noise
        
    def get_edge_cases(self) -> Dict[str, np.ndarray]:
        """Generate edge case audio for robustness testing."""
        duration = 1.0
        
        return {
            "silence": generate_test_audio("silence", duration=duration, 
                                         sample_rate=self.sample_rate),
            "clipping": np.clip(generate_test_audio("sine", duration=duration,
                                                  sample_rate=self.sample_rate,
                                                  amplitude=2.0), -1.0, 1.0),
            "dc_offset": generate_test_audio("sine", duration=duration,
                                           sample_rate=self.sample_rate) + 0.5,
            "very_quiet": generate_test_audio("speech_like", duration=duration,
                                            sample_rate=self.sample_rate,
                                            amplitude=0.001),
            "high_frequency": generate_test_audio("sine", duration=duration,
                                                sample_rate=self.sample_rate,
                                                frequency=7000, amplitude=0.3)
        }