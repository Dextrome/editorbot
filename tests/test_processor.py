"""Tests for audio processing functionality."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.audio.processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor(sample_rate=44100)
        # Create a simple sine wave for testing
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(44100 * duration), dtype=np.float32)
        self.test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    def test_init(self):
        """Test processor initialization."""
        assert self.processor.sample_rate == 44100
        assert self.processor.audio_data is None

    def test_normalize(self):
        """Test audio normalization."""
        # Create audio with peak at 0.5
        audio = self.test_audio * 0.5
        normalized = self.processor.normalize(audio)
        
        assert np.max(np.abs(normalized)) == pytest.approx(1.0, rel=1e-5)

    def test_normalize_zero_audio(self):
        """Test normalization of silent audio."""
        silent_audio = np.zeros(1000)
        normalized = self.processor.normalize(silent_audio)
        
        assert np.all(normalized == 0)

    @patch('src.audio.processor.librosa')
    def test_trim_silence(self, mock_librosa):
        """Test silence trimming."""
        mock_librosa.effects.trim.return_value = (self.test_audio[:1000], (0, 1000))
        
        trimmed = self.processor.trim_silence(self.test_audio)
        
        mock_librosa.effects.trim.assert_called_once()
        assert len(trimmed) == 1000

    @patch('src.audio.processor.librosa')
    def test_resample(self, mock_librosa):
        """Test audio resampling."""
        target_sr = 22050
        expected_audio = np.zeros(int(len(self.test_audio) * target_sr / 44100))
        mock_librosa.resample.return_value = expected_audio
        
        self.processor.audio_data = self.test_audio
        resampled = self.processor.resample(target_sr)
        
        mock_librosa.resample.assert_called_once()

    def test_normalize_no_data_raises(self):
        """Test that normalize raises error when no data available."""
        with pytest.raises(ValueError, match="No audio data"):
            self.processor.normalize()


class TestAudioProcessorLoad:
    """Test cases for AudioProcessor load functionality."""

    @patch('src.audio.processor.librosa')
    def test_load_success(self, mock_librosa):
        """Test successful audio loading."""
        mock_audio = np.zeros(44100)
        mock_librosa.load.return_value = (mock_audio, 44100)
        
        processor = AudioProcessor()
        
        with patch('pathlib.Path.exists', return_value=True):
            audio, sr = processor.load("test.wav")
        
        assert sr == 44100
        assert len(audio) == 44100
        assert processor.audio_data is not None

    def test_load_file_not_found(self):
        """Test loading non-existent file raises error."""
        processor = AudioProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load("nonexistent.wav")


class TestAudioProcessorSave:
    """Test cases for AudioProcessor save functionality."""

    @patch('src.audio.processor.sf')
    def test_save_success(self, mock_sf):
        """Test successful audio saving."""
        processor = AudioProcessor()
        audio = np.zeros(44100)
        
        with patch('pathlib.Path.mkdir'):
            processor.save("output.wav", audio)
        
        mock_sf.write.assert_called_once()

    def test_save_no_data_raises(self):
        """Test that save raises error when no data available."""
        processor = AudioProcessor()
        
        with pytest.raises(ValueError, match="No audio data"):
            processor.save("output.wav")
