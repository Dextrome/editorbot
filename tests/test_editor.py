"""Tests for the AI editor module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.ai.editor import AIEditor


class TestAIEditor:
    """Test cases for AIEditor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.editor = AIEditor(sample_rate=44100)
        # Create test audio
        duration = 2.0
        t = np.linspace(0, duration, int(44100 * duration), dtype=np.float32)
        self.test_audio = np.sin(2 * np.pi * 440 * t) * 0.5

    def test_init(self):
        """Test editor initialization."""
        assert self.editor.sample_rate == 44100
        assert self.editor.processor is not None
        assert self.editor.analyzer is not None
        assert self.editor.effects is not None

    @patch.object(AIEditor, '_generate_recommendations')
    def test_analyze_recording(self, mock_recommend):
        """Test recording analysis."""
        mock_recommend.return_value = ["normalize_audio"]
        
        with patch.object(self.editor.analyzer, 'detect_tempo', return_value=120.0):
            with patch.object(self.editor.analyzer, 'detect_key', return_value="C major"):
                with patch.object(self.editor.analyzer, 'get_loudness', return_value=-12.0):
                    with patch.object(self.editor.analyzer, 'detect_beats', return_value=np.array([0.5, 1.0, 1.5])):
                        with patch.object(self.editor.analyzer, 'extract_features', return_value={"mfcc": np.zeros((13, 10))}):
                            with patch.object(self.editor.analyzer, 'detect_sections', return_value=(np.array([0, 1]), np.array([0, 1]))):
                                analysis = self.editor.analyze_recording(self.test_audio)

        assert "tempo" in analysis
        assert "key" in analysis
        assert "loudness" in analysis
        assert "recommendations" in analysis

    def test_get_preset_params_balanced(self):
        """Test balanced preset parameters."""
        params = self.editor._get_preset_params("balanced")
        
        assert "gate_threshold" in params
        assert "comp_threshold" in params
        assert "eq_low" in params
        assert params["reverb_wet"] == 0.2

    def test_get_preset_params_warm(self):
        """Test warm preset parameters."""
        params = self.editor._get_preset_params("warm")
        
        assert params["eq_low"] == 3.0
        assert params["eq_high"] == -1.0

    def test_get_preset_params_unknown_returns_balanced(self):
        """Test unknown preset returns balanced."""
        params = self.editor._get_preset_params("unknown")
        balanced = self.editor._get_preset_params("balanced")
        
        assert params == balanced

    def test_generate_recommendations_quiet_audio(self):
        """Test recommendations for quiet audio."""
        analysis = {
            "loudness": -25.0,
            "features": {"rms": np.array([0.01, 0.02, 0.01])}
        }
        
        recommendations = self.editor._generate_recommendations(analysis)
        
        assert "normalize_audio" in recommendations
        assert "apply_compression" in recommendations

    def test_generate_recommendations_loud_audio(self):
        """Test recommendations for loud audio."""
        analysis = {
            "loudness": -3.0,
            "features": {"rms": np.array([0.5, 0.5, 0.5])}
        }
        
        recommendations = self.editor._generate_recommendations(analysis)
        
        assert "reduce_gain" in recommendations
        assert "normalize_audio" not in recommendations
