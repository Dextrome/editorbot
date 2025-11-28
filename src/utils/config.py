"""Configuration management for the AI audio editor."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import json


@dataclass
class Config:
    """Configuration settings for the audio editor."""

    # Audio settings
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2

    # Processing settings
    default_preset: str = "balanced"
    normalize_output: bool = True
    trim_silence: bool = True
    silence_threshold_db: float = -40.0

    # AI settings
    use_gpu: bool = True
    model_path: Optional[str] = None
    batch_size: int = 1

    # Output settings
    output_format: str = "wav"
    output_quality: int = 320  # For lossy formats (kbps)

    # Paths
    temp_dir: str = "./temp"
    output_dir: str = "./output"

    # Effect defaults
    eq_settings: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.0, "mid": 0.0, "high": 0.0}
    )
    compression_settings: Dict[str, float] = field(
        default_factory=lambda: {
            "threshold": -18.0,
            "ratio": 4.0,
            "attack": 5.0,
            "release": 50.0,
        }
    )
    reverb_settings: Dict[str, float] = field(
        default_factory=lambda: {"room_size": 0.4, "damping": 0.5, "wet": 0.2}
    )

    @classmethod
    def from_file(cls, config_path: str | Path) -> "Config":
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Config instance with loaded settings.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def to_file(self, config_path: str | Path) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config_path: Path to save the configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "default_preset": self.default_preset,
            "normalize_output": self.normalize_output,
            "trim_silence": self.trim_silence,
            "silence_threshold_db": self.silence_threshold_db,
            "use_gpu": self.use_gpu,
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "output_format": self.output_format,
            "output_quality": self.output_quality,
            "temp_dir": self.temp_dir,
            "output_dir": self.output_dir,
            "eq_settings": self.eq_settings,
            "compression_settings": self.compression_settings,
            "reverb_settings": self.reverb_settings,
        }

    def update(self, **kwargs: Any) -> None:
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration values to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
