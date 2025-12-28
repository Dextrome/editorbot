"""Feature caching system for audio processing.

Provides disk-based caching for all computed features:
- Beat-level features (spectral, MFCCs, chroma, etc.)
- Stem separation (Demucs)
- Mel spectrograms
- Edit labels

This dramatically speeds up training by avoiding recomputation.
Shared between rl_editor and super_editor.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default cache directory (under shared module)
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"


class FeatureCache:
    """Disk-based cache for audio features.

    Cache structure:
        cache_dir/
            features/           # Beat-level features
                {hash}.npz
            stems/              # Demucs stem separation
                {hash}_stems.npz
            mel/                # Mel spectrograms
                {hash}_mel.npz
            labels/             # Edit labels
                {hash}_labels.npz
            metadata/           # Track metadata
                {hash}_meta.json
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        enabled: bool = True,
    ):
        """Initialize cache.

        Args:
            cache_dir: Cache directory (default: editorbot/cache/)
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.enabled = enabled

        if enabled:
            self._ensure_dirs()
            logger.info(f"Feature cache initialized at: {self.cache_dir}")

    def _ensure_dirs(self) -> None:
        """Ensure cache directories exist."""
        for subdir in ["features", "stems", "mel", "labels", "metadata", "full"]:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _get_file_hash(self, filepath: Union[str, Path]) -> str:
        """Generate hash for a file based on path and modification time.

        This ensures cache invalidation when files change.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            # Use just the name for non-existent files (e.g., temp arrays)
            return hashlib.md5(str(filepath).encode()).hexdigest()[:16]

        # Hash based on: absolute path + file size + modification time
        stat = filepath.stat()
        key = f"{filepath.absolute()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _get_array_hash(self, arr: np.ndarray, name: str = "") -> str:
        """Generate hash for a numpy array."""
        key = f"{name}|{arr.shape}|{arr.dtype}|{arr.sum():.6f}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    # === Beat Features ===

    def get_features_path(self, audio_path: Union[str, Path]) -> Path:
        """Get cache path for beat features."""
        file_hash = self._get_file_hash(audio_path)
        name = Path(audio_path).stem
        return self.cache_dir / "features" / f"{name}_{file_hash}.npz"

    def load_features(
        self,
        audio_path: Union[str, Path],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load cached beat features.

        Returns:
            Dict with 'beat_features', 'beat_times', 'beats', 'tempo', or None
        """
        if not self.enabled:
            return None

        cache_path = self.get_features_path(audio_path)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            result = {key: data[key] for key in data.files}
            logger.debug(f"Loaded cached features from {cache_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load cached features: {e}")
            return None

    def save_features(
        self,
        audio_path: Union[str, Path],
        beat_features: np.ndarray,
        beat_times: np.ndarray,
        beats: np.ndarray,
        tempo: float,
        feature_config: Optional[Dict] = None,
    ) -> None:
        """Save beat features to cache."""
        if not self.enabled:
            return

        cache_path = self.get_features_path(audio_path)
        try:
            np.savez_compressed(
                cache_path,
                beat_features=beat_features,
                beat_times=beat_times,
                beats=beats,
                tempo=np.array(tempo),
                feature_config=json.dumps(feature_config) if feature_config else "",
            )
            logger.debug(f"Saved features to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")

    # === Stem Separation ===

    def get_stems_path(self, audio_path: Union[str, Path]) -> Path:
        """Get cache path for stems."""
        file_hash = self._get_file_hash(audio_path)
        name = Path(audio_path).stem
        return self.cache_dir / "stems" / f"{name}_{file_hash}_stems.npz"

    def load_stems(
        self,
        audio_path: Union[str, Path],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load cached stems.

        Returns:
            Dict mapping stem name to audio array, or None
        """
        if not self.enabled:
            return None

        cache_path = self.get_stems_path(audio_path)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path)
            stems = {key: data[key] for key in data.files if not key.startswith('_')}
            logger.debug(f"Loaded cached stems from {cache_path}")
            return stems
        except Exception as e:
            logger.warning(f"Failed to load cached stems: {e}")
            return None

    def save_stems(
        self,
        audio_path: Union[str, Path],
        stems: Dict[str, np.ndarray],
    ) -> None:
        """Save stems to cache."""
        if not self.enabled:
            return

        cache_path = self.get_stems_path(audio_path)
        try:
            np.savez_compressed(cache_path, **stems)
            logger.debug(f"Saved stems to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache stems: {e}")

    # === Mel Spectrograms ===

    def get_mel_path(self, audio_path: Union[str, Path]) -> Path:
        """Get cache path for mel spectrogram."""
        file_hash = self._get_file_hash(audio_path)
        name = Path(audio_path).stem
        return self.cache_dir / "mel" / f"{name}_{file_hash}_mel.npz"

    def load_mel(
        self,
        audio_path: Union[str, Path],
    ) -> Optional[np.ndarray]:
        """Load cached mel spectrogram."""
        if not self.enabled:
            return None

        cache_path = self.get_mel_path(audio_path)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path)
            logger.debug(f"Loaded cached mel from {cache_path}")
            return data['mel']
        except Exception as e:
            logger.warning(f"Failed to load cached mel: {e}")
            return None

    def save_mel(
        self,
        audio_path: Union[str, Path],
        mel: np.ndarray,
    ) -> None:
        """Save mel spectrogram to cache."""
        if not self.enabled:
            return

        cache_path = self.get_mel_path(audio_path)
        try:
            np.savez_compressed(cache_path, mel=mel)
            logger.debug(f"Saved mel to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache mel: {e}")

    # === Edit Labels ===

    def get_labels_path(
        self,
        raw_path: Union[str, Path],
        edited_path: Union[str, Path],
    ) -> Path:
        """Get cache path for edit labels."""
        raw_hash = self._get_file_hash(raw_path)
        edited_hash = self._get_file_hash(edited_path)
        raw_name = Path(raw_path).stem
        return self.cache_dir / "labels" / f"{raw_name}_{raw_hash}_{edited_hash}_labels.npz"

    def load_labels(
        self,
        raw_path: Union[str, Path],
        edited_path: Union[str, Path],
    ) -> Optional[np.ndarray]:
        """Load cached edit labels."""
        if not self.enabled:
            return None

        cache_path = self.get_labels_path(raw_path, edited_path)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path)
            logger.debug(f"Loaded cached labels from {cache_path}")
            return data['labels']
        except Exception as e:
            logger.warning(f"Failed to load cached labels: {e}")
            return None

    def save_labels(
        self,
        raw_path: Union[str, Path],
        edited_path: Union[str, Path],
        labels: np.ndarray,
    ) -> None:
        """Save edit labels to cache."""
        if not self.enabled:
            return

        cache_path = self.get_labels_path(raw_path, edited_path)
        try:
            np.savez_compressed(cache_path, labels=labels)
            logger.debug(f"Saved labels to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache labels: {e}")

    # === Full Processed Data ===

    def get_full_path(self, audio_path: Union[str, Path]) -> Path:
        """Get cache path for full processed data."""
        file_hash = self._get_file_hash(audio_path)
        name = Path(audio_path).stem
        return self.cache_dir / "full" / f"{name}_{file_hash}_full.npz"

    def load_full(
        self,
        audio_path: Union[str, Path],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load full cached processed data."""
        if not self.enabled:
            return None

        cache_path = self.get_full_path(audio_path)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            result = {key: data[key] for key in data.files}
            logger.debug(f"Loaded full cached data from {cache_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load full cached data: {e}")
            return None

    def save_full(
        self,
        audio_path: Union[str, Path],
        data: Dict[str, np.ndarray],
    ) -> None:
        """Save full processed data to cache."""
        if not self.enabled:
            return

        cache_path = self.get_full_path(audio_path)
        try:
            np.savez_compressed(cache_path, **data)
            logger.debug(f"Saved full data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache full data: {e}")

    # === Generic Key-Value Cache ===

    def get_generic_path(self, key: str, subdir: str = "generic") -> Path:
        """Get cache path for a generic key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{key_hash}.npz"

    def load_generic(self, key: str, subdir: str = "generic") -> Optional[Dict[str, np.ndarray]]:
        """Load generic cached data by key."""
        if not self.enabled:
            return None

        cache_path = self.get_generic_path(key, subdir)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception as e:
            logger.warning(f"Failed to load cached data for key {key}: {e}")
            return None

    def save_generic(self, key: str, data: Dict[str, np.ndarray], subdir: str = "generic") -> None:
        """Save generic data to cache."""
        if not self.enabled:
            return

        cache_path = self.get_generic_path(key, subdir)
        try:
            np.savez_compressed(cache_path, **data)
            logger.debug(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data for key {key}: {e}")

    # === Utility Methods ===

    def clear(self, subdir: Optional[str] = None) -> int:
        """Clear cache.

        Args:
            subdir: If provided, only clear this subdirectory

        Returns:
            Number of files deleted
        """
        count = 0
        if subdir:
            target = self.cache_dir / subdir
            if target.exists():
                for f in target.glob("*"):
                    if f.is_file():
                        f.unlink()
                        count += 1
        else:
            for subdir_name in ["features", "stems", "mel", "labels", "metadata", "full", "generic"]:
                target = self.cache_dir / subdir_name
                if target.exists():
                    for f in target.glob("*"):
                        if f.is_file():
                            f.unlink()
                            count += 1

        logger.info(f"Cleared {count} cached files")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "cache_dir": str(self.cache_dir),
            "enabled": self.enabled,
            "subdirs": {},
        }

        total_size = 0
        total_files = 0

        for subdir in ["features", "stems", "mel", "labels", "full", "generic"]:
            target = self.cache_dir / subdir
            if target.exists():
                files = list(target.glob("*.npz"))
                size = sum(f.stat().st_size for f in files)
                stats["subdirs"][subdir] = {
                    "files": len(files),
                    "size_mb": size / (1024 * 1024),
                }
                total_size += size
                total_files += len(files)

        stats["total_files"] = total_files
        stats["total_size_mb"] = total_size / (1024 * 1024)

        return stats

    def has_cached(self, audio_path: Union[str, Path], cache_type: str = "features") -> bool:
        """Check if a file has cached data.

        Args:
            audio_path: Path to audio file
            cache_type: Type of cache ('features', 'stems', 'mel', 'full')
        """
        if not self.enabled:
            return False

        if cache_type == "features":
            return self.get_features_path(audio_path).exists()
        elif cache_type == "stems":
            return self.get_stems_path(audio_path).exists()
        elif cache_type == "mel":
            return self.get_mel_path(audio_path).exists()
        elif cache_type == "full":
            return self.get_full_path(audio_path).exists()
        return False


# Global cache instance
_global_cache: Optional[FeatureCache] = None


def get_cache(cache_dir: Optional[str] = None) -> FeatureCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None or (cache_dir and Path(cache_dir) != _global_cache.cache_dir):
        _global_cache = FeatureCache(cache_dir=cache_dir)
    return _global_cache


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable caching globally."""
    global _global_cache
    if _global_cache:
        _global_cache.enabled = enabled
