"""
Iterative Style Transfer - Main interface for transforming recordings into songs.

This implements the iterative refinement loop:
1. Generate initial remix based on style embedding
2. Score the result against target style
3. Adjust and regenerate until quality threshold is met
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
import sys
import os

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from style_encoder import StyleEncoder
from discriminator import StyleDiscriminator
from remix_policy import RemixPolicy, RemixPlan

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result of style transfer."""
    output_path: str
    iterations: int
    final_score: float
    score_history: List[float]
    detailed_scores: Dict[str, float]


class IterativeStyleTransfer:
    """
    Main interface for style transfer with iterative refinement.
    
    Usage:
        transfer = IterativeStyleTransfer()
        transfer.load_models("models/")
        
        result = transfer.transform(
            source="my_jam.wav",
            target_style="reference_song.wav",
            output="output.wav",
            max_iterations=10,
            threshold=0.8
        )
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        style_dim: int = 256,
        hidden_dim: int = 512
    ):
        self.device = device
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # Models (initialized but not loaded)
        self.style_encoder: Optional[StyleEncoder] = None
        self.discriminator: Optional[StyleDiscriminator] = None
        self.policy: Optional[RemixPolicy] = None
        
        # Learned style embedding (average of training data)
        self.learned_style: Optional[np.ndarray] = None
        
        # Remix engine (will use loopremix)
        self.remix_engine = None
    
    def load_models(self, model_dir: str):
        """Load all trained models."""
        model_dir = Path(model_dir)
        
        # Style encoder
        encoder_path = model_dir / "style_encoder.pt"
        self.style_encoder = StyleEncoder(
            model_path=str(encoder_path) if encoder_path.exists() else None,
            device=self.device,
            embedding_dim=self.style_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Discriminator
        disc_path = model_dir / "discriminator.pt"
        self.discriminator = StyleDiscriminator(
            model_path=str(disc_path) if disc_path.exists() else None,
            device=self.device,
            style_dim=self.style_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Policy
        policy_path = model_dir / "policy.pt"
        self.policy = RemixPolicy(
            model_path=str(policy_path) if policy_path.exists() else None,
            device=self.device,
            style_dim=self.style_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Load learned style embedding if available
        style_path = model_dir / "learned_style.npy"
        if style_path.exists():
            self.learned_style = np.load(str(style_path))
            logger.info(f"Loaded learned style embedding from {style_path}")
        
        logger.info(f"Loaded models from {model_dir}")
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """Load all models from a single training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Initialize models
        self.style_encoder = StyleEncoder(device=self.device, embedding_dim=self.style_dim, hidden_dim=self.hidden_dim)
        self.discriminator = StyleDiscriminator(device=self.device, style_dim=self.style_dim, hidden_dim=self.hidden_dim)
        self.policy = RemixPolicy(device=self.device, style_dim=self.style_dim, hidden_dim=self.hidden_dim)
        
        # Load weights
        self.style_encoder.model.load_state_dict(checkpoint['style_encoder'])
        self.discriminator.model.load_state_dict(checkpoint['discriminator'])
        self.policy.model.load_state_dict(checkpoint['policy'])
        
        logger.info(f"Loaded models from checkpoint {checkpoint_path}")
    
    def extract_style(self, audio_path: str) -> np.ndarray:
        """Extract style embedding from a reference song."""
        if self.style_encoder is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        return self.style_encoder.encode_file(audio_path)
    
    def score_similarity(
        self, 
        source_embedding: np.ndarray, 
        target_embedding: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Score how similar two style embeddings are."""
        if self.discriminator is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        detailed = self.discriminator.detailed_score(source_embedding, target_embedding)
        return detailed['overall'], detailed
    
    def transform(
        self,
        source: str,
        target_style: Optional[str] = None,
        output: str = "output.wav",
        max_iterations: int = 10,
        threshold: float = 0.8,
        target_duration: float = 180.0,
        verbose: bool = True
    ) -> TransferResult:
        """
        Transform a source recording to match a target style.
        
        Args:
            source: Path to source recording (raw jam/loop)
            target_style: Path to reference song (optional - uses learned style if None)
            output: Path for output file
            max_iterations: Maximum refinement iterations
            threshold: Stop when similarity exceeds this (0-1)
            target_duration: Target output duration in seconds
            verbose: Print progress
            
        Returns:
            TransferResult with output path and metrics
        """
        if self.style_encoder is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Get target style embedding
        if target_style is not None:
            # Use provided reference song
            if verbose:
                logger.info(f"Extracting style from reference: {target_style}")
            target_embedding = self.extract_style(target_style)
        elif self.learned_style is not None:
            # Use the learned style from training
            if verbose:
                logger.info("Using learned style from training data")
            target_embedding = self.learned_style
        else:
            raise ValueError(
                "No target style provided and no learned style available. "
                "Either provide a reference song or train with --save-style"
            )
        
        # Extract source features for remix planning
        if verbose:
            logger.info(f"Analyzing source: {source}")
        source_phrases = self._extract_phrases(source)
        source_embedding = self.extract_style(source)
        
        # Initial score
        initial_score, _ = self.score_similarity(source_embedding, target_embedding)
        if verbose:
            logger.info(f"Initial style similarity: {initial_score:.3f}")
        
        # Iterative refinement
        score_history = [initial_score]
        best_score = initial_score
        best_output = None
        
        for iteration in range(max_iterations):
            if verbose:
                logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Generate remix plan
            temperature = 1.0 - (iteration / max_iterations) * 0.5  # Decrease over time
            plan = self._generate_remix_plan(
                source_phrases, 
                target_embedding,
                target_duration,
                temperature
            )
            
            # Execute remix
            temp_output = f"{output}.iter{iteration}.wav"
            self._execute_remix(source, plan, temp_output)
            
            # Score result
            result_embedding = self.extract_style(temp_output)
            score, detailed = self.score_similarity(result_embedding, target_embedding)
            score_history.append(score)
            
            if verbose:
                logger.info(f"Iteration {iteration + 1} score: {score:.3f}")
                for k, v in detailed.items():
                    if k != 'overall':
                        logger.info(f"  {k}: {v:.3f}")
            
            # Track best
            if score > best_score:
                best_score = score
                best_output = temp_output
                if verbose:
                    logger.info(f"New best score: {best_score:.3f}")
            
            # Check threshold
            if score >= threshold:
                if verbose:
                    logger.info(f"Threshold {threshold} reached!")
                break
            
            # Cleanup non-best iterations
            if temp_output != best_output and os.path.exists(temp_output):
                os.remove(temp_output)
        
        # Copy best to final output
        if best_output and best_output != output:
            import shutil
            shutil.move(best_output, output)
        
        # Cleanup any remaining temp files
        for i in range(max_iterations):
            temp = f"{output}.iter{i}.wav"
            if os.path.exists(temp):
                os.remove(temp)
        
        # Final scoring
        final_embedding = self.extract_style(output)
        final_score, final_detailed = self.score_similarity(final_embedding, target_embedding)
        
        if verbose:
            logger.info(f"\n=== Final Results ===")
            logger.info(f"Output: {output}")
            logger.info(f"Final score: {final_score:.3f}")
            logger.info(f"Iterations: {len(score_history)}")
            logger.info(f"Score improvement: {final_score - initial_score:+.3f}")
        
        return TransferResult(
            output_path=output,
            iterations=len(score_history),
            final_score=final_score,
            score_history=score_history,
            detailed_scores=final_detailed
        )
    
    def _extract_phrases(self, source_path: str) -> List[Dict]:
        """Extract phrases from source using loopremix."""
        try:
            # Try to use loopremix for phrase detection
            loopremix_path = str(Path(__file__).parent.parent / "loopremix")
            if loopremix_path not in sys.path:
                sys.path.insert(0, loopremix_path)
            from loopremix import LoopAnalyzer, PhraseDetector
            
            # Load audio - keep original format (mono or stereo)
            audio, sr = librosa.load(source_path, sr=44100, mono=False)
            # librosa with mono=False returns (channels, samples) for stereo, (samples,) for mono
            
            # Analyze loop structure
            analyzer = LoopAnalyzer(sample_rate=sr)
            loop_info = analyzer.analyze(audio)
            
            # Detect phrases
            detector = PhraseDetector(sample_rate=sr)
            detected_phrases = detector.detect_phrases(audio, loop_info)
            
            # Store for later use
            self._source_audio = audio
            self._source_sr = sr
            self._loop_info = loop_info
            self._detected_phrases = detected_phrases
            
            # Convert to dict format
            phrases = []
            for p in detected_phrases:
                phrases.append({
                    'start_time': p.start_time,
                    'end_time': p.end_time,
                    'duration': p.end_time - p.start_time,
                    'energy': p.energy,
                    'brightness': p.spectral_centroid / 5000.0,  # Normalize
                    'onset_density': p.onset_density,
                    'start_energy': p.start_energy,
                    'end_energy': p.end_energy,
                    'quality_score': p.quality_score,
                    'avg_chroma': p.avg_chroma
                })
            
            logger.info(f"Detected {len(phrases)} phrases using loopremix")
            return phrases
            
        except ImportError as e:
            logger.warning(f"loopremix not available ({e}), using basic phrase detection")
            return self._basic_phrase_detection(source_path)
        except Exception as e:
            logger.warning(f"loopremix failed ({e}), using basic phrase detection")
            return self._basic_phrase_detection(source_path)
    
    def _basic_phrase_detection(self, source_path: str) -> List[Dict]:
        """Fallback phrase detection without loopremix."""
        audio, sr = librosa.load(source_path, sr=44100, mono=True)
        duration = len(audio) / sr
        
        # Simple fixed-length phrase detection
        phrase_duration = 8.0  # 8 seconds per phrase
        phrases = []
        
        t = 0.0
        while t < duration - phrase_duration / 2:
            end_t = min(t + phrase_duration, duration)
            
            start_sample = int(t * sr)
            end_sample = int(end_t * sr)
            segment = audio[start_sample:end_sample]
            
            energy = float(np.sqrt(np.mean(segment ** 2)))
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            
            phrases.append({
                'start_time': t,
                'end_time': end_t,
                'duration': end_t - t,
                'energy': energy,
                'brightness': float(np.mean(centroid)) / 5000.0,
                'onset_density': 5.0,
                'start_energy': energy,
                'end_energy': energy,
                'quality_score': 0.8,
                'avg_chroma': None
            })
            
            t = end_t
        
        return phrases
    
    def _generate_remix_plan(
        self,
        source_phrases: List[Dict],
        target_embedding: np.ndarray,
        target_duration: float,
        temperature: float
    ) -> RemixPlan:
        """Generate remix plan using policy network."""
        if self.policy is not None:
            try:
                return self.policy.generate_plan(
                    source_phrases,
                    target_embedding,
                    target_duration,
                    temperature
                )
            except Exception as e:
                logger.warning(f"Policy generation failed: {e}, using heuristic")
        
        # Fallback: heuristic-based planning
        return self._heuristic_remix_plan(source_phrases, target_duration)
    
    def _heuristic_remix_plan(
        self,
        source_phrases: List[Dict],
        target_duration: float
    ) -> RemixPlan:
        """Heuristic remix planning (no trained model needed)."""
        from .remix_policy import RemixAction
        
        actions = []
        current_duration = 0.0
        used_phrases = set()
        
        # Sort by quality
        sorted_indices = sorted(
            range(len(source_phrases)),
            key=lambda i: source_phrases[i]['quality_score'],
            reverse=True
        )
        
        for idx in sorted_indices:
            if current_duration >= target_duration:
                break
            
            if idx in used_phrases:
                continue
            
            phrase = source_phrases[idx]
            actions.append(RemixAction(
                phrase_index=idx,
                crossfade_duration=0.5,
                energy_adjust=1.0,
                pitch_shift=0.0
            ))
            
            current_duration += phrase['duration']
            used_phrases.add(idx)
        
        return RemixPlan(
            actions=actions,
            expected_duration=current_duration,
            style_match_score=0.0
        )
    
    def _execute_remix(
        self,
        source_path: str,
        plan: RemixPlan,
        output_path: str
    ):
        """Execute a remix plan to generate output audio."""
        try:
            # Try to use loopremix
            loopremix_path = str(Path(__file__).parent.parent / "loopremix")
            if loopremix_path not in sys.path:
                sys.path.insert(0, loopremix_path)
            from loopremix import SongBuilder
            
            # Check if we have cached data from phrase extraction
            if hasattr(self, '_source_audio') and hasattr(self, '_loop_info') and hasattr(self, '_detected_phrases'):
                audio = self._source_audio
                sr = self._source_sr
                loop_info = self._loop_info
                detected_phrases = self._detected_phrases
            else:
                # Load and analyze fresh
                from loopremix import LoopAnalyzer, PhraseDetector
                audio, sr = librosa.load(source_path, sr=44100, mono=False)
                # librosa with mono=False returns (channels, samples) for stereo, (samples,) for mono
                analyzer = LoopAnalyzer(sample_rate=sr)
                loop_info = analyzer.analyze(audio)
                detector = PhraseDetector(sample_rate=sr)
                detected_phrases = detector.detect_phrases(audio, loop_info)
            
            # Reorder phrases according to plan
            ordered_phrases = []
            for action in plan.actions:
                if action.phrase_index < len(detected_phrases):
                    ordered_phrases.append(detected_phrases[action.phrase_index])
            
            if not ordered_phrases:
                logger.warning("No phrases to remix, using original audio")
                sf.write(output_path, audio.T if audio.ndim == 2 else audio, sr)
                return
            
            # Build remix using SongBuilder
            builder = SongBuilder(sample_rate=sr)
            output_audio = builder.build_remix(
                audio=audio,
                phrases=ordered_phrases,
                loop_info=loop_info,
                target_duration=plan.expected_duration
            )
            
            # Ensure proper format for soundfile (float32, contiguous)
            if output_audio.ndim == 2:
                # Stereo: ensure (samples, channels) format
                output_audio = np.ascontiguousarray(output_audio.astype(np.float32))
            else:
                # Mono
                output_audio = np.ascontiguousarray(output_audio.astype(np.float32))
            
            sf.write(output_path, output_audio, sr)
            logger.info(f"Built remix using loopremix ({len(ordered_phrases)} phrases)")
            
        except ImportError as e:
            logger.warning(f"loopremix not available ({e}), using basic concatenation")
            self._basic_remix(source_path, plan, output_path)
        except Exception as e:
            logger.warning(f"loopremix failed ({e}), using basic concatenation")
            self._basic_remix(source_path, plan, output_path)
    
    def _basic_remix(
        self,
        source_path: str,
        plan: RemixPlan,
        output_path: str
    ):
        """Fallback remix execution without loopremix."""
        audio, sr = librosa.load(source_path, sr=44100, mono=True)
        
        # Get phrases from plan
        phrases = self._basic_phrase_detection(source_path)
        
        # Concatenate selected phrases with crossfade
        output_segments = []
        
        for action in plan.actions:
            if action.phrase_index >= len(phrases):
                continue
            
            phrase = phrases[action.phrase_index]
            start_sample = int(phrase['start_time'] * sr)
            end_sample = int(phrase['end_time'] * sr)
            
            segment = audio[start_sample:end_sample]
            
            # Apply energy adjustment
            segment = segment * action.energy_adjust
            
            output_segments.append(segment)
        
        # Simple concatenation with crossfade
        if not output_segments:
            output_audio = audio[:int(180 * sr)]  # Fallback: first 3 minutes
        else:
            crossfade_samples = int(0.5 * sr)
            output_audio = output_segments[0]
            
            for seg in output_segments[1:]:
                if len(output_audio) < crossfade_samples:
                    output_audio = np.concatenate([output_audio, seg])
                else:
                    # Crossfade
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    
                    output_audio[-crossfade_samples:] *= fade_out
                    seg[:crossfade_samples] *= fade_in
                    
                    output_audio[-crossfade_samples:] += seg[:crossfade_samples]
                    output_audio = np.concatenate([output_audio, seg[crossfade_samples:]])
        
        sf.write(output_path, output_audio, sr)


def main():
    """CLI interface for style transfer."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Transform a recording to match a target style")
    parser.add_argument("source", help="Source recording (raw jam/loop)")
    parser.add_argument("target", help="Target style reference song")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("-m", "--models", default="models/", help="Model directory")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Quality threshold")
    parser.add_argument("-d", "--duration", type=float, default=180.0, help="Target duration")
    
    args = parser.parse_args()
    
    transfer = IterativeStyleTransfer()
    
    # Try to load models if available
    model_dir = Path(args.models)
    if model_dir.exists():
        transfer.load_models(str(model_dir))
    else:
        logger.warning(f"Model directory {model_dir} not found, using heuristics only")
        # Initialize with untrained models
        transfer.style_encoder = StyleEncoder(device=transfer.device)
        transfer.discriminator = StyleDiscriminator(device=transfer.device)
        transfer.policy = RemixPolicy(device=transfer.device)
    
    result = transfer.transform(
        source=args.source,
        target_style=args.target,
        output=args.output,
        max_iterations=args.iterations,
        threshold=args.threshold,
        target_duration=args.duration
    )
    
    print(f"\nDone! Output saved to: {result.output_path}")
    print(f"Final style match score: {result.final_score:.3f}")


if __name__ == "__main__":
    main()
