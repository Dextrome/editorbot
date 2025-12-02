"""
Style Discriminator - Scores how well a remix matches a target style.

This is the "critic" that evaluates remixes and provides training signal
for the policy network. It learns to distinguish between:
- Original songs in the target style
- Remixes that successfully match the style
- Remixes that don't match the style
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StyleDiscriminatorNet(nn.Module):
    """
    Discriminator that scores style similarity.
    
    Takes two style embeddings and outputs a similarity score.
    Can also take raw features for end-to-end training.
    """
    
    def __init__(
        self,
        style_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.style_dim = style_dim
        
        # Comparison network
        # Takes concatenated embeddings and predicts similarity
        self.comparator = nn.Sequential(
            nn.Linear(style_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Feature-level comparison (for more fine-grained feedback)
        self.feature_comparator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, style_dim),
            nn.Tanh()
        )
    
    def forward(
        self,
        source_embedding: torch.Tensor,   # (B, style_dim)
        target_embedding: torch.Tensor    # (B, style_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compare two style embeddings.
        
        Returns:
            similarity_score: (B, 1) how similar the styles are (0-1)
            feature_diff: (B, style_dim) per-dimension difference for feedback
        """
        # Concatenate and compare
        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        similarity_logit = self.comparator(combined)
        similarity_score = torch.sigmoid(similarity_logit)
        
        # Feature-level difference
        source_proj = self.feature_comparator(source_embedding)
        target_proj = self.feature_comparator(target_embedding)
        feature_diff = target_proj - source_proj
        
        return similarity_score, feature_diff
    
    def score(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Get just the similarity score."""
        similarity, _ = self.forward(source_embedding, target_embedding)
        return similarity


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for more robust style comparison.
    
    Compares at multiple levels:
    - Global structure (overall energy arc, length)
    - Phrase-level patterns (transition types, phrase lengths)
    - Local features (chroma, brightness transitions)
    """
    
    def __init__(self, style_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        # Global discriminator (full embedding)
        self.global_disc = StyleDiscriminatorNet(style_dim, hidden_dim)
        
        # Structure discriminator (focuses on structure features)
        self.structure_disc = nn.Sequential(
            nn.Linear(style_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Texture discriminator (focuses on local features)
        self.texture_disc = nn.Sequential(
            nn.Linear(style_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Harmonic discriminator (focuses on chroma/key)
        self.harmonic_disc = nn.Sequential(
            nn.Linear(style_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Energy discriminator (focuses on dynamics)
        self.energy_disc = nn.Sequential(
            nn.Linear(style_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Combine scores
        self.combiner = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Multi-scale style comparison.
        
        Returns:
            final_score: (B, 1) combined similarity score
            sub_scores: dict of individual discriminator scores
        """
        B = source_embedding.size(0)
        D = source_embedding.size(1)
        
        # Split embedding into quarters for sub-discriminators
        chunk_size = D // 4
        src_chunks = torch.split(source_embedding, chunk_size, dim=-1)
        tgt_chunks = torch.split(target_embedding, chunk_size, dim=-1)
        
        # Global comparison
        global_score, _ = self.global_disc(source_embedding, target_embedding)
        
        # Sub-comparisons (using absolute difference as input)
        structure_diff = torch.abs(src_chunks[0] - tgt_chunks[0])
        texture_diff = torch.abs(src_chunks[1] - tgt_chunks[1])
        harmonic_diff = torch.abs(src_chunks[2] - tgt_chunks[2])
        energy_diff = torch.abs(src_chunks[3] - tgt_chunks[3])
        
        structure_score = torch.sigmoid(self.structure_disc(structure_diff))
        texture_score = torch.sigmoid(self.texture_disc(texture_diff))
        harmonic_score = torch.sigmoid(self.harmonic_disc(harmonic_diff))
        energy_score = torch.sigmoid(self.energy_disc(energy_diff))
        
        # Combine all scores
        all_scores = torch.cat([
            global_score, 
            structure_score, 
            texture_score, 
            harmonic_score, 
            energy_score
        ], dim=-1)
        
        final_score = torch.sigmoid(self.combiner(all_scores))
        
        sub_scores = {
            'global': global_score.mean().item(),
            'structure': structure_score.mean().item(),
            'texture': texture_score.mean().item(),
            'harmonic': harmonic_score.mean().item(),
            'energy': energy_score.mean().item()
        }
        
        return final_score, sub_scores


class StyleDiscriminator:
    """High-level interface for style discrimination."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        style_dim: int = 256,
        hidden_dim: int = 512,
        use_multiscale: bool = True
    ):
        self.device = device
        self.style_dim = style_dim
        
        if use_multiscale:
            self.model = MultiScaleDiscriminator(style_dim, hidden_dim)
        else:
            self.model = StyleDiscriminatorNet(style_dim, hidden_dim)
        
        self.model.to(device)
        self.use_multiscale = use_multiscale
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: str):
        """Load trained model."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded style discriminator from {path}")
    
    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved style discriminator to {path}")
    
    @torch.no_grad()
    def score(
        self,
        source_embedding: np.ndarray,
        target_embedding: np.ndarray
    ) -> float:
        """
        Score how well source matches target style.
        
        Args:
            source_embedding: Embedding of generated remix
            target_embedding: Embedding of target style
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        self.model.eval()
        
        src_t = torch.FloatTensor(source_embedding).unsqueeze(0).to(self.device)
        tgt_t = torch.FloatTensor(target_embedding).unsqueeze(0).to(self.device)
        
        if self.use_multiscale:
            score, _ = self.model(src_t, tgt_t)
        else:
            score = self.model.score(src_t, tgt_t)
        
        return score.item()
    
    @torch.no_grad()
    def detailed_score(
        self,
        source_embedding: np.ndarray,
        target_embedding: np.ndarray
    ) -> dict:
        """
        Get detailed breakdown of style similarity.
        
        Returns dict with overall score and sub-scores.
        """
        self.model.eval()
        
        src_t = torch.FloatTensor(source_embedding).unsqueeze(0).to(self.device)
        tgt_t = torch.FloatTensor(target_embedding).unsqueeze(0).to(self.device)
        
        if self.use_multiscale:
            score, sub_scores = self.model(src_t, tgt_t)
            return {
                'overall': score.item(),
                **sub_scores
            }
        else:
            score, feature_diff = self.model(src_t, tgt_t)
            return {
                'overall': score.item(),
                'feature_diff_norm': torch.norm(feature_diff).item()
            }
