"""Edit Predictor for Phase 2 RL training.

Predicts edit labels from raw mel spectrogram.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.distributions import Categorical

from ..config import Phase2Config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EditPredictor(nn.Module):
    """Policy network that predicts edit labels from raw mel spectrogram.

    Architecture:
        raw_mel -> Conv layers -> Transformer -> Linear -> edit_logits
    """

    def __init__(self, config: Phase2Config):
        super().__init__()
        self.config = config
        self.n_labels = config.n_edit_labels

        # Convolutional front-end for local features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.audio.n_mels, config.predictor_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.predictor_dim // 2, config.predictor_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.predictor_dim // 2, config.predictor_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.predictor_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.predictor_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_predictor_layers,
            norm=nn.LayerNorm(config.predictor_dim),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.predictor_dim, config.predictor_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_dim, config.n_edit_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        raw_mel: torch.Tensor,  # (B, T, n_mels)
        mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> torch.Tensor:
        """
        Args:
            raw_mel: Raw mel spectrogram (B, T, n_mels)
            mask: Valid frame mask (B, T)

        Returns:
            logits: Edit label logits (B, T, n_labels)
        """
        # Conv expects (B, C, T)
        x = raw_mel.transpose(1, 2)  # (B, n_mels, T)
        x = self.conv_layers(x)       # (B, predictor_dim, T)
        x = x.transpose(1, 2)         # (B, T, predictor_dim)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        x = self.transformer(x, src_key_padding_mask=attn_mask)  # (B, T, predictor_dim)

        # Output logits
        logits = self.output_head(x)  # (B, T, n_labels)

        return logits

    def get_action(
        self,
        raw_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from policy.

        Args:
            raw_mel: Raw mel spectrogram (B, T, n_mels)
            mask: Valid frame mask
            deterministic: If True, use argmax instead of sampling

        Returns:
            actions: Sampled edit labels (B, T)
            log_probs: Log probabilities of actions (B,) - summed over T
            entropy: Policy entropy (scalar)
        """
        logits = self.forward(raw_mel, mask)  # (B, T, n_labels)

        dist = Categorical(logits=logits)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()

        # Log prob summed over sequence
        log_probs = dist.log_prob(actions)  # (B, T)
        if mask is not None:
            log_probs = log_probs * mask.float()
        log_probs = log_probs.sum(dim=1)  # (B,)

        # Entropy averaged over sequence
        entropy = dist.entropy()  # (B, T)
        if mask is not None:
            entropy = (entropy * mask.float()).sum() / mask.float().sum()
        else:
            entropy = entropy.mean()

        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        raw_mel: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given actions.

        Args:
            raw_mel: Raw mel spectrogram (B, T, n_mels)
            actions: Edit labels to evaluate (B, T)
            mask: Valid frame mask

        Returns:
            log_probs: Log probabilities (B,)
            entropy: Policy entropy (scalar)
        """
        logits = self.forward(raw_mel, mask)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        if mask is not None:
            log_probs = log_probs * mask.float()
        log_probs = log_probs.sum(dim=1)

        entropy = dist.entropy()
        if mask is not None:
            entropy = (entropy * mask.float()).sum() / mask.float().sum()
        else:
            entropy = entropy.mean()

        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Value network for PPO.

    Estimates expected return from raw mel state.
    """

    def __init__(self, config: Phase2Config):
        super().__init__()
        self.config = config

        # Convolutional encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.audio.n_mels, config.value_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.value_dim // 2, config.value_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.value_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_value_layers,
            norm=nn.LayerNorm(config.value_dim),
        )

        # Global pooling + value head
        self.value_head = nn.Sequential(
            nn.Linear(config.value_dim, config.value_dim // 2),
            nn.GELU(),
            nn.Linear(config.value_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        raw_mel: torch.Tensor,  # (B, T, n_mels)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            raw_mel: Raw mel spectrogram (B, T, n_mels)
            mask: Valid frame mask (B, T)

        Returns:
            value: Estimated value (B,)
        """
        # Conv
        x = raw_mel.transpose(1, 2)  # (B, n_mels, T)
        x = self.conv_layers(x)       # (B, value_dim, T)
        x = x.transpose(1, 2)         # (B, T, value_dim)

        # Transformer
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Global average pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)  # (B, value_dim)

        # Value prediction
        value = self.value_head(x).squeeze(-1)  # (B,)

        return value


class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""

    def __init__(self, config: Phase2Config):
        super().__init__()
        self.actor = EditPredictor(config)
        self.critic = ValueNetwork(config)

    def forward(
        self,
        raw_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: Edit label logits (B, T, n_labels)
            value: State value (B,)
        """
        logits = self.actor(raw_mel, mask)
        value = self.critic(raw_mel, mask)
        return logits, value

    def get_action_and_value(
        self,
        raw_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value in one forward pass.

        Args:
            raw_mel: Raw mel spectrogram
            mask: Valid frame mask
            action: If provided, evaluate this action instead of sampling
            deterministic: Use argmax instead of sampling

        Returns:
            action: Edit labels (B, T)
            log_prob: Log probability (B,)
            entropy: Policy entropy (scalar)
            value: State value (B,)
        """
        logits = self.actor(raw_mel, mask)
        value = self.critic(raw_mel, mask)

        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        if mask is not None:
            log_prob = log_prob * mask.float()
        log_prob = log_prob.sum(dim=1)

        entropy = dist.entropy()
        if mask is not None:
            entropy = (entropy * mask.float()).sum() / mask.float().sum()
        else:
            entropy = entropy.mean()

        return action, log_prob, entropy, value
