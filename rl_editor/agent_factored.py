"""Factored Agent with multi-head policy network.

Uses 3 output heads for factored action space:
1. Type head: What action to take (18 outputs)
2. Size head: How many beats (5 outputs)  
3. Amount head: Intensity/direction (5 outputs)

The combined log_prob is: log_prob = log(P(type)) + log(P(size|type)) + log(P(amount|type))
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from .config import Config
from .agent import HybridNATTENEncoder, ValueNetwork
from .actions_factored import (
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace, EditHistoryFactored,
)

logger = logging.getLogger(__name__)


class FactoredPolicyNetwork(nn.Module):
    """Multi-head policy network for factored action space.
    
    Outputs logits for 3 action components:
    - Type: What action (18 options)
    - Size: How many beats (5 options)
    - Amount: Intensity/direction (5 options)
    
    Uses shared encoder with separate heads for each component.
    """

    def __init__(
        self, config: Config, input_dim: int
    ) -> None:
        """Initialize factored policy network.

        Args:
            config: Configuration object
            input_dim: Input state dimension
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        model_config = config.model

        # Shared encoder (same as original policy)
        self.encoder = HybridNATTENEncoder(
            input_dim=input_dim,
            hidden_dim=model_config.policy_hidden_dim,
            n_heads=model_config.natten_n_heads,
            n_layers=model_config.natten_n_layers,
            kernel_size=model_config.natten_kernel_size,
            dilation=model_config.natten_dilation,
            dropout=model_config.policy_dropout,
        )
        
        hidden_dim = model_config.policy_hidden_dim
        
        # Separate heads for each action component
        # Type head: What action to take
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_ACTION_TYPES),
        )
        
        # Size head: How many beats (conditioned on type via concatenation)
        # We concatenate encoded features with type embedding
        self.type_embedding = nn.Embedding(N_ACTION_TYPES, 32)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_ACTION_SIZES),
        )
        
        # Amount head: Intensity (conditioned on type)
        self.amount_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_ACTION_AMOUNTS),
        )
        
        logger.info(f"FactoredPolicyNetwork: {N_ACTION_TYPES} types, {N_ACTION_SIZES} sizes, {N_ACTION_AMOUNTS} amounts")

    def forward(
        self,
        state: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        size_mask: Optional[torch.Tensor] = None,
        amount_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for all three heads.

        Args:
            state: State tensor (batch_size, input_dim)
            type_mask: Optional mask for types (batch_size, N_ACTION_TYPES)
            size_mask: Optional mask for sizes (batch_size, N_ACTION_SIZES)
            amount_mask: Optional mask for amounts (batch_size, N_ACTION_AMOUNTS)
            temperature: Softmax temperature

        Returns:
            Tuple of (type_logits, size_logits, amount_logits)
        """
        # Encode state
        encoded = self.encoder(state)  # (B, hidden_dim)
        
        # Type logits
        type_logits = self.type_head(encoded)
        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask.bool(), float("-inf"))
        type_logits = type_logits / max(temperature, 0.1)
        
        # For size and amount, we need to sample or use the most likely type
        # During forward, we use soft attention over types
        type_probs = F.softmax(type_logits, dim=-1)  # (B, N_types)
        type_embed = torch.matmul(type_probs, self.type_embedding.weight)  # (B, 32)
        
        # Size logits (conditioned on type)
        size_input = torch.cat([encoded, type_embed], dim=-1)
        size_logits = self.size_head(size_input)
        if size_mask is not None:
            size_logits = size_logits.masked_fill(~size_mask.bool(), float("-inf"))
        size_logits = size_logits / max(temperature, 0.1)
        
        # Amount logits (conditioned on type)
        amount_input = torch.cat([encoded, type_embed], dim=-1)
        amount_logits = self.amount_head(amount_input)
        if amount_mask is not None:
            amount_logits = amount_logits.masked_fill(~amount_mask.bool(), float("-inf"))
        amount_logits = amount_logits / max(temperature, 0.1)
        
        return type_logits, size_logits, amount_logits
    
    def get_action_and_log_prob(
        self,
        state: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        size_mask: Optional[torch.Tensor] = None,
        amount_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities.

        Args:
            state: State tensor (batch_size, input_dim)
            type_mask, size_mask, amount_mask: Optional masks
            deterministic: If True, return argmax instead of sample

        Returns:
            Tuple of (type_action, size_action, amount_action, combined_log_prob)
        """
        # Get encoded state
        encoded = self.encoder(state)  # (B, hidden_dim)
        B = encoded.shape[0]
        
        # === Type selection ===
        type_logits = self.type_head(encoded)
        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask.bool(), float("-inf"))
        
        type_probs = F.softmax(type_logits, dim=-1)
        type_probs = self._fix_probs(type_probs)
        
        if deterministic:
            type_action = type_logits.argmax(dim=-1)
        else:
            type_dist = torch.distributions.Categorical(type_probs)
            type_action = type_dist.sample()
        
        type_log_prob = torch.log(type_probs.gather(-1, type_action.unsqueeze(-1)) + 1e-10).squeeze(-1)
        
        # === Size selection (conditioned on chosen type) ===
        type_embed = self.type_embedding(type_action)  # (B, 32)
        size_input = torch.cat([encoded, type_embed], dim=-1)
        size_logits = self.size_head(size_input)
        if size_mask is not None:
            size_logits = size_logits.masked_fill(~size_mask.bool(), float("-inf"))
        
        size_probs = F.softmax(size_logits, dim=-1)
        size_probs = self._fix_probs(size_probs)
        
        if deterministic:
            size_action = size_logits.argmax(dim=-1)
        else:
            size_dist = torch.distributions.Categorical(size_probs)
            size_action = size_dist.sample()
        
        size_log_prob = torch.log(size_probs.gather(-1, size_action.unsqueeze(-1)) + 1e-10).squeeze(-1)
        
        # === Amount selection (conditioned on chosen type) ===
        amount_input = torch.cat([encoded, type_embed], dim=-1)
        amount_logits = self.amount_head(amount_input)
        if amount_mask is not None:
            amount_logits = amount_logits.masked_fill(~amount_mask.bool(), float("-inf"))
        
        amount_probs = F.softmax(amount_logits, dim=-1)
        amount_probs = self._fix_probs(amount_probs)
        
        if deterministic:
            amount_action = amount_logits.argmax(dim=-1)
        else:
            amount_dist = torch.distributions.Categorical(amount_probs)
            amount_action = amount_dist.sample()
        
        amount_log_prob = torch.log(amount_probs.gather(-1, amount_action.unsqueeze(-1)) + 1e-10).squeeze(-1)
        
        # Combined log probability: log P(type) + log P(size|type) + log P(amount|type)
        combined_log_prob = type_log_prob + size_log_prob + amount_log_prob
        
        return type_action, size_action, amount_action, combined_log_prob
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        type_action: torch.Tensor,
        size_action: torch.Tensor,
        amount_action: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        size_mask: Optional[torch.Tensor] = None,
        amount_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given actions.

        Args:
            state: State tensor (batch_size, input_dim)
            type_action, size_action, amount_action: Action tensors
            type_mask, size_mask, amount_mask: Optional masks

        Returns:
            Tuple of (combined_log_prob, combined_entropy)
        """
        # Get encoded state
        encoded = self.encoder(state)
        
        # === Type evaluation ===
        type_logits = self.type_head(encoded)
        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask.bool(), float("-inf"))
        type_logits = torch.clamp(type_logits, -20.0, 20.0)
        
        # Add exploration noise during training
        type_logits = type_logits + torch.randn_like(type_logits) * 0.1
        type_logits = type_logits / 1.5  # Temperature softening
        
        type_dist = torch.distributions.Categorical(logits=type_logits)
        type_log_prob = type_dist.log_prob(type_action)
        type_entropy = type_dist.entropy()
        
        # === Size evaluation (conditioned on actual type taken) ===
        type_embed = self.type_embedding(type_action)
        size_input = torch.cat([encoded, type_embed], dim=-1)
        size_logits = self.size_head(size_input)
        if size_mask is not None:
            size_logits = size_logits.masked_fill(~size_mask.bool(), float("-inf"))
        size_logits = torch.clamp(size_logits, -20.0, 20.0)
        size_logits = size_logits + torch.randn_like(size_logits) * 0.1
        size_logits = size_logits / 1.5
        
        size_dist = torch.distributions.Categorical(logits=size_logits)
        size_log_prob = size_dist.log_prob(size_action)
        size_entropy = size_dist.entropy()
        
        # === Amount evaluation ===
        amount_input = torch.cat([encoded, type_embed], dim=-1)
        amount_logits = self.amount_head(amount_input)
        if amount_mask is not None:
            amount_logits = amount_logits.masked_fill(~amount_mask.bool(), float("-inf"))
        amount_logits = torch.clamp(amount_logits, -20.0, 20.0)
        amount_logits = amount_logits + torch.randn_like(amount_logits) * 0.1
        amount_logits = amount_logits / 1.5
        
        amount_dist = torch.distributions.Categorical(logits=amount_logits)
        amount_log_prob = amount_dist.log_prob(amount_action)
        amount_entropy = amount_dist.entropy()
        
        # Combined
        combined_log_prob = type_log_prob + size_log_prob + amount_log_prob
        combined_entropy = type_entropy + size_entropy + amount_entropy
        
        # Handle NaN
        combined_log_prob = torch.nan_to_num(combined_log_prob, nan=-5.0, posinf=0.0, neginf=-15.0)
        combined_entropy = torch.nan_to_num(combined_entropy, nan=0.5, posinf=5.0, neginf=0.0)
        combined_entropy = torch.clamp(combined_entropy, min=0.1)
        
        return combined_log_prob, combined_entropy
    
    def _fix_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Fix probability distributions that sum to 0 or have NaN."""
        probs_sum = probs.sum(dim=-1, keepdim=True)
        invalid = (probs_sum < 1e-8) | torch.isnan(probs_sum)
        if invalid.any():
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = torch.where(invalid.expand_as(probs), uniform, probs)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs


class FactoredAgent:
    """Agent using factored action space with multi-head policy."""

    def __init__(
        self,
        config: Config,
        input_dim: int,
        beat_feature_dim: int = 121,
        use_auxiliary_tasks: bool = True,
    ) -> None:
        """Initialize factored agent.

        Args:
            config: Configuration object
            input_dim: Input state dimension
            beat_feature_dim: For auxiliary tasks
            use_auxiliary_tasks: Whether to use auxiliary tasks
        """
        self.config = config
        self.device = torch.device(config.training.device)
        self.use_auxiliary_tasks = use_auxiliary_tasks

        self.policy_net = FactoredPolicyNetwork(config, input_dim).to(self.device)
        self.value_net = ValueNetwork(config, input_dim).to(self.device)
        
        # Auxiliary task module (optional)
        self.auxiliary_module = None
        if use_auxiliary_tasks:
            try:
                from .auxiliary_tasks import AuxiliaryTaskModule, AuxiliaryConfig
                aux_config = AuxiliaryConfig()
                hidden_dim = config.model.policy_hidden_dim
                self.auxiliary_module = AuxiliaryTaskModule(
                    hidden_dim=hidden_dim,
                    beat_feature_dim=beat_feature_dim,
                    config=aux_config,
                ).to(self.device)
                logger.info(f"Initialized auxiliary task module with hidden_dim={hidden_dim}")
            except Exception as e:
                logger.warning(f"Failed to initialize auxiliary tasks: {e}")
                self.auxiliary_module = None

        logger.info(
            f"Initialized FactoredAgent with {input_dim} input dims, "
            f"{N_ACTION_TYPES} types, {N_ACTION_SIZES} sizes, {N_ACTION_AMOUNTS} amounts"
        )

    def select_action(
        self,
        state: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        size_mask: Optional[torch.Tensor] = None,
        amount_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tuple[int, int, int], float]:
        """Select factored action given state.

        Returns:
            Tuple of ((type, size, amount), log_prob)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            type_act, size_act, amount_act, log_prob = self.policy_net.get_action_and_log_prob(
                state, type_mask, size_mask, amount_mask, deterministic
            )

            return (
                (int(type_act[0].item()), int(size_act[0].item()), int(amount_act[0].item())),
                float(log_prob[0].item())
            )

    def select_action_batch(
        self,
        states: torch.Tensor,
        type_masks: torch.Tensor,
        size_masks: torch.Tensor,
        amount_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select actions for batch of states.

        Returns:
            Tuple of (type_actions, size_actions, amount_actions, log_probs)
        """
        with torch.no_grad():
            return self.policy_net.get_action_and_log_prob(
                states, type_masks, size_masks, amount_masks, deterministic
            )

    def compute_value_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Compute values for batch of states."""
        values = self.value_net(states)
        values = values.float()
        values = torch.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
        values = torch.clamp(values, -100.0, 100.0)
        return values.squeeze(-1)

    def compute_value(self, state: torch.Tensor) -> float:
        """Compute value of single state."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            value = self.value_net(state)
            return float(value[0].item())

    def evaluate_actions(
        self,
        states: torch.Tensor,
        type_actions: torch.Tensor,
        size_actions: torch.Tensor,
        amount_actions: torch.Tensor,
        type_masks: Optional[torch.Tensor] = None,
        size_masks: Optional[torch.Tensor] = None,
        amount_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given actions."""
        return self.policy_net.evaluate_actions(
            states, type_actions, size_actions, amount_actions,
            type_masks, size_masks, amount_masks
        )

    def get_encoder_output(self, states: torch.Tensor) -> torch.Tensor:
        """Get encoded representation from policy encoder."""
        return self.policy_net.encoder(states)

    def compute_auxiliary_loss(
        self,
        states: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute auxiliary task losses."""
        if self.auxiliary_module is None:
            return torch.tensor(0.0, device=self.device), {}
        
        encoded = self.get_encoder_output(states)
        predictions = self.auxiliary_module(encoded)
        return self.auxiliary_module.compute_losses(predictions, targets, epoch)

    def get_auxiliary_predictions(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from auxiliary task heads without computing loss."""
        if self.auxiliary_module is None:
            return {}
        encoded = self.get_encoder_output(states)
        return self.auxiliary_module(encoded)

    def get_policy_parameters(self):
        return self.policy_net.parameters()

    def get_value_parameters(self):
        return self.value_net.parameters()

    def get_auxiliary_parameters(self):
        if self.auxiliary_module is None:
            return []
        return self.auxiliary_module.parameters()

    def get_all_parameters(self):
        return list(self.policy_net.parameters()) + list(self.value_net.parameters())

    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "factored": True,  # Mark as factored agent
        }
        if self.auxiliary_module is not None:
            checkpoint["auxiliary_module"] = self.auxiliary_module.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Saved factored agent to {path}")

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check if loading old discrete checkpoint
        if not checkpoint.get("factored", False):
            logger.warning("Loading non-factored checkpoint into factored agent - encoder weights only")
            # Try to load encoder weights only
            old_policy = checkpoint.get("policy_net", {})
            new_policy = self.policy_net.state_dict()
            
            # Transfer encoder weights
            for key in list(old_policy.keys()):
                if key.startswith("encoder.") and key in new_policy:
                    new_policy[key] = old_policy[key]
            
            self.policy_net.load_state_dict(new_policy)
            self.value_net.load_state_dict(checkpoint["value_net"])
        else:
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.value_net.load_state_dict(checkpoint["value_net"])
        
        if self.auxiliary_module is not None and "auxiliary_module" in checkpoint:
            self.auxiliary_module.load_state_dict(checkpoint["auxiliary_module"])
        
        logger.info(f"Loaded factored agent from {path}")

    def train(self) -> None:
        self.policy_net.train()
        self.value_net.train()
        if self.auxiliary_module is not None:
            self.auxiliary_module.train()

    def eval(self) -> None:
        self.policy_net.eval()
        self.value_net.eval()
        if self.auxiliary_module is not None:
            self.auxiliary_module.eval()
