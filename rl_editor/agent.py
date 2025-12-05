"""Policy and value networks for RL agent.

Includes transformer-based option for better context understanding.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .config import Config

logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_heads: int = 4, n_layers: int = 2
    ) -> None:
        """Initialize transformer encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)

        Returns:
            Encoded tensor (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
        """
        # Handle both 2D and 3D input
        if x.dim() == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        x = self.input_projection(x)
        x = self.transformer(x)

        if squeeze_output:
            x = x.squeeze(1)

        return x


class PolicyNetwork(nn.Module):
    """Policy network for action selection.

    Outputs action logits for discrete action space.
    """

    def __init__(
        self, config: Config, input_dim: int, n_actions: int
    ) -> None:
        """Initialize policy network.

        Args:
            config: Configuration object
            input_dim: Input state dimension
            n_actions: Number of actions
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.n_actions = n_actions

        model_config = config.model

        if model_config.policy_use_transformer:
            # Transformer-based policy
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=model_config.policy_hidden_dim,
                n_heads=model_config.transformer_n_heads,
                n_layers=model_config.transformer_n_layers,
            )
            self.head = nn.Linear(model_config.policy_hidden_dim, n_actions)
        else:
            # MLP-based policy
            layers = []
            prev_dim = input_dim
            for _ in range(model_config.policy_n_layers):
                layers.append(nn.Linear(prev_dim, model_config.policy_hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(model_config.policy_dropout))
                prev_dim = model_config.policy_hidden_dim

            self.encoder = nn.Sequential(*layers)
            self.head = nn.Linear(prev_dim, n_actions)

    def forward(
        self, state: torch.Tensor, mask: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: State tensor (batch_size, input_dim)
            mask: Optional action mask (batch_size, n_actions) binary
            temperature: Softmax temperature for exploration (higher = more random)

        Returns:
            Tuple of (action_logits, action_probs)
        """
        # Encode state
        encoded = self.encoder(state)

        # Compute logits
        logits = self.head(encoded)

        # Apply mask if provided (set invalid actions to -inf)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))

        # Apply temperature scaling for exploration (higher temp = more exploration)
        scaled_logits = logits / max(temperature, 0.1)

        # Compute probabilities
        probs = torch.softmax(scaled_logits, dim=-1)

        return logits, probs

    def get_logits(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get action logits for a batch of states.

        Args:
            state: State tensor (batch_size, input_dim)
            mask: Optional action mask (batch_size, n_actions)

        Returns:
            Logits tensor (batch_size, n_actions)
        """
        # Handle NaN in input states
        if torch.isnan(state).any():
            state = torch.nan_to_num(state, nan=0.0)
            
        encoded = self.encoder(state)
        
        # Handle NaN in encoded
        if torch.isnan(encoded).any():
            encoded = torch.nan_to_num(encoded, nan=0.0)
            
        logits = self.head(encoded)
        
        # Handle NaN in logits
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)
            
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return logits

    def get_action_and_value(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability.

        Args:
            state: State tensor
            mask: Optional action mask
            deterministic: If True, return greedy action; else sample from distribution

        Returns:
            Tuple of (action, log_prob)
        """
        logits, probs = self.forward(state, mask)
        
        # Handle case where all actions are masked (probs sum to 0 or have NaN)
        # Replace NaN/zero rows with uniform distribution
        probs_sum = probs.sum(dim=-1, keepdim=True)
        invalid_mask = (probs_sum < 1e-8) | torch.isnan(probs_sum)
        if invalid_mask.any():
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = torch.where(invalid_mask.expand_as(probs), uniform, probs)
            # Re-normalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


class ValueNetwork(nn.Module):
    """Value network for state evaluation.

    Outputs state value for TD learning.
    """

    def __init__(self, config: Config, input_dim: int) -> None:
        """Initialize value network.

        Args:
            config: Configuration object
            input_dim: Input state dimension
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        model_config = config.model

        # MLP architecture for value network (transformer typically used for policy only)
        layers = []
        prev_dim = input_dim
        for _ in range(model_config.value_n_layers):
            layers.append(nn.Linear(prev_dim, model_config.value_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_config.value_dropout))
            prev_dim = model_config.value_hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor (batch_size, input_dim)

        Returns:
            State value tensor (batch_size,)
        """
        value = self.network(state).squeeze(-1)
        return value


class Agent:
    """RL agent combining policy and value networks."""

    def __init__(self, config: Config, input_dim: int, n_actions: int) -> None:
        """Initialize agent.

        Args:
            config: Configuration object
            input_dim: Input state dimension
            n_actions: Number of actions
        """
        self.config = config
        self.device = torch.device(config.training.device)

        self.policy_net = PolicyNetwork(config, input_dim, n_actions).to(self.device)
        self.value_net = ValueNetwork(config, input_dim).to(self.device)

        logger.info(
            f"Initialized agent with {input_dim} input dims, {n_actions} actions on device {self.device}"
        )

    def select_action(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Select action given state.

        Args:
            state: State tensor
            mask: Optional action mask
            deterministic: Use greedy policy

        Returns:
            Tuple of (action_index, log_prob)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action, log_prob = self.policy_net.get_action_and_value(
                state, mask, deterministic
            )

            return int(action[0].item()), float(log_prob[0].item())

    def select_action_batch(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select actions for a batch of states.

        Args:
            states: Batch of state tensors (B, state_dim)
            masks: Batch of action masks (B, n_actions)
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (actions, log_probs) tensors
        """
        with torch.no_grad():
            actions, log_probs = self.policy_net.get_action_and_value(
                states, masks, deterministic
            )
            return actions, log_probs

    def compute_value(self, state: torch.Tensor) -> float:
        """Compute value of state.

        Args:
            state: State tensor

        Returns:
            State value
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            value = self.value_net(state)
            return float(value[0].item())

    def compute_value_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Compute values for a batch of states.

        Args:
            states: Batch of state tensors (B, state_dim)

        Returns:
            Values tensor (B,)
        """
        values = self.value_net(states)
        return values.squeeze(-1)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        min_entropy: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given state-action pairs.

        Args:
            states: Batch of states (B, state_dim)
            actions: Batch of actions (B,)
            min_entropy: Minimum entropy to maintain exploration

        Returns:
            Tuple of (log_probs, entropy) tensors
        """
        logits = self.policy_net.get_logits(states)
        
        # Handle NaN values in logits
        if torch.isnan(logits).any():
            logger.warning("NaN in logits during evaluate_actions, replacing with zeros")
            logits = torch.nan_to_num(logits, nan=0.0)
        
        # Add small noise to logits to prevent distribution from collapsing
        # This keeps some exploration even when policy becomes confident
        noise = torch.randn_like(logits) * 0.01
        logits = logits + noise
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Ensure minimum entropy to prevent policy collapse
        entropy = torch.clamp(entropy, min=min_entropy)
        
        return log_probs, entropy

    def get_policy_parameters(self):
        """Get policy network parameters for optimization."""
        return self.policy_net.parameters()

    def get_value_parameters(self):
        """Get value network parameters for optimization."""
        return self.value_net.parameters()

    def get_all_parameters(self):
        """Get all network parameters."""
        return list(self.policy_net.parameters()) + list(self.value_net.parameters())

    def save(self, path: str) -> None:
        """Save agent networks.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved agent checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load agent networks.

        Args:
            path: Path to load checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        logger.info(f"Loaded agent checkpoint from {path}")

    def train(self) -> None:
        """Set networks to train mode."""
        self.policy_net.train()
        self.value_net.train()

    def eval(self) -> None:
        """Set networks to eval mode."""
        self.policy_net.eval()
        self.value_net.eval()
