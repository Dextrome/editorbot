"""Policy and value networks for RL agent.

Uses hybrid NATTEN encoder: local neighborhood attention + global pooling.
This provides efficient local context (O(n*k) instead of O(n²)) while
maintaining global awareness through pooled summary features.

Architecture:
- 3-head factored policy: type (18), size (5), amount (5) = 450 combinations
- Shared encoder between policy heads
- Separate value network with same encoder architecture
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .config import Config
from .actions import (
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace, EditHistoryFactored,
)

logger = logging.getLogger(__name__)

# Check if NATTEN is available and has a working backend
NATTEN_AVAILABLE = False
try:
    import natten
    # Test if NATTEN actually works (has backend)
    _test_q = torch.randn(1, 8, 1, 8)
    try:
        natten.na1d(_test_q, _test_q, _test_q, kernel_size=3)
        NATTEN_AVAILABLE = True
        logger.info("NATTEN available with working backend")
    except (NotImplementedError, RuntimeError):
        logger.warning("NATTEN installed but no working backend (needs PyTorch 2.7+ or build from source). Using sliding window fallback.")
    del _test_q
except ImportError:
    logger.info("NATTEN not installed. Using sliding window attention fallback.")

# Explicitly log which attention implementation will be used at module import time.
if NATTEN_AVAILABLE:
    logger.info("Using natten neighborhood-attention implementation for HybridNATTENEncoder.")
else:
    logger.info("Using sliding-window attention fallback (pure PyTorch) for HybridNATTENEncoder.")


class NATTENLayer(nn.Module):
    """Single NATTEN layer with neighborhood attention.

    Performs local self-attention within a sliding window (kernel_size),
    which is O(n*k) instead of O(n²) for standard attention.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        kernel_size: int = 31,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kernel_size = kernel_size
        self.dilation = dilation

        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with neighborhood attention.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        B, L, D = x.shape
        residual = x
        
        # QKV projection
        qkv = self.qkv(x)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, B, L, H, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, L, H, head_dim)
        
        # Determine effective kernel size (must be <= sequence length, odd, and >= 3)
        effective_kernel = min(self.kernel_size, L)
        if effective_kernel % 2 == 0:
            effective_kernel = max(3, effective_kernel - 1)
        effective_kernel = max(3, effective_kernel)  # NATTEN requires kernel >= 2, we use 3 for safety
        
        # Apply NATTEN neighborhood attention or sliding window fallback
        if NATTEN_AVAILABLE and L >= effective_kernel:
            out = natten.na1d(q, k, v, kernel_size=effective_kernel, dilation=self.dilation)
        else:
            # Sliding window attention fallback (provides same locality as NATTEN)
            out = self._sliding_window_attention(q, k, v)
        
        # Reshape and project
        out = out.reshape(B, L, D)
        out = self.proj(out)
        out = self.dropout(out)
        
        # Residual + LayerNorm
        out = self.norm(out + residual)
        
        return out
    
    def _sliding_window_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Sliding window attention fallback when NATTEN is not available.
        
        Args:
            q, k, v: Query, Key, Value tensors (B, L, H, head_dim)
            
        Returns:
            Output tensor (B, L, H, head_dim)
        """
        B, L, H, D = q.shape
        kernel = self.kernel_size
        half_k = kernel // 2
        
        # Create sliding window attention mask
        positions = torch.arange(L, device=q.device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        mask = dist <= half_k  # (L, L)
        
        # Apply dilation if specified
        if self.dilation > 1:
            dilated_dist = dist // self.dilation
            mask = (dilated_dist <= half_k) & (dist % self.dilation == 0)
        
        # Transpose for attention computation: (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = D ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)
        
        # Apply sliding window mask
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax over keys
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, L, D)
        out = out.transpose(1, 2)  # (B, L, H, D)
        
        return out


class HybridNATTENEncoder(nn.Module):
    """Hybrid encoder combining local NATTEN attention with global pooling.
    
    Architecture:
    1. Project input features to hidden dimension
    2. Apply N layers of NATTEN (local neighborhood attention)
    3. Pool global summary (mean over sequence)
    4. Concatenate local features with global summary
    5. Final projection to output dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        kernel_size: int = 31,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Stack of NATTEN layers for local context
        self.natten_layers = nn.ModuleList([
            NATTENLayer(
                dim=hidden_dim,
                n_heads=n_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Global pooling projection
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Final combination: local + global -> hidden_dim
        self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Feed-forward network after combination
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(
            f"HybridNATTENEncoder: input={input_dim}, hidden={hidden_dim}, "
            f"heads={n_heads}, layers={n_layers}, kernel={kernel_size}, dilation={dilation}"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hybrid local-global encoding.
        
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            
        Returns:
            Encoded tensor (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
        """
        # Use autocast for FP16 speedup (guards against NaN below)
        with torch.amp.autocast('cuda', enabled=True):
            original_dtype = x.dtype
            
            # Handle 2D input (single position, no sequence)
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_output = True
            else:
                squeeze_output = False
                
            B, L, _ = x.shape
            
            # Project to hidden dimension
            x = self.input_projection(x)
            
            # Apply NATTEN layers for local context
            local_features = x
            for layer in self.natten_layers:
                local_features = layer(local_features)
            
            # Global pooling
            global_summary = x.mean(dim=1, keepdim=True)
            global_summary = self.global_proj(global_summary)
            global_summary = global_summary.expand(-1, L, -1)
            
            # Combine local and global
            combined = torch.cat([local_features, global_summary], dim=-1)
            combined = self.combine_proj(combined)
            
            # Feed-forward + residual
            out = combined + self.ffn(combined)
            out = self.final_norm(out)
            
            if squeeze_output:
                out = out.squeeze(1)
            
            out = torch.clamp(out, -100.0, 100.0)
                
        return out


class ValueNetwork(nn.Module):
    """Value network for state evaluation."""

    def __init__(self, config: Config, input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        model_config = config.model

        self.encoder = HybridNATTENEncoder(
            input_dim=input_dim,
            hidden_dim=model_config.value_hidden_dim,
            n_heads=model_config.natten_n_heads,
            n_layers=model_config.natten_n_layers,
            kernel_size=model_config.natten_kernel_size,
            dilation=model_config.natten_dilation,
            dropout=model_config.value_dropout,
        )
        
        self.head = nn.Linear(model_config.value_hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state)
        value = self.head(encoded)
        return value


class PolicyNetwork(nn.Module):
    """Multi-head policy network for factored action space.
    
    Outputs logits for 3 action components:
    - Type: What action (18 options)
    - Size: How many beats (5 options)
    - Amount: Intensity/direction (5 options)
    """

    def __init__(self, config: Config, input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        model_config = config.model

        # Shared encoder
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
        
        # Type head: What action to take
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_ACTION_TYPES),
        )
        
        # Type embedding for conditioning size/amount heads
        self.type_embedding = nn.Embedding(N_ACTION_TYPES, 32)
        
        # Size head: How many beats (conditioned on type)
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
        
        logger.info(f"PolicyNetwork: {N_ACTION_TYPES} types, {N_ACTION_SIZES} sizes, {N_ACTION_AMOUNTS} amounts")

    def forward(
        self,
        state: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        size_mask: Optional[torch.Tensor] = None,
        amount_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for all three heads."""
        encoded = self.encoder(state)
        
        # Type logits
        type_logits = self.type_head(encoded)
        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask.bool(), float("-inf"))
        type_logits = type_logits / max(temperature, 0.1)
        
        # Soft attention over types for conditioning
        type_probs = F.softmax(type_logits, dim=-1)
        type_embed = torch.matmul(type_probs, self.type_embedding.weight)
        
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
        """Sample actions and compute log probabilities."""
        encoded = self.encoder(state)
        
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
        type_embed = self.type_embedding(type_action)
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
        
        # Combined log probability
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
        """Evaluate log probability and entropy for given actions."""
        encoded = self.encoder(state)
        
        # === Type evaluation ===
        type_logits = self.type_head(encoded)
        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask.bool(), float("-inf"))
        type_logits = torch.clamp(type_logits, -20.0, 20.0)
        
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
        
        size_dist = torch.distributions.Categorical(logits=size_logits)
        size_log_prob = size_dist.log_prob(size_action)
        size_entropy = size_dist.entropy()
        
        # === Amount evaluation ===
        amount_input = torch.cat([encoded, type_embed], dim=-1)
        amount_logits = self.amount_head(amount_input)
        if amount_mask is not None:
            amount_logits = amount_logits.masked_fill(~amount_mask.bool(), float("-inf"))
        amount_logits = torch.clamp(amount_logits, -20.0, 20.0)
        
        amount_dist = torch.distributions.Categorical(logits=amount_logits)
        amount_log_prob = amount_dist.log_prob(amount_action)
        amount_entropy = amount_dist.entropy()
        
        # Combined
        combined_log_prob = type_log_prob + size_log_prob + amount_log_prob
        # Normalize entropy by number of heads (3) so it's comparable to single-head policy
        # Without this, max entropy = ln(20)+ln(5)+ln(5) ≈ 6.2, overwhelming reward signal
        combined_entropy = (type_entropy + size_entropy + amount_entropy) / 3.0
        
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


class Agent:
    """RL Agent with multi-head factored policy and value networks."""

    def __init__(
        self,
        config: Config,
        input_dim: int,
        beat_feature_dim: int = 121,
        use_auxiliary_tasks: bool = True,
    ) -> None:
        self.config = config
        self.device = torch.device(config.training.device)
        self.use_auxiliary_tasks = use_auxiliary_tasks

        self.policy_net = PolicyNetwork(config, input_dim).to(self.device)
        self.value_net = ValueNetwork(config, input_dim).to(self.device)
        self._compiled = False
        
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
            f"Initialized Agent with {input_dim} input dims, "
            f"{N_ACTION_TYPES} types, {N_ACTION_SIZES} sizes, {N_ACTION_AMOUNTS} amounts"
        )

    def compile_models(self):
        """Compile models with torch.compile for faster inference.

        Call this AFTER loading checkpoint weights.
        Note: Disabled on Windows due to Triton not being available.
        """
        if self._compiled:
            return
        # torch.compile requires Triton which isn't available on Windows
        # Skip compilation - eager mode is fine
        import sys
        if sys.platform == "win32":
            logger.info("Skipping torch.compile on Windows (Triton not available)")
            return
        try:
            self.policy_net = torch.compile(self.policy_net, mode="reduce-overhead")
            self.value_net = torch.compile(self.value_net, mode="reduce-overhead")
            self._compiled = True
            logger.info("Models compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager mode: {e}")

    def select_action(
        self,
        state: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        size_mask: Optional[torch.Tensor] = None,
        amount_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tuple[int, int, int], float]:
        """Select factored action given state."""
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
        """Select actions for batch of states."""
        with torch.no_grad():
            return self.policy_net.get_action_and_log_prob(
                states, type_masks, size_masks, amount_masks, deterministic
            )

    def compute_value_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Compute values for batch of states."""
        # Ensure states has a batch dimension
        if states.dim() == 1:
            states = states.unsqueeze(0)
        values = self.value_net(states)
        values = values.float()
        values = torch.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
        values = torch.clamp(values, -100.0, 100.0)
        out = values.squeeze(-1)
        # Defensive logging for unexpected shapes
        if out.dim() == 0:
            logger.warning(f"compute_value_batch produced scalar output; states.shape={states.shape}, values.shape={values.shape}, out.shape={out.shape}")
        return out

    def compute_value(self, state: torch.Tensor) -> float:
        """Compute value of single state."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            value = self.value_net(state)
            # Ensure shape is (B,) after squeezing trailing dim
            value = value.squeeze(-1)
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
        """Get predictions from auxiliary task heads."""
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
            "version": "merged",
        }
        if self.auxiliary_module is not None:
            checkpoint["auxiliary_module"] = self.auxiliary_module.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Saved agent to {path}")

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        version = checkpoint.get("version", "unknown")
        
        if version in ("factored", "merged"):
            # New format from train.py
            policy_key = "policy_state_dict" if "policy_state_dict" in checkpoint else "policy_net"
            value_key = "value_state_dict" if "value_state_dict" in checkpoint else "value_net"
            aux_key = "auxiliary_state_dict" if "auxiliary_state_dict" in checkpoint else "auxiliary_module"
            
            self.policy_net.load_state_dict(checkpoint[policy_key])
            self.value_net.load_state_dict(checkpoint[value_key])
            if self.auxiliary_module is not None and checkpoint.get(aux_key):
                self.auxiliary_module.load_state_dict(checkpoint[aux_key])
            logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('current_epoch', 'N/A')})")
        else:
            # Legacy checkpoint - transfer encoder weights only
            logger.warning("Loading legacy checkpoint - encoder weights only")
            old_policy = checkpoint.get("policy_state_dict", checkpoint.get("policy_net", {}))
            new_policy = self.policy_net.state_dict()
            
            transferred = 0
            for key in list(old_policy.keys()):
                if key.startswith("encoder.") and key in new_policy:
                    new_policy[key] = old_policy[key]
                    transferred += 1
            
            self.policy_net.load_state_dict(new_policy)
            old_value = checkpoint.get("value_state_dict", checkpoint.get("value_net", {}))
            if old_value:
                self.value_net.load_state_dict(old_value)
            logger.info(f"Transferred {transferred} encoder weights from legacy checkpoint")

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


# Aliases for backward compatibility
FactoredAgent = Agent
FactoredPolicyNetwork = PolicyNetwork
