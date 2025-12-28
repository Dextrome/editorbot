"""Logging utilities for training with TensorBoard and Weights & Biases.

Shared between rl_editor and super_editor for consistent logging.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Unified training logger supporting TensorBoard and W&B."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "training",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "audio-editor",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            config: Configuration dict to log
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
                logger.info(f"TensorBoard logging to {self.run_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # Initialize W&B
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=f"{experiment_name}_{timestamp}",
                    config=config,
                    dir=str(self.run_dir),
                )
                logger.info(f"W&B logging to project {wandb_project}")
            except ImportError:
                logger.warning("W&B not available. Install with: pip install wandb")
                self.use_wandb = False

        self.step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the metric
            value: Value to log
            step: Global step (uses internal counter if None)
        """
        if step is None:
            step = self.step

        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)

        if self.wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple related scalars.

        Args:
            main_tag: Main category name
            tag_scalar_dict: Dict of tag -> value
            step: Global step
        """
        if step is None:
            step = self.step

        if self.tb_writer:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)

        if self.wandb_run:
            import wandb
            wandb.log({f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log a histogram of values.

        Args:
            tag: Name of the histogram
            values: Array of values
            step: Global step
        """
        if step is None:
            step = self.step

        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_audio(self, tag: str, audio: np.ndarray, sample_rate: int, step: Optional[int] = None) -> None:
        """Log audio sample.

        Args:
            tag: Name of the audio
            audio: Audio array
            sample_rate: Sample rate
            step: Global step
        """
        if step is None:
            step = self.step

        if self.tb_writer:
            # TensorBoard expects shape (1, L) or (2, L)
            if audio.ndim == 1:
                audio_tb = audio.reshape(1, -1)
            else:
                audio_tb = audio
            self.tb_writer.add_audio(tag, audio_tb, step, sample_rate=sample_rate)

        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Audio(audio, sample_rate=sample_rate)}, step=step)

    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log image.

        Args:
            tag: Name of the image
            image: Image array (H, W, C) or (H, W)
            step: Global step
        """
        if step is None:
            step = self.step

        if self.tb_writer:
            # TensorBoard expects (C, H, W)
            if image.ndim == 2:
                image_tb = image[np.newaxis, :, :]
            elif image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
                image_tb = image.transpose(2, 0, 1)
            else:
                image_tb = image
            self.tb_writer.add_image(tag, image_tb, step)

        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_figure(self, tag: str, figure, step: Optional[int] = None) -> None:
        """Log a matplotlib figure.

        Args:
            tag: Name of the figure
            figure: Matplotlib figure
            step: Global step
        """
        if step is None:
            step = self.step

        if self.tb_writer:
            self.tb_writer.add_figure(tag, figure, step)

        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Image(figure)}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dict of metric_name -> value
            step: Global step
        """
        if step is None:
            step = self.step

        for tag, value in metrics.items():
            # Handle nested dicts
            if isinstance(value, dict):
                for subk, subv in value.items():
                    out_tag = f"{tag}/{subk}"
                    try:
                        self.log_scalar(out_tag, float(subv), step)
                    except (TypeError, ValueError):
                        pass
                continue

            # Default: log scalar
            try:
                self.log_scalar(tag, float(value), step)
            except (TypeError, ValueError):
                pass

    def log_training_step(
        self,
        loss: float,
        learning_rate: Optional[float] = None,
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Log standard training step metrics.

        Args:
            loss: Training loss
            learning_rate: Current learning rate
            step: Global step
            **kwargs: Additional metrics to log
        """
        if step is None:
            step = self.step

        metrics = {"train/loss": loss}

        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        # Add any additional metrics
        for key, value in kwargs.items():
            if not key.startswith("train/"):
                key = f"train/{key}"
            metrics[key] = value

        self.log_metrics(metrics, step)

    def log_validation(
        self,
        val_loss: float,
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Log validation metrics.

        Args:
            val_loss: Validation loss
            step: Global step
            **kwargs: Additional metrics to log
        """
        if step is None:
            step = self.step

        metrics = {"val/loss": val_loss}

        for key, value in kwargs.items():
            if not key.startswith("val/"):
                key = f"val/{key}"
            metrics[key] = value

        self.log_metrics(metrics, step)

    def log_evaluation(
        self,
        mean_reward: float,
        std_reward: float,
        min_reward: float,
        max_reward: float,
        mean_length: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log RL evaluation metrics.

        Args:
            mean_reward: Mean evaluation reward
            std_reward: Std of evaluation reward
            min_reward: Minimum reward
            max_reward: Maximum reward
            mean_length: Mean episode length (optional)
            step: Global step
        """
        if step is None:
            step = self.step

        metrics = {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/min_reward": min_reward,
            "eval/max_reward": max_reward,
        }

        if mean_length is not None:
            metrics["eval/mean_length"] = mean_length

        self.log_metrics(metrics, step)

    def increment_step(self) -> None:
        """Increment internal step counter."""
        self.step += 1

    def set_step(self, step: int) -> None:
        """Set internal step counter."""
        self.step = step

    def flush(self) -> None:
        """Flush any buffered data."""
        if self.tb_writer:
            self.tb_writer.flush()

    def close(self) -> None:
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.wandb_run:
            import wandb
            wandb.finish()

        logger.info("Training logger closed")


def create_logger(
    log_dir: str = "./logs",
    experiment_name: str = "training",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "audio-editor",
    config: Optional[Dict[str, Any]] = None,
) -> TrainingLogger:
    """Create training logger.

    Args:
        log_dir: Directory for logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use W&B
        wandb_project: W&B project name
        config: Configuration to log

    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        config=config,
    )


def create_logger_from_config(config) -> TrainingLogger:
    """Create training logger from a config object.

    Works with both rl_editor.config.Config and super_editor.config.Phase1Config/Phase2Config.

    Args:
        config: Config object with training settings

    Returns:
        TrainingLogger instance
    """
    # Handle rl_editor config
    if hasattr(config, 'training'):
        return TrainingLogger(
            log_dir=config.training.log_dir,
            experiment_name="rl_audio_editor",
            use_tensorboard=config.training.use_tensorboard,
            use_wandb=config.training.use_wandb,
            wandb_project=config.training.wandb_project,
            config=config.to_dict() if hasattr(config, 'to_dict') else None,
        )

    # Handle super_editor config (Phase1Config or Phase2Config)
    log_dir = getattr(config, 'log_dir', './logs')
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name="super_editor",
        use_tensorboard=True,
        use_wandb=False,
        config=None,
    )
