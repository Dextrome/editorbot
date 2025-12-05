"""Tests for logging utilities."""

import pytest
import numpy as np
from pathlib import Path
from rl_editor.config import Config
from rl_editor.logging_utils import TrainingLogger, create_logger


class TestTrainingLogger:
    """Test TrainingLogger class."""

    @pytest.fixture
    def log_dir(self, tmp_path):
        return str(tmp_path / "logs")

    def test_logger_init_no_backends(self, log_dir):
        """Test logger initialization without any backends."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        assert logger.step == 0
        assert logger.run_dir.exists()
        logger.close()

    def test_logger_init_tensorboard(self, log_dir):
        """Test logger initialization with TensorBoard."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test_tb",
            use_tensorboard=True,
            use_wandb=False,
        )
        # TensorBoard may or may not be available
        assert logger.run_dir.exists()
        logger.close()

    def test_log_scalar(self, log_dir):
        """Test scalar logging."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        # Should not raise even without backends
        logger.log_scalar("test/value", 1.0, step=0)
        logger.log_scalar("test/value", 2.0)  # Uses internal step
        logger.close()

    def test_log_metrics(self, log_dir):
        """Test batch metric logging."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        metrics = {"loss": 0.5, "accuracy": 0.9, "reward": 10.0}
        logger.log_metrics(metrics, step=0)
        logger.close()

    def test_log_training_step(self, log_dir):
        """Test training step logging."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        logger.log_training_step(
            policy_loss=0.1,
            value_loss=0.2,
            episode_reward=100.0,
            episode_length=50,
            entropy=0.5,
            learning_rate=0.001,
            step=0,
        )
        logger.close()

    def test_log_evaluation(self, log_dir):
        """Test evaluation logging."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        logger.log_evaluation(
            mean_reward=100.0,
            std_reward=10.0,
            min_reward=50.0,
            max_reward=150.0,
            mean_length=100.0,
            step=0,
        )
        logger.close()

    def test_log_histogram(self, log_dir):
        """Test histogram logging."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        values = np.random.randn(100)
        logger.log_histogram("test/values", values, step=0)
        logger.close()

    def test_step_counter(self, log_dir):
        """Test step counter."""
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="test",
            use_tensorboard=False,
            use_wandb=False,
        )
        assert logger.step == 0
        logger.increment_step()
        assert logger.step == 1
        logger.set_step(10)
        assert logger.step == 10
        logger.close()

    def test_create_logger_from_config(self, log_dir):
        """Test creating logger from config."""
        config = Config()
        config.training.log_dir = log_dir
        config.training.use_tensorboard = False
        config.training.use_wandb = False
        
        logger = create_logger(config)
        assert logger is not None
        logger.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
