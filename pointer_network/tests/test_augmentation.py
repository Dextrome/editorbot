"""Unit tests for pointer network data augmentation."""

import pytest
import torch
import numpy as np
from pointer_network.data.augmentation import (
    AugmentationConfig,
    Augmentor,
    add_noise,
    freq_mask,
    time_mask,
    apply_freq_mask,
    apply_time_mask,
    scale_amplitude,
    channel_dropout,
    apply_channel_dropout,
    chunk_shuffle,
    random_crop,
)


class TestSafeAugmentations:
    """Tests for Category A augmentations (no pointer adjustment needed)."""

    def test_add_noise_shape_preserved(self):
        """Noise augmentation should preserve tensor shape."""
        mel = torch.randn(128, 1000)
        augmented = add_noise(mel, 0.05, 0.15)
        assert augmented.shape == mel.shape

    def test_add_noise_modifies_values(self):
        """Noise should actually change the values."""
        mel = torch.randn(128, 1000)
        augmented = add_noise(mel, 0.1, 0.2)
        assert not torch.allclose(mel, augmented)

    def test_add_noise_within_reasonable_range(self):
        """Noise should be within expected magnitude."""
        mel = torch.ones(128, 1000) * 10
        augmented = add_noise(mel, 0.1, 0.1)
        diff = (augmented - mel).abs()
        # Noise should be roughly 10 * 0.1 * randn std ≈ 1.0
        assert diff.mean() < 5.0  # Very generous bound

    def test_freq_mask_shape_preserved(self):
        """Frequency masking should preserve shape."""
        mel = torch.randn(128, 1000)
        augmented, masks = freq_mask(mel, num_masks=2, max_width=20)
        assert augmented.shape == mel.shape

    def test_freq_mask_creates_zeros(self):
        """Frequency masking should create zero bands."""
        mel = torch.ones(128, 1000)
        augmented, masks = freq_mask(mel, num_masks=2, max_width=20)
        # Should have some zeros
        assert (augmented == 0).any()
        # But not all zeros
        assert (augmented != 0).any()

    def test_freq_mask_returns_reproducible_masks(self):
        """Should be able to apply same masks to another tensor."""
        mel1 = torch.ones(128, 1000)
        mel2 = torch.ones(128, 1000) * 2
        _, masks = freq_mask(mel1, num_masks=2, max_width=20)
        mel2_masked = apply_freq_mask(mel2, masks)
        # Same positions should be zeroed
        assert (mel2_masked == 0).any()

    def test_time_mask_shape_preserved(self):
        """Time masking should preserve shape."""
        mel = torch.randn(128, 1000)
        augmented, masks = time_mask(mel, num_masks=2, max_width=50)
        assert augmented.shape == mel.shape

    def test_time_mask_creates_zeros(self):
        """Time masking should create zero time segments."""
        mel = torch.ones(128, 1000)
        augmented, masks = time_mask(mel, num_masks=2, max_width=50)
        assert (augmented == 0).any()
        assert (augmented != 0).any()

    def test_scale_amplitude_shape_preserved(self):
        """Amplitude scaling should preserve shape."""
        mel = torch.randn(128, 1000)
        augmented, scale = scale_amplitude(mel, 0.7, 1.3)
        assert augmented.shape == mel.shape

    def test_scale_amplitude_correct_scaling(self):
        """Amplitude scaling should use correct factor."""
        mel = torch.ones(128, 1000)
        augmented, scale = scale_amplitude(mel, 0.5, 0.5)  # Fixed scale
        assert torch.allclose(augmented, mel * 0.5)

    def test_channel_dropout_shape_preserved(self):
        """Channel dropout should preserve shape."""
        mel = torch.randn(128, 1000)
        augmented, mask = channel_dropout(mel, dropout_prob=0.1)
        assert augmented.shape == mel.shape

    def test_channel_dropout_zeros_some_channels(self):
        """Channel dropout should zero some channels."""
        # Use high dropout to ensure some channels are dropped
        mel = torch.ones(128, 1000)
        augmented, mask = channel_dropout(mel, dropout_prob=0.5)
        # Some rows should be all zeros
        row_sums = augmented.sum(dim=1)
        assert (row_sums == 0).any()
        assert (row_sums != 0).any()

    def test_channel_dropout_reproducible(self):
        """Should be able to apply same dropout mask to another tensor."""
        mel1 = torch.ones(128, 1000)
        mel2 = torch.ones(128, 1000) * 2
        mel1_dropped, mask = channel_dropout(mel1, dropout_prob=0.5)
        mel2_dropped = apply_channel_dropout(mel2, mask)
        # Same channels should be dropped in both
        row_sums_1 = mel1_dropped.sum(dim=1)
        row_sums_2 = mel2_dropped.sum(dim=1)
        # Zeros should be in same positions
        assert ((row_sums_1 == 0) == (row_sums_2 == 0)).all()


class TestPointerAwareAugmentations:
    """Tests for Category B augmentations (require pointer adjustment)."""

    def test_chunk_shuffle_shapes_preserved(self):
        """Chunk shuffling should preserve all shapes."""
        raw_mel = torch.randn(128, 5000)
        edit_mel = torch.randn(128, 3000)
        pointers = torch.randint(0, 5000, (3000,))

        shuffled_raw, shuffled_edit, new_pointers = chunk_shuffle(
            raw_mel, edit_mel, pointers, chunk_size=1000
        )

        assert shuffled_raw.shape == raw_mel.shape
        assert shuffled_edit.shape == edit_mel.shape
        assert new_pointers.shape == pointers.shape

    def test_chunk_shuffle_pointers_valid(self):
        """After shuffling, pointers should still be valid indices."""
        raw_mel = torch.randn(128, 5000)
        edit_mel = torch.randn(128, 3000)
        pointers = torch.randint(0, 5000, (3000,))

        shuffled_raw, _, new_pointers = chunk_shuffle(
            raw_mel, edit_mel, pointers, chunk_size=1000
        )

        assert (new_pointers >= 0).all()
        assert (new_pointers < shuffled_raw.shape[1]).all()

    def test_chunk_shuffle_content_preserved(self):
        """Content should be preserved, just reordered."""
        raw_mel = torch.randn(128, 5000)
        edit_mel = torch.randn(128, 3000)
        pointers = torch.randint(0, 5000, (3000,))

        shuffled_raw, _, _ = chunk_shuffle(
            raw_mel, edit_mel, pointers, chunk_size=1000
        )

        # Total content should be the same (just reordered)
        assert torch.allclose(raw_mel.sum(), shuffled_raw.sum())

    def test_chunk_shuffle_pointer_content_preserved(self):
        """Pointers should still point to same content after shuffle."""
        raw_mel = torch.randn(128, 5000)
        edit_mel = torch.randn(128, 3000)
        # Use sequential pointers for easy verification
        pointers = torch.arange(3000)

        # Get original pointed content
        original_content = raw_mel[:, pointers]

        shuffled_raw, _, new_pointers = chunk_shuffle(
            raw_mel, edit_mel, pointers, chunk_size=1000
        )

        # Get new pointed content
        new_content = shuffled_raw[:, new_pointers]

        # Content should match
        assert torch.allclose(original_content, new_content)

    def test_chunk_shuffle_short_sequence(self):
        """Short sequences (< 2 chunks) should be unchanged."""
        raw_mel = torch.randn(128, 500)
        edit_mel = torch.randn(128, 300)
        pointers = torch.randint(0, 500, (300,))

        shuffled_raw, shuffled_edit, new_pointers = chunk_shuffle(
            raw_mel, edit_mel, pointers, chunk_size=1000
        )

        assert torch.equal(raw_mel, shuffled_raw)
        assert torch.equal(pointers, new_pointers)

    def test_random_crop_output_shorter(self):
        """Cropped sequences should be shorter or equal."""
        raw_mel = torch.randn(128, 5000)
        edit_mel = torch.randn(128, 3000)
        pointers = torch.randint(0, 5000, (3000,))

        cropped_raw, cropped_edit, new_pointers = random_crop(
            raw_mel, edit_mel, pointers,
            min_crop_len=500, max_crop_len=1000
        )

        assert cropped_edit.shape[1] <= edit_mel.shape[1]
        assert len(new_pointers) <= len(pointers)

    def test_random_crop_pointers_valid(self):
        """After cropping, pointers should be valid indices."""
        raw_mel = torch.randn(128, 5000)
        edit_mel = torch.randn(128, 3000)
        pointers = torch.randint(0, 5000, (3000,))

        cropped_raw, _, new_pointers = random_crop(
            raw_mel, edit_mel, pointers,
            min_crop_len=500, max_crop_len=1000
        )

        assert (new_pointers >= 0).all()
        assert (new_pointers < cropped_raw.shape[1]).all()

    def test_random_crop_short_sequence(self):
        """Sequences shorter than min_crop should be unchanged."""
        raw_mel = torch.randn(128, 500)
        edit_mel = torch.randn(128, 300)
        pointers = torch.randint(0, 500, (300,))

        cropped_raw, cropped_edit, new_pointers = random_crop(
            raw_mel, edit_mel, pointers,
            min_crop_len=500, max_crop_len=1000
        )

        # Should be unchanged
        assert cropped_edit.shape == edit_mel.shape


class TestAugmentor:
    """Tests for the Augmentor class."""

    def test_augmentor_disabled(self):
        """Augmentor with disabled config should not modify inputs."""
        config = AugmentationConfig(enabled=False)
        augmentor = Augmentor(config)

        raw_mel = torch.randn(128, 1000)
        edit_mel = torch.randn(128, 500)
        pointers = torch.randint(0, 1000, (500,))

        aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

        assert torch.equal(raw_mel, aug_raw)
        assert torch.equal(edit_mel, aug_edit)
        assert torch.equal(pointers, aug_ptr)

    def test_augmentor_shapes_preserved(self):
        """Augmentor should preserve tensor shapes (with possible cropping)."""
        config = AugmentationConfig(
            enabled=True,
            crop_enabled=False,  # Disable crop to test shape preservation
        )
        augmentor = Augmentor(config)

        raw_mel = torch.randn(128, 1000)
        edit_mel = torch.randn(128, 500)
        pointers = torch.randint(0, 1000, (500,))

        aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

        assert aug_raw.shape[0] == 128
        assert aug_edit.shape[0] == 128
        # Length might change due to shuffle but should stay reasonable

    def test_augmentor_pointers_valid(self):
        """Augmented pointers should always be valid indices."""
        config = AugmentationConfig(enabled=True)
        augmentor = Augmentor(config)

        for _ in range(10):  # Run multiple times due to randomness
            raw_mel = torch.randn(128, 2000)
            edit_mel = torch.randn(128, 1000)
            pointers = torch.randint(0, 2000, (1000,))

            aug_raw, _, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

            assert (aug_ptr >= 0).all(), "Pointers should be non-negative"
            assert (aug_ptr < aug_raw.shape[1]).all(), "Pointers should be within raw_mel bounds"

    def test_validate_pointers(self):
        """validate_pointers should correctly check pointer validity."""
        augmentor = Augmentor()

        raw_mel = torch.randn(128, 1000)
        valid_pointers = torch.randint(0, 1000, (500,))
        invalid_pointers = torch.randint(1000, 2000, (500,))

        assert augmentor.validate_pointers(raw_mel, valid_pointers)
        assert not augmentor.validate_pointers(raw_mel, invalid_pointers)

    def test_augmentor_no_nan(self):
        """Augmentation should not introduce NaN values."""
        config = AugmentationConfig(enabled=True)
        augmentor = Augmentor(config)

        for _ in range(10):
            raw_mel = torch.randn(128, 2000)
            edit_mel = torch.randn(128, 1000)
            pointers = torch.randint(0, 2000, (1000,))

            aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

            assert not torch.isnan(aug_raw).any(), "raw_mel has NaN"
            assert not torch.isnan(aug_edit).any(), "edit_mel has NaN"


class TestEdgeCases:
    """Tests for edge cases and potential failure modes."""

    def test_empty_sequence(self):
        """Handle empty sequences gracefully."""
        raw_mel = torch.randn(128, 100)
        edit_mel = torch.randn(128, 0)
        pointers = torch.tensor([], dtype=torch.long)

        # Should not crash
        config = AugmentationConfig(enabled=True, crop_enabled=False)
        augmentor = Augmentor(config)
        aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

    def test_very_short_sequence(self):
        """Handle very short sequences."""
        raw_mel = torch.randn(128, 10)
        edit_mel = torch.randn(128, 5)
        pointers = torch.randint(0, 10, (5,))

        config = AugmentationConfig(enabled=True)
        augmentor = Augmentor(config)
        aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

        # Should still have valid pointers
        assert (aug_ptr >= 0).all()
        assert (aug_ptr < aug_raw.shape[1]).all()

    def test_single_frame(self):
        """Handle single-frame sequences."""
        raw_mel = torch.randn(128, 1)
        edit_mel = torch.randn(128, 1)
        pointers = torch.tensor([0])

        config = AugmentationConfig(enabled=True)
        augmentor = Augmentor(config)
        aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

        assert aug_ptr[0] == 0 or aug_ptr[0] < aug_raw.shape[1]

    def test_all_pointers_same(self):
        """Handle all pointers pointing to same frame."""
        raw_mel = torch.randn(128, 1000)
        edit_mel = torch.randn(128, 500)
        pointers = torch.zeros(500, dtype=torch.long)  # All point to frame 0

        config = AugmentationConfig(enabled=True)
        augmentor = Augmentor(config)
        aug_raw, aug_edit, aug_ptr = augmentor(raw_mel, edit_mel, pointers)

        # All pointers should still be valid
        assert (aug_ptr >= 0).all()
        assert (aug_ptr < aug_raw.shape[1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
