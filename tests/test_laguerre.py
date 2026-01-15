"""Tests for EMA and LaguerreFeatures."""

import math

import pytest
import torch

from box_pricer import EMA, LaguerreFeatures


class TestEMA:
    """Tests for the EMA class."""

    def test_init_with_alpha(self):
        """Test initialization with alpha parameter."""
        ema = EMA(alpha=0.1)
        assert ema.alpha == 0.1

    def test_init_with_characteristic_time(self):
        """Test initialization with characteristic_time parameter."""
        ct = 10.0
        ema = EMA(characteristic_time=ct)
        expected_alpha = 1 - math.exp(-1 / ct)
        assert abs(ema.alpha - expected_alpha) < 1e-10

    def test_init_requires_one_param(self):
        """Test that exactly one of alpha or characteristic_time is required."""
        with pytest.raises(ValueError, match="Exactly one"):
            EMA()
        with pytest.raises(ValueError, match="Exactly one"):
            EMA(alpha=0.1, characteristic_time=10.0)

    def test_alpha_bounds(self):
        """Test that alpha must be in (0, 1)."""
        with pytest.raises(ValueError, match="must be in the range"):
            EMA(alpha=0.0)
        with pytest.raises(ValueError, match="must be in the range"):
            EMA(alpha=1.0)
        with pytest.raises(ValueError, match="must be in the range"):
            EMA(alpha=-0.1)

    def test_alpha_too_large(self):
        """Test warning for unexpectedly large alpha."""
        with pytest.raises(ValueError, match="unexpectedly large"):
            EMA(alpha=0.9)

    def test_output_shape_2d(self):
        """Test that output shape matches input for 2D tensors."""
        ema = EMA(characteristic_time=10.0)
        x = torch.randn(8, 100)
        y = ema(x)
        assert y.shape == x.shape

    def test_output_shape_3d(self):
        """Test that output shape matches input for 3D tensors."""
        ema = EMA(characteristic_time=10.0)
        x = torch.randn(4, 50, 3)
        y = ema(x)
        assert y.shape == x.shape

    def test_invalid_ndim(self):
        """Test that 1D and 4D+ inputs raise errors."""
        ema = EMA(characteristic_time=10.0)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            ema(torch.randn(100))
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            ema(torch.randn(2, 3, 4, 5))

    def test_differentiable(self):
        """Test that gradients flow through EMA."""
        torch.manual_seed(42)
        ema = EMA(characteristic_time=10.0)
        # Use positive data to avoid log(0) issues near the minimum shift
        x = (torch.rand(4, 50) + 1.0).requires_grad_(True)
        y = ema(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_ema_smoothing_effect(self):
        """Test that EMA actually smooths the signal."""
        ema = EMA(characteristic_time=20.0)
        # Create a noisy signal
        t = torch.linspace(0, 10, 200)
        signal = torch.sin(t) + 0.5 * torch.randn_like(t)
        signal = signal.unsqueeze(0)  # (1, 200)

        smoothed = ema(signal)

        # Smoothed signal should have lower variance
        assert smoothed.std() < signal.std()

    def test_constant_signal_unchanged(self):
        """Test that a constant signal is approximately unchanged by EMA."""
        ema = EMA(characteristic_time=10.0, warm_start=5.0)
        x = torch.full((4, 100), 5.0)
        y = ema(x)
        # After warm-up, should converge to constant
        assert torch.allclose(y[:, -10:], x[:, -10:], atol=1e-3)

    def test_adjust_mode(self):
        """Test that adjust mode produces valid output."""
        ema = EMA(characteristic_time=10.0, adjust=True)
        x = torch.randn(4, 50)
        y = ema(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()


class TestLaguerreFeatures:
    """Tests for the LaguerreFeatures class."""

    def test_output_shape_2d(self):
        """Test output shape for 2D input."""
        lf = LaguerreFeatures(embedding_dim=5, characteristic_time=10.0)
        x = torch.randn(8, 100)
        y = lf(x)
        assert y.shape == (8, 100, 5)

    def test_output_shape_3d(self):
        """Test output shape for 3D input."""
        lf = LaguerreFeatures(embedding_dim=4, characteristic_time=10.0)
        x = torch.randn(2, 50, 3)
        y = lf(x)
        assert y.shape == (2, 50, 3, 4)

    def test_invalid_ndim(self):
        """Test that 1D and 4D+ inputs raise errors."""
        lf = LaguerreFeatures(embedding_dim=3, characteristic_time=10.0)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            lf(torch.randn(100))
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            lf(torch.randn(2, 3, 4, 5))

    def test_differentiable(self):
        """Test that gradients flow through Laguerre features."""
        torch.manual_seed(42)
        lf = LaguerreFeatures(embedding_dim=4, characteristic_time=10.0)
        # Use positive data to avoid log(0) issues near the minimum shift
        x = (torch.rand(4, 50) + 1.0).requires_grad_(True)
        y = lf(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_warm_start_vector_list(self):
        """Test initialization with warm_start_vector as list."""
        lf = LaguerreFeatures(
            embedding_dim=3,
            characteristic_time=10.0,
            warm_start_vector=[0.1, 0.2, 0.3],
        )
        x = torch.randn(2, 20)
        y = lf(x)
        assert y.shape == (2, 20, 3)

    def test_warm_start_vector_tensor(self):
        """Test initialization with warm_start_vector as tensor."""
        lf = LaguerreFeatures(
            embedding_dim=3,
            characteristic_time=10.0,
            warm_start_vector=torch.tensor([0.1, 0.2, 0.3]),
        )
        x = torch.randn(2, 20)
        y = lf(x)
        assert y.shape == (2, 20, 3)

    def test_warm_start_vector_wrong_size(self):
        """Test that mismatched warm_start_vector raises error."""
        with pytest.raises(ValueError, match="must match embedding_dim"):
            LaguerreFeatures(
                embedding_dim=3,
                characteristic_time=10.0,
                warm_start_vector=[0.1, 0.2],  # Wrong size
            )

    def test_first_feature_is_ema(self):
        """Test that first Laguerre feature approximates EMA."""
        ct = 10.0
        ema = EMA(characteristic_time=ct)
        lf = LaguerreFeatures(embedding_dim=3, characteristic_time=ct)

        x = torch.randn(4, 100)
        ema_out = ema(x)
        lf_out = lf(x)

        # First Laguerre feature should be close to EMA
        # (not exact due to different warm start handling)
        assert torch.allclose(ema_out[:, 10:], lf_out[:, 10:, 0], atol=1e-5)

    def test_features_are_different(self):
        """Test that different Laguerre dimensions produce different features."""
        lf = LaguerreFeatures(embedding_dim=4, characteristic_time=10.0)
        x = torch.randn(4, 100)
        y = lf(x)

        # Each dimension should be different
        for i in range(3):
            for j in range(i + 1, 4):
                # They shouldn't be identical
                assert not torch.allclose(y[..., i], y[..., j], atol=1e-6)

    def test_gpu_if_available(self):
        """Test that computation works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        lf = LaguerreFeatures(embedding_dim=3, characteristic_time=10.0)
        x = torch.randn(4, 50, device="cuda", requires_grad=True)
        y = lf(x)

        assert y.device.type == "cuda"
        assert y.shape == (4, 50, 3)

        # Test gradients on GPU
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_ema_large_values(self):
        """Test EMA with large input values."""
        ema = EMA(characteristic_time=10.0)
        x = torch.randn(4, 100) * 1e6
        y = ema(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_ema_small_values(self):
        """Test EMA with small input values."""
        ema = EMA(characteristic_time=10.0)
        x = torch.randn(4, 100) * 1e-6
        y = ema(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_laguerre_long_sequence(self):
        """Test Laguerre with long sequences."""
        lf = LaguerreFeatures(embedding_dim=5, characteristic_time=50.0)
        x = torch.randn(2, 10000)
        y = lf(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_laguerre_many_dimensions(self):
        """Test Laguerre with many embedding dimensions."""
        lf = LaguerreFeatures(embedding_dim=20, characteristic_time=10.0)
        x = torch.randn(4, 100)
        y = lf(x)
        assert y.shape == (4, 100, 20)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
