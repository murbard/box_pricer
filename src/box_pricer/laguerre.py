"""
Efficient, differentiable Laguerre-like features for time series.

This module provides fast, GPU-compatible implementations of exponential moving
averages (EMA) and Laguerre polynomial features using log-space cumulative sums
for numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class EMA:
    """
    Efficient, differentiable exponential moving average.

    Uses log-space cumulative sums for numerical stability and full
    differentiability. Supports both recursive and normalized (adjusted) modes.

    Args:
        alpha: Smoothing factor in (0, 1). Mutually exclusive with characteristic_time.
        characteristic_time: Time constant (in samples) for the EMA. The alpha is
            computed as 1 - exp(-1/characteristic_time). Mutually exclusive with alpha.
        adjust: If False (default), uses the recursive approach. If True, uses
            normalized weights that sum to 1.
        warm_start: Initial value(s) for padding when adjust=False. Can be a scalar,
            a 1D tensor of shape (batch,), or None for no padding.
        eps: Small value for numerical stability. Defaults to float64 epsilon.
    """

    alpha: float | None = None
    characteristic_time: float | None = None
    adjust: bool = False
    warm_start: torch.Tensor | float | None = 0.0
    eps: float | None = None

    # Computed fields
    _log_alpha: torch.Tensor = field(init=False, repr=False)
    _eps: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        # Validate and compute alpha
        specified = sum(p is not None for p in (self.alpha, self.characteristic_time))
        if specified != 1:
            raise ValueError("Exactly one of alpha or characteristic_time must be specified")

        if self.alpha is None:
            if not isinstance(self.characteristic_time, (float, int)):
                raise TypeError("characteristic_time must be a float or int")
            if self.characteristic_time <= 0:
                raise ValueError("characteristic_time must be positive")
            self.alpha = 1 - math.exp(-1 / self.characteristic_time)

        if not isinstance(self.alpha, float):
            raise TypeError("alpha must be a float")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in the range (0, 1)")

        # Warn about unexpectedly large alpha (< 1 second characteristic time)
        if self.alpha > 1 - math.exp(-1):
            raise ValueError(
                f"alpha={self.alpha} is unexpectedly large (characteristic time < 1). "
                "Did you mean to use (1 - alpha) instead?"
            )

        self._log_alpha = torch.tensor(self.alpha, dtype=torch.float64).log()

        if self.eps is not None:
            if self.eps <= 0:
                raise ValueError("eps must be positive")
            self._eps = torch.tensor(self.eps, dtype=torch.float64)
        else:
            self._eps = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64)

        # Convert warm_start to tensor if needed
        if isinstance(self.warm_start, list):
            self.warm_start = torch.tensor(self.warm_start, dtype=torch.float64)

    def _get_normalization_vectors(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute log-space normalization vectors for EMA."""
        effective_len = seq_len + 1 if (self.warm_start is not None and not self.adjust) else seq_len

        # log([(1-α)^{T-1}, ..., (1-α)^0])
        log_one_minus_alpha = torch.log(1 - torch.exp(self._log_alpha.to(device)))
        norm_vector = torch.arange(effective_len - 1, -1, -1, device=device) * log_one_minus_alpha
        norm_vector = norm_vector.view(1, -1)

        if self.adjust:
            norm_vector_logcumsumexp = norm_vector.logcumsumexp(dim=-1)
        else:
            norm_vector_logcumsumexp = None

        return norm_vector, norm_vector_logcumsumexp

    def _prepare_warm_start(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend warm start values to the input tensor."""
        if self.warm_start is None:
            return x

        if isinstance(self.warm_start, (int, float)):
            return F.pad(x, (1, 0), value=float(self.warm_start))

        if isinstance(self.warm_start, torch.Tensor):
            n = x.shape[0]
            ws_shape = tuple(self.warm_start.shape)

            if not ws_shape or ws_shape == (1,):
                return F.pad(x, (1, 0), value=self.warm_start.item())
            elif ws_shape in ((n, 1), (n,)):
                return torch.cat([self.warm_start.view(-1, 1).to(x.device, x.dtype), x], dim=-1)
            else:
                raise ValueError(
                    f"warm_start shape {ws_shape} incompatible with input shape {x.shape}. "
                    "Expected scalar, (batch,), or (batch, 1)."
                )
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute EMA of input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, features).

        Returns:
            EMA of input with same shape.
        """
        # Handle 3D input by flattening
        if x.ndim == 3:
            batch, seq_len, features = x.shape
            x_flat = x.permute(0, 2, 1).reshape(batch * features, seq_len)
            result = self._compute_ema(x_flat)
            return result.view(batch, features, seq_len).permute(0, 2, 1)
        elif x.ndim == 2:
            return self._compute_ema(x)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    def _compute_ema(self, x: torch.Tensor) -> torch.Tensor:
        """Core EMA computation on 2D tensor (batch, seq_len)."""
        orig_dtype = x.dtype
        x = x.to(torch.float64)

        norm_vector, norm_vector_logcumsumexp = self._get_normalization_vectors(
            seq_len=x.shape[-1], device=x.device
        )

        if self.warm_start is not None and not self.adjust:
            x = self._prepare_warm_start(x)

        # Shift to positive values for log-space computation
        shift = x.min() - self._eps.to(x.device)
        x = x - shift

        # Log-space weighted computation
        x = x.log() + norm_vector

        if self.adjust:
            x = (x.logcumsumexp(dim=-1) - norm_vector_logcumsumexp).exp()
        else:
            x[..., 1:] += self._log_alpha.to(x.device)
            x = (x.logcumsumexp(dim=-1) - norm_vector).exp()
            if self.warm_start is not None:
                x = x[..., 1:]  # Remove prepended warm start

        x = x + shift
        return x.to(orig_dtype)


@dataclass
class LaguerreFeatures:
    """
    Efficient, differentiable Laguerre-like features for time series.

    Computes a sequence of orthogonal-like features by iteratively applying
    EMA to residuals. The result is similar to Laguerre polynomial basis
    functions with a tunable time scale.

    Args:
        embedding_dim: Number of Laguerre features to compute.
        alpha: Smoothing factor in (0, 1). Mutually exclusive with characteristic_time.
        characteristic_time: Time constant (in samples). Mutually exclusive with alpha.
        adjust: If False (default), uses recursive EMA. If True, uses normalized weights.
        warm_start_vector: Initial values for each Laguerre dimension. Shape should be
            (embedding_dim,) or (embedding_dim, batch). Defaults to zeros.
    """

    embedding_dim: int
    alpha: float | None = None
    characteristic_time: float | None = None
    adjust: bool = False
    warm_start_vector: torch.Tensor | list[float] | None = None

    # Internal EMA instance
    _ema: EMA = field(init=False, repr=False)
    _warm_start_vector: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        # Initialize the base EMA (warm_start will be set per-dimension)
        self._ema = EMA(
            alpha=self.alpha,
            characteristic_time=self.characteristic_time,
            adjust=self.adjust,
            warm_start=0.0,  # Will be overwritten per dimension
        )

        # Process warm start vector
        if self.warm_start_vector is None:
            self._warm_start_vector = torch.zeros(self.embedding_dim, dtype=torch.float64)
        elif isinstance(self.warm_start_vector, list):
            self._warm_start_vector = torch.tensor(self.warm_start_vector, dtype=torch.float64)
        else:
            self._warm_start_vector = self.warm_start_vector.to(torch.float64)

        if len(self._warm_start_vector) != self.embedding_dim:
            raise ValueError(
                f"warm_start_vector length ({len(self._warm_start_vector)}) "
                f"must match embedding_dim ({self.embedding_dim})"
            )

        # Reshape to (embedding_dim, 1) for broadcasting
        if self._warm_start_vector.ndim == 1:
            self._warm_start_vector = self._warm_start_vector.view(-1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laguerre features of input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, features).

        Returns:
            Laguerre features of shape (batch, seq_len, embedding_dim) for 2D input,
            or (batch, seq_len, features, embedding_dim) for 3D input.
        """
        # Handle 3D input
        if x.ndim == 3:
            batch, seq_len, features = x.shape
            x_flat = x.permute(0, 2, 1).reshape(batch * features, seq_len)
            result = self._compute_laguerre(x_flat)
            # result shape: (embedding_dim, batch * features, seq_len)
            result = result.view(self.embedding_dim, batch, features, seq_len)
            return result.permute(1, 3, 2, 0)  # (batch, seq_len, features, embedding_dim)
        elif x.ndim == 2:
            result = self._compute_laguerre(x)
            # result shape: (embedding_dim, batch, seq_len)
            return result.permute(1, 2, 0)  # (batch, seq_len, embedding_dim)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    def _compute_laguerre(self, x: torch.Tensor) -> torch.Tensor:
        """Core Laguerre computation on 2D tensor (batch, seq_len)."""
        orig_dtype = x.dtype
        batch, seq_len = x.shape
        x = x.to(torch.float64)

        # Output buffer with warm start column
        outputs = torch.zeros(
            self.embedding_dim, batch, seq_len + 1,
            device=x.device, dtype=torch.float64
        )
        outputs[..., 0] = self._warm_start_vector.to(x.device)

        # Iteratively compute Laguerre features:
        # y_{t,i} = EMA(x_t - sum_{j<i} y_{t-1,j})
        residual = x
        for d in range(self.embedding_dim):
            self._ema.warm_start = self._warm_start_vector[d]
            outputs[d, :, 1:] = self._ema._compute_ema(residual)
            residual = residual - outputs[d, :, :-1]

        # Remove warm start column
        outputs = outputs[..., 1:]
        return outputs.to(orig_dtype)
