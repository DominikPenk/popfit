"""
Integer parametrization using a straight-through estimator (STE).

This module provides a parametrization that maps continuous latent variables
to discrete integer values in the forward pass while allowing gradients to
flow unchanged during backpropagation.

The implementation is based on the straight-through estimator (STE) commonly
used in quantization-aware training and relaxed discrete optimization. In the
forward pass, values are rounded to the nearest integer. In the backward pass,
the rounding operation is treated as the identity function.

⚠️ Important:
    This parametrization enables gradient-based optimization over integer-valued
    variables, but it does NOT produce true gradients of the discrete objective.
    Instead, it provides a heuristic relaxation that works well for small-scale
    or smooth discrete problems but may fail for highly combinatorial tasks.

Typical use cases include:
- Integer hyperparameters
- Architectural counts (layers, units, steps)
- Mixed discrete/continuous optimization problems
"""

import torch

from .base import Parametrization


class StraightThroughRound(torch.autograd.Function):
    """Straight-through rounding operation.

    This autograd function implements rounding in the forward pass while
    passing gradients through unchanged in the backward pass.

    The backward behavior corresponds to treating the rounding operation
    as the identity function during differentiation, enabling gradient-based
    optimization of discretized variables.

    This technique is known as the straight-through estimator (STE).
    """

    @staticmethod
    def forward(ctx, x):
        """Round input values to the nearest integer.

        Args:
            x: Continuous input tensor.

        Returns:
            Tensor with values rounded to the nearest integer.
        """
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradients through unchanged.

        Args:
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input, identical to
            `grad_output`.
        """
        return grad_output


class Integer(Parametrization):
    """Integer-valued parametrization using straight-through rounding.

    This parametrization maps a continuous latent variable to an integer-valued
    variable by rounding in the forward pass. During backpropagation, gradients
    are passed through unchanged, allowing gradient-based optimizers (e.g. Adam)
    to operate on the latent continuous representation.

    Optimization happens in latent space, while the model observes integer
    values during evaluation.

    Notes:
        - This parametrization provides a heuristic relaxation of discrete
          optimization.
        - Gradients are not true gradients of the discrete objective.
        - Works best for small-scale or approximately smooth discrete problems.
        - Population-based optimizers (e.g. CMA-ES) do not rely on this behavior.

    See Also:
        Bengio et al., "Estimating or Propagating Gradients Through Stochastic
        Neurons", arXiv:1308.3432
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent values to integer values.

        Args:
            z: Latent continuous tensor.

        Returns:
            Tensor with values rounded to the nearest integer.
        """
        return StraightThroughRound.apply(z)  # type: ignore[reportReturnType]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map integer values back to latent space.

        This parametrization uses the identity mapping for the inverse, meaning
        integer values are treated as valid latent values.

        Args:
            x: Integer-valued tensor.

        Returns:
            Tensor interpreted as latent values.
        """
        return x

    def forward_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        """Transform bounds from latent space to integer space.

        Bounds are rounded to the nearest integers to match the forward
        transformation applied to values.

        Args:
            bounds: Tensor of shape (2, ...) containing lower and upper bounds
                in latent space.

        Returns:
            Tensor of rounded bounds in integer space.
        """
        return bounds.round()

    def inverse_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        """Transform bounds from integer space back to latent space.

        This parametrization uses the identity mapping for bounds inversion.

        Args:
            bounds: Tensor of shape (2, ...) containing integer-space bounds.

        Returns:
            Tensor interpreted as latent-space bounds.
        """
        return bounds
