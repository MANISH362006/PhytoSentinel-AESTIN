"""
SENR0 — Spectral Epidemic Network R0

Computes the basic reproduction number R0 for plant disease spread directly
from the DAGCA-constructed adjacency matrix using the Next-Generation Matrix
(NGM) formalism.

Epidemiological foundation:
  In a network SEIR model, the NGM K is defined as:
      K[i,j] = (β/γ) · A[i,j]
  where A is the dispersal adjacency matrix from DAGCA.

  R0 = ρ(K) — the spectral radius (dominant eigenvalue) of K.

  When R0 < 1: disease dies out.
  When R0 > 1: epidemic spreads.

This provides a physically meaningful, differentiable link between the
learned graph structure and epidemic threshold behavior.

Reference: Diekmann et al. (1990). "On the definition and the computation
           of the basic reproduction ratio R0 in models for infectious
           diseases in heterogeneous populations."
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class SENR0(nn.Module):
    """
    Spectral Epidemic Network R0 calculator.

    Given the DAGCA adjacency matrix A (learned dispersal weights),
    computes R0 = ρ(β/γ · A) where ρ is the spectral radius.

    The computation is differentiable w.r.t. A via torch.linalg.eigvals,
    enabling R0 to be included as a regularization or constraint in the loss.
    """

    def __init__(self, gamma: float = 0.1, beta_scale: float = 1.0):
        """
        Args:
            gamma:      recovery rate (1/infectious_period)
            beta_scale: global transmission rate scaling (learned or fixed)
        """
        super().__init__()
        self.gamma      = gamma
        self.log_beta   = nn.Parameter(torch.tensor(beta_scale).log())

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    def forward(self, A: Tensor) -> Tensor:
        """
        Args:
            A: (N, N) weighted adjacency matrix from DAGCA.get_adjacency_matrix()

        Returns:
            R0: scalar — basic reproduction number
        """
        K = (self.beta / self.gamma) * A   # Next-Generation Matrix

        # power iteration: more stable than full eigen for large N,
        # and fully differentiable
        R0 = _power_iteration_spectral_radius(K)
        return R0

    def threshold_loss(self, A: Tensor, target_r0: float = 1.0) -> Tensor:
        """
        Optional regularization: penalize R0 deviating from a target value.
        Useful when you want the model to learn a specific epidemic regime.
        """
        R0 = self(A)
        return (R0 - target_r0).pow(2)


def _power_iteration_spectral_radius(M: Tensor, num_iter: int = 20,
                                      eps: float = 1e-8) -> Tensor:
    """
    Differentiable spectral radius via power iteration.

    Computes ρ(M) = lim_{k→∞} ||M^k v|| / ||M^{k-1} v||.
    Avoids full eigendecomposition; gradient flows through matrix-vector products.
    """
    N = M.size(0)
    v = torch.ones(N, 1, device=M.device, dtype=M.dtype)
    v = v / v.norm()

    for _ in range(num_iter):
        v_new = M @ v
        norm  = v_new.norm().clamp(min=eps)
        v     = v_new / norm

    # Rayleigh quotient: ρ ≈ vᵀ M v
    R0 = (v.T @ M @ v).squeeze() / (v.T @ v).squeeze().clamp(min=eps)
    return R0.abs()


class NGMDerivation:
    """
    Static utility class: provides the formal NGM derivation for documentation
    and verification purposes (not used in training, but critical for the patent
    and conference paper appendix).
    """

    @staticmethod
    def compute_ngm(A: np.ndarray, beta: float, gamma: float) -> np.ndarray:
        """Compute the Next-Generation Matrix K = (β/γ) · A."""
        return (beta / gamma) * A

    @staticmethod
    def compute_r0_numpy(A: np.ndarray, beta: float, gamma: float) -> float:
        """Compute R0 = ρ(K) using numpy (for validation / reporting)."""
        K = NGMDerivation.compute_ngm(A, beta, gamma)
        eigenvalues = np.linalg.eigvals(K)
        return float(np.max(np.abs(eigenvalues)))

    @staticmethod
    def epidemic_threshold_analysis(A: np.ndarray, beta: float,
                                     gamma: float) -> dict:
        """
        Full threshold analysis: returns R0, critical beta for threshold,
        and whether epidemic is expected.
        """
        R0 = NGMDerivation.compute_r0_numpy(A, beta, gamma)
        rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
        beta_critical = gamma / (rho_A + 1e-10)

        return {
            "R0":             R0,
            "rho_A":          rho_A,           # spectral radius of adjacency
            "beta_critical":  beta_critical,   # min beta for epidemic
            "epidemic":       R0 > 1.0,
            "gamma":          gamma,
            "beta":           beta,
        }
