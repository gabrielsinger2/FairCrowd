"""
Here we introduce positive dependence between annotators through an item-level common
random component. For each example i, with probability corr_s depending on S_i, the
annotators use a shared latent difficulty variable U_i. This makes their errors correlated.

When corr_s = 0, the model reduces to the usual conditionally independent annotators.
When corr_s = 1, errors are maximally positively dependent through the shared variable.

Notation
--------
S_i in {0, 1}: sensitive group.
Y_i in {0, 1}: latent true label.
Y_annotators[i, r]: noisy label provided by annotator r on example i.

The returned confusion tensor has shape (R, 2, 2, 2), with convention:
    confusion[r, y_hat, y_true, s] = P(annotator r outputs y_hat | Y=y_true, S=s)

The marginal error rates of each annotator are preserved by the correlated mechanism.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union, Optional
import numpy as np


RhoSpec = Union[float, Tuple[float, float], np.ndarray]


def _as_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a NumPy random generator."""
    return np.random.default_rng(seed)


def _sample_rho_values(rng: np.random.Generator, rho: RhoSpec, R: int, name: str) -> np.ndarray:
    """
    Sample or validate annotator-specific error rates.

    Parameters
    ----------
    rho:
        - float: each annotator rate is sampled uniformly in [0, rho].
        - pair (low, high): each annotator rate is sampled uniformly in [low, high].
        - array of length R: fixed annotator-specific rates.
    R:
        Number of annotators.
    name:
        Name used in error messages.

    Returns
    -------
    rates:
        Array of shape (R,).
    """
    if np.isscalar(rho):
        high = float(rho)
        if not (0.0 <= high <= 1.0):
            raise ValueError(f"{name} as a scalar must be in [0, 1]. Got {rho}.")
        return rng.uniform(0.0, high, size=R)

    rho_array = np.asarray(rho, dtype=float)

    if rho_array.shape == (2,):
        low, high = float(rho_array[0]), float(rho_array[1])
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(
                f"{name} as an interval must satisfy 0 <= low <= high <= 1. "
                f"Got {rho_array}."
            )
        return rng.uniform(low, high, size=R)

    if rho_array.shape == (R,):
        if np.any((rho_array < 0.0) | (rho_array > 1.0)):
            raise ValueError(f"{name} as an array must contain values in [0, 1].")
        return rho_array.copy()

    raise ValueError(
        f"{name} must be either a scalar, an interval of length 2, "
        f"or an array of length R={R}. Got shape {rho_array.shape}."
    )


def _build_confusion_tensor(rho_by_group: np.ndarray) -> np.ndarray:
    """
    Build the confusion tensor from annotator-specific and group-specific error rates.

    Parameters
    ----------
    rho_by_group:
        Array of shape (R, 2), where rho_by_group[r, s] is the error probability
        of annotator r on group s.

    Returns
    -------
    confusion:
        Tensor of shape (R, 2, 2, 2), with
        confusion[r, y_hat, y_true, s] = P(Y_hat_r=y_hat | Y=y_true, S=s).
    """
    R = rho_by_group.shape[0]
    confusion = np.zeros((R, 2, 2, 2), dtype=float)

    for r in range(R):
        for s in range(2):
            rho = rho_by_group[r, s]

            # If true label is 0
            confusion[r, 0, 0, s] = 1.0 - rho
            confusion[r, 1, 0, s] = rho

            # If true label is 1
            confusion[r, 1, 1, s] = 1.0 - rho
            confusion[r, 0, 1, s] = rho

    return confusion


def generate_synthetic_binary_correlated(
    N: int = 20_000,
    R: int = 5,
    bias_toward_1: float = 0.2,
    prop_s_1: float = 0.5,
    rho_0: RhoSpec = 0.2,
    rho_1: RhoSpec = 0.1,
    corr_0: float = 0.3,
    corr_1: float = 0.3,
    seed: Optional[int] = 0,
    return_details: bool = False,
):
    """
    Generate binary crowdsourcing data with non-independent annotators.

    Parameters
    ----------
    N:
        Number of examples.
    R:
        Number of annotators.
    bias_toward_1:
        Controls the difference between P(Y=1 | S=0) and P(Y=1 | S=1).

        The code sets:
            P(Y=1 | S=0) = 1/2 + bias_toward_1 / 2
            P(Y=1 | S=1) = 1/2 - bias_toward_1 / 2

    prop_s_1:
        Probability of belonging to group S=1.

    rho_0:
        Error rates for group S=0.
        Can be:
            - scalar: rates sampled uniformly in [0, rho_0],
            - interval (low, high): rates sampled uniformly in [low, high],
            - array of length R: fixed error rates.

    rho_1:
        Same as rho_0, but for group S=1.

    corr_0:
        Dependence strength for group S=0.
        corr_0=0 means independent annotator errors.
        corr_0=1 means fully shared item-level latent error component.

    corr_1:
        Dependence strength for group S=1.

    seed:
        Random seed.

    return_details:
        If False, returns exactly:
            Y_annotators, S, Y, confusion, prop

        If True, returns:
            Y_annotators, S, Y, confusion, prop, details

        where details contains rho_by_group, shared_mode, and error_matrix.

    Returns
    -------
    Y_annotators:
        Array of shape (N, R), with entries in {0, 1}.

    S:
        Array of shape (N,), sensitive group in {0, 1}.

    Y:
        Array of shape (N,), latent true label in {0, 1}.

    confusion:
        Tensor of shape (R, 2, 2, 2), with
        confusion[r, y_hat, y_true, s].

    prop:
        Array of shape (2,), where prop[s] = P(Y=1 | S=s).

    details:
        Returned only if return_details=True.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if R <= 0:
        raise ValueError("R must be positive.")
    if not (-1.0 <= bias_toward_1 <= 1.0):
        raise ValueError("bias_toward_1 must be in [-1, 1].")
    if not (0.0 <= prop_s_1 <= 1.0):
        raise ValueError("prop_s_1 must be in [0, 1].")
    if not (0.0 <= corr_0 <= 1.0):
        raise ValueError("corr_0 must be in [0, 1].")
    if not (0.0 <= corr_1 <= 1.0):
        raise ValueError("corr_1 must be in [0, 1].")

    rng = _as_rng(seed)

    # Sensitive group
    S = rng.binomial(1, prop_s_1, size=N).astype(int)

    # True label distribution
    prop = np.array(
        [
            0.5 + bias_toward_1 / 2.0,
            0.5 - bias_toward_1 / 2.0,
        ],
        dtype=float,
    )

    if np.any((prop < 0.0) | (prop > 1.0)):
        raise ValueError(
            "The generated class probabilities are outside [0, 1]. "
            "Check bias_toward_1."
        )

    Y = rng.binomial(1, prop[S], size=N).astype(int)

    # Annotator-specific and group-specific error rates.
    rho_by_group = np.zeros((R, 2), dtype=float)
    rho_by_group[:, 0] = _sample_rho_values(rng, rho_0, R, "rho_0")
    rho_by_group[:, 1] = _sample_rho_values(rng, rho_1, R, "rho_1")

    confusion = _build_confusion_tensor(rho_by_group)

    # For each item i and annotator r, get rho_{r, S_i}.
    # Shape: (N, R)
    rho_items = rho_by_group[:, S].T

    # Independent component:
    # E_ind[i, r] ~ Bernoulli(rho_{r, S_i})
    independent_errors = rng.random((N, R)) < rho_items

    # Shared component:
    # One U_i per item. Given U_i, annotator r fails if U_i < rho_{r, S_i}.
    # This preserves marginal P(E_ir=1 | S_i=s) = rho_{r,s},
    # but creates positive dependence between annotators.
    shared_u = rng.random(N)
    shared_errors = shared_u[:, None] < rho_items

    # Item-level switch between independent and shared mechanism.
    corr_by_group = np.array([corr_0, corr_1], dtype=float)
    shared_mode = rng.random(N) < corr_by_group[S]

    error_matrix = np.where(shared_mode[:, None], shared_errors, independent_errors)

    # If error is True, annotator flips the true label.
    Y_annotators = np.where(error_matrix, 1 - Y[:, None], Y[:, None]).astype(float)

    if not return_details:
        return Y_annotators, S, Y, confusion, prop

    details: Dict[str, np.ndarray] = {
        "rho_by_group": rho_by_group,
        "shared_mode": shared_mode,
        "error_matrix": error_matrix,
    }

    return Y_annotators, S, Y, confusion, prop, details


def generate_synthetic_binary_correlated_restricted(
    N: int = 20_000,
    R: int = 8,
    R_anot_max: int = 5,
    bias_toward_1: float = 0.2,
    prop_s_1: float = 0.5,
    rho_0: RhoSpec = (0.0, 0.2),
    rho_1: RhoSpec = (0.0, 0.1),
    corr_0: float = 0.3,
    corr_1: float = 0.3,
    seed: Optional[int] = 0,
    return_details: bool = False,
):
    """
    Same as generate_synthetic_binary_correlated, but each item is observed by at most
    R_anot_max annotators. Missing annotations are encoded as np.nan.

    This is useful to simulate incomplete crowdsourcing matrices.
    """
    if not (1 <= R_anot_max <= R):
        raise ValueError("R_anot_max must satisfy 1 <= R_anot_max <= R.")

    output = generate_synthetic_binary_correlated(
        N=N,
        R=R,
        bias_toward_1=bias_toward_1,
        prop_s_1=prop_s_1,
        rho_0=rho_0,
        rho_1=rho_1,
        corr_0=corr_0,
        corr_1=corr_1,
        seed=seed,
        return_details=True,
    )

    Y_annotators, S, Y, confusion, prop, details = output

    rng = _as_rng(None if seed is None else seed + 10_000)

    n_missing_per_row = R - R_anot_max
    if n_missing_per_row > 0:
        for i in range(N):
            missing_idx = rng.choice(R, size=n_missing_per_row, replace=False)
            Y_annotators[i, missing_idx] = np.nan

    if not return_details:
        return Y_annotators, S, Y, confusion, prop

    observed_mask = ~np.isnan(Y_annotators)
    details["observed_mask"] = observed_mask

    return Y_annotators, S, Y, confusion, prop, details


def empirical_error_correlation(Y_annotators: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Estimate the empirical pairwise correlation matrix of annotator errors.

    Missing values np.nan are handled pairwise.

    Parameters
    ----------
    Y_annotators:
        Array of shape (N, R), possibly with np.nan values.

    Y:
        Array of shape (N,), true labels.

    Returns
    -------
    corr:
        Array of shape (R, R). Entry corr[r, q] is the empirical correlation
        between the error indicators of annotators r and q.
    """
    Y_annotators = np.asarray(Y_annotators, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if Y_annotators.ndim != 2:
        raise ValueError("Y_annotators must be a 2D array.")
    if Y.ndim != 1:
        raise ValueError("Y must be a 1D array.")
    if Y_annotators.shape[0] != Y.shape[0]:
        raise ValueError("Y_annotators and Y must have the same number of rows.")

    errors = (Y_annotators != Y[:, None]).astype(float)
    errors[np.isnan(Y_annotators)] = np.nan

    R = Y_annotators.shape[1]
    corr = np.eye(R, dtype=float)

    for r in range(R):
        for q in range(r + 1, R):
            mask = ~np.isnan(errors[:, r]) & ~np.isnan(errors[:, q])
            if mask.sum() < 2:
                value = np.nan
            else:
                er = errors[mask, r]
                eq = errors[mask, q]
                if np.std(er) == 0.0 or np.std(eq) == 0.0:
                    value = np.nan
                else:
                    value = float(np.corrcoef(er, eq)[0, 1])
            corr[r, q] = value
            corr[q, r] = value

    return corr


# Backward-compatible aliases using the typo already present in the original repo.
generate_synhtetic_binary_correlated = generate_synthetic_binary_correlated
generate_synhtetic_binary_correlated_restricted = generate_synthetic_binary_correlated_restricted


if __name__ == "__main__":
    Y_annotators, S, Y, confusion, prop, details = generate_synthetic_binary_correlated(
        N=20_000,
        R=5,
        bias_toward_1=0.2,
        prop_s_1=0.5,
        rho_0=0.2,
        rho_1=0.1,
        corr_0=0.5,
        corr_1=0.5,
        seed=0,
        return_details=True,
    )

    print("Y_annotators shape:", Y_annotators.shape)
    print("S shape:", S.shape)
    print("Y shape:", Y.shape)
    print("confusion shape:", confusion.shape)
    print("prop:", prop)
    print("rho_by_group:")
    print(details["rho_by_group"])

    corr = empirical_error_correlation(Y_annotators, Y)
    print("Empirical error correlation:")
    print(np.round(corr, 3))
