"""Module to iteratively compute derivatives of conformal blocks."""

import ctypes
from dataclasses import dataclass
from enum import Enum
from functools import cache
import json
from pathlib import Path
from typing import Literal
import warnings

import mpmath as mp
import numpy as np
from scipy.special import comb
from scipy.special import eval_gegenbauer
from scipy.special import factorial
from tqdm import tqdm

from multicorrelator.blocks.base import BlockType
from multicorrelator.blocks.derivatives.partitions import generate_partitions
from multicorrelator.blocks.derivatives.phi1 import Phi1Numeric
from multicorrelator.blocks.derivatives.r_eta_derivatives import rEtaDerivativesSymbolic
from multicorrelator.blocks.recursion import PolesData
from multicorrelator.blocks.recursion import RecursiveG


parent_dir = Path(__file__).resolve().parents[3]
lib_path = parent_dir / "goblocks" / "server" / "lib" / "librecursive.so"
lib = ctypes.cdll.LoadLibrary(lib_path)

# Function signatures
CreateRD = lib.CreateRD
CreateRD.argtypes = [ctypes.c_char_p]
CreateRD.restype = ctypes.c_longlong

FreeRD = lib.FreeRD
FreeRD.argtypes = [ctypes.c_longlong]
FreeRD.restype = None

RunRequest = lib.RunRequest
RunRequest.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong)]
RunRequest.restype = ctypes.POINTER(ctypes.c_double)

FreeResult = lib.FreeResult
FreeResult.argtypes = [ctypes.POINTER(ctypes.c_double)]
FreeResult.restype = None


class FType(str, Enum):

    F_PLUS = "f_plus"
    F_MINUS = "f_minus"


@dataclass
class ConvergedBlockDerivativeData:

    delta_12: float
    delta_34: float

    unique_poles_map: dict[tuple[float, float], int]
    poles_data: dict[float, list[PolesData]]

    dh_tilde: np.ndarray
    dg_final: np.ndarray


class RecursiveDerivatives:

    def __init__(
        self,
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        n_max: int,
        d: int,
        prepopulate_cache: bool = True
    ):
        """Initialize the RecursiveDerivatives class.

        Args:
            k_1_max: Maximum order of Type I poles to retain
            k_2_max: Maximum order of Type II poles to retain
            ell_min: Minimum spin
            ell_max: Maximum spin
            n_max: Maximum derivative order
            d: Spacetime dimension
            prepopulate_cache: Whether to prepopulate caches for efficiency

        """
        # This object is used to get the poles
        self.recursive_g = RecursiveG(k_1_max, k_2_max, ell_min, ell_max, d)

        self.phi1_obj = Phi1Numeric()
        self.r_eta_derivs = rEtaDerivativesSymbolic()

        # Crossing-symmetric point
        self.r_star = 3 - 2 * np.sqrt(2)
        self.eta_star = 1.0
        self.z_star = 0.5
        self.zbar_star = 0.5

        # Get list of (m, n) derivatives to compute in both (z, zbar) and (r, eta) space
        self.derivative_orders_z_zbar = self._compute_derivative_orders_z_zbar(n_max)
        self.derivative_orders_r_eta = self._compute_derivative_orders_r_eta(self.derivative_orders_z_zbar)

        for (m, n) in self.derivative_orders_r_eta:
            for s in range(m):
                if (s, n) not in self.derivative_orders_r_eta:
                    raise Exception(f"Missing derivative order (s={s}, n={n}) in r, eta space")

        # Break up derivative orders into those for F_+ and F_-
        self.derivative_orders_z_zbar_F_plus = [(m, n) for (m, n) in self.derivative_orders_z_zbar if (m + n) % 2 == 0]
        self.derivative_orders_z_zbar_F_minus = [(m, n) for (m, n) in self.derivative_orders_z_zbar if (m + n) % 2 == 1]

        if prepopulate_cache:
            self.prepopulate_cache()

        if len(self.derivative_orders_z_zbar_F_plus) != len(self.derivative_orders_z_zbar_F_minus):
            raise ValueError("Number of F_plus and F_minus derivatives must be equal.")

        self.num_derivs = len(self.derivative_orders_r_eta)

    def recurse(
        self,
        delta_12: float,
        delta_34: float,
        max_iterations: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ):
        """Run the recursion to compute d_r^m d_n^n h_{Delta, l} at the crossing-symmetric point.

        The result is returned as a matrix where rows are (Delta, l) and columns are (m, n).

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence
            verbose: Whether to print debug information

        Returns:
            The computed derivatives matrix, along with the poles data

        """
        alpha = (1 + delta_12 - delta_34) / 2
        beta = (1 - delta_12 + delta_34) / 2

        # Compute the poles
        self.recursive_g.unique_poles_map = {}
        poles_data = self.recursive_g.get_all_poles_data(delta_12, delta_34)
        unique_poles_items = list(self.recursive_g.unique_poles_map.items())
        num_poles = len(unique_poles_items)

        # Compute derivative of h tilde
        dh_tilde = np.zeros((self.num_derivs, num_poles))

        m_n_ell_cache = {}

        iterator = range(self.num_derivs * num_poles)
        if verbose:
            iterator = tqdm(iterator, desc="Computing dh_tilde")

        for flat_idx in iterator:
            idx_i = flat_idx // num_poles
            idx_j = flat_idx % num_poles

            m, n = self.derivative_orders_r_eta[idx_i]
            (_, ell), _ = unique_poles_items[idx_j]

            if (m, n, ell) in m_n_ell_cache:
                dh_tilde[idx_i, idx_j] = m_n_ell_cache[(m, n, ell)]
                continue

            # Protect evaluation behind a cache, because h tilde is independent of delta
            dh_tilde[idx_i, idx_j] = self.derivative_h_tilde(
                m,
                n,
                ell,
                self.r_star,
                self.eta_star,
                self.recursive_g.nu,
                alpha,
                beta
            )

            m_n_ell_cache[(m, n, ell)] = dh_tilde[idx_i, idx_j]

        # Compute dg tilde by multiplying each column by its corresponding r^Delta
        # but make sure to not modify dh_tilde because it needs to be used in the `evaluate` step
        dg_tilde = dh_tilde.copy()

        for (delta, _), idx in self.recursive_g.unique_poles_map.items():
            dg_tilde[:, idx] = self.r_delta_mat(delta) @ dg_tilde[:, idx]

        # Initialize dg to dg tilde
        dg = dg_tilde.copy()

        for iter in range(max_iterations):
            dg_new = dg_tilde.copy()

            # Loop over scaling dimensions and spins, updating h
            for (delta, ell), idx in self.recursive_g.unique_poles_map.items():
                for pole in poles_data[ell]:
                    delta_diff = delta - pole.delta
                    term = self.r_delta_mat(delta_diff) @ dg[:, pole.idx]
                    term *= float(pole.c / delta_diff)
                    dg_new[:, idx] += term

            # Print difference between old and new h
            diff = np.max(np.abs(dg_new - dg))
            if verbose:
                print(f"Iteration {iter}: max difference = {diff}")

            # Check convergence
            if diff < tol:
                break

            dg = dg_new

        # Save last iteration
        self.converged_data = ConvergedBlockDerivativeData(
            delta_12=delta_12,
            delta_34=delta_34,
            unique_poles_map=self.recursive_g.unique_poles_map,
            poles_data=poles_data,
            dh_tilde=dh_tilde,
            dg_final=dg
        )

    def evaluate(self, delta: float, ell: float) -> dict[tuple, float]:
        """Evaluate the recursion result for a specific spin and scaling dimension.

        Args:
            delta: The scaling dimension to evaluate
            ell: The spin to evaluate

        Returns:
            The evaluated conformal block derivatives

        """
        if self.converged_data is None:
            raise ValueError("Recursion has not been run yet. Call `recurse` first.")

        if ell not in self.converged_data.poles_data:
            raise ValueError(f"No poles data available for spin {ell}. Run `recurse` first.")

        # Find index with given ell
        dh_tilde_index = None
        for (_, ell_val), idx in self.converged_data.unique_poles_map.items():
            if ell_val == ell:
                dh_tilde_index = idx
                break

        if dh_tilde_index is None:
            raise ValueError(f"No poles data available for spin {ell}. Run `recurse` first.")

        # Multiply by r^Delta
        dg = self.r_delta_mat(delta) @ self.converged_data.dh_tilde[:, dh_tilde_index]

        for pole in self.converged_data.poles_data[ell]:
            r_mat = self.r_delta_mat(delta - pole.delta)
            term = r_mat @ self.converged_data.dg_final[:, pole.idx]
            term *= float(pole.c / (delta - pole.delta))
            dg += term

        # Convert to dictionary where keys are (m, n) tuples and values are conformal block derivatives
        dg = {(m, n): dg[idx] for idx, (m, n) in enumerate(self.derivative_orders_r_eta)}

        return dg

    def recurse_and_evaluate_dg(
        self,
        delta_12: float,
        delta_34: float,
        deltas: np.ndarray,
        spins: np.ndarray,
        max_iterations: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> list[dict[tuple, float]]:
        """Run the recursion and evaluate dg in one step.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            deltas: The scaling dimensions to evaluate
            spins: The spins to evaluate (must be the same length as deltas)
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence
            verbose: Whether to print debug information

        Returns:
            The evaluated dg values for the given parameters

        """
        self.recurse(delta_12, delta_34, max_iterations, tol, verbose)
        dg = [self.evaluate(delta, spin) for delta, spin in zip(deltas, spins)]
        return dg

    def recurse_and_evaluate_dF(
        self,
        block_types: list[BlockType],
        delta_12: float,
        delta_34: float,
        delta_ave_23: float,
        deltas: np.ndarray,
        spins: np.ndarray,
        max_iterations: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        normalize: bool = False
    ) -> np.ndarray:
        """Run the recursion and evaluate dF in one step.

        Args:
            block_types: List of block types (e.g., '+' or '-')
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            delta_ave_23: Average scaling dimension between the second and third operators
            deltas: The scaling dimensions to evaluate
            spins: The spins to evaluate (must be the same length as deltas)
            n_max: Maximum derivative order to compute
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence
            verbose: Whether to print debug information
            normalize: Whether to normalize the final result by 1 / (2^(m+n) m! n!)

        Returns:
            The evaluated dF values for the given parameters

        """
        dg = self.recurse_and_evaluate_dg(delta_12, delta_34, deltas, spins, max_iterations, tol, verbose)

        F_derivs = np.zeros((len(block_types), len(spins), len(self.derivative_orders_z_zbar_F_plus)))

        for i, block_type in enumerate(block_types):
            for j in range(len(spins)):
                F_derivs[i, j] = self._compute_F_derivatives_wrt_z_zbar(block_type, dg[j], delta_ave_23, normalize)

        return F_derivs

    def gen_block(
        self,
        block_type: BlockType,
        spins: np.ndarray,
        deltas: np.ndarray,
        delta_ij: float,
        delta_kl: float,
        delta_ave_kj: float,
        max_iterations: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        use_alternative_normalization: bool = False
    ) -> np.ndarray:
        """Wrapper around recurse_and_evaludate_dF that mimics the interface of FBlocks."""
        return self.recurse_and_evaluate_dF(
            block_types=[block_type],
            delta_12=delta_ij,
            delta_34=delta_kl,
            delta_ave_23=delta_ave_kj,
            deltas=deltas,
            spins=spins,
            max_iterations=max_iterations,
            tol=tol,
            verbose=verbose,
            normalize=use_alternative_normalization
        )[0]

    def derivative_h_tilde(
        self,
        m: int,
        n: int,
        ell: int,
        r: float,
        eta: float,
        nu: float,
        alpha: float,
        beta: float,
    ) -> float:
        """Compute d_r^m d_eta^n (h_tilde(ell, r, eta)).

        Args:
            m: The order of the derivative with respect to r (must be a non-negative integer)
            n: The order of the derivative with respect to eta (must be a non-negative integer)
            ell: The spin (must be a non-negative integer)
            r: The variable r
            eta: The variable eta
            nu: The exponent of the Gegenbauer polynomial (can be float)
            alpha: The exponent of the second function (can be float)
            beta: The exponent of the third function (can be float)

        Returns:
            The computed derivative value

        """
        total = 0

        for i in range(n + 1):
            term = self.binomial(n, i) * 2 ** (n - i) * self.rf(nu, n - i)
            term *= self.derivative_phi_1_2_3(m, i, r, eta, nu, alpha, beta)
            term *= self.gegenbauer(ell - n + i, nu + n - i, eta)
            total += term

        total *= (-1) ** ell * self.factorial(ell) / self.rf(2 * nu, ell)

        return total

    @cache
    def derivative_phi_1_2_3(
        self,
        i: int,
        j: int,
        r: float,
        eta: float,
        nu: float,
        alpha: float,
        beta: float
    ) -> float:
        """Compute d_r^i d_eta^j ((1 - r^2)^(-nu) * (1 + r^2 + 2 r eta)^(-alpha) * (1 + r^2 - 2 r eta)^(-beta)).

        Args:
            i: The order of the derivative with respect to r (must be a non-negative integer)
            j: The order of the derivative with respect to eta (must be a non-negative integer)
            r: The variable r
            eta: The variable eta
            nu: The exponent of the first function (can be float)
            alpha: The exponent of the second function (can be float)
            beta: The exponent of the third function (can be float)

        Returns:
            The computed derivative value

        """
        result = 0

        # Loop over partitions of i into i1 + i2 + i3 = i
        for i1 in range(i + 1):
            for i2 in range(i - i1 + 1):
                i3 = i - i1 - i2

                # Loop over partitions of j into j2 + j3 = j (j1 must be zero)
                for j2 in range(j + 1):
                    j3 = j - j2

                    coef_i = self.factorial(i) / (self.factorial(i1) * self.factorial(i2) * self.factorial(i3))
                    coef_j = self.factorial(j) / (self.factorial(j2) * self.factorial(j3))

                    term = coef_i * coef_j
                    term *= self.phi1_obj.eval(i1)
                    term *= self.derivative_f_wrt_r_eta(i2, j2, r, eta, alpha, FType.F_PLUS)
                    term *= self.derivative_f_wrt_r_eta(i3, j3, r, eta, beta, FType.F_MINUS)

                    result += term

        return result

    @cache
    def derivative_f_wrt_r_eta(self, i: int, j: int, r: float, eta: float, exponent: float, f_type: FType) -> float:
        """Compute d_r^i d_eta^j ((1 + r^2 +/- 2 r eta)^(-exponent)).

        Args:
            i: The order of the derivative with respect to r (must be a non-negative integer)
            j: The order of the derivative with respect to eta (must be a non-negative integer)
            r: The variable r
            eta: The variable eta
            exponent: The exponent of the function f(r, eta) (can be float)
            f_type: Type of function, either F_PLUS or F_MINUS

        Returns:
            The computed derivative value

        """
        result = 0

        for k in range(i + 1):
            term = self.binomial(i, k) * self.ff(j, k) * r ** (j - k)
            term *= self.derivative_f_wrt_r(r, eta, i - k, -exponent - j, f_type)
            result += term

        result *= 2 ** j * self.rf(exponent, j)

        if f_type == FType.F_PLUS:
            result *= (-1) ** j

        return result

    @cache
    def derivative_f_wrt_r(self, r: float, eta: float, n_val: float, m_val: float, f_type: FType) -> float:
        """Compute d_r^n (f(r, eta)^m) where f(r, eta) = 1 + r^2 +/- 2 r eta.

        Args:
            r: The variable r
            eta: The variable eta
            n_val: The order of the derivative with respect to r (must be a non-negative integer)
            m_val: The exponent of the function f(r, eta) (can be float)
            f_type: Type of function, either F_PLUS or F_MINUS

        Returns:
            The computed derivative value

        """
        fac = 1 if f_type == FType.F_PLUS else -1

        # Compute function at specified r and eta point
        f0 = 1 + r ** 2 + fac * 2 * r * eta

        # Compute derivative at specified r and eta point
        f1 = 2 * r + fac * 2 * eta

        result = 0.0

        for k in range((n_val + 1) // 2, n_val + 1):
            j1 = 2 * k - n_val
            j2 = n_val - k
            bell = self.factorial(n_val) / (self.factorial(j1) * self.factorial(j2))
            term = self.ff(m_val, k) * f0 ** (m_val - k) * bell * f1 ** j1
            result += term

        return result

    def _compute_derivative_orders_z_zbar(self, n_max: int) -> list[tuple[int, int]]:
        """Compute the list of (m, n) derivative orders in (z, zbar) space up to 2 * n_max - 1.

        Args:
            n_max: Maximum derivative order in (z, zbar) space

        Returns:
            List of (m, n) derivative orders in (z, zbar) space

        """
        n_max_deriv = 2 * n_max - 1

        return [
            (m, n)
            for m in range(n_max_deriv + 1)
            for n in range(min(m, n_max_deriv - m) + 1)
        ]

    def _compute_derivative_orders_r_eta(
        self,
        derivative_orders_z_zbar: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Compute the list of (m, n) derivative orders in (r, eta) space corresponding to the given (z, zbar) orders.

        Args:
            derivative_orders_z_zbar: List of (m, n) derivative orders in (z, zbar) space

        Returns:
            List of (m, n) derivative orders in (r, eta) space

        """
        derivative_orders = []

        for (m, n) in derivative_orders_z_zbar:
            for i in range(m + 1):
                for j in range(n + 1):
                    sum = m + n - i - j
                    derivative_orders.extend([
                        (p, q)
                        for p in range(sum + 1)
                        for q in range(sum + 1)
                        if p + q <= sum
                    ])

        # Get unique derivative orders
        derivative_orders = list(set(derivative_orders))

        # Lexicographic sort
        derivative_orders.sort()

        return derivative_orders

    def _compute_F_derivatives_wrt_z_zbar(
        self,
        block_type: BlockType,
        dg: dict[tuple, float],
        delta_ave_23: float,
        normalize: bool
    ) -> np.ndarray:
        """Compute derivatives of F with respect to z, zbar from derivatives of g with respect to r, eta.

        Args:
            block_type: Type of block, either PLUS or MINUS
            dg: Dictionary of derivatives of g with respect to r and eta. Keys are (m, n) tuples
            delta_ave_23: Average scaling dimension between the second and third operators
            normalize: Whether to normalize the final result by 1 / (2^(m+n) m! n!)

        Returns:
            Array of F derivatives with respect to z, zbar

        """
        F_derivs = []

        if block_type == BlockType.PLUS:
            derivative_orders = self.derivative_orders_z_zbar_F_plus
        else:
            derivative_orders = self.derivative_orders_z_zbar_F_minus

        # Store block derivatives with respect to z, zbar
        dg_z_zbar = {}

        for (m, n) in derivative_orders:
            sum = 0

            for i in range(m + 1):
                for j in range(n + 1):
                    if (m - i, n - j) not in dg_z_zbar:
                        # Convert dg to derivatives with respect to z, zbar
                        dg_z_zbar[(m - i, n - j)] = self.partial_z_derivative(dg, m - i, n - j)

                    term = 2 * (-1) ** (i + j) * 2 ** (i + j - 2 * delta_ave_23)
                    term *= self.rf(1 + delta_ave_23 - i, i) * self.rf(1 + delta_ave_23 - j, j)
                    term *= self.binomial(m, i) * self.binomial(n, j)
                    term *= dg_z_zbar[(m - i, n - j)]

                    sum += term

            if normalize:
                sum /= 2 ** (m + n) * self.factorial(m) * self.factorial(n)

            F_derivs.append(sum)

        return np.array(F_derivs)

    def partial_z_derivative(self, dg_dr_eta: dict[tuple, float], m: int, n: int) -> float:
        """Compute d_z^m d_zbar^n g(r(z, zbar), eta(z, zbar)) using the multivariate Faà di Bruno formula.

        Args:
            dg_dr_eta: Dictionary of derivatives of h with respect to r and eta, keys are (p, q) tuples
            m: The order of the derivative with respect to z (must be a non-negative integer)
            n: The order of the derivative with respect to zbar (must be a non-negative integer)

        Returns:
            Value of partial_z^m partial_zbar^n g

        """
        pq_pairs = [(p, q) for p in range(m + n + 1) for q in range(m + n + 1) if p + q <= m + n]
        result = sum([self.partition_factor(p, q, m, n) * dg_dr_eta[(p, q)] for (p, q) in pq_pairs])
        result *= self.factorial(m) * self.factorial(n)
        return result

    def prepopulate_cache(self):
        """Prepopulate caches with partitions and r, eta derivatives."""
        unique_args = set()

        for (m, n) in self.derivative_orders_z_zbar:
            for i in range(m + 1):
                for j in range(n + 1):
                    sum_ = m + n - i - j
                    for p in range(sum_ + 1):
                        for q in range(sum_ + 1 - p):  # ensures p + q <= sum_
                            unique_args.add((p, q, m - i, n - j))

        for p, q, dm, dn in tqdm(sorted(unique_args), desc="Prepopulating cache"):
            self.partition_factor(p, q, dm, dn)

    @cache
    def binomial(self, m, s) -> int:
        """Cached binomial coefficient."""
        return comb(m, s, exact=True)

    @cache
    def factorial(self, n) -> int:
        """Cached factorial."""
        return factorial(n)

    @cache
    def rf(self, x, n) -> mp.mpf:
        """Cached rising factorial."""
        return mp.rf(x, n)

    @cache
    def ff(self, x, n) -> mp.mpf:
        """Cached falling factorial."""
        return mp.ff(x, n)

    @cache
    def gegenbauer(self, ell, nu, eta) -> float:
        """Cached Gegenbauer polynomial evaluation."""
        return eval_gegenbauer(ell, nu, eta)

    @cache
    def r_delta_mat(self, delta: float) -> np.ndarray:
        """Cached R matrix for a given r_power."""
        mat = np.zeros((self.num_derivs, self.num_derivs))

        for i, (m, n_out) in enumerate(self.derivative_orders_r_eta):
            for j, (s, n_in) in enumerate(self.derivative_orders_r_eta):
                if n_out == n_in and m >= s:
                    mat[i, j] = self.binomial(m, s) * self.ff(delta, m - s) * self.r_star ** (delta - m + s)

        return mat

    @cache
    def r_deriv_cache(self, a: int, b: int) -> float:
        return self.r_eta_derivs.eval_r(a, b, self.z_star, self.zbar_star)

    @cache
    def eta_deriv_cache(self, a: int, b: int) -> float:
        return self.r_eta_derivs.eval_eta(a, b, self.z_star, self.zbar_star)

    @cache
    def r_deriv_to_pow_cache(self, a: int, b: int, k: int) -> float:
        return self.r_deriv_cache(a, b) ** k

    @cache
    def eta_deriv_to_pow_cache(self, a: int, b: int, k: int) -> float:
        return self.eta_deriv_cache(a, b) ** k

    @cache
    def partition_factor(self, p: int, q: int, m: int, n: int) -> float:
        """Compute factor associated with given partition."""
        ab_pairs = [(a, b) for a in range(m + n + 1) for b in range(m + n + 1) if 1 <= a + b <= m + n]
        partitions = generate_partitions(ab_pairs, p, q, m, n)

        total_pq = 0

        # Sum contributions across partitions
        for k_tuple, l_tuple in partitions:
            if sum(k_tuple) != p or sum(l_tuple) != q:
                raise ValueError(
                    f"Partition does not satisfy p, q constraints: p={p}, q={q}, "
                    f"sum_k={sum(k_tuple)}, sum_l={sum(l_tuple)}"
                )

            sum_a = sum(a * (k + ell) for (a, _), k, ell in zip(ab_pairs, k_tuple, l_tuple))
            sum_b = sum(b * (k + ell) for (_, b), k, ell in zip(ab_pairs, k_tuple, l_tuple))
            if sum_a != m or sum_b != n:
                raise ValueError(
                    f"Partition does not satisfy derivative constraints: m={m}, n={n}, "
                    f"sum_a={sum_a}, sum_b={sum_b}"
                )

            # Compute combinatorial factor
            coeff_comb = 1
            coeff_deriv = 1

            for (a, b), k, ell in zip(ab_pairs, k_tuple, l_tuple):
                coeff_comb /= self.factorial(a) ** (k + ell)
                coeff_comb /= self.factorial(b) ** (k + ell)
                coeff_comb /= self.factorial(k)
                coeff_comb /= self.factorial(ell)
                coeff_deriv *= self.r_deriv_to_pow_cache(a, b, k) * self.eta_deriv_to_pow_cache(a, b, ell)

            total_pq += coeff_comb * coeff_deriv

        return total_pq


class ConvolvedDerivativeBlocksRecursivePython:
    """Derivatives of 3D conformal blocks using the Python recursive implementation."""

    def __init__(
        self,
        spins: list[int],
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        n_max: int,
        d: int = 3,
        prepopulate_cache: bool = True,
        max_iterations: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        normalize: bool = True,
        num_derivs_to_keep: int = None
    ):
        """Initialize the ConvolvedDerivativeBlocksRecursivePython class.

        Args:
            k_1_max: Maximum order of Type I poles to retain
            k_2_max: Maximum order of Type II poles to retain
            ell_min: Minimum spin
            ell_max: Maximum spin
            n_max: Maximum derivative order
            d: Spacetime dimension
            prepopulate_cache: Whether to prepopulate caches for efficiency
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence
            verbose: Whether to print debug information
            normalize: Whether to normalize the final result by 1 / (2^(m+n) m! n!)
            num_derivs_to_keep: Number of derivatives to keep

        """
        self.max_iterations = max_iterations
        self.tol = tol
        self.verbose = verbose
        self.normalize = normalize
        self.num_derivs_to_keep = num_derivs_to_keep

        self.recursion_derivs = RecursiveDerivatives(
            k_1_max,
            k_2_max,
            ell_min,
            ell_max,
            n_max,
            d,
            prepopulate_cache
        )

    def evaluate(
        self,
        block_types: list[BlockType],
        spins: np.ndarray,
        deltas: np.ndarray,
        delta_ij: float,
        delta_kl: float,
        delta_ave_kj: float,
    ) -> np.ndarray:
        """Evaluate using the Python recursive implementation."""
        blocks = self.recursion_derivs.recurse_and_evaluate_dF(
            block_types,
            delta_ij,
            delta_kl,
            delta_ave_kj,
            deltas,
            spins,
            max_iterations=self.max_iterations,
            tol=self.tol,
            verbose=self.verbose,
            normalize=self.normalize
        )

        if self.num_derivs_to_keep is not None:
            blocks = blocks[:, :, :self.num_derivs_to_keep]

        return blocks


class ConvolvedDerivativeBlocksRecursiveGoBlocks:
    """Derivatives of 3D conformal blocks using the GoBlocks recursive implementation."""

    def __init__(
        self,
        spins: list[int],
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        n_max: int,
        d: int = 3,
        max_iterations: int = 100,
        tol: float = 1e-6,
        normalize: Literal["norm_1", "norm_2"] | None = None,
        use_precomputed_phi1: bool = True,
        use_numeric_derivs: bool = False,
        cache_dir: str = "",
        num_derivs_to_keep: int | None = None
    ):
        """Initialize the ConvolvedDerivativeBlocksRecursiveGoBlocks class.

        Args:
            k_1_max: Maximum order of Type I poles to retain
            k_2_max: Maximum order of Type II poles to retain
            ell_min: Minimum spin
            ell_max: Maximum spin
            n_max: Maximum derivative order
            d: Spacetime dimension
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence
            normalize: Normalization type or None for no normalization
            use_precomputed_phi1: Whether to use precomputed phi1 values
            use_numeric_derivs: Whether to use numeric derivatives
            cache_dir: Directory to use for caching (relative to the bootstop root directory)
            num_derivs_to_keep: Number of derivatives to keep

        """
        self.n_max = n_max
        self.max_iterations = max_iterations
        self.tol = tol
        self.use_precomputed_phi1 = use_precomputed_phi1
        self.use_numeric_derivs = use_numeric_derivs
        self.num_derivs_to_keep = num_derivs_to_keep

        self.r_star = 3 - 2 * np.sqrt(2)
        self.eta_star = 1.0

        self.handle_params = {
            "k1_max": k_1_max,
            "k2_max": k_2_max,
            "ell_min": ell_min,
            "ell_max": ell_max,
            "d": d,
            "nmax": n_max,
            "cache_dir": str(parent_dir / cache_dir),
            "use_precomputed_phi1": use_precomputed_phi1
        }
        self.handle = CreateRD(json.dumps(self.handle_params).encode("utf-8"))

        # Set derivative-order normalization factors
        self._set_normalization_factors(normalize, n_max)

    def evaluate(
        self,
        block_types: list[BlockType],
        spins: np.ndarray,
        deltas: np.ndarray,
        delta_ij: float,
        delta_kl: float,
        delta_ave_kj: float,
    ) -> np.ndarray:
        """Evaluate using the GoBlocks recursive implementation."""
        if self.tol < 1e-12:
            warnings.warn("Using a tolerance value less than 1e-12 may result in numerical instability")

        if isinstance(spins, np.ndarray):
            spins = spins.tolist()

        if isinstance(deltas, np.ndarray):
            deltas = deltas.tolist()

        req = {
            "command": "recurse_and_evaluate_df",
            "handle": 0,  # Load the cache from disk each call (this is necessary because of PyGMO constraints)
            "deltas": deltas,
            "ells": spins,
            "block_types": [bt.value for bt in block_types],
            "delta_12": delta_ij,
            "delta_34": delta_kl,
            "delta_ave_23": delta_ave_kj,
            "max_iterations": self.max_iterations,
            "tol": self.tol,
            "r": self.r_star,
            "eta": self.eta_star,
            "nmax": self.n_max,
            "normalise": False,  # Normalization is handled separately
            "use_numeric_derivs": self.use_numeric_derivs
        }

        result = self._run_request(req | self.handle_params)

        if result is not None:
            result = np.array(result)
            num_derivs = len(result) // (len(block_types) * len(deltas))
            result = result.reshape(len(block_types), len(deltas), num_derivs)

            for idx, block_type in enumerate(block_types):
                if block_type == BlockType.PLUS:
                    result[idx] /= self.F_plus_norm
                elif block_type == BlockType.MINUS:
                    result[idx] /= self.F_minus_norm

            if self.num_derivs_to_keep is not None:
                result = result[:, :, :self.num_derivs_to_keep]

        return result

    @staticmethod
    def _run_request(req_obj) -> list[float] | None:
        s = json.dumps(req_obj).encode("utf-8")

        out_len = ctypes.c_longlong()
        ptr = RunRequest(ctypes.c_char_p(s), ctypes.byref(out_len))

        if not ptr:
            print("RunRequest returned NULL or error")
            return None

        length = out_len.value

        # Create Python list from C array
        arr = [ptr[i] for i in range(length)]

        # Free C memory
        FreeResult(ptr)

        return arr

    def _set_normalization_factors(self, normalize: Literal["norm_1", "norm_2"] | None, n_max: int):
        """Set normalization factors for F derivatives.

        Args:
            normalize: Normalization type
            n_max: Maximum derivative order

        """
        if normalize is None:
            self.F_plus_norm = 1
            self.F_minus_norm = 1
            return

        n_max_deriv = 2 * n_max - 1
        derivative_orders_z_zbar = [
            (m, n)
            for m in range(n_max_deriv + 1)
            for n in range(min(m, n_max_deriv - m) + 1)
        ]

        derivative_orders_z_zbar_F_plus = [(m, n) for (m, n) in derivative_orders_z_zbar if (m + n) % 2 == 0]
        derivative_orders_z_zbar_F_minus = [(m, n) for (m, n) in derivative_orders_z_zbar if (m + n) % 2 == 1]

        def norm_1(m: int, n: int) -> float:
            return 2 ** (m + n) * factorial(m) * factorial(n)

        def norm_2(m: int, n: int) -> float:
            return 4 ** (m + n) * factorial(m) * factorial(n)

        if normalize == "norm_1":
            self.F_plus_norm = np.array([norm_1(m, n) for (m, n) in derivative_orders_z_zbar_F_plus])
            self.F_minus_norm = np.array([norm_1(m, n) for (m, n) in derivative_orders_z_zbar_F_minus])
        elif normalize == "norm_2":
            self.F_plus_norm = np.array([norm_2(m, n) for (m, n) in derivative_orders_z_zbar_F_plus])
            self.F_minus_norm = np.array([norm_2(m, n) for (m, n) in derivative_orders_z_zbar_F_minus])
        else:
            raise ValueError(f"Unknown normalization type: {normalize}")


class ConvolvedDerivativeBlocksRecursiveGoBlocksDg:
    """Derivatives of 3D conformal blocks using the GoBlocks recursive implementation for dg."""

    def __init__(
        self,
        spins: list[int],
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        n_max: int,
        d: int = 3,
        max_iterations: int = 100,
        tol: float = 1e-6,
        normalize: Literal["norm_1", "norm_2"] | None = None,
        use_precomputed_phi1: bool = True,
        use_numeric_derivs: bool = False,
        cache_dir: str = "",
        num_derivs_to_keep: int | None = None
    ):
        """Initialize the ConvolvedDerivativeBlocksRecursiveGoBlocksDg class.

        Args:
            k_1_max: Maximum order of Type I poles to retain
            k_2_max: Maximum order of Type II poles to retain
            ell_min: Minimum spin
            ell_max: Maximum spin
            n_max: Maximum derivative order
            d: Spacetime dimension
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence
            normalize: Normalization type or None for no normalization
            use_precomputed_phi1: Whether to use precomputed phi1 values
            use_numeric_derivs: Whether to use numeric derivatives
            cache_dir: Directory to use for caching (relative to the bootstop root directory)
            num_derivs_to_keep: Number of derivatives to keep

        """
        self.n_max = n_max
        self.max_iterations = max_iterations
        self.tol = tol
        self.use_precomputed_phi1 = use_precomputed_phi1
        self.use_numeric_derivs = use_numeric_derivs
        self.num_derivs_to_keep = num_derivs_to_keep

        self.r_star = 3 - 2 * np.sqrt(2)
        self.eta_star = 1.0

        self.handle_params = {
            "k1_max": k_1_max,
            "k2_max": k_2_max,
            "ell_min": ell_min,
            "ell_max": ell_max,
            "d": d,
            "nmax": n_max,
            "cache_dir": str(parent_dir / cache_dir),
            "use_precomputed_phi1": use_precomputed_phi1
        }
        self.handle = CreateRD(json.dumps(self.handle_params).encode("utf-8"))

        # Set derivative-order normalization factors
        self._set_normalization_factors(normalize, n_max)

    def evaluate(
        self,
        spins: np.ndarray,
        deltas: np.ndarray,
        delta_ij: float,
        delta_kl: float,
        delta_ave_kj: float,
    ) -> np.ndarray:
        """Evaluate using the GoBlocks recursive implementation."""
        if self.tol < 1e-12:
            warnings.warn("Using a tolerance value less than 1e-12 may result in numerical instability")

        if isinstance(spins, np.ndarray):
            spins = spins.tolist()

        if isinstance(deltas, np.ndarray):
            deltas = deltas.tolist()

        req = {
            "command": "recurse_and_evaluate_dg",
            "handle": 0,  # Load the cache from disk each call (this is necessary because of PyGMO constraints)
            "deltas": deltas,
            "ells": spins,
            "delta_12": delta_ij,
            "delta_34": delta_kl,
            "delta_ave_23": delta_ave_kj,
            "max_iterations": self.max_iterations,
            "tol": self.tol,
            "r": self.r_star,
            "eta": self.eta_star,
            "nmax": self.n_max,
            "normalise": False,  # Normalization is handled separately
            "use_numeric_derivs": self.use_numeric_derivs
        }

        result = self._run_request(req | self.handle_params)

        if result is not None:
            result = np.array(result)
            num_derivs = len(result) // len(deltas)
            result = result.reshape(len(deltas), num_derivs)

            # Normalize by derivative order if requested
            result /= self.deriv_norm

            if self.num_derivs_to_keep is not None:
                result = result[:, :self.num_derivs_to_keep]

        return result

    @staticmethod
    def _run_request(req_obj) -> list[float] | None:
        s = json.dumps(req_obj).encode("utf-8")

        out_len = ctypes.c_longlong()
        ptr = RunRequest(ctypes.c_char_p(s), ctypes.byref(out_len))

        if not ptr:
            print("RunRequest returned NULL or error")
            return None

        length = out_len.value

        # Create Python list from C array
        arr = [ptr[i] for i in range(length)]

        # Free C memory
        FreeResult(ptr)

        return arr

    def _set_normalization_factors(self, normalize: Literal["norm_1", "norm_2"] | None, n_max: int):
        """Set normalization factors for F derivatives.

        Args:
            normalize: Normalization type
            n_max: Maximum derivative order

        """
        if normalize is None:
            self.deriv_norm = 1
            return

        n_max_deriv = 2 * n_max - 1
        derivative_orders_z_zbar = [
            (m, n)
            for m in range(n_max_deriv + 1)
            for n in range(min(m, n_max_deriv - m) + 1)
        ]

        def norm_1(m: int, n: int) -> float:
            return 2 ** (m + n) * factorial(m) * factorial(n)

        def norm_2(m: int, n: int) -> float:
            return 4 ** (m + n) * factorial(m) * factorial(n)

        if normalize == "norm_1":
            self.deriv_norm = np.array([norm_1(m, n) for (m, n) in derivative_orders_z_zbar])
        elif normalize == "norm_2":
            self.deriv_norm = np.array([norm_2(m, n) for (m, n) in derivative_orders_z_zbar])
        else:
            raise ValueError(f"Unknown normalization type: {normalize}")
