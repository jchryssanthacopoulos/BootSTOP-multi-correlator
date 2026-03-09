"""Module for retrieving precomputed derivatives of r(z, zbar) and eta(z, zbar) at the crossing symmetric point."""

from functools import cache
from typing import Literal

import jax
import jax.numpy as jnp
import mpmath as mp
from scipy.special import comb
from scipy.special import factorial

from multicorrelator.blocks.derivatives.partitions import generate_partitions


class rDerivativesNumeric:
    """This class precomputes derivatives of r(z, zbar) with respect to z and zbar at the crossing symmetric point."""

    def __init__(self):
        self.derivatives = {
            (0, 0): 0.1715728753,
            (0, 1): 0.2426406871,
            (0, 2): 0.1005050634,
            (0, 3): 1.1543289326,
            (0, 4): 1.8608229691,
            (0, 5): 42.8993462598,
            (0, 6): 124.5747209591,
            (0, 7): 4345.2039257042,
            (0, 8): 18327.3952888535,
            (0, 9): 857510.6385313133,
            (1, 1): 0.3431457505,
            (1, 2): 0.1421356237,
            (1, 3): 1.6324676319,
            (1, 4): 2.6316010801,
            (1, 5): 60.6688372976,
            (1, 6): 176.1752599092,
            (1, 7): 6145.0463230076,
            (1, 8): 25918.8509804694,
            (1, 9): 1212703.1748901960,
            (2, 2): 0.0588745030,
            (2, 3): 0.6761902332,
            (2, 4): 1.0900448581,
            (2, 5): 25.1298552221,
            (2, 6): 72.9741820090,
            (2, 7): 2545.3615284007,
            (2, 8): 10735.9395972373,
            (2, 9): 502318.1021724199,
            (3, 3): 7.7662350914,
            (3, 4): 12.5194719061,
            (3, 5): 288.6234581193,
            (3, 6): 838.1290134284,
            (3, 7): 29234.1933528438,
            (3, 8): 123305.2870911192,
            (3, 9): 5769264.74282372,
            (4, 4): 20.1818738376,
            (4, 5): 465.2721985427,
            (4, 6): 1351.0964468420,
            (4, 7): 47126.6524991831,
            (4, 8): 198772.9008270586,
            (4, 9): 9300278.3223660290,
            (5, 5): 10726.3686453867,
            (5, 6): 31148.1292234618,
            (5, 7): 1086456.1633229163,
            (5, 8): 4582503.3553225324,
            (5, 9): 214408284.2309231758,
            (6, 6): 90450.5509926548,
            (6, 7): 3154942.5615885705,
            (6, 8): 13307057.7180386782,
            (6, 9): 622616764.7831015587,
            (7, 7): 110045350.2815852761,
            (7, 8): 464154195.9917950630,
            (7, 9): 21717060969.4115142822,
            (8, 8): 1957730309.6224517822,
            (8, 9): 91599190193.3742675781,
            (9, 9): 4285785232396.5854492188
        }

    def eval(self, m: int, n: int) -> float:
        """Evaluate d_z^m d_zbar^n r(z, zbar) at the crossing symmetric point.

        Args:
            m: The order of the derivative with respect to z
            n: The order of the derivative with respect to zbar

        Returns:
            The value of the (m, n)-th derivative of r(z, zbar) at the crossing symmetric point

        """
        if (m, n) not in self.derivatives and (n, m) not in self.derivatives:
            raise ValueError(f"Derivative of order {m}, {n} not found.")

        if (m, n) in self.derivatives:
            return self.derivatives[(m, n)]

        # Derivatives are symmetric in m and n
        return self.derivatives[(n, m)]


class etaDerivativesNumeric:
    """This class precomputes derivatives of eta(z, zbar) with respect to z and zbar at the crossing symmetric point."""

    def __init__(self):
        self.derivatives = {
            (0, 0): 1.0,
            (0, 1): 0.0,
            (0, 2): 2.0,
            (0, 3): -6.0,
            (0, 4): 66.0,
            (0, 5): -450.0,
            (0, 6): 6390.0,
            (0, 7): -68670.0,
            (0, 8): 1231650.0,
            (0, 9): -18115650.0,
            (1, 1): -2.0,
            (1, 2): 2.0,
            (1, 3): -18.0,
            (1, 4): 78.0,
            (1, 5): -990.0,
            (1, 6): 8010.0,
            (1, 7): -132930.0,
            (1, 8): 1590750.0,
            (1, 9): -32687550.0,
            (2, 2): 2.0,
            (2, 3): 6.0,
            (2, 4): 54.0,
            (2, 5): 90.0,
            (2, 6): 4770.0,
            (2, 7): -4410.0,
            (2, 8): 872550.0,
            (2, 9): -3543750.0,
            (3, 3): -126.0,
            (3, 4): 306.0,
            (3, 5): -6210.0,
            (3, 6): 33750.0,
            (3, 7): -784350.0,
            (3, 8): 6926850.0,
            (3, 9): -185494000.0,
            (4, 4): 1314.0,
            (4, 5): 8910.0,
            (4, 6): 109350.0,
            (4, 7): 652050.0,
            (4, 8): 19249650,
            (4, 9): 79181550.0,
            (5, 5): -287550.0,
            (5, 6): 1089450.0,
            (5, 7): -34898850.0,
            (5, 8): 233178750.0,
            (5, 9): -8028294755.6,
            (6, 6): 8752050.0,
            (6, 7): 93583349.95,
            (6, 8): 1499289754.5,
            (6, 9): 15154633907.0,
            (7, 7): -4119623546.23,
            (7, 8): 21151793137.0,
            (7, 9): -928576320832.0,
            (8, 8): 251718941164.0
        }

    def eval(self, m: int, n: int) -> float:
        """Evaluate d_z^m d_zbar^n eta(z, zbar) at the crossing symmetric point.

        Args:
            m: The order of the derivative with respect to z
            n: The order of the derivative with respect to zbar

        Returns:
            The value of the (m, n)-th derivative of r at the crossing symmetric point

        """
        if (m, n) not in self.derivatives and (n, m) not in self.derivatives:
            raise ValueError(f"Derivative of order {m}, {n} not found.")

        if (m, n) in self.derivatives:
            return self.derivatives[(m, n)]

        # Derivatives are symmetric in m and n
        return self.derivatives[(n, m)]


def nth_order_derivative(func: callable, m: int, n: int, z: float, zbar: float) -> float:
    """Compute nth-order derivative using JAX.

    Args:
        func : function R^2 -> R
        m: order of derivative wrt z
        n: order of derivative wrt zbar
        z: point in z
        zbar: point in zbar

    Returns:
        The value of the (m, n)-th derivative of func at the point (z, zbar)

    """
    deriv_func = func

    for _ in range(m):
        deriv_func = jax.grad(deriv_func, 0)  # derivative wrt z (first arg)

    for _ in range(n):
        deriv_func = jax.grad(deriv_func, 1)  # derivative wrt zbar (second arg)

    return deriv_func(z, zbar)


class etaDerivativesNumericJax:
    """This class computes derivatives of eta(z, zbar) on the fly using JAX."""

    def eta(self, z: float, zbar: float) -> float:
        term1 = jnp.sqrt((z * (1 + jnp.sqrt(1 - zbar)) ** 2) / (zbar * (1 + jnp.sqrt(1 - z)) ** 2))
        term2 = jnp.sqrt((zbar * (1 + jnp.sqrt(1 - z)) ** 2) / (z * (1 + jnp.sqrt(1 - zbar)) ** 2))
        return 0.5 * (term1 + term2)

    def eval(self, m: int, n: int) -> float:
        """Evaluate d_z^m d_zbar^n eta(z, zbar) at the crossing symmetric point."""
        return nth_order_derivative(self.eta, m, n, 0.5, 0.5)


class rEtaDerivativesSymbolic:
    """This class computes derivatives of r(z, zbar) and eta(z, zbar) with respect to z and zbar symbolically."""

    @cache
    def eval_r(self, m: int, n: int, z: float, zbar: float) -> float:
        """Evaluate d_z^m d_zbar^n r(z, zbar) at (z, zbar).

        Examples:
            eval_r(0, 0, 0.5, 0.5) = 0.1715728753
            eval_r(0, 1, 0.5, 0.5) = 0.2426406871
            eval_r(0, 2, 0.5, 0.5) = 0.1005050634
            eval_r(0, 3, 0.5, 0.5) = 1.1543289326

            eval_r(1, 1, 0.5, 0.5) = 0.3431457505
            eval_r(1, 2, 0.5, 0.5) = 0.1421356237
            eval_r(1, 3, 0.5, 0.5) = 1.6324676319

            eval_r(2, 2, 0.5, 0.5) = 0.0588745030
            eval_r(2, 3, 0.5, 0.5) = 0.6761902332
            eval_r(2, 4, 0.5, 0.5) = 1.0900448581

        Args:
            m: The order of the derivative with respect to z
            n: The order of the derivative with respect to zbar
            z: Point in z
            zbar: Point in zbar

        Returns:
            The value of the (m, n)-th derivative of r(z, zbar) at (z, zbar)

        """
        sum = 0

        for m1, m2, n1, n2 in self.pair_splits(m, n):
            term = self.ff(0.5, m1) * self.ff(0.5, n1)
            term *= z ** (0.5 - m1) * zbar ** (0.5 - n1)
            term *= self.f_m_z(m2, z) * self.f_m_z(n2, zbar)
            term /= self.factorial(m1) * self.factorial(m2) * self.factorial(n1) * self.factorial(n2)
            sum += term

        sum *= self.factorial(m) * self.factorial(n)

        return sum

    @cache
    def eval_eta(self, m: int, n: int, z: float, zbar: float) -> float:
        """Evaluate d_z^m d_zbar^n eta(z, zbar) at (z, zbar).

        Examples:
            eval_eta(0, 0, 0.5, 0.5) = 1.0
            eval_eta(0, 1, 0.5, 0.5) = 0.0
            eval_eta(0, 2, 0.5, 0.5) = 2.0
            eval_eta(0, 3, 0.5, 0.5) = -6.0

            eval_eta(1, 1, 0.5, 0.5) = -2.0
            eval_eta(1, 2, 0.5, 0.5) = 2.0
            eval_eta(1, 3, 0.5, 0.5) = -18.0

            eval_eta(2, 2, 0.5, 0.5) = 2.0
            eval_eta(2, 3, 0.5, 0.5) = 6.0
            eval_eta(2, 4, 0.5, 0.5) = 54.0

        Args:
            m: The order of the derivative with respect to z
            n: The order of the derivative with respect to zbar
            z: Point in z
            zbar: Point in zbar

        Returns:
            The value of the (m, n)-th derivative of eta(z, zbar) at (z, zbar)

        """
        deriv_left = self.phi_3_derivative(m, n, z, zbar, direction='left')
        deriv_right = self.phi_3_derivative(m, n, z, zbar, direction='right')
        return (deriv_left + deriv_right) / 2

    @cache
    def phi_3_derivative(self, m: int, n: int, z: float, zbar: float, direction: Literal['left', 'right']) -> float:
        """Compute d_m d_n_bar [phi_3(z, zbar) * phi_3(zbar, z)^(-1)] at (z, zbar).

        Args:
            m: The order of the derivative with respect to z
            n: The order of the derivative with respect to zbar
            z: Point in z
            zbar: Point in zbar
            direction: 'left' for phi_3(z, zbar) / phi_3(zbar, z), 'right' for phi_3(zbar, z) / phi_3(z, zbar)

        Returns:
            The value of the (m, n)-th derivative

        """
        sum = 0

        if direction == 'left':
            for i in range(0, m + 1):
                fac_i = self.binomial(m, i) * self.ff(0.5, i) * z ** (0.5 - i) * self.f_m_z(m - i, z)
                for j in range(0, n + 1):
                    delta_j0 = 1 if j == 0 else 0
                    term = fac_i * self.binomial(n, j) * self.ff(-0.5, n - j)
                    term *= zbar ** (-0.5 - n + j) * (delta_j0 + (-1) ** j * self.ff(0.5, j) * (1 - zbar) ** (0.5 - j))
                    sum += term

            return sum

        for i in range(0, m + 1):
            delta_i0 = 1 if i == 0 else 0
            fac_i = self.binomial(m, i) * self.ff(-0.5, m - i) * z ** (-0.5 - m + i)
            fac_i *= (delta_i0 + (-1) ** i * self.ff(0.5, i) * (1 - z) ** (0.5 - i))
            for j in range(0, n + 1):
                term = fac_i * self.binomial(n, j) * self.ff(0.5, j)
                term *= zbar ** (0.5 - j) * self.f_m_z(n - j, zbar)
                sum += term

        return sum

    @cache
    def f_m_z(self, m: int, z: float) -> float:
        """Compute d^m/dz^m of f(z) = (1 + sqrt(1 - z))^(-1) at z.

        Examples:
            f_m_z(0, 0.5) = 0.585786
            f_m_z(1, 0.5) = 0.242640
            f_m_z(2, 0.5) = 0.443650
            f_m_z(3, 0.5) = 1.580735
            f_m_z(4, 0.5) = 8.567317
            f_m_z(5, 0.5) = 62.81925

        Args:
            m: Order of derivative
            z: Point in z

        Returns:
            The value of the m-th derivative of f(z) at z

        """
        sum = 0

        for k in range(m + 1):
            # Generate partitions satisfying sum_i j_i = k and sum_i i * j_i = m
            partitions = generate_partitions([(i, 0) for i in range(1, m - k + 2)], k, 0, m, 0)

            bell_poly = 0

            for partition, _ in partitions:
                prod = 1
                for i, j_i in enumerate(partition, start=1):
                    fac_i = (self.ff(0.5, i) * (-1) ** i * (1 - z) ** (0.5 - i)) ** j_i
                    fac_i /= self.factorial(j_i) * self.factorial(i) ** j_i
                    prod *= fac_i
                bell_poly += prod

            sum += self.ff(-1, k) * (1 + mp.sqrt(1 - z)) ** (-1 - k) * bell_poly

        sum *= self.factorial(m)

        return sum

    def pair_splits(self, m: int, n: int):
        """Yield all (m1, m2, n1, n2) with m1 + m2 = m and n1 + n2 = n."""
        for m1 in range(m + 1):
            m2 = m - m1
            for n1 in range(n + 1):
                n2 = n - n1
                yield (m1, m2, n1, n2)

    @cache
    def binomial(self, m, s) -> int:
        """Cached binomial coefficient."""
        return comb(m, s, exact=True)

    @cache
    def factorial(self, n) -> int:
        """Cached factorial."""
        return factorial(n)

    @cache
    def ff(self, x, n) -> mp.mpf:
        """Cached falling factorial."""
        return mp.ff(x, n)
