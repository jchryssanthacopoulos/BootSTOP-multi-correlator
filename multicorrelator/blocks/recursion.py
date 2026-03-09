"""Class to recursively compute conformal blocks based on Equation (4.5) of 1406.4858."""

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import mpmath as mp
import numpy as np
from scipy.special import eval_gegenbauer, factorial

from multicorrelator.blocks.base import BlockType, ConvolvedBlocks3D


parent_dir = Path(__file__).resolve().parents[2]
lib_path = parent_dir / "goblocks" / "server" / "lib" / "librecursive.so"

GOBLOCKS_LIB = ctypes.CDLL(str(lib_path))
GOBLOCKS_LIB.RunRequest.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong)]
GOBLOCKS_LIB.RunRequest.restype = ctypes.POINTER(ctypes.c_double)
GOBLOCKS_LIB.FreeResult.argtypes = [ctypes.POINTER(ctypes.c_double)]
GOBLOCKS_LIB.FreeResult.restype = None


@dataclass
class PolesData:
    n: int
    delta: float
    ell: int
    c: float
    idx: int


@dataclass
class ConvergedBlockData:
    r: float
    eta: float
    delta_12: float
    delta_34: float

    poles_data: list[PolesData]
    h_final: np.ndarray


class RecursiveG:
    def __init__(self, k_1_max: int, k_2_max: int, ell_min: float, ell_max: float, d: int):
        """Initialize the class.

        Args:
            k_1_max: Maximum order of Type I poles to retain
            k_2_max: Maximum order of Type II poles to retain
            ell_min: Minimum spin
            ell_max: Maximum spin
            d: Spacetime dimension

        """
        self.k_1_max = k_1_max
        self.k_2_max = k_2_max
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.nu = (d - 2) / 2

        # This keeps track of the unique combination of (delta, ell) of the poles
        self.unique_poles_map = {}

        self.converged_data = None

    def recurse(
        self,
        delta_12: float,
        delta_34: float,
        r: float,
        eta: float,
        max_iterations: int = 100,
        tol: float = 1e-6,
    ):
        """Run the recursion to compute h.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            r: Radial coordinate
            eta: Angle in the complex plane
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence

        Returns:
            The computed g matrix at each iteration, the scaling dimensions, and the spins

        """
        # Compute the poles
        self.unique_poles_map = {}
        poles_data = self.get_all_poles_data(delta_12, delta_34)

        # Compute h tilde
        h_tilde = np.zeros(len(self.unique_poles_map))
        for (
            _,
            ell,
        ), idx in self.unique_poles_map.items():  # h_tilde is independent of delta
            h_tilde[idx] = self._htilde(delta_12, delta_34, ell, r, eta)

        # Initialize h to h tilde
        h = h_tilde.copy()

        for iter in range(max_iterations):
            h_new = h_tilde.copy()

            # Loop over scaling dimensions and spins, updating h
            for (delta, ell), idx in self.unique_poles_map.items():
                for poles in poles_data[ell]:
                    # Note that division by zero protection is not required
                    # here because delta will never equal poles.delta
                    h_new[idx] += (
                        poles.c * r**poles.n * h[poles.idx] / (delta - poles.delta)
                    )

            # Print difference between old and new h
            diff = np.max(np.abs(h_new - h))
            print(f"Iteration {iter}: max difference = {diff}")

            # Check convergence
            if diff < tol:
                break

            h = h_new

        # Save last iteration
        self.converged_data = ConvergedBlockData(
            r=r,
            eta=eta,
            delta_12=delta_12,
            delta_34=delta_34,
            poles_data=poles_data,
            h_final=h,
        )

    def evaluate_g(
        self,
        delta_12: float,
        delta_34: float,
        delta: float,
        ell: float,
        r: float,
        eta: float,
    ) -> float:
        """Evaluate the recursion result for a specific spin and scaling dimension.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            delta: The scaling dimension to evaluate
            ell: The spin to evaluate
            r: Radial coordinate
            eta: Angle in the complex plane

        Returns:
            The evaluated g value for the given parameters

        """
        if self.converged_data is None:
            raise ValueError("Recursion has not been run yet. Call `recurse` first.")

        if r != self.converged_data.r or eta != self.converged_data.eta:
            raise ValueError("r and eta do not match the ones used in the recursion.")

        if (
            delta_12 != self.converged_data.delta_12
            or delta_34 != self.converged_data.delta_34
        ):
            raise ValueError(
                "Delta values do not match the ones used in the recursion."
            )

        if ell not in self.converged_data.poles_data:
            raise ValueError(
                f"No poles data available for spin {ell}. Run `recurse` first."
            )

        g = self._htilde(delta_12, delta_34, ell, r, eta)

        for poles in self.converged_data.poles_data[ell]:
            if delta != poles.delta:
                # Divide by zero protection is required because delta can be anything
                g += (
                    poles.c
                    * r**poles.n
                    * self.converged_data.h_final[poles.idx]
                    / (delta - poles.delta)
                )

        g *= r**delta

        return g

    def recurse_and_evaluate_g(
        self,
        delta_12: float,
        delta_34: float,
        deltas: np.ndarray,
        spins: np.ndarray,
        r_vals: np.ndarray,
        eta_vals: np.ndarray,
        max_iterations: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Run the recursion and evaluate g in one step.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            deltas: The scaling dimensions to evaluate
            spins: The spins to evaluate (must be the same length as deltas)
            r_vals: Radial coordinates
            eta_vals: Angles in the complex plane (must be the same length as r_vals)
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence

        Returns:
            The evaluated g values for the given parameters with the shape
                (len(deltas), len(r_vals))

        """
        if len(deltas) != len(spins):
            raise ValueError("Length of delta and ell must match.")

        if len(r_vals) != len(eta_vals):
            raise ValueError("Length of r_vals and eta_vals must match.")

        result = np.zeros((len(deltas), len(r_vals)))

        for i, (r, eta) in enumerate(zip(r_vals, eta_vals)):
            # Call recurse once for each r and eta point
            self.recurse(delta_12, delta_34, r, eta, max_iterations, tol)

            # Evaluate the blocks for each delta and spin
            result[:, i] = [
                self.evaluate_g(delta_12, delta_34, delta, ell, r, eta)
                for delta, ell in zip(deltas, spins)
            ]

        return result

    def recurse_and_evaluate_g_using_z(
        self,
        delta_12: float,
        delta_34: float,
        deltas: np.ndarray,
        spins: np.ndarray,
        z_vals: np.ndarray,
        max_iterations: int = 100,
        tol: float = 1e-6,
    ) -> float:
        """Run the recursion and evaluate g in one step using complex variable z.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            deltas: The scaling dimensions to evaluate
            spins: The spins to evaluate (must be the same length as deltas)
            z_vals: Array of z points to evaluate
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence

        Returns:
            The evaluated g value for the given parameters with the shape
                (len(deltas), len(z_vals))

        """
        r_vals, eta_vals = self._z_to_r_eta(z_vals)

        return self.recurse_and_evaluate_g(
            delta_12, delta_34, deltas, spins, r_vals, eta_vals, max_iterations, tol
        )

    def recurse_and_evaluate_F_using_z(
        self,
        block_types: list[BlockType],
        delta_12: float,
        delta_34: float,
        delta_ave_23: float,
        deltas: np.ndarray,
        spins: np.ndarray,
        z_vals: np.ndarray,
        max_iterations: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Run the recursion and evaluate F_+/- in one step.

        This implements Equation (3.7) of 1406.4858.

        Args:
            block_types: List of block types (e.g., '+' or '-')
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            delta_ave_23: Average scaling dimension between the second and third operators
            deltas: The scaling dimensions to evaluate
            spins: The spins to evaluate (must be the same length as deltas)
            z_vals: Array of z points to evaluate
            max_iterations: Maximum number of iterations for convergence
            tol: Tolerance for convergence

        Returns:
            The evaluated g value for the given parameters with shape
                (len(block_types), len(spins), len(z_vals))

        """
        g_u_v = self.recurse_and_evaluate_g_using_z(
            delta_12, delta_34, deltas, spins, z_vals, max_iterations, tol
        )

        # Evaluate g(v, u). u, v -> v, u when z -> 1 - z
        if not np.array_equal(z_vals, 1 - z_vals):
            g_v_u = self.recurse_and_evaluate_g_using_z(
                delta_12, delta_34, deltas, spins, 1 - z_vals, max_iterations, tol
            )
        else:
            g_v_u = g_u_v

        u_vals, v_vals = self._z_to_u_v(z_vals)

        u_expanded = u_vals[None, :]
        v_expanded = v_vals[None, :]

        F = np.zeros((len(block_types), len(spins), len(z_vals)))

        for i, block_type in enumerate(block_types):
            term1 = v_expanded**delta_ave_23 * g_u_v
            term2 = u_expanded**delta_ave_23 * g_v_u

            if block_type == BlockType.MINUS:
                term2 *= -1

            val = term1 + term2

            if np.any(np.abs(np.imag(val)) > 1e-10):
                raise ValueError("Imaginary part found in F")

            F[i] = val.real

        return F

    def get_all_poles_data(
        self, delta_12: float, delta_34: float
    ) -> dict[float, list[PolesData]]:
        """Get all poles data.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators

        Returns:
            A dictionary mapping spins to lists of PolesData objects

        """
        ell_vals = np.arange(self.ell_min, self.ell_max + 1, 1)

        poles_total = {}

        for ell in ell_vals:
            poles_total[ell] = self.get_delta_1_poles(ell, delta_12, delta_34)

            poles_total[ell].extend(self.get_delta_2_poles(ell, delta_12, delta_34))

            poles_total[ell].extend(self.get_delta_3_poles(ell, delta_12, delta_34))

        return poles_total

    def get_delta_1_poles(
        self, ell: int, delta_12: float, delta_34: float
    ) -> list[PolesData]:
        """Get the poles data for Type I.

        This implements row 1 of Table 1 of 1406.4858.

        Args:
            ell: The spin
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators

        Returns:
            A list of PolesData objects for Type I poles

        """

        def c_1(ell: int, k: int) -> float:
            """Compute the coefficient for Type I poles."""
            denom = factorial(k) ** 2 * mp.rf(ell + self.nu, k)

            numerator = mp.rf(ell + 2 * self.nu, k)
            numerator *= mp.rf((1 - k + delta_12) / 2, k)
            numerator *= mp.rf((1 - k + delta_34) / 2, k)
            numerator *= -(4**k) * k * (-1) ** k

            return numerator / denom

        poles = []

        for k in range(1, self.k_1_max + 1):
            ell_val = ell + k

            if ell_val > self.ell_max or ell_val < self.ell_min:
                continue

            n = k
            delta = 1 - ell - k

            idx = self._get_or_add_pole_idx(delta + n, ell_val)

            poles.append(
                PolesData(n=n, delta=delta, ell=ell_val, c=c_1(ell, k), idx=idx)
            )

        return poles

    def get_delta_2_poles(
        self, ell: int, delta_12: float, delta_34: float
    ) -> list[PolesData]:
        """Get the poles data for Type II.

        This implements row 2 of Table 1 of 1406.4858.

        Args:
            ell: The spin
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators

        Returns:
            A list of PolesData objects for Type II poles

        """

        def c_2(ell: int, k: int) -> float:
            """Compute the coefficient for Type II poles."""
            denom = (
                factorial(k) ** 2
                * mp.rf(ell + self.nu - k, 2 * k)
                * mp.rf(ell + self.nu + 1 - k, 2 * k)
            )

            numerator = mp.rf(self.nu - k, 2 * k)
            numerator *= mp.rf((1 - k + ell - delta_12 + self.nu) / 2, k)
            numerator *= mp.rf((1 - k + ell + delta_12 + self.nu) / 2, k)
            numerator *= mp.rf((1 - k + ell - delta_34 + self.nu) / 2, k)
            numerator *= mp.rf((1 - k + ell + delta_34 + self.nu) / 2, k)
            numerator *= -(4 ** (2 * k)) * k * (-1) ** k

            return numerator / denom

        poles = []

        for k in range(1, self.k_2_max + 1):
            ell_val = ell

            if ell_val > self.ell_max or ell_val < self.ell_min:
                continue

            n = 2 * k
            delta = 1 + self.nu - k

            idx = self._get_or_add_pole_idx(delta + n, ell_val)

            poles.append(
                PolesData(n=n, delta=delta, ell=ell_val, c=c_2(ell, k), idx=idx)
            )

        return poles

    def get_delta_3_poles(
        self, ell: int, delta_12: float, delta_34: float
    ) -> list[PolesData]:
        """Get the poles data for Type III.

        This implements row 3 of Table 1 of 1406.4858.

        Args:
            ell: The spin
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators

        Returns:
            A list of PolesData objects for Type III poles

        """

        def c_3(ell: int, k: int) -> float:
            """Compute the coefficient for Type III poles."""
            denom = factorial(k) ** 2 * mp.rf(ell + self.nu + 1 - k, k)

            numerator = mp.rf(ell + 1 - k, k)
            numerator *= mp.rf((1 - k + delta_12) / 2, k)
            numerator *= mp.rf((1 - k + delta_34) / 2, k)
            numerator *= -(4**k) * k * (-1) ** k

            return numerator / denom

        poles = []

        for k in range(1, ell + 1):
            ell_val = ell - k

            if ell_val > self.ell_max or ell_val < self.ell_min:
                continue

            n = k
            delta = 1 + ell + 2 * self.nu - k

            idx = self._get_or_add_pole_idx(delta + n, ell_val)

            poles.append(
                PolesData(n=n, delta=delta, ell=ell_val, c=c_3(ell, k), idx=idx)
            )

        return poles

    def _htilde(
        self, delta_12: float, delta_34: float, ell: float, r: float, eta: float
    ) -> float:
        """Compute htilde for given parameters.

        This implements Equation (4.6) of 1406.4858.

        Args:
            delta_12: Difference between scaling dimensions of first two external operators
            delta_34: Difference between scaling dimensions of last two external operators
            ell: The spin
            r: Radial coordinate
            eta: Angle in the complex plane

        Returns:
            The computed htilde matrix for the given spin

        """
        coeff = factorial(ell)
        denom = mp.rf(2 * self.nu, ell)
        gegen = eval_gegenbauer(ell, self.nu, eta)

        prefactor = coeff / denom * (-1) ** int(ell) * gegen

        factor1 = (1 - r**2) ** self.nu
        factor2 = (1 + r**2 + 2 * r * eta) ** (0.5 * (1 + delta_12 - delta_34))
        factor3 = (1 + r**2 - 2 * r * eta) ** (0.5 * (1 - delta_12 + delta_34))

        return prefactor / (factor1 * factor2 * factor3)

    def _get_or_add_pole_idx(self, delta: float, ell: float) -> int:
        """Return the index of a pole given by `key`, adding it if it's new.

        Args:
            delta: The scaling dimension of the pole
            ell: The spin of the pole

        Returns:
            The index of the pole in the unique_poles_map

        """
        key = (delta, ell)

        if key not in self.unique_poles_map:
            idx = len(self.unique_poles_map)
            self.unique_poles_map[key] = idx
            return idx

        return self.unique_poles_map[key]

    def _z_to_r_eta(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion from complex z to (r, eta) arrays.

        Args:
            z: Array of complex values (can also be a scalar)

        Returns:
            r: Array of |r exp(i theta)|
            eta: Array of cos(theta)

        """
        r_exp_i_theta = z / (1 + np.sqrt(1 - z)) ** 2

        r = np.abs(r_exp_i_theta)
        eta = np.where(r != 0, r_exp_i_theta.real / r, 0.0)

        return r, eta

    def _z_to_u_v(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion from complex z to (u, v) arrays.

        Args:
            z: np.ndarray of complex values (or scalar)

        Returns:
            u: Array of real values
            v: Array of real values

        """
        z_star = np.conj(z)

        u = z * z_star
        v = (1 - z) * (1 - z_star)

        return u, v


class BaseConvolvedBlocksRecursive(ConvolvedBlocks3D):
    """Base class with shared initialization logic for recursive 3D convolved blocks."""

    Z_FILE_PATH = (
        Path(__file__).parent / "../block_lattices/z_pts/output"
    )

    def __init__(
        self,
        spins: list[int],
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        d: int = 3,
        num_z_points: int = 100,
        max_iterations: int = 100,
        tol: float = 1e-6,
        z_pts_filename: str | None = None,
    ):
        super().__init__(spins)

        self.spins = spins
        self.k_1_max = k_1_max
        self.k_2_max = k_2_max
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.d = d
        self.num_z_points = num_z_points
        self.max_iterations = max_iterations
        self.tol = tol

        # If no filename given, use default
        if z_pts_filename is None:
            z_pts_filename = "sampled_points.npy"

        self.z_file = self.Z_FILE_PATH / z_pts_filename

        # Load z points in the "peanut"
        self.z_vals = self._load_sampled_zpoints()

        # Limit z points to num_z_points
        if len(self.z_vals) < num_z_points:
            raise RuntimeError("There are not enough points in the sample file.")

        self.z_vals = self.z_vals[:num_z_points]
        self.z_vals_str = [f"{z.real}{z.imag:+}i" for z in np.ravel(self.z_vals)]

    def _load_sampled_zpoints(self, textfile: bool = False) -> np.ndarray:
        """Load sampled z-points from file into a NumPy complex array."""
        if textfile:
            z_pts = np.loadtxt(self.z_file, comments="#")
        else:
            z_pts = np.load(self.z_file)
        return z_pts


class ConvolvedBlocksRecursivePython(BaseConvolvedBlocksRecursive):
    """3D convolved blocks evaluated using Python recursion."""

    def __init__(
        self,
        spins: list[int],
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        d: int = 3,
        num_z_points: int = 100,
        max_iterations: int = 100,
        tol: float = 1e-6,
        z_pts_filename: str | None = None,
    ):
        super().__init__(
            spins,
            k_1_max,
            k_2_max,
            ell_min,
            ell_max,
            d,
            num_z_points,
            max_iterations,
            tol,
            z_pts_filename,
        )
        self.recursive_g = RecursiveG(
            self.k_1_max, self.k_2_max, self.ell_min, self.ell_max, self.d
        )

    def evaluate(
        self,
        block_types: list["BlockType"],
        spins: np.ndarray,
        deltas: np.ndarray,
        delta_ij: float,
        delta_kl: float,
        delta_ave_kj: float,
    ) -> np.ndarray:
        """Evaluate using the Python recursive implementation."""
        return self.recursive_g.recurse_and_evaluate_F_using_z(
            block_types,
            delta_ij,
            delta_kl,
            delta_ave_kj,
            deltas,
            spins,
            self.z_vals,
            self.max_iterations,
            self.tol,
        )


class ConvolvedBlocksRecursiveGoBlocks(BaseConvolvedBlocksRecursive):
    """3D convolved blocks evaluated using GoBlocks client."""

    def __init__(
        self,
        spins: list[int],
        k_1_max: int,
        k_2_max: int,
        ell_min: float,
        ell_max: float,
        d: int = 3,
        num_z_points: int = 100,
        max_iterations: int = 100,
        tol: float = 1e-12,
        z_pts_filename: str | None = None,
    ):
        super().__init__(
            spins,
            k_1_max,
            k_2_max,
            ell_min,
            ell_max,
            d,
            num_z_points,
            max_iterations,
            tol,
            z_pts_filename,
        )

    def evaluate(
        self,
        block_types: list[BlockType],
        spins: list[int] | np.ndarray,
        deltas: list[float] | np.ndarray,
        delta_ij: float,
        delta_kl: float,
        delta_ave_kj: float,
    ) -> np.ndarray:
        """Evaluate using GoBlocks."""
        if self.tol < 1e-12:
            warnings.warn("Using a tolerance value less than 1e-12 may result in numerical instability")

        if isinstance(spins, np.ndarray):
            spins = spins.tolist()

        if isinstance(deltas, np.ndarray):
            deltas = deltas.tolist()

        req = {
            "command": "recurse_and_evaluate_f",
            "k1_max": self.k_1_max,
            "k2_max": self.k_2_max,
            "ell_min": self.ell_min,
            "ell_max": self.ell_max,
            "d": self.d,
            "delta_12": delta_ij,
            "delta_34": delta_kl,
            "delta_ave_23": delta_ave_kj,
            "zsstr": self.z_vals_str,
            "deltas": deltas,
            "ells": spins,
            "block_types": [bt.value for bt in block_types],
            "max_iterations": self.max_iterations,
            "tol": self.tol,
        }

        # call into Go
        length = ctypes.c_longlong()
        res_ptr = GOBLOCKS_LIB.RunRequest(json.dumps(req).encode("utf-8"), ctypes.byref(length))

        # convert to numpy array (takes ownership of copy)
        if res_ptr and length.value > 0:
            res_np = np.ctypeslib.as_array(res_ptr, shape=(length.value,))
            result = res_np.reshape(len(block_types), len(deltas), len(self.z_vals_str)).copy()
        else:
            result = None

        # free memory
        if res_ptr:
            GOBLOCKS_LIB.FreeResult(res_ptr)

        return result
