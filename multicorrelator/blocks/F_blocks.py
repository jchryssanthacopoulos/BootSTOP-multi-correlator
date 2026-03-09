"""Class to compute derivatives of F blocks based on scalar_blocks."""

from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Callable

from jax import config as jax_config
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from multicorrelator.blocks.base import BlockType


# Set MPMath precision
mp.mp.prec = 665


# Enable 64 bit Jax to enable higher float bit number (stops overflows)
jax_config.update("jax_enable_x64", True)


# Set up safe fork behaviour for Multiprocessing (default on MacOS/Windows)
multiprocessing.set_start_method('spawn', force=True)


# Define the scalar_blocks executable
SCALAR_BLOCKS_EXECUTABLE = Path(__file__).resolve().parent.parent.parent / "executables" / "scalar_blocks"

if not SCALAR_BLOCKS_EXECUTABLE.exists():
    raise FileNotFoundError(f"Executable not found: {SCALAR_BLOCKS_EXECUTABLE}")


@dataclass
class ParsedBlock:
    """A class used to hold the parsed block data and methods to manipulate it."""

    R_0 = 3 - 2 * mp.sqrt(2)

    spin: int
    dim: int

    expansion_coeffs: np.ndarray
    keys: np.ndarray
    output_poles: dict

    def convolve(self, exponent: float, symmetric: bool):
        """Generate the expansion coefficients for derivatives of a convolved conformal block.

        This method deletes rows from `expansion_coeff` which are zero by symmetry and also deletes the corresponding
            entries in `key`.

        Args:
            exponent: The exponent of u and v in the convolved block
            symmetric: Whether the convolved block is symmetric under (z, z*) -> (1 - z, 1 - z*)

        TECHNICAL DETAILS:
            Note the convolution process keeps blocks independent of delta and x.

            If :math:`g_{Delta,ell}^{Delta_{12},Delta_{34}}(z,z^*)` is
            an ordinary conformal block then the convolved block is either:

            .. math::

            F^{alpha}_{pm, Delta, ell}(z,z^*) = v^{alpha}
            g_{Delta,ell}^{Delta_{12},Delta_{34}}(z,z^*)
            pm (z rightarrow 1-z, z^* \rightarrow 1-z^*)

            or

            .. math::

            H^{alpha}_{pm, Delta,ell}(z,z^*) = u^{alpha}
            g_{Delta,ell}^{Delta_{12},Delta_{34}}(z,z^*)
            pm (z rightarrow 1-z, z^* rightarrow 1-z^*)

            with the :math:`u,v` cross-ratios defined by
            :math:`u= z z^*` and :math:`v= (1-z)(1-z^*)`.

            The derivative :math:`partial_{z}^{m} partial_{z^*}^{n}` of
            :math:`F^{alpha}_{pm, Delta, ell}(z,z^*)` or
            :math:`H^{alpha}_{pm, Delta, ell}(z,z^*)` evaluated at the
            crossing symmetric point :math:`z=z^*=0.5`
            is computed using the generalised Leibnitz rule:

            .. math::

            partial_z^m partial_{z^*}^n [f(z,z^*) g(z,z^*)] =
            sum_{i=0}^{m} sum_{j=0}^{n} binom{m}{i} binom{n}{j}
            partial_z^i partial_{z^*}^j f(z,z^*)
            partial_z^{m-i} partial_{z^*}^{n-j} g(z,z^*)

            The final expressions are

            .. math::

            partial_{z}^{m} partial_{z^*}^{n}
            F^{alpha}_{pm, Delta,ell}(z,z^*) =
            [1 pm (-1)^{m+n}] sum_{i=0}^{m} sum_{j=0}^{n}
            binom{m}{i} binom{n}{j} c^{alpha}_{ij}
            partial_z^{m-i} partial_{z^*}^{n-j}
            g_{Delta,ell}^{Delta_{12},Delta_{34}}(z,z^*)

            and

            .. math::

            partial_{z}^{m} partial_{z^*}^{n}
            H^{alpha}_{pm, Delta,ell}(z,z^*) =
            [1 pm (-1)^{m+n}] sum_{i=0}^{m} sum_{j=0}^{n}
            binom{m}{i} binom{n}{j} (-1)^{i+j} c^{alpha}_{ij}
            partial_z^{m-i} partial_{z^*}^{n-j}
            g_{Delta,ell}^{Delta_{12},Delta_{34}}(z,z^*)

            with the coefficients appearing in the sum being

            .. math::

            c^alpha_{ij} = (-1)^{i+j} 2^{i+j-2alpha}
            (1+alpha-i)_i (1+alpha-j)_j .

        """
        summation_variable_values = np.arange(1 + self.keys.max(), dtype=int)

        # Pochhammer in SciPy can overflow, switch scipy.special.poch -> mp.rf
        mprf_vectorized = np.vectorize(mp.rf)
        pochhammers = mprf_vectorized(1 + exponent - summation_variable_values, summation_variable_values)

        convolved_expansion_coefficients = np.zeros_like(self.expansion_coeffs)
        mask = np.ones(self.keys.shape[0], dtype=int)
        import pdb; pdb.set_trace()

        for location, entry in enumerate(self.keys):
            m, n = entry[0], entry[1]

            if (symmetric and (m + n) % 2 == 1) or (not symmetric and (m + n) % 2 == 0):
                convolved_expansion_coefficients[location] = 0
                mask[location] = 0
                continue

            running_sum = 0

            for i in range(0, m + 1):
                for j in range(0, n + 1):
                    running_sum += (
                        2 * (-1) ** (i + j) * 2 ** (i + j - 2 * exponent) *
                        mp.binomial(m, i) * mp.binomial(n, j) * pochhammers[i] * pochhammers[j] *
                        self._deriv_block(m - i, n - j)
                    )

            convolved_expansion_coefficients[location] = running_sum

        # Delete the zero coefficients to help reduce file size when saving and update attributes
        self.expansion_coeffs = np.delete(convolved_expansion_coefficients, mask == 0, axis=0)
        self.keys = np.delete(self.keys, mask == 0, axis=0)
        import pdb; pdb.set_trace()

    def normalize(self, use_alternative_normalization: bool | None = False):
        """Normalize the conformal block.

        A separate normalization factor is broadcast over each row in the attribute `expansion_coeffs`. Normalization
            should be applied only once and must not be applied before calling _convolve (as convolution mixes
            different orders of derivatives).

        Args:
            use_alternative_normalization: Use the alternative normalization method (if None, do not normalize)

        TECHNICAL DETAILS:
            The normalization factor :math:`f_{m,n}` associated with an order (m,n) derivative is

            .. math::

            f_{m,n} = [2^{m+n}(m+n)! ]^{-1}.

            If the alternative method is used, it is f_{m,n} = [m! n! 2^{m+n}]^{-1}.

        """
        if use_alternative_normalization is None:
            return

        m, n = np.hsplit(self.keys, 2)
        import pdb; pdb.set_trace()

        # Fastest and cleanest way to convert key entries to mpf
        m, n = mp.mpf(1) * m, mp.mpf(1) * n

        # Create vectorised powers of power and factorial functions
        vectorized_2_power = np.vectorize(lambda a: mp.power(2, a))
        vectorized_factorial = np.vectorize(lambda a: mp.factorial(a))

        if use_alternative_normalization:
            normalization = vectorized_factorial(m) * vectorized_factorial(n) * vectorized_2_power(m + n)
        else:
            normalization = vectorized_factorial(m + n) * vectorized_2_power(m + n)

        self.expansion_coeffs = self.expansion_coeffs / normalization

    def evaluate(
            self,
            delta_lattice: np.ndarray,
            coefficient_func: Callable = None,
            use_jax: bool = False,
            no_cores: int = 1
    ) -> np.ndarray:
        """Numerically evaluate the conformal block at the given conformal weights.

        Args:
            delta_lattice: The lattice of conformal weights the block is to be evaluated at
            coefficient_func: A weight dependent coefficient function
            use_jax: Use jax where appropriate to speed up calculations
            no_cores: The number of cores to use for parallel processing

        Returns:
            Evaluated block

        """
        # If use_jax, convert array to jax array to speed up computation through allow GPU handling of arrays
        if use_jax:
            # This has to be float64 as Jax supports no larger
            # https://github.com/google/jax/issues/5091#issuecomment-738239005
            expansion_coeffs = jnp.array(self.expansion_coeffs.astype(np.float64))
        else:
            expansion_coeffs = self.expansion_coeffs

        # Compute x lattice
        x_lattice = np.array([delta - (self.spin + self.dim - 2) for delta in delta_lattice])

        # Break up the lattice into no_cores (roughly equal) pieces
        broken_lattice = np.array_split(x_lattice, no_cores)

        # Give each chunk an index. If evaluations don't complete in order, we can track where they came frome
        broken_lattice = list(enumerate(broken_lattice))

        # Create a multiprocessing pool with the specified number of cores
        pool = multiprocessing.Pool(processes=no_cores)

        # Create a list of arguments for the _core_evaluate function on each sublattice
        arguments = [
            (chunk_index, sublattice, expansion_coeffs, coefficient_func, use_jax)
            for chunk_index, sublattice in broken_lattice
        ]

        # Process data chunks in parallel using the multiprocessing pool
        results = pool.starmap(self._core_evaluate, arguments)

        # Close the multiprocessing pool
        pool.close()
        pool.join()

        # Ensure the results came in the correct order
        results = sorted(results, key=lambda x: x[0])

        # Reconstruct the evaluated block
        evaluated_block = np.concatenate(list(zip(*results))[1], axis=1)

        # Transpose block to be more in line with BootSTOP convention
        evaluated_block = np.transpose(evaluated_block)

        # Convert the mpmath array to a NumPy array of floats
        evaluated_block = np.array(evaluated_block, dtype=float)

        return evaluated_block

    def _core_evaluate(
            self,
            chunk_index: int,
            sublattice: np.ndarray,
            expansion_coeffs: np.ndarray,
            coefficient_func: Callable,
            use_jax: bool
    ) -> tuple[int, np.ndarray]:
        """Evaluate the conformal block at a single conformal weight or over an evenly spaced interval.

        Args:
            chunk_index: The index of the chunk
            sublattice: The sublattice to evaluate the block at
            expansion_coeffs: The expansion coefficients
            coefficient_func: A weight dependent coefficient function
            use_jax: Use jax where appropriate to speed up calculations

        Returns:
            The evaluated sub block

        """
        evaluated_sub_block = np.empty(shape=(self.keys.shape[0], len(sublattice)), dtype=object)

        for index, x in enumerate(sublattice):
            # Result = [zzbDeriv[0,0] zzbDeriv[0,1] zzbDeriv[1,0] ... ] [TRSPOSE]
            # Cast the expansion coefficients to floats for quicker eval.
            if use_jax:
                weight_dependent_terms = jnp.array(
                    self._weight_dependent_terms_array(x, coefficient_func).astype(np.float64)
                )
            else:
                weight_dependent_terms = self._weight_dependent_terms_array(x, coefficient_func)
            evaluated_sub_block[:, index] = expansion_coeffs @ weight_dependent_terms

        return chunk_index, evaluated_sub_block

    def _weight_dependent_terms_array(self, x: float, coefficient_func: Callable = None) -> np.ndarray:
        """Evaluate all arrays which are functions of the conformal weight.

        Args:
            x: The value of the x coordinate
            coefficient_func: A weight dependent coefficient function.

        Returns:
            The array of weight dependent terms

        """
        if isinstance(x, (np.generic,)):
            x = x.item()

        x = mp.mpf(x)

        r_0_term = self.R_0 ** (x + self.spin + self.dim - 2)
        recip_poles_product = self._poles_at_x(x + self.spin + self.dim - 2)
        x_monomials = np.power(x, np.arange(self.expansion_coeffs.shape[1]), dtype=object)

        weight_dependent_terms = r_0_term * recip_poles_product * x_monomials.T

        if coefficient_func is not None:
            weight_dependent_terms = coefficient_func(x + self.spin + self.dim - 2) * weight_dependent_terms

        return weight_dependent_terms

    def _deriv_block(self, a: int, b: int) -> np.ndarray:
        """Locate the expansion coefficients for the given derivative orders a and b.

        Args:
            a: The order of the derivative in z
            b: The order of the derivative in z*

        Returns:
            The expansion coefficients

        """
        if a >= b:
            location = np.where((self.keys == np.array([a, b], dtype='int')).all(axis=1))[0]
        else:
            # We can see this in eq. 7 of the scalar_blocks convention file.
            # scalar_blocks uses symmetry to perform less ops
            location = np.where((self.keys == np.array([b, a], dtype='int')).all(axis=1))[0]

        # If location is empty (i.e. invalid m, n supplied)
        if not location.size:
            print(
                'The passed in values for m and n do NOT correspond to a valid derivative in the supplied block file. '
                'Have you tried computing more derivatives with scalar_blocks?'
            )
            raise ValueError('InvalidDerivative')

        return self.expansion_coeffs[location]

    def _poles_at_x(self, delta: float) -> float:
        """Compute the reciprocal Q function at the given delta.

        Args:
            delta: The delta to compute the reciprocal Q function at

        Returns:
            The reciprocal Q function at the given delta

        """
        delta = mp.mpf(delta)
        reciprocal_Q = 1

        for order, poles_fixed_order in self.output_poles.items():
            # delta is defined in scalar_blocks conventions eq. 6
            vectorised_delta = delta * np.ones_like(poles_fixed_order)
            reciprocal_Q *= 1 / np.prod((vectorised_delta - poles_fixed_order) ** order)

        return reciprocal_Q


class FBlocks:
    """A class used to hold derivatives of a conformal block."""

    def __init__(
            self,
            dim: int,
            order: int,
            max_derivs: int,
            poles: int,
            precision: int
    ):
        """Initialises the class with the given parameters.

            dim: The spacetime dimension
            order: The order of the radial coefficient expansion
            max_derivs: The maximum order of derivatives computed as :math:`m+n \\leq 2nmax-1` and :math:`m \\geq n`
            poles: The maximum order in poles
            spin: The spin
            precision: The precision for the block generation

        """
        self.dim = dim
        self.order = order
        self.max_derivs = max_derivs
        self.poles = poles
        self.precision = precision

    def gen_block(
            self,
            block_type: BlockType | None,
            spins: np.ndarray,
            deltas: np.ndarray,
            delta_ij: float,
            delta_kl: float,
            delta_ave_kj: float,
            num_threads: int = 8,
            use_jax: bool = False,
            no_cores: int = 1,
            use_alternative_normalization: bool | None = False,
            verbose: bool = False
    ) -> np.ndarray:
        """Generates the block using the external executable and processes the output file.

        Args:
            block_type: The type of the blocks (e.g., '+' or '-') or None (if None, return dg directly)
            spins: Spins to evaluate the block at
            deltas: The scaling dimensions of the exchanged operator to evaluate the block at
            delta_ij: The difference in the scaling dimension of the first and second operators
            delta_kl: The difference in the scaling dimension of the third and fourth operators
            delta_ave_kj: The average of the scaling dimensions of the second and third operators
            num_threads: Number of threads to use to compute the block
            use_jax: Use jax in evaluation to speed up calculations
            no_cores: The number of cores to use for parallel processing in evaluation step
            use_alternative_normalization: Use the alternative normalization method (if None, do not normalize)
            verbose: Whether to print debug information

        Returns:
            A matrix of evaluated blocks

        """
        # Ensure spins has the same length as deltas
        assert len(spins) == len(deltas), "spins and deltas must have the same length"

        # Get unique spins
        unique_spins, inverse_indices = np.unique(spins, return_inverse=True)

        # Split the deltas into groups with the same spin
        deltas_by_spin = {spin: deltas[inverse_indices == i] for i, spin in enumerate(unique_spins)}

        # Generate and process the scalar block file for each unique spin
        parsed_blocks = self._gen_and_process_scalar_blocks_file(unique_spins, delta_ij, delta_kl, num_threads, verbose)

        evaluated_blocks = {}

        for spin, parsed_block in parsed_blocks.items():
            # Convolve the blocks with u, v
            if block_type is not None:
                # If the block type is "+" then the block is symmetric under (z, z*) -> (1 - z, 1 - z*)
                parsed_block.convolve(delta_ave_kj, block_type == BlockType.PLUS)

            # Normalize the block
            parsed_block.normalize(use_alternative_normalization)

            # Evaluate the block
            evaluated_blocks[spin] = parsed_block.evaluate(
                deltas_by_spin[spin],
                use_jax=use_jax,
                no_cores=no_cores
            )

        # Combine the evaluated blocks
        num_cols = next(iter(evaluated_blocks.values())).shape[1]
        final_matrix = np.empty((len(deltas), num_cols), dtype=float)

        for i, spin in enumerate(unique_spins):
            final_matrix[inverse_indices == i] = evaluated_blocks[spin]

        return final_matrix

    def _gen_and_process_scalar_blocks_file(
            self,
            spins: np.ndarray,
            delta_12: float,
            delta_34: float,
            num_threads: int = 8,
            verbose: bool = False
    ) -> dict[int, ParsedBlock]:
        """Generates the scalar block file and processes the output file.

        Args:
            spins: The spins to generate the block for
            delta_12: The difference in the scaling dimension of the first and second operators
            delta_34: The difference in the scaling dimension of the third and fourth operators
            num_threads: Number of threads to use to compute the block
            verbose: Whether to print debug information

        Returns:
            A dictionary of parsed blocks, keyed by spin

        """
        with tempfile.TemporaryDirectory(prefix="fblocks_") as temp_dir:
            temp_dir = Path(temp_dir)

            # Generate a comma-separated string of spins
            spins_str = ",".join(map(str, spins))

            # Define the command
            scalar_blocks_command = [
                str(SCALAR_BLOCKS_EXECUTABLE),
                '--dim', str(self.dim),
                '--order', str(self.order),
                '--max-derivs', str(self.max_derivs),
                '--spin-ranges', spins_str,
                '--poles', str(self.poles),
                '--delta-12', str(delta_12),
                '--delta-34', str(delta_34),
                f'--num-threads={num_threads}',
                '-o', str(temp_dir),
                f'--precision={self.precision}'
            ]

            # Execute the command
            start = time.time()

            try:
                subprocess.run(scalar_blocks_command)
            except subprocess.CalledProcessError as e:
                raise SystemExit("Error executing scalar_blocks") from e

            if verbose:
                print("Time taken to execute scalar_blocks:", time.time() - start)

            parsed_blocks = {}

            for spin in spins:
                # Define the expected output file name
                data_file = temp_dir / (
                    f'zzbDerivTable-d{self.dim}-delta12-{delta_12}-delta34-{delta_34}-L{spin}-nmax{self.max_derivs}-'
                    f'keptPoleOrder{self.poles}-order{self.order}.m'
                )

                # Process the output file
                if not data_file.exists():
                    raise FileNotFoundError(f"Expected output file not found: {data_file}")

                parsed_blocks[spin] = self._process_blocks_file(spin, data_file)

        return parsed_blocks

    def _process_blocks_file(self, spin: int, data_file: Path) -> ParsedBlock:
        """Creates an array of polynomial coefficients read from scalar_block file.

        Args:
            spin: The spin of the conformal block
            data_file: The path to the scalar_block file

        """
        # Open the source file
        with open(data_file, 'r') as f:
            source_file = f.read()

        # Split the data into poles and polynomials
        split_data = re.split(r'shiftedPoles -> ', source_file)

        if len(split_data) != 1:
            raise SystemExit('Invalid input file')

        # the only item at the split (the derivs)
        formatted_data = split_data[0]

        # Remove all the extraneous line breaks.
        # Replace is faster than regex subs for simple replacements
        formatted_data = formatted_data.replace('\n', '')

        # Create a 'key' which we will later process to check which zzbDerivs we have
        key = formatted_data

        # Insert a new line for each zzbDeriv and remove comma from line above
        formatted_data = formatted_data.replace(', zzb', '\n zzb')

        # Remove the leading and trailing { }
        formatted_data = re.sub(r'[\{\}]', '', formatted_data)

        # Remove the -> between the expansion and the zzbDeriv
        formatted_data = formatted_data.replace('] ->', '] =')

        # Remove the strange indentation in the octave file
        formatted_data = formatted_data.replace('   ', ' ')

        # Split each z, z\bar deriv onto a new line
        formatted_data = formatted_data.split('\n')

        # Generate a list with the format [ (eq_index, coeff, power), ...]
        indices = [
            (index, *t)
            for index, eq in enumerate(formatted_data)
            for t in self._parse_lines(eq)
        ]

        # Generate an array with size = num of eqs x max power; _=highest coef
        m, _, n = np.array(indices, dtype=np.double).max(axis=0)
        expansion_coeffs = np.zeros((int(m + 1), int(n + 1)), dtype=object)

        # Populate the new blank array with the coefficients
        for row, coeff, col in indices:
            expansion_coeffs[row, col] = coeff

        # Now deal with the key. We need to extract the zzbDeriv[{},{}]
        search_term = r'zzbDeriv\[(\d+),(\d+)\]'
        keys = np.array(re.findall(search_term, key), dtype=int)

        # Generate the poles
        output_poles = self._generate_poles(spin)

        return ParsedBlock(spin, self.dim, expansion_coeffs, keys, output_poles)

    def _generate_poles(self, spin) -> dict:
        """Generate the poles for the conformal block.

        Args:
            spin: The spin to generate the poles for

        Returns:
            A dictionary of poles for the conformal block

        """
        poles_dict = {}

        nu = 0.5 * (self.dim - 2)

        if self.dim % 2 != 0:
            poles_1st_order = []

            for k in range(1, 1 + self.poles):
                poles_1st_order.append(1 - spin - k)
            for k in range(1, 1 + int(np.floor(0.5 * self.poles))):
                poles_1st_order.append(1 + nu - k)
            for k in range(1, 1 + min(self.poles, spin)):
                poles_1st_order.append(1 + (self.dim - 2) + spin - k)

            poles_dict.update({1: poles_1st_order})

            return poles_dict

        poles_1st_order = []
        poles_2nd_order = []

        nu = int(nu)

        for k in range(1, self.poles + 1):
            if 2 * nu + 2 * spin + 2 * k <= self.poles:
                poles_2nd_order.append(1 - spin - k)
            else:
                poles_1st_order.append(1 - spin - k)

        for k in range(1, 1 + min(nu - 1, int(0.5 * self.poles))):
            poles_1st_order.append(1 + nu - k)

        for k in range(1, 1 + min(2 * spin + 2 * nu, self.poles)):
            if not (k > spin and k < spin + 2 * nu):
                poles_1st_order.append(1 + 2 * nu + spin - k)

        poles_dict.update({1: poles_1st_order})
        poles_dict.update({2: poles_2nd_order})

        return poles_dict

    def _parse_lines(self, formatted_data: str) -> list[tuple[int, mp.mpf, int]]:
        """Parse scalar blocks output into an array of polynomial coefficients.

        Args:
            formatted_data: The formatted data to parse

        Returns:
            A list of tuples containing the equation index, coefficient, and power

        """
        # split on =. Extract the LHS
        _, eq = formatted_data.split('=')

        return [self._stripper(item) for item in eq.split('+')]

    def _stripper(self, val: str) -> tuple[mp.mpf, int]:
        """Extract the coefficient and power from a string.

        Args:
            val: The string to extract the coefficient and power from

        Returns:
            A tuple of the coefficient and power

        """
        # Extract the coefficient and power by spliting
        coef, power, *_ = *val.split('*'), '^'

        # Handle unit coefficients
        _, power, *_ = *power.split('^'), '1'

        # Handle the (x^0) part (no power is given explicitly)
        power = 0 if not power else power

        return mp.mpf(coef), int(power)
