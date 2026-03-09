"""Base class for the 3D Ising CFT."""

import mpmath as mp
import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict

from multicorrelator.blocks.base import BlockType
from multicorrelator.blocks.base import ConvolvedBlocks3D
from multicorrelator.spin_partition import SpinPartition


class CrossingViolation(BaseModel):
    """Model encapsulating crossing violation across several channels."""

    ssss: np.ndarray | None = None
    eeee: np.ndarray | None = None
    sese: np.ndarray | None = None
    ssee_minus: np.ndarray | None = None
    ssee_plus: np.ndarray | None = None
    regularizer: np.float64 | None = None

    # Enable arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def scalar_violation(self) -> np.float64:
        """Return the scalar version of the violation."""
        violation = 0

        if self.ssss is not None:
            violation += np.linalg.norm(self.ssss)

        if self.eeee is not None:
            violation += np.linalg.norm(self.eeee)

        if self.sese is not None:
            violation += np.linalg.norm(self.sese)

        if self.ssee_minus is not None:
            violation += np.linalg.norm(self.ssee_minus)

        if self.ssee_plus is not None:
            violation += np.linalg.norm(self.ssee_plus)

        if self.regularizer is not None:
            violation += self.regularizer

        return violation


class CFT3DIsingBase:
    """Base class for the 3D Ising model CFT."""

    def __init__(
            self,
            blocks: ConvolvedBlocks3D,
            multiply_by_scaling: bool = False,
            multiply_by_spin_scaling: bool = False
    ):
        """Initialise the 3D Ising model CFT.

        Args:
            blocks: The convolved blocks for the 3D Ising model
            multiply_by_scaling: Whether to multiply the F blocks by the scaling factor
            multiply_by_spin_scaling: Whether to multiply the F blocks by a spin-dependent scaling factor

        """
        self.blocks = blocks
        self.multiply_by_scaling = multiply_by_scaling
        self.multiply_by_spin_scaling = multiply_by_spin_scaling

    def get_crossing_violation(self, spin_partition: SpinPartition) -> CrossingViolation:
        """Compute the crossing equations for the 3D Ising model.

        Args:
            spin_partition: The spin partition

        Returns:
            The crossing violations for the various channels

        """
        raise NotImplementedError

    def get_mixed_correlators_crossing_violation(
            self,
            delta_sigma: float,
            delta_epsilon: float,
            positive_parity_spins: list[int],
            negative_parity_spins: list[int],
            positive_parity_deltas: np.ndarray,
            negative_parity_deltas: np.ndarray,
            lambdas_ssss: np.ndarray,
            lambdas_eeee: np.ndarray,
            lambdas_ssee: np.ndarray,
            lambdas_sese: np.ndarray
    ) -> CrossingViolation:
        """Compute the crossing violation for the mixed correlators.

        Args:
            delta_sigma: The scaling dimension of the sigma operator
            delta_epsilon: The scaling dimension of the epsilon operator
            positive_parity_spins: The spins of the positive parity spectrum
            negative_parity_spins: The spins of the negative parity spectrum
            positive_parity_deltas: The scaling dimensions of the positive parity spectrum
            negative_parity_deltas: The scaling dimensions of the negative parity spectrum
            lambdas_ssss: The OPE coefficients for the ssss channel
            lambdas_eeee: The OPE coefficients for the eeee channel
            lambdas_ssee: The OPE coefficients for the ssee channel
            lambdas_sese: The OPE coefficients for the sese channel

        Returns:
            The crossing violations for the various channels

        """
        violation_ssss = self.get_single_spectrum_crossing_violation(
            block_types=[BlockType.MINUS],
            spins=positive_parity_spins,
            deltas_exchanged=positive_parity_deltas,
            deltas_external=[delta_sigma] * 4,
            lambda_square=lambdas_ssss,
            include_identity_contribution=True
        )

        violation_eeee = self.get_single_spectrum_crossing_violation(
            block_types=[BlockType.MINUS],
            spins=positive_parity_spins,
            deltas_exchanged=positive_parity_deltas,
            deltas_external=[delta_epsilon] * 4,
            lambda_square=lambdas_eeee,
            include_identity_contribution=True
        )

        violation_sese = self.get_single_spectrum_crossing_violation(
            block_types=[BlockType.MINUS],
            spins=negative_parity_spins,
            deltas_exchanged=negative_parity_deltas,
            deltas_external=[delta_sigma, delta_epsilon, delta_sigma, delta_epsilon],
            lambda_square=lambdas_sese
        )

        violation_ssee = self._get_crossing_violation_ssee(
            delta_sigma=delta_sigma,
            delta_epsilon=delta_epsilon,
            positive_parity_spins=positive_parity_spins,
            negative_parity_spins=negative_parity_spins,
            positive_parity_deltas=positive_parity_deltas,
            negative_parity_deltas=negative_parity_deltas,
            lambdas_ssee=lambdas_ssee,
            lambdas_sese=lambdas_sese
        )

        return CrossingViolation(
            ssss=violation_ssss[0],
            eeee=violation_eeee[0],
            sese=violation_sese[0],
            ssee_minus=violation_ssee[0],
            ssee_plus=violation_ssee[1]
        )

    def get_single_spectrum_crossing_violation(
            self,
            block_types: list[BlockType],
            spins: list[int],
            deltas_exchanged: np.ndarray,
            deltas_external: list[float],
            lambda_square: np.ndarray,
            include_identity_contribution: bool = False
    ) -> np.ndarray:
        """Compute the crossing violation for the given F type (i.e., '+' or '-') over a single spectrum.

        Args:
            block_types: List of block types (i.e., '+' or '-')
            spin_partition: The spin partition for the given channel
            deltas_exchanged: The exchanged operator dimensions
            deltas_external: The external operator dimensions
            lambda_square: The OPE coefficients squared
            include_identity_contribution: Whether to include the contribution stemming from the identity operator

        Returns:
            The crossing violation for the given operators and block types with shape (num_block_types, num_eval_points)

        """
        if len(spins) != len(deltas_exchanged):
            raise ValueError("Incompatible lengths for spins and deltas_exchanged")

        delta_i, delta_j, delta_k, delta_l = deltas_external

        delta_minus_ij = delta_i - delta_j
        delta_minus_kl = delta_k - delta_l

        delta_plus_jk_ave = (delta_j + delta_k) / 2

        # Make a copy of the spins, deltas, and lambdas so we don't modify the originals
        spins = spins.copy()
        deltas_exchanged = deltas_exchanged.copy()
        lambda_square = lambda_square.copy()

        if include_identity_contribution:
            # Add the identity operator to list of spins, deltas, and lambdas
            # Use 1e-15 for the delta to avoid issues with evaluating blocks at exactly zero
            spins += [0]
            deltas_exchanged = np.append(deltas_exchanged, 1e-15)
            lambda_square = np.append(lambda_square, 1)

        # Generate scaling factors
        if self.multiply_by_scaling:
            for idx, spin in enumerate(spins):
                lambda_square[idx] *= (
                    4 ** deltas_exchanged[idx] * float(mp.rf(1, spin)) / float(mp.rf(0.5, spin))
                )
        if self.multiply_by_spin_scaling:
            for idx, spin in enumerate(spins):
                lambda_square[idx] *= (-1) ** spin

        # This returns an array of shape = (num_block_types, num_spins/deltas, num_eval_points)
        block_vals = self.blocks.evaluate(
            block_types,
            spins,
            deltas_exchanged,
            delta_minus_ij,
            delta_minus_kl,
            delta_plus_jk_ave
        )

        # Reshape lambda_square for broadcasting
        lambda_square_expanded = lambda_square[None, :, None]

        # Sum over operators (i.e., spins/deltas)
        violation = np.sum(block_vals * lambda_square_expanded, axis=1)

        return violation

    def _get_crossing_violation_ssee(
            self,
            delta_sigma: float,
            delta_epsilon: float,
            positive_parity_spins: list[int],
            negative_parity_spins: list[int],
            positive_parity_deltas: np.ndarray,
            negative_parity_deltas: np.ndarray,
            lambdas_ssee: np.ndarray,
            lambdas_sese: np.ndarray
    ) -> np.ndarray:
        """Get the crossing violation error for the ssee channel.

        This computes the last equation of (3.10) in 1406.4858

        Args:
            delta_sigma: Scaling dimension of the sigma operator
            delta_epsilon: Scaling dimension of the epsilon operator
            positive_parity_spins: Spins of the positive parity spectrum
            negative_parity_spins: Spins of the negative parity spectrum
            positive_parity_deltas: Scaling dimensions of the positive parity spectrum
            negative_parity_deltas: Scaling dimensions of the negative parity spectrum
            lambdas_ssee: OPE coefficients for the ssee channel
            lambdas_sese: OPE coefficients for the sese channel

        Returns:
            Array of crossing violation errors of shape (2, num_eval_points)

        """
        # Get first term
        term1 = self.get_single_spectrum_crossing_violation(
            block_types=[BlockType.MINUS, BlockType.PLUS],
            spins=positive_parity_spins,
            deltas_exchanged=positive_parity_deltas,
            deltas_external=[delta_sigma, delta_sigma, delta_epsilon, delta_epsilon],  # ssee
            lambda_square=lambdas_ssee,
            include_identity_contribution=True
        )

        # Get second term
        lambdas_sese_times_spin = lambdas_sese * [(-1) ** spin for spin in negative_parity_spins]

        term2 = self.get_single_spectrum_crossing_violation(
            block_types=[BlockType.MINUS, BlockType.PLUS],
            spins=negative_parity_spins,
            deltas_exchanged=negative_parity_deltas,
            deltas_external=[delta_epsilon, delta_sigma, delta_sigma, delta_epsilon],  # esse
            lambda_square=lambdas_sese_times_spin
        )

        # If we're evaluating F_+, we need to multiply the second term by -1
        term2[1] *= -1

        return term1 + term2
