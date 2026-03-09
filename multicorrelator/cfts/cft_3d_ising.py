"""CFT data for the 3D Ising model."""

from multicorrelator.blocks.base import ConvolvedBlocks3D
from multicorrelator.cfts.cft_base import CFT3DIsingBase
from multicorrelator.cfts.cft_base import CrossingViolation
from multicorrelator.spin_partition import SpinPartition


class CFT3DIsingFourValueLambdaModel(CFT3DIsingBase):
    """3D Ising model CFT using the four-value lambda model."""

    def __init__(
            self,
            spin_partition: SpinPartition,
            blocks: ConvolvedBlocks3D,
            multiply_by_scaling: bool = False,
            multiply_by_spin_scaling: bool = False
    ):
        """Initialise the CFT for the 3D Ising model using the given data.

        Args:
            spin_partition: The spin partition
            blocks: The convolved blocks for the 3D Ising model
            multiply_by_scaling: Whether to multiply the F blocks by a scaling factor
            multiply_by_spin_scaling: Whether to multiply the F blocks by a spin-dependent scaling factor

        """
        super().__init__(blocks, multiply_by_scaling, multiply_by_spin_scaling)

        external_operators = spin_partition.get_external_operator_names()

        if "sigma" not in external_operators:
            raise ValueError("Missing external operator 'sigma'")

        if "epsilon" not in external_operators:
            raise ValueError("Missing external operator 'epsilon'")

        ope_labels = spin_partition.get_positive_parity_spectrum_ope_labels()

        if "ssss" not in ope_labels:
            raise ValueError("Missing operator product expansion 'ssss'")

        if "eeee" not in ope_labels:
            raise ValueError("Missing operator product expansion 'eeee'")

        if "ssee" not in ope_labels:
            raise ValueError("Missing operator product expansion 'ssee'")

        if "sese" not in spin_partition.get_negative_parity_spectrum_ope_labels():
            raise ValueError("Missing operator product expansion 'sese'")

    def get_crossing_violation(self, spin_partition: SpinPartition) -> CrossingViolation:
        """Compute the crossing equations for the 3D Ising model.

        Args:
            spin_partition: The spin partition

        Returns:
            The crossing violations for the various channels

        """
        # Retrieve all necessary data ONCE
        delta_sigma = spin_partition.get_external_operator_delta("sigma")
        delta_epsilon = spin_partition.get_external_operator_delta("epsilon")

        positive_parity_spins = spin_partition.get_positive_parity_spectrum_spins()
        negative_parity_spins = spin_partition.get_negative_parity_spectrum_spins()

        positive_parity_deltas = spin_partition.get_positive_parity_spectrum_deltas()
        negative_parity_deltas = spin_partition.get_negative_parity_spectrum_deltas()

        lambdas_ssss = spin_partition.get_positive_parity_spectrum_ope_coefficients("ssss")
        lambdas_eeee = spin_partition.get_positive_parity_spectrum_ope_coefficients("eeee")
        lambdas_ssee = spin_partition.get_positive_parity_spectrum_ope_coefficients("ssee")
        lambdas_sese = spin_partition.get_negative_parity_spectrum_ope_coefficients("sese")

        return self.get_mixed_correlators_crossing_violation(
            delta_sigma=delta_sigma,
            delta_epsilon=delta_epsilon,
            positive_parity_spins=positive_parity_spins,
            negative_parity_spins=negative_parity_spins,
            positive_parity_deltas=positive_parity_deltas,
            negative_parity_deltas=negative_parity_deltas,
            lambdas_ssss=lambdas_ssss,
            lambdas_eeee=lambdas_eeee,
            lambdas_ssee=lambdas_ssee,
            lambdas_sese=lambdas_sese
        )


class CFT3DIsingThreeValueLambdaModel(CFT3DIsingBase):
    """3D Ising model CFT using the SpinPartitionFull structure and assuming that the OPEs are not squared."""

    def __init__(
            self,
            spin_partition: SpinPartition,
            blocks: ConvolvedBlocks3D,
            multiply_by_scaling: bool = False,
            multiply_by_spin_scaling: bool = False
    ):
        """Initialise the CFT for the 3D Ising model using the given data.

        Args:
            spin_partition: The spin partition
            blocks: The convolved blocks for the 3D Ising model
            multiply_by_scaling: Whether to multiply the F blocks by a scaling factor
            multiply_by_spin_scaling: Whether to multiply the F blocks by a spin-dependent scaling factor

        """
        super().__init__(blocks, multiply_by_scaling, multiply_by_spin_scaling)

        external_operators = spin_partition.get_external_operator_names()

        if "sigma" not in external_operators:
            raise ValueError("Missing external operator 'sigma'")

        if "epsilon" not in external_operators:
            raise ValueError("Missing external operator 'epsilon'")

        ope_labels = spin_partition.get_positive_parity_spectrum_ope_labels()

        if "ssss" not in ope_labels:
            raise ValueError("Missing operator product expansion 'ssss'")

        if "eeee" not in ope_labels:
            raise ValueError("Missing operator product expansion 'eeee'")

        if "sese" not in spin_partition.get_negative_parity_spectrum_ope_labels():
            raise ValueError("Missing operator product expansion 'sese'")

    def get_crossing_violation(self, spin_partition: SpinPartition) -> CrossingViolation:
        """Compute the crossing equations for the 3D Ising model.

        Args:
            spin_partition: The spin partition

        Returns:
            The crossing violations for the various channels

        """
        # Retrieve all necessary data ONCE
        delta_sigma = spin_partition.get_external_operator_delta("sigma")
        delta_epsilon = spin_partition.get_external_operator_delta("epsilon")

        positive_parity_spins = spin_partition.get_positive_parity_spectrum_spins()
        negative_parity_spins = spin_partition.get_negative_parity_spectrum_spins()

        positive_parity_deltas = spin_partition.get_positive_parity_spectrum_deltas()
        negative_parity_deltas = spin_partition.get_negative_parity_spectrum_deltas()

        lambdas_ssss = spin_partition.get_positive_parity_spectrum_ope_coefficients("ssss")
        lambdas_eeee = spin_partition.get_positive_parity_spectrum_ope_coefficients("eeee")
        lambdas_sese = spin_partition.get_negative_parity_spectrum_ope_coefficients("sese")

        # Square the OPE coefficients
        lambdas_ssee = lambdas_ssss * lambdas_eeee
        lambdas_ssss **= 2
        lambdas_eeee **= 2
        lambdas_sese **= 2

        return self.get_mixed_correlators_crossing_violation(
            delta_sigma=delta_sigma,
            delta_epsilon=delta_epsilon,
            positive_parity_spins=positive_parity_spins,
            negative_parity_spins=negative_parity_spins,
            positive_parity_deltas=positive_parity_deltas,
            negative_parity_deltas=negative_parity_deltas,
            lambdas_ssss=lambdas_ssss,
            lambdas_eeee=lambdas_eeee,
            lambdas_ssee=lambdas_ssee,
            lambdas_sese=lambdas_sese
        )
