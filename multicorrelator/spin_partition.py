"""Spin partition of operators in a CFT."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic import validator


class SpectrumType(str, Enum):
    """Enum class to represent the type of spectrum available."""

    POSITIVE_PARITY = "positive_parity"
    NEGATIVE_PARITY = "negative_parity"


class Operator(BaseModel):
    """Generic operator with spin and scaling dimension."""

    # metadata
    name: str
    spin: int
    min_delta: float
    max_delta: float

    # data
    delta: float = None

    def variables(self) -> List[str]:
        """Get the variables for the operator."""
        return [f"delta_{self.name}"]

    def bounds(self) -> List[Tuple[float, float]]:
        """Get the bounds on the scaling dimension."""
        return [(self.min_delta, self.max_delta)]


class OperatorInSpectrumSpinGroup(BaseModel):
    """Operator in a given spectrum and spin group."""

    name: str
    spectrum: SpectrumType
    spin_group: int
    operator: int


class ExchangedOperator(Operator):
    """Exchanged operator containing scaling dimension and OPE coefficients in several channels."""

    # metadata
    ope_labels: List[str]
    min_ope_coefficients: List[float]
    max_ope_coefficients: List[float]

    # data
    ope_coefficients: List[Optional[float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_ope_coefficients_length(cls, values: ExchangedOperator) -> ExchangedOperator:
        """Check that the ope_coefficients have the same length as ope_labels, or default them to None.

        Args:
            values: The values to validate

        Returns:
            The validated values

        """
        ope_labels = values.ope_labels
        ope_coefficients = values.ope_coefficients
        min_ope_coefficients = values.min_ope_coefficients
        max_ope_coefficients = values.max_ope_coefficients

        if len(values.min_ope_coefficients) != len(ope_labels):
            raise ValueError(
                f"min_ope_coefficients must have the same length as ope_labels, but got {len(min_ope_coefficients)} "
                f"and {len(ope_labels)}"
            )

        if len(values.max_ope_coefficients) != len(ope_labels):
            raise ValueError(
                f"max_ope_coefficients must have the same length as ope_labels, but got {len(max_ope_coefficients)} "
                f"and {len(ope_labels)}"
            )

        if len(ope_coefficients) == 0:
            # Default ope_coefficients to a list of None's of the same length as ope_labels
            values.ope_coefficients = [None] * len(ope_labels)
            return values

        if len(ope_coefficients) != len(ope_labels):
            raise ValueError(
                f"ope_coefficients must have the same length as ope_labels, but got {len(ope_coefficients)} and "
                f"{len(ope_labels)}."
            )

        return values

    def variables(self) -> List[str]:
        """Get the variables for the exchanged operator."""
        variables = super().variables()
        variables.extend([f"ope_coeff_{self.name}_{ope}" for ope in self.ope_labels])
        return variables

    def bounds(self) -> Tuple[float, float]:
        """Get the bounds on the scaling dimension and OPE coefficients."""
        bounds = super().bounds()
        bounds.extend([
            (min_coeff, max_coeff)
            for min_coeff, max_coeff in zip(self.min_ope_coefficients, self.max_ope_coefficients)
        ])
        return bounds

    def get_ope_coefficient(self, ope_label: str) -> float | None:
        """Get the OPE coefficient for the given OPE label.

        Args:
            ope_label: The OPE label to get the coefficient for

        Returns:
            The OPE coefficient for the given OPE label

        """
        if ope_label not in self.ope_labels:
            return

        idx = self.ope_labels.index(ope_label)

        return self.ope_coefficients[idx]

    def lambda_inequality_constraints(self, epsilon: float = 0.001) -> List[Optional[float]]:
        """Return any inequality constraints between OPE coefficients.

        This helps enforce lambda_ssss * lambda_eeee - lambda_ssee ** 2

        Args:
            epsilon: The epsilon value to use for the inequality constraints

        Returns:
            A list of inequality constraints between OPE coefficients, or None if any coefficient is None

        """
        if "ssss" not in self.ope_labels:
            return []

        if "eeee" not in self.ope_labels:
            return []

        if "ssee" not in self.ope_labels:
            return []

        lambda_ssss = self.get_ope_coefficient("ssss")
        if lambda_ssss is None:
            return [None]

        lambda_eeee = self.get_ope_coefficient("eeee")
        if lambda_eeee is None:
            return [None]

        lambda_ssee = self.get_ope_coefficient("ssee")
        if lambda_ssee is None:
            return [None]

        return [np.abs(lambda_ssss * lambda_eeee - lambda_ssee ** 2) - epsilon]

    def print_dataframe(self) -> pd.DataFrame:
        """Return the operator data in a dataframe with columns: operator, spin, delta, lambda for each OPE label."""
        columns = ["operator", "spin", "delta"] + [f"lambda_{label}" for label in self.ope_labels]

        # Create a row with the operator's data
        row = [self.name, self.spin, self.delta] + self.ope_coefficients

        return pd.DataFrame([row], columns=columns)

    def sample(self):
        """Sample the scaling dimension and OPE coefficients from a uniform distribution within the bounds."""
        self.delta = np.random.uniform(self.min_delta, self.max_delta)

        self.ope_coefficients = [
            np.random.uniform(min_coeff, max_coeff)
            for min_coeff, max_coeff in zip(self.min_ope_coefficients, self.max_ope_coefficients)
        ]

    def has_ope_label(self, ope_label: str) -> bool:
        """Check if the operator has the given OPE label.

        Args:
            ope_label: The OPE label to check

        Returns:
            True if the operator has the OPE label, False otherwise

        """
        return ope_label in self.ope_labels


class SpinGroup(BaseModel):
    """Group of operators with the same spin."""

    name: str
    spin: int
    num_operators: int
    ope_labels: List[str]
    min_delta: float
    max_delta: float
    min_ope_coefficients: List[float]
    max_ope_coefficients: List[float]

    operators: List[ExchangedOperator] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_operators(cls, values: SpinGroup) -> SpinGroup:
        """Automatically generate operators after validation.

        Args:
            values: The values to validate

        Returns:
            The validated values

        """
        if not values.operators:
            values.operators = [
                ExchangedOperator(
                    name=f"{values.name}_op_{idx + 1}",
                    spin=values.spin,
                    ope_labels=values.ope_labels,
                    min_delta=values.min_delta,
                    max_delta=values.max_delta,
                    min_ope_coefficients=values.min_ope_coefficients,
                    max_ope_coefficients=values.max_ope_coefficients
                )
                for idx in range(values.num_operators)
            ]
            return values

        if len(values.operators) != values.num_operators:
            raise ValueError(
                f"Number of operators must be equal to num_operators, but got {len(values.operators)} and "
                f"{values.num_operators}."
            )

        return values

    def variables(self) -> List[str]:
        """Get the variables for the spin group."""
        variables = []
        for operator in self.operators:
            variables.extend(operator.variables())
        return variables

    def bounds(self) -> List[Tuple[float, float]]:
        """Get the bounds on the scaling dimension and OPE coefficients for all operators in the spin group."""
        bounds = []
        for operator in self.operators:
            bounds.extend(operator.bounds())
        return bounds

    def get_spins(self) -> List[int]:
        """Get the spins of the operators in the spin group."""
        return [operator.spin for operator in self.operators]

    def get_deltas(self) -> List[float]:
        """Get the scaling dimensions of the operators in the spin group."""
        return [operator.delta for operator in self.operators]

    def get_ope_coefficients(self, ope_label: str) -> np.ndarray:
        """Get the OPE coefficients of the operators in the spin group for the given OPE label.

        Args:
            ope_label: The OPE label to get the coefficients for

        Returns:
            The OPE coefficients of the operators in the spin group for the given OPE label

        """
        return np.array([operator.get_ope_coefficient(ope_label) for operator in self.operators])

    def print_dataframe(self) -> pd.DataFrame:
        """Return the data of all the operators in the spin group in a dataframe."""
        dataframes = [operator.print_dataframe() for operator in self.operators]
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def delta_inequality_constraints(self) -> List[Optional[float]]:
        """Return the inequality constraints between delta values for all operators in the group.

        If delta is None for any operator, the constraint for that pair is set to None.

        Returns:
            A list of inequality constraints between successive operators, or None if either delta is None

        """
        inequality_constraints = []

        for i in range(1, len(self.operators)):
            prev_operator = self.operators[i - 1]
            curr_operator = self.operators[i]

            # Get the delta values for the operators
            delta_prev = prev_operator.delta
            delta_curr = curr_operator.delta

            # If either delta is None, set the constraint to None
            if delta_prev is None or delta_curr is None:
                inequality_constraints.append(None)
            else:
                # Calculate the difference if both deltas are available
                inequality_constraints.append(delta_prev - delta_curr)

        return inequality_constraints

    def lambda_inequality_constraints(self, epsilon: float = 0.001) -> List[Optional[float]]:
        """Return any inequality constraints between OPE coefficients for all operators in the group."""
        lambda_constraints = []

        for operator in self.operators:
            lambda_constraints.extend(operator.lambda_inequality_constraints(epsilon))

        return lambda_constraints

    def sample(self):
        """Sample the scaling dimensions and OPE coefficients for all operators in the spin group."""
        for operator in self.operators:
            operator.sample()


class Spectrum(BaseModel):
    """Spectrum of operators in given spin groups with given OPE labels."""

    name: str
    ope_labels: List[str]
    spin_groups: List[SpinGroup]

    @validator("spin_groups", pre=True)
    def attach_ope_labels(cls, spin_groups: List[dict], values: List[dict]) -> List[dict]:
        """Attach ope_labels to each spin group.

        Args:
            spin_groups: The spin groups to validate
            values: The values to validate

        Returns:
            The validated spin groups

        """
        ope_labels = values.get("ope_labels")
        if not ope_labels:
            raise ValueError("opes are required for spin_groups")

        # Attach ope_labels to each spin group
        return [
            {
                **spin_group,
                "name": f"{values['name']}_{spin_group['name']}",  # Add spectrum name to spin group name
                "ope_labels": ope_labels
            }
            for spin_group in spin_groups
        ]

    def variables(self) -> List[str]:
        """Get the variables for the spectrum."""
        variables = []
        for spin_group in self.spin_groups:
            variables.extend(spin_group.variables())
        return variables

    def bounds(self) -> List[Tuple[float, float]]:
        """Get the bounds on the scaling dimension and OPE coefficients for all operators in the spectrum."""
        bounds = []
        for spin_group in self.spin_groups:
            bounds.extend(spin_group.bounds())
        return bounds

    def get_spins(self) -> List[int]:
        """Get the spins of the operators in the spectrum."""
        spins = []
        for spin_group in self.spin_groups:
            spins.extend(spin_group.get_spins())
        return spins

    def get_deltas(self, spin: Optional[int] = None) -> List[float]:
        """Get the scaling dimensions of the operators in the spectrum.

        Args:
            spin: The spin to get the scaling dimensions for (if None, get all scaling dimensions)

        Returns:
            The scaling dimensions of the operators in the spectrum

        """
        deltas = []
        for spin_group in self.spin_groups:
            if spin is None or spin == spin_group.spin:
                deltas.extend(spin_group.get_deltas())
        return deltas

    def get_ope_coefficients(self, ope_label: str, spin: Optional[int] = None) -> np.ndarray:
        """Get the OPE coefficients of the operators in the spectrum for the given OPE label.

        Args:
            ope_label: The OPE label to get the coefficients for
            spin: The spin to get the coefficients for (if None, get all coefficients)

        Returns:
            The OPE coefficients of the operators in the spectrum for the given OPE label

        """
        ope_coefficients = []

        for spin_group in self.spin_groups:
            if spin is None or spin == spin_group.spin:
                ope_coefficients.append(spin_group.get_ope_coefficients(ope_label))

        return np.hstack(ope_coefficients)

    def print_dataframe(self) -> pd.DataFrame:
        """Return the data of all the operators in all the spin groups in a dataframe."""
        dataframes = [spin_group.print_dataframe() for spin_group in self.spin_groups]
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def delta_inequality_constraints(self) -> List[Optional[float]]:
        """Return the inequality constraints between delta values for all operators in the spin groups.

        Returns:
            A list of inequality constraints between successive operators

        """
        inequality_constraints = []
        spin_groups_by_spin = defaultdict(list)

        # Collect intra-group constraints and group SpinGroups by spin
        for spin_group in self.spin_groups:
            inequality_constraints.extend(spin_group.delta_inequality_constraints())
            spin_groups_by_spin[spin_group.spin].append(spin_group)

        # Process inter-group constraints (between last operator of one group and first of the next)
        for _, groups in spin_groups_by_spin.items():
            for i in range(1, len(groups)):
                prev_group = groups[i - 1]
                curr_group = groups[i]

                # Get last operator from previous group and first operator from current group
                delta_prev = prev_group.operators[-1].delta if prev_group.operators else None
                delta_curr = curr_group.operators[0].delta if curr_group.operators else None

                # If either delta is None, append None; otherwise, compute difference
                if delta_prev is None or delta_curr is None:
                    inequality_constraints.append(None)
                else:
                    inequality_constraints.append(delta_prev - delta_curr)

        return inequality_constraints

    def lambda_inequality_constraints(self, epsilon: float = 0.001) -> List[Optional[float]]:
        """Return any inequality constraints between OPE coefficients for all operators in the spectrum."""
        lambda_constraints = []

        for spin_group in self.spin_groups:
            lambda_constraints.extend(spin_group.lambda_inequality_constraints(epsilon))

        return lambda_constraints

    def sample(self):
        """Sample the scaling dimensions and OPE coefficients for all operators in the spectrum."""
        for spin_group in self.spin_groups:
            spin_group.sample()


class SpinPartition(BaseModel):
    """Spin partition for operators where the external operators are among the exchanged operators in the spectrum."""

    external_delta_ij_constraint: float | None = Field(
        description=(
            "If provided, enforces |delta_i - delta_j| <= external_delta_ij_constraint for all external operators "
            "i, j"
        ),
        default=None
    )
    external_operators: List[OperatorInSpectrumSpinGroup]
    positive_parity_spectrum: Spectrum
    negative_parity_spectrum: Spectrum

    @model_validator(mode="after")
    def validate_external_operators(cls, values: SpinPartition) -> SpinPartition:
        """Validate that the external operators can be located in the given spectra.

        Args:
            values: The values to validate

        Returns:
            The validated values

        """
        for external_operator in values.external_operators:
            name = external_operator.name

            if external_operator.spectrum == SpectrumType.POSITIVE_PARITY:
                spectrum = values.positive_parity_spectrum
            else:
                spectrum = values.negative_parity_spectrum

            if external_operator.spin_group < 1:
                raise ValueError(f"Spin group for {name} must be greater than 0")

            if external_operator.spin_group > len(spectrum.spin_groups):
                raise ValueError(f"Spin group for {name} must be less than the number of spin groups in the spectrum")

            if external_operator.operator < 1:
                raise ValueError(f"Operator for {name} must be greater than 0")

            if external_operator.operator > spectrum.spin_groups[external_operator.spin_group - 1].num_operators:
                raise ValueError(f"Operator for {name} must be less than the number of operators in the spin group")

        return values

    def variables(self) -> List[str]:
        """Get the variables for the spin partition."""
        variables = self.positive_parity_spectrum.variables()
        variables.extend(self.negative_parity_spectrum.variables())
        return variables

    def bounds(self) -> List[Tuple[float, float]]:
        """Get the bounds on the scaling dimension and OPE coefficients for all operators in the spin partition."""
        bounds = self.positive_parity_spectrum.bounds()
        bounds.extend(self.negative_parity_spectrum.bounds())
        return bounds

    def get_external_operator_names(self) -> List[str]:
        """Get the names of the external operators.

        Returns:
            The names of the external operators

        """
        return [operator.name for operator in self.external_operators]

    def get_external_operator_delta(self, name: str) -> float | None:
        """Get the scaling dimension of the external operator with the given name.

        Args:
            name: The name of the external operator

        Returns:
            The scaling dimension of the external operator, or None if not found

        """
        operator = self._locate_external_operator_in_spin_group(name)
        if operator is None:
            return

        return operator.delta

    def get_positive_parity_spectrum_ope_labels(self) -> List[str]:
        """Get the OPE labels of the operators in the positive parity spectrum."""
        return self.positive_parity_spectrum.ope_labels

    def get_positive_parity_spectrum_spins(self) -> List[int]:
        """Get the spins of the operators in the positive parity spectrum."""
        return self.positive_parity_spectrum.get_spins()

    def get_positive_parity_spectrum_deltas(self, spin: Optional[int] = None) -> List[float]:
        """Get the scaling dimensions of the operators in the positive parity spectrum."""
        return self.positive_parity_spectrum.get_deltas(spin)

    def get_unique_spins(self) -> List[int]:
        """Get the unique spins of the operators in the positive parity spectrum."""
        positive_parity_spins = self.get_positive_parity_spectrum_spins()
        negative_parity_spins = self.get_negative_parity_spectrum_spins()
        return list(set(positive_parity_spins + negative_parity_spins))

    def get_positive_parity_spectrum_ope_coefficients(self, ope_label: str, spin: Optional[int] = None) -> np.ndarray:
        """Get the OPE coefficients of the operators in the positive parity spectrum for the given OPE label.

        Args:
            ope_label: The OPE label to get the coefficients for
            spin: The spin to get the coefficients for (if None, get all coefficients)

        Returns:
            The OPE coefficients

        """
        return self.positive_parity_spectrum.get_ope_coefficients(ope_label, spin)

    def get_negative_parity_spectrum_ope_labels(self) -> List[str]:
        """Get the OPE labels of the operators in the negative parity spectrum."""
        return self.negative_parity_spectrum.ope_labels

    def get_negative_parity_spectrum_spins(self) -> List[int]:
        """Get the spins of the operators in the negative parity spectrum."""
        return self.negative_parity_spectrum.get_spins()

    def get_negative_parity_spectrum_deltas(self, spin: Optional[int] = None) -> List[float]:
        """Get the scaling dimensions of the operators in the negative parity spectrum."""
        return self.negative_parity_spectrum.get_deltas(spin)

    def get_negative_parity_spectrum_ope_coefficients(self, ope_label: str, spin: Optional[int] = None) -> np.ndarray:
        """Get the OPE coefficients of the operators in the negative parity spectrum for the given OPE label.

        Args:
            ope_label: The OPE label to get the coefficients for
            spin: The spin to get the coefficients for (if None, get all coefficients)

        Returns:
            The OPE coefficients

        """
        return self.negative_parity_spectrum.get_ope_coefficients(ope_label, spin)

    def from_array(self, x: np.ndarray):
        """Populate data for the spin partition from a unidimensional numpy array.

        Args:
            x: A numpy array containing the variables for the SpinPartition

        """
        # Validate input array size
        total_size = len(self.variables())
        if len(x) != total_size:
            raise ValueError(f"Input array must have size {total_size}, but got {len(x)}")

        current_idx = 0

        # Populate deltas and OPE coefficients for exchanged operators in the spectrum
        for spectrum in [self.positive_parity_spectrum, self.negative_parity_spectrum]:
            for spin_group in spectrum.spin_groups:
                for operator in spin_group.operators:
                    # Set delta
                    operator.delta = x[current_idx]
                    current_idx += 1

                    # Set OPE coefficients
                    num_ope_coeffs = len(operator.ope_labels)
                    operator.ope_coefficients = x[current_idx:current_idx + num_ope_coeffs].tolist()
                    current_idx += num_ope_coeffs

    def print_positive_parity_dataframe(self) -> pd.DataFrame:
        """Return the data of all the operators in the positive parity spectrum in a dataframe."""
        return self.positive_parity_spectrum.print_dataframe()

    def print_negative_parity_dataframe(self) -> pd.DataFrame:
        """Return the data of all the operators in the negative parity spectrum in a dataframe."""
        return self.negative_parity_spectrum.print_dataframe()

    def delta_inequality_constraints(self) -> List[Optional[float]]:
        """Return the inequality constraints between delta values for all operators in the spectra.

        Returns:
            A list of inequality constraints between successive operators

        """
        constraints = self.positive_parity_spectrum.delta_inequality_constraints()
        constraints.extend(self.negative_parity_spectrum.delta_inequality_constraints())

        # add another constraint ensuring |delta_i - delta_j| <= external_delta_ij_constraint
        # for all external operators i, j
        if self.external_delta_ij_constraint is not None:
            external_deltas = [self.get_external_operator_delta(name) for name in self.get_external_operator_names()]

            for i in range(len(external_deltas)):
                for j in range(i + 1, len(external_deltas)):
                    delta_i = external_deltas[i]
                    delta_j = external_deltas[j]

                    if delta_i is None or delta_j is None:
                        constraints.append(None)
                    else:
                        constraints.append(np.abs(delta_i - delta_j) - self.external_delta_ij_constraint)

        return constraints

    def lambda_inequality_constraints(self, epsilon: float = 0.001) -> List[Optional[float]]:
        """Return inequality constraints between OPE coefficients for all operators in the spectra."""
        constraints = self.positive_parity_spectrum.lambda_inequality_constraints(epsilon)
        constraints.extend(self.negative_parity_spectrum.lambda_inequality_constraints(epsilon))

        # add another constraint ensuring lambda_ssss for epsilon = lambda_sese for sigma
        epsilon_operator = self._locate_external_operator_in_spin_group("epsilon")
        if epsilon_operator is None or not epsilon_operator.has_ope_label("ssss"):
            return constraints

        sigma_operator = self._locate_external_operator_in_spin_group("sigma")
        if sigma_operator is None or not sigma_operator.has_ope_label("sese"):
            return constraints

        # at this point, we've located the operators and they have the given OPE labels,
        # but they might not have initialized OPE coefficients
        lambda_ssss = epsilon_operator.get_ope_coefficient("ssss")
        if lambda_ssss is None:
            return constraints + [None]

        lambda_sese = sigma_operator.get_ope_coefficient("sese")
        if lambda_sese is None:
            return constraints + [None]

        return constraints + [np.abs(lambda_ssss - lambda_sese) - epsilon]

    def sample(self):
        """Sample the scaling dimensions and OPE coefficients for all operators in the spectra."""
        self.positive_parity_spectrum.sample()
        self.negative_parity_spectrum.sample()

    def _locate_external_operator_in_spin_group(self, name: str) -> Optional[ExchangedOperator]:
        """Locate the given external operator in its corresponding spin group.

        Args:
            name: The name of the external operator

        Returns:
            The external operator if found, or None

        """
        operator_names = self.get_external_operator_names()

        if name not in operator_names:
            return

        operator_idx = operator_names.index(name)
        external_operator = self.external_operators[operator_idx]

        # locate operator in the spectrum
        if external_operator.spectrum == SpectrumType.POSITIVE_PARITY:
            spectrum = self.positive_parity_spectrum
        else:
            spectrum = self.negative_parity_spectrum

        spin_group = spectrum.spin_groups[external_operator.spin_group - 1]

        return spin_group.operators[external_operator.operator - 1]
