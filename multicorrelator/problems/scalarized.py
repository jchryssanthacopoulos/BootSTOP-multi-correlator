"""Scalarized multi-correlator objective for various models."""

import copy
from typing import Tuple

import numpy as np

from multicorrelator.cfts.cft_3d_ising import CFT3DIsingBase
from multicorrelator.problems.base import Problem
from multicorrelator.spin_partition import SpinPartition


class ThreeDIsingProblemScalarized(Problem):
    """Scalarized 3D Ising problem using the SpinPartition structure."""

    def __init__(self, cft: CFT3DIsingBase, spin_partition_dict: dict):
        """Solve the 3D Ising problem with multiple correlators simultaneously.

        Args:
            cft: The CFT object containing the crossing violation equations
            spin_partition_dict: The dictionary containing the spin partition data

        """
        self.cft = cft
        self.spin_partition_dict = spin_partition_dict

        spin_partition = SpinPartition(**spin_partition_dict)

        self.variables = spin_partition.variables()
        self.bounds = spin_partition.bounds()
        self.output_variables = ['crossing_error']

        # Handle delta inequality constraints
        num_delta_inequality_constraints = len(spin_partition.delta_inequality_constraints())
        self.output_variables += [f'delta_inequality_{i}' for i in range(num_delta_inequality_constraints)]

        # Handle lambda inequality constraints
        num_lambda_inequality_constraints = len(spin_partition.lambda_inequality_constraints())
        self.output_variables += [f'lambda_inequality_{i}' for i in range(num_lambda_inequality_constraints)]

        self.num_inequality_constraints = num_delta_inequality_constraints + num_lambda_inequality_constraints

    def fitness(self, x: np.ndarray) -> list:
        """Get the fitness for the given vector.

        Args:
            x: State space vector to get the fitness for

        Returns:
            Array of fitness variables

        """
        # To be extra safe, we deepcopy the spin partition dictionary
        spin_partition_dict_copy = copy.deepcopy(self.spin_partition_dict)

        # Create the spin partition object. To be safe x2, we deepcopy the spin partition object
        spin_partition = SpinPartition(**spin_partition_dict_copy)
        spin_partition = spin_partition.copy(deep=True)

        # Update the spin partition with the new values. To be safe x3, we deepcopy the x array
        x_copy = x.copy()
        spin_partition.from_array(x_copy)

        # Get the crossing violation
        crossing_violation = self.cft.get_crossing_violation(spin_partition)
        fitness_vector = [crossing_violation.scalar_violation()]

        # Add delta inequality constraints
        fitness_vector += spin_partition.delta_inequality_constraints()

        # Add lambda inequality constraints
        fitness_vector += spin_partition.lambda_inequality_constraints()

        return fitness_vector

    def decision_variables(self) -> list[str]:
        """Get variables describing the decision vector."""
        return self.variables

    def fitness_variables(self) -> list[str]:
        """Get variables describing the fitness output."""
        return self.output_variables

    def get_nic(self) -> int:
        """Get number of inequality constraints."""
        return self.num_inequality_constraints

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds."""
        lower_bounds = np.array([bound[0] for bound in self.bounds])
        upper_bounds = np.array([bound[1] for bound in self.bounds])

        return lower_bounds, upper_bounds
