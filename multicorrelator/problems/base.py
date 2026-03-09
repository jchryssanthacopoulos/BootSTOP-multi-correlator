"""Base problem class that expose certain methods to get fitness and metadata."""

import numpy as np
import pygmo as pg


class Problem:
    """Base problem class."""

    def fitness(self, x: np.ndarray) -> list:
        """Get the fitness for the decision vector.

        Args:
            x: Decision vector to get the fitness for

        Returns:
            Array of fitness variables

        """
        raise NotImplementedError()

    def decision_variables(self) -> list[str]:
        """Get variables describing the decision vector."""
        raise NotImplementedError()

    def fitness_variables(self) -> list[str]:
        """Get variables describing the fitness output."""
        raise NotImplementedError()

    def get_nic(self) -> int:
        """Get number of inequality constraints."""
        return 0

    def get_nec(self) -> int:
        """Get number of equality constraints."""
        return 0

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds."""
        raise NotImplementedError()

    def gradient(self, x: np.ndarray) -> float:
        """Calculate gradient."""
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
