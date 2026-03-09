"""Class to load 3D convolved blocks and interpolate between them."""

from enum import Enum
import os

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class BlockType(Enum):

    PLUS = "+"
    MINUS = "-"


class ConvolvedBlocks3D:
    """Base class to evaluate the 3D convolved blocks."""

    def __init__(self, spins: list[int]):
        """Initialise the 3D convolved blocks for the given spins.

        Args:
            spins: The spins of the blocks

        """
        self.spins = spins

    def evaluate(
            self,
            block_types: list[BlockType],
            spins: list[int],
            deltas: list[float],
            delta_ij: float,
            delta_kl: float,
            delta_ave_kj: float
    ) -> np.ndarray:
        """Evaluate the 3D convolved blocks at the given points.

        Args:
            block_types: List of blocks types (e.g., '+' or '-')
            spins: List of spins for the blocks
            deltas: List of scaling dimensions of the exchanged operator
            delta_ij: The difference in the scaling dimension of the first and second operators
            delta_kl: The difference in the scaling dimension of the third and fourth operators
            delta_ave_kj: The average of the scaling dimensions of the second and third operators

        Returns:
            The interpolated 3D convolved block with shape (num_block_types, num_spins/deltas, num_eval_points)

        """
        raise NotImplementedError
