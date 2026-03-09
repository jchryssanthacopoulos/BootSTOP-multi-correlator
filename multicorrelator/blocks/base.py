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


class ConvolvedBlocks3DFiles(ConvolvedBlocks3D):
    """Base class to load the 3D convolved blocks from files."""

    F_PLUS_BLOCKS_FILENAME_PATTERN = '../block_lattices/3d/F+_blocks_spin_{spin}_concatenated.npy'
    F_MINUS_BLOCKS_FILENAME_PATTERN = '../block_lattices/3d/F-_blocks_spin_{spin}_concatenated.npy'
    SCALING_DIMENSIONS_FILENAME_PATTERN = '../block_lattices/3d/scaling_dimensions_spin_{spin}.npy'

    def __init__(self, spins: list[int]):
        """Initialise the 3D convolved blocks for the given spins.

        Args:
            spins: The spins of the blocks

        """
        super().__init__(spins)

        self.F_minus_blocks = {}
        self.F_plus_blocks = {}
        self.scaling_dimensions = {}

        self.num_eval_points = None

        for spin in spins:
            self._load_blocks(spin)

    def _load_blocks(self, spin: int):
        """Load the F blocks and scaling dimensions for a given spin."""
        F_plus_blocks_filename = os.path.join(CURRENT_DIR, self.F_PLUS_BLOCKS_FILENAME_PATTERN.format(spin=spin))
        F_minus_blocks_filename = os.path.join(CURRENT_DIR, self.F_MINUS_BLOCKS_FILENAME_PATTERN.format(spin=spin))
        scaling_dimensions_filename = os.path.join(
            CURRENT_DIR, self.SCALING_DIMENSIONS_FILENAME_PATTERN.format(spin=spin)
        )

        # Load the F blocks in "memory-mapped" mode, which doesn't require loading the entire file into memory
        # instead, it only accesses the indices it needs based on the results of the interpolator
        self.F_plus_blocks[spin] = np.load(F_plus_blocks_filename, mmap_mode='r')
        self.F_minus_blocks[spin] = np.load(F_minus_blocks_filename, mmap_mode='r')

        self.scaling_dimensions[spin] = np.load(scaling_dimensions_filename, allow_pickle=True)
        self.scaling_dimensions[spin] = self.scaling_dimensions[spin].item()["scaling_dims"].squeeze()

        if self.num_eval_points is None:
            self.num_eval_points = self.F_plus_blocks[spin].shape[1]

    def __del__(self):
        """Explicitly delete the blocks and scaling dimensions.

        This might be overkill but it may help with memory issues.

        """
        del self.F_plus_blocks
        del self.F_minus_blocks
        del self.scaling_dimensions
