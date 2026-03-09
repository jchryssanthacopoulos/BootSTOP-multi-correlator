"""Class to load 3D convolved blocks and interpolate between them using a KDTree."""

import numpy as np
from scipy.spatial import KDTree

from multicorrelator.blocks.base import BlockType
from multicorrelator.blocks.base import ConvolvedBlocks3DFiles


class ConvolvedBlocks3DKDTree(ConvolvedBlocks3DFiles):
    """Class to interpolate between 3D convolved blocks using a KDTree."""

    def __init__(self, spins: list[int], num_neighbors: int = 16):
        """Initialize the 3D convolved blocks for the given spins.

        Args:
            spins: The spins of the blocks
            num_neighbors: The number of nearest neighbors to use to interpolate F blocks

        """
        super().__init__(spins)

        self.num_neighbors = num_neighbors

        # Build a KDTree from the grid points in the scaling dimensions
        self.kdtree = {spin: KDTree(self.scaling_dimensions[spin]) for spin in self.spins}

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
        if len(spins) != len(deltas):
            raise ValueError("Incompatible lengths for spins and deltas")

        # Ensure all spins present
        for spin in spins:
            if spin not in self.spins:
                raise ValueError(f"Spin {spin} not found")

        result = np.zeros((len(block_types), len(spins), self.num_eval_points))

        for i, block_type in enumerate(block_types):
            # Note: This can probably be optimized
            for j, (spin, delta) in enumerate(zip(spins, deltas)):
                result[i, j] = self._evaluate_at_point(
                    block_type,
                    spin,
                    delta,
                    delta_ij,
                    delta_kl,
                    delta_ave_kj
                )

        return result

    def _evaluate_at_point(
            self,
            block_type: BlockType,
            spin: int,
            delta: float,
            delta_ij: float,
            delta_kl: float,
            delta_ave_kj: float
    ) -> np.ndarray:
        """Interpolate between the 3D convolved blocks.

        Args:
            block_type: The type of the blocks (e.g., '+' or '-')
            delta: The scaling dimension of the exchanged operator
            delta_ij: The difference in the scaling dimension of the first and second operators
            delta_kl: The difference in the scaling dimension of the third and fourth operators
            delta_ave_kj: The average of the scaling dimensions of the second and third operators

        Returns:
            The interpolated 3D convolved block as a function of the z points

        """
        point = [delta_ij, delta_kl, delta_ave_kj, delta]

        # Find the k nearest neighbors to the given point
        distances, indices = self.kdtree[spin].query(
            point,
            k=min(self.num_neighbors, len(self.scaling_dimensions[spin]))
        )

        # If there is only one neighbor, return the block directly
        if not isinstance(indices, np.ndarray):
            if block_type == BlockType.PLUS:
                return self.F_plus_blocks[spin][indices]
            return self.F_minus_blocks[spin][indices]

        weights = self._get_weights(distances)

        if block_type == BlockType.PLUS:
            return np.dot(weights, self.F_plus_blocks[spin][indices])

        return np.dot(weights, self.F_minus_blocks[spin][indices])

    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        """Get weights from distances.

        Args:
            distances: The distances to the k nearest neighbors

        Returns:
            The weights for the k nearest neighbors

        """
        if np.any(distances == 0):
            weights = np.zeros_like(distances)
            weights[distances == 0] = 1.0
            return weights

        weights = 1.0 / distances
        weights /= weights.sum()

        return weights
