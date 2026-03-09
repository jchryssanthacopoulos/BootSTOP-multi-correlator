"""Class to load 3D convolved blocks and interpolate between them using polynomial regression."""

import numpy as np
from numpy.linalg import lstsq
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures

from multicorrelator.blocks.base import BlockType
from multicorrelator.blocks.base import ConvolvedBlocks3DFiles


class ConvolvedBlocks3DPolynomial(ConvolvedBlocks3DFiles):
    """Class to interpolate between 3D convolved blocks using polynomial regression."""

    def __init__(self, spins: list[int], num_neighbors: int = 16, degree: int = 2, regularization: float = 0.0):
        """Initialize the 3D convolved blocks for a given spin.

        TODO: Add LASSO regression as an option

        Args:
            spins: The spins of the blocks
            num_neighbors: The number of nearest neighbors to use to interpolate F blocks
            degree: The degree of the polynomial to fit to the nearest neighbors
            regularization: The regularization parameter for the polynomial fit (0 for no regularization)

        """
        super().__init__(spins)

        self.num_neighbors = num_neighbors
        self.regularization = regularization

        # Build a KDTree from the grid points in the scaling dimensions
        self.kdtree = {spin: KDTree(self.scaling_dimensions[spin]) for spin in self.spins}

        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

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
            spin: The spin of the blocks
            delta: The scaling dimension of the exchanged operator
            delta_ij: The difference in the scaling dimension of the first and second operators
            delta_kl: The difference in the scaling dimension of the third and fourth operators
            delta_ave_kj: The average of the scaling dimensions of the second and third operators

        Returns:
            The interpolated 3D convolved block as a function of the z points

        """
        point = np.array([delta_ij, delta_kl, delta_ave_kj, delta])

        _, indices = self.kdtree[spin].query(point, k=min(self.num_neighbors, len(self.scaling_dimensions[spin])))

        neighbor_x = self.scaling_dimensions[spin][indices]
        neighbor_y = (
            self.F_plus_blocks[spin][indices] if block_type == BlockType.PLUS else self.F_minus_blocks[spin][indices]
        )

        # Add a singleton dimension at the beginning if necessary
        if neighbor_x.ndim == 1:
            neighbor_x = neighbor_x.reshape(1, -1)
        if neighbor_y.ndim == 1:
            neighbor_y = neighbor_y.reshape(1, -1)

        # Design matrix for polynomial features
        X_poly = self.poly.fit_transform(neighbor_x)

        # Fit polynomial via least squares for each output dim
        if self.regularization > 0:
            # Ridge-style regression: (X^T X + λI) β = X^T y
            XtX = X_poly.T @ X_poly + self.regularization * np.eye(X_poly.shape[1])
            Xty = X_poly.T @ neighbor_y
            coeffs = np.linalg.solve(XtX, Xty)
        else:
            coeffs, _, _, _ = lstsq(X_poly, neighbor_y, rcond=None)

        # Evaluate at query point
        x0_poly = self.poly.transform([point])
        y_interp = x0_poly @ coeffs

        return y_interp.ravel()
