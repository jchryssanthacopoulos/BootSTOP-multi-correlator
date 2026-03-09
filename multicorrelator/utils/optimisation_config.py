from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from multicorrelator.blocks.base import ConvolvedBlocks3D
from multicorrelator.blocks.derivatives.recursion_derivatives import ConvolvedDerivativeBlocksRecursiveGoBlocks
from multicorrelator.blocks.derivatives.recursion_derivatives import ConvolvedDerivativeBlocksRecursivePython
from multicorrelator.blocks.kdtree import ConvolvedBlocks3DKDTree
from multicorrelator.blocks.polynomial import ConvolvedBlocks3DPolynomial
from multicorrelator.blocks.recursion import ConvolvedBlocksRecursiveGoBlocks
from multicorrelator.blocks.recursion import ConvolvedBlocksRecursivePython


class LambdaModelEnum(str, Enum):

    FOUR_VALUE_LAMBDAS = "four_value_lambdas"
    THREE_VALUE_LAMBDAS = "three_value_lambdas"


class BlockInterpolationTypeEnum(str, Enum):

    MULTILINEAR = "multilinear"
    POLYNOMIAL = "polynomial"
    NEURAL_OPERATOR = "neural_operator"
    MULTILINEAR_TENSOR = "multilinear_tensor"
    RECURSIVE = "recursive"
    GOBLOCKS = "goblocks"
    RECURSIVE_DERIVS = "recursive_derivs"
    GOBLOCKS_DERIVS = "goblocks_derivs"


class FBlockInterpolation(BaseModel):

    name: BlockInterpolationTypeEnum = Field(..., description="Type of interpolation for F blocks")
    params: dict = Field(default_factory=dict, description="Parameters for the F block interpolation method")

    def interpolation_obj(self, spins: list[int]) -> ConvolvedBlocks3D:
        """Get the interpolation object for the given spins.

        Args:
            spins: The spins for which to get the interpolation object

        Returns:
            The convolved blocks interpolation object for the given spins

        """
        interpolation_type_to_method = {
            BlockInterpolationTypeEnum.MULTILINEAR: ConvolvedBlocks3DKDTree,
            BlockInterpolationTypeEnum.POLYNOMIAL: ConvolvedBlocks3DPolynomial,
            BlockInterpolationTypeEnum.RECURSIVE: ConvolvedBlocksRecursivePython,
            BlockInterpolationTypeEnum.GOBLOCKS: ConvolvedBlocksRecursiveGoBlocks,
            BlockInterpolationTypeEnum.RECURSIVE_DERIVS: ConvolvedDerivativeBlocksRecursivePython,
            BlockInterpolationTypeEnum.GOBLOCKS_DERIVS: ConvolvedDerivativeBlocksRecursiveGoBlocks
        }

        if self.name not in interpolation_type_to_method:
            raise ValueError(f"Unsupported interpolation type: {self.name}")

        return interpolation_type_to_method[self.name](spins, **self.params)


class OptimisationConfig(BaseModel):

    pop_size: int = Field(..., description="Population size for PyGMO islands")
    outdir: Path = Field(..., description="Output directory for results")
    use_wandb: bool = False
    num_islands: int = 1
    max_iter: int = 10
    evolutions: int = 1
    verbosity: int = 1
    save_frequency: int = 1
    debug_output_file: Optional[Path] = None
    lambda_model: LambdaModelEnum = LambdaModelEnum.FOUR_VALUE_LAMBDAS
    F_block_interpolation: FBlockInterpolation
    scaling: bool = False
    spin_scaling: bool = False
