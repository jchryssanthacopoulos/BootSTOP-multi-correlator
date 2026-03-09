"""
Run the multi-scalar mixed 3D Ising model using PyGMO, with optimiser and spin config files.
"""

import argparse
import os
from pathlib import Path
import time
import warnings

import wandb

from multicorrelator.cfts.cft_3d_ising import (
    CFT3DIsingFourValueLambdaModel,
    CFT3DIsingThreeValueLambdaModel,
)
from multicorrelator.optimizer import Optimizer
from multicorrelator.problems.scalarized import ThreeDIsingProblemScalarized
from multicorrelator.spin_partition import SpinPartition
from multicorrelator.utils.config import load_config_file
from multicorrelator.utils.optimisation_config import LambdaModelEnum
from multicorrelator.utils.optimisation_config import OptimisationConfig


def main(optimiser_config_path: Path, spin_config_path: Path):
    """Run optimization from optimiser and spin configuration files.

    Args:
        optimiser_config_path: Path to the optimiser configuration file
        spin_config_path: Path to the spin partition configuration file

    """
    optimiser_config_dict = load_config_file(optimiser_config_path)
    spin_partition_dict = load_config_file(spin_config_path)

    assert optimiser_config_dict is not None and spin_partition_dict is not None

    use_wandb = optimiser_config_dict.get("use_wandb", False)

    if use_wandb:
        # Start wandb first so wandb.config is populated (for sweeps)
        # The initial config is just the one set in the sweep code.
        wandb.init(
            entity="bootstoppers",
            project="BootSTOP-multi-objective-3d",
            dir=str(optimiser_config_dict["outdir"]),
        )
        # Merge sweep params into config dict (flat override)
        optimiser_config_dict.update(wandb.config)

    # Create validated config
    optimiser_config = OptimisationConfig(**optimiser_config_dict)

    if use_wandb:
        # Change the run optimiser log directory to be unique to the WandB run
        optimiser_config.outdir = optimiser_config.outdir / f"{wandb.run.name}-{wandb.run.id}"

        # Push final config back to wandb (ensures merged view is logged)
        wandb.config.update(
            {
                "optimiser_config": optimiser_config.__dict__,
                "spin_partition_config": spin_partition_dict,
            },
            allow_val_change=True,
        )

    if not optimiser_config.outdir.exists():
        optimiser_config.outdir.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            f"Output directory '{optimiser_config.outdir}' did not exist and was created.",
            UserWarning,
        )
    elif not optimiser_config.outdir.is_dir():
        raise NotADirectoryError(
            f"Path '{optimiser_config.outdir}' exists but is not a directory."
        )

    filename = (
        f"scalar_ising_3d_pop_size_{optimiser_config.pop_size}_evolutions_{optimiser_config.evolutions}_islands_"
        f"{optimiser_config.num_islands}_max_iter_{optimiser_config.max_iter}_{time.strftime('%Y%m%d-%H%M%S')}_"
        f"{os.getpid()}.csv"
    )
    filename = optimiser_config.outdir / filename

    spin_partition = SpinPartition(**spin_partition_dict)
    blocks = optimiser_config.F_block_interpolation.interpolation_obj(spin_partition.get_unique_spins())

    if optimiser_config.lambda_model == LambdaModelEnum.FOUR_VALUE_LAMBDAS:
        cft = CFT3DIsingFourValueLambdaModel(spin_partition, blocks, optimiser_config.scaling)
    else:
        cft = CFT3DIsingThreeValueLambdaModel(spin_partition, blocks, optimiser_config.scaling)

    problem = ThreeDIsingProblemScalarized(cft, spin_partition_dict)

    optimizer = Optimizer(
        problem,
        optimiser_config.pop_size,
        optimiser_config.num_islands,
        optimiser_config.max_iter,
        verbosity=optimiser_config.verbosity,
        debug_output_file=str(optimiser_config.debug_output_file),
    )

    start_timestamp = time.time()
    optimizer.run(optimiser_config.evolutions, filename, optimiser_config.save_frequency)
    running_time_seconds = time.time() - start_timestamp

    _print_results(optimizer, spin_partition_dict, filename, running_time_seconds)

    if use_wandb:
        _log_results_to_wandb(optimizer, spin_partition, running_time_seconds)
        wandb.finish()


def _log_results_to_wandb(optimizer: Optimizer, spin_partition: SpinPartition, running_time_seconds: float):
    """Logs the results of the optimization run to WandB.

    Args:
        optimizer: The Optimizer instance containing the results
        spin_partition: The SpinPartition instance containing the spin partition configuration
        running_time_seconds: The total runtime of the optimization in seconds

    """
    champ_x, champ_f = optimizer.get_best_champion()
    spin_partition.from_array(champ_x)

    pos_spec_champ = spin_partition.print_positive_parity_dataframe().to_dict(orient="records")
    neg_spec_champ = spin_partition.print_negative_parity_dataframe().to_dict(orient="records")

    x_min_global_loss, f_min_global_loss = optimizer.get_individual_with_lowest_loss()
    spin_partition.from_array(x_min_global_loss)

    pos_spec_loss = spin_partition.print_positive_parity_dataframe().to_dict(orient="records")
    neg_spec_loss = spin_partition.print_negative_parity_dataframe().to_dict(orient="records")

    wandb.log(
        {
            "runtime_seconds": running_time_seconds,
            "champion_fitness": champ_f.tolist(),
            "champion_positive_spectrum": pos_spec_champ,
            "champion_negative_spectrum": neg_spec_champ,
            "lowest_loss_fitness": f_min_global_loss.tolist(),
            "lowest_loss_positive_spectrum": pos_spec_loss,
            "lowest_loss_negative_spectrum": neg_spec_loss,
        }
    )


def _print_results(optimizer: Optimizer, spin_partition_dict: dict, filename: Path, running_time_seconds: float):
    """Prints the results of the optimization run.

    Args:
        optimizer: The Optimizer instance containing the results
        spin_partition_dict: The dictionary containing the spin partition configuration
        filename: The path to the output file where results are saved
        running_time_seconds: The total runtime of the optimization in seconds

    """
    spin_partition = SpinPartition(**spin_partition_dict)

    champ_x, champ_f = optimizer.get_best_champion()
    spin_partition.from_array(champ_x)

    print("\nRESULTS FOR CHAMPION")
    print("====================")
    print("- Fitness: " + " ".join(f"{x:.4f}" for x in champ_f) + "\n")
    print("- Positve Z2 spectrum:")
    print(spin_partition.print_positive_parity_dataframe().to_string(index=False) + "\n")
    print("- Negative Z2 spectrum:")
    print(spin_partition.print_negative_parity_dataframe().to_string(index=False) + "\n")

    x_min_global_loss, f_min_global_loss = optimizer.get_individual_with_lowest_loss()
    spin_partition.from_array(x_min_global_loss)

    print("RESULTS FOR INDIVIDUAL WITH LOWEST LOSS")
    print("=======================================")
    print("- Fitness: " + " ".join(f"{x:.4f}" for x in f_min_global_loss) + "\n")
    print("- Positve Z2 spectrum:")
    print(spin_partition.print_positive_parity_dataframe().to_string(index=False) + "\n")
    print("- Negative Z2 spectrum:")
    print(spin_partition.print_negative_parity_dataframe().to_string(index=False) + "\n")

    print(f"RESULTS FILE: {filename}")
    print(f"RUNTIME: {running_time_seconds:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Ising model optimization using optimiser and spin configs.")
    parser.add_argument(
        "--optimiser-config",
        "-o",
        type=Path,
        required=True,
        help="Path to the optimiser config YAML/JSON file.",
    )
    parser.add_argument(
        "--spin-config",
        "-s",
        type=Path,
        required=True,
        help="Path to the spin partition config YAML/JSON file.",
    )

    # Use known_args here as WandB gives all config options as args (which we don't use)
    args, unknown = parser.parse_known_args()
    main(args.optimiser_config, args.spin_config)
