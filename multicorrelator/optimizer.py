"""Class to optimize a given PyGMO problem."""

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pygmo as pg

from multicorrelator.problems.base import Problem


class Optimizer:
    """Optimize a PyGMO problem."""

    def __init__(
            self,
            problem: Problem,
            pop_size: int,
            num_islands: int,
            max_iter: int,
            verbosity: int = 100,
            debug_output_file: str | Path | None = None
    ):
        """Initialise the optimizer.

        Args:
            problem: Base PyGMO problem object
            pop_size: Number of individuals in the population
            num_islands: Number of islands in the archipelago
            max_iter: Maximum number of iterations for IPOpt
            verbosity: Verbosity level for the PyGMO algorithm
            debug_output_file: The name of the debug output file

        """
        self.num_islands = num_islands
        self.max_iter = max_iter

        # Instantiate ipopt
        self.ipopt = pg.ipopt()
        self.ipopt.set_integer_option("max_iter", max_iter)

        if debug_output_file is not None:
            self.ipopt.set_string_option("output_file", debug_output_file)
            self.ipopt.set_integer_option("file_print_level", 5)

        # Define the optimization algorithm. We like IPOpt!
        self.algo = pg.algorithm(self.ipopt)

        if verbosity is not None:
            self.algo.set_verbosity(verbosity)

        # Create a PyGMO problem
        self.shared_problem = pg.problem(problem)

        self.pop = pg.population(self.shared_problem, pop_size)
        self.archi = pg.archipelago(
            n=num_islands,
            t=pg.fully_connected(),
            algo=self.algo,
            prob=self.shared_problem,
            pop_size=pop_size
        )

        self.decision_variables = problem.decision_variables()
        self.fitness_variables = problem.fitness_variables()

    def run(self, evolutions: int, filename: Path, save_frequency: int):
        """Run the optimization.

        Args:
            evolutions: Number of times to evolve the algorithm
            filename: File to save the results in
            save_frequency: Frequency of evolutions to save on

        """
        # Write the header
        with filename.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ['evolution', 'island', 'solution_type'] +
                self.decision_variables +
                self.fitness_variables
            )

        # Evolve the archipelago
        for i in range(evolutions):
            self.archi.evolve(n=1)
            self.archi.wait()

            if i % save_frequency == 0:
                self.save_results_to_file(filename, i + 1)

            if i != evolutions - 1:
                self.reset()

    def get_champ(self) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """Get the champion decision and fitness vectors."""
        return self.archi.get_champions_x(), self.archi.get_champions_f()

    def get_best_champion(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the best champion decision and fitness vectors."""
        champ_x, champ_f = self.get_champ()

        losses = [f[0] for f in champ_f]

        # Get index of the lowest loss
        best_index = int(np.argmin(losses))

        # Return corresponding x and f
        return champ_x[best_index], champ_f[best_index]

    def get_individual_with_lowest_loss(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the individual with the lowest loss (i.e., ignoring constraints, etc.)."""
        best_f = None
        best_x = None

        for isl in self.archi:
            pop = isl.get_population()

            fs = pop.get_f()
            xs = pop.get_x()

            # This is assuming a single-objective problem
            idx = np.argmin(fs[:, 0])
            f = fs[idx]
            x = xs[idx]

            if best_f is None or f[0] < best_f[0]:
                best_f = f
                best_x = x

        return best_x, best_f

    def reset(self):
        """Reset the population to the one with the best fitness.

        On each call of evolve, it seems like the archi is resetting the pop to that at initialisation. As such,
            we re-initialise on each evolution.

        NOTE: The standard behaviour is to provide an arg to the evolve method for the number of evolutions. This
            doesn't provide any interim logging capability, however (so we do it our way).

        """
        buffered_pops = [
            island.get_population() for island in self.archi
        ]

        # We can only initialise the archi with one population, so we pick the one with the max fitness
        best_population = min(
            buffered_pops,
            key=lambda pop: min(pop.champion_f),
        )

        ipopt = pg.ipopt()
        ipopt.set_integer_option("max_iter", self.max_iter)
        algo = pg.algorithm(ipopt)
        algo.set_verbosity(1)

        self.archi = pg.archipelago(
            pop=best_population,
            algo=algo,
            n=self.num_islands,
            t=pg.fully_connected()
        )

    def save_results_to_file(self, filename: Path, evol_num: int):
        """Save the champion and individual with minimum loss to a file.

        Args:
            filename: File to save the results in
            evol_num: Evolution number

        """
        champ_x, champ_f = self.get_champ()
        x_min_global_loss, f_min_global_loss = self.get_individual_with_lowest_loss()

        with filename.open("a", newline="") as file:
            writer = csv.writer(file)

            # Write the champions per island
            for i in range(len(champ_f)):
                writer.writerow(np.concatenate(([evol_num], [i], ['champion'], champ_x[i], champ_f[i])))

            # Write the individual with lowest loss across all islands
            writer.writerow(np.concatenate(([evol_num], [0], ['min_loss'], x_min_global_loss, f_min_global_loss)))
