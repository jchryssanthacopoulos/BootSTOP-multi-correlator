from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# ------------------ CONFIGURATION ------------------
STOCHASTIC: bool = False  # Use random sampling or grid sampling
NUM_POINTS: int = 100  # Number of points to sample if STOCHASTIC
LAMBDA_THRESHOLD: float = 0.6  # Only keep points where lambda(z) ≤ threshold
SEED: int = 42  # RNG seed for reproducibility

# Grid sampling parameters
GRID_RE_RANGE: tuple[float, float] = (0.5, 0.6)
GRID_IM_RANGE: tuple[float, float] = (0.2, 0.7)
GRID_RESOLUTION: int = 10

# Random sampling parameters
RAND_RE_RANGE: tuple[float, float] = (0.5, 1.5)
RAND_IM_RANGE: tuple[float, float] = (0.0, 1.5)
RAND_TOTAL: int = 100_000

# Output paths
OUTPUT_DIR: Path = Path("output")
TXT_FILENAME: str = "sampled_points_first_quadrant.txt"
NPY_FILENAME: str = "sampled_points_first_quadrant.npy"

# Plotting
PLOT = True

# Save to file
SAVE_TO_FILE = True
# ----------------------------------------------------


def rho(z: np.ndarray) -> np.ndarray:
    """
    Compute the rho transformation for conformal cross-ratio z.
    (see Effective Bootstrap (1606.02771v2 eq. 2.6)

    Args:
        z: Complex points for transformation.

    Returns:
        Transformed complex points.
    """
    return z / (1 + np.sqrt(1 - z)) ** 2


def lambda_measure(z: np.ndarray) -> np.ndarray:
    """
    Compute the lambda(z) measure = |rho(z)| + |rho(1 - z)|.
    (see Effective Bootstrap (1606.02771v2 eq. 3.11)

    Args:
        z: Complex points.

    Returns:
        Array of lambda values.
    """
    return np.abs(rho(z)) + np.abs(rho(1 - z))


def generate_candidates_stochastic() -> np.ndarray:
    """
    Generate candidate points by random sampling.

    Returns:
        A filtered complex array of candidate points satisfying the lambda
        threshold.
    """
    np.random.seed(SEED)
    re = np.random.uniform(*RAND_RE_RANGE, RAND_TOTAL)
    im = np.random.uniform(*RAND_IM_RANGE, RAND_TOTAL)
    z = re + 1j * im
    return z[lambda_measure(z) <= LAMBDA_THRESHOLD]


def generate_candidates_grid() -> np.ndarray:
    """
    Generate candidate points on a regular grid.

    Returns:
        A filtered complex array of candidate points satisfying the lambda
        threshold.
    """
    re = np.linspace(*GRID_RE_RANGE, GRID_RESOLUTION)
    im = np.linspace(*GRID_IM_RANGE, GRID_RESOLUTION)
    Re, Im = np.meshgrid(re, im)
    z = Re + 1j * Im
    return z[lambda_measure(z) <= LAMBDA_THRESHOLD]


def save_points(
    z: np.ndarray,
    output_dir: Path,
    txt_filename: str,
    npy_filename: Optional[str] = None,
) -> None:
    """
    Save sampled points to text and (optionally) NumPy binary files.

    Args:
        z: Complex array of sampled points.
        output_dir: Directory to save files in.
        txt_filename: Name of the .txt file.
        npy_filename: Name of the .npy file (optional).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / txt_filename
    np.savetxt(
        txt_path,
        np.column_stack((np.real(z), np.imag(z))),
        header="Re(z)\tIm(z)",
        fmt="%.8f",
    )
    print(f"Saved sampled points to {txt_path.resolve()}")

    if npy_filename:
        npy_path = output_dir / npy_filename
        np.save(npy_path, z)
        print(f"Saved NumPy array to {npy_path.resolve()}")


def plot_points(z_sampled: np.ndarray, lambda_threshold: float) -> None:
    """
    Plot sampled points in the complex z-plane.

    Args:
        z_sampled: Complex numpy array of points.
        lambda_threshold: Threshold used for lambda(z), shown in the title.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(np.real(z_sampled), np.imag(z_sampled), ".", label="Sampled Points")
    # plt.axhline(y=0, color="gray", linestyle="--", label="Branch cuts")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-1, 1)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title(f"Sampled Points in z-plane (lambda ≤ {lambda_threshold})")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.show()


def main() -> None:
    """
    Main entry point: generate, save, and plot sampled points.
    """
    if STOCHASTIC:
        z_candidates = generate_candidates_stochastic()
        if len(z_candidates) < NUM_POINTS:
            raise ValueError("Not enough points satisfy the lambda condition.")
        z_sampled = np.random.choice(z_candidates, size=NUM_POINTS, replace=False)
    else:
        z_sampled = generate_candidates_grid()

    print(f"Number of sampled points: {len(z_sampled)}")

    if SAVE_TO_FILE:
        save_points(z_sampled, OUTPUT_DIR, TXT_FILENAME, NPY_FILENAME)

    if PLOT:
        plot_points(z_sampled, LAMBDA_THRESHOLD)


if __name__ == "__main__":
    main()
