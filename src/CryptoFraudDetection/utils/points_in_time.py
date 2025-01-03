"""Cut data to random points in time for each coin.

Random points in time are picked for each coin and the data is sliced from the
beginning to the selected points. Models can then be trained on these
slices.
"""

from collections.abc import Generator

import torch

from CryptoFraudDetection.utils import logger


def single_coin(
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    n_splits: int,
    logger_: logger.Logger,
) -> list[int]:
    """Generate random cutoff points in time for a single coin.

    Divides the total time range into `n_splits` segments, randomly
    selects points within each segment, and appends the final cutoff
    point (equal to the length of the input tensor).

    Args:
        x: 2D tensor [features, time].
        y: 1D tensor [time], matching x in time length.
        n: The total number of points to generate.
        n_splits: Number of segments to split the time range into.
        logger_: Logger instance for error reporting.

    Returns:
        Sorted list of randomly selected cutoff points in time.

    Raises:
        SystemExit: If there are not enough points to sample in any segment.

    """
    time_steps = x.shape[1]
    points_per_split = n // n_splits
    boundaries = torch.linspace(0, time_steps, steps=n_splits + 1).to(int)

    all_points = []
    for i in range(n_splits):
        low, high = boundaries[i].item(), boundaries[i + 1].item()
        size = points_per_split if i < n_splits - 1 else (points_per_split - 1)

        num_available = max(0, high - low)
        if num_available < size:
            logger_.error(
                f"Not enough points in range {low}-{high} to sample {size}.",
            )

        sampled = torch.randperm(num_available)[:size] + low
        all_points.extend(sampled.tolist())

    # Convert points to cutoff indices by adding 1
    all_points = [p + 1 for p in all_points]

    # Append the final cutoff index (equal to the total length of the input tensor)
    all_points.append(time_steps)
    return sorted(all_points)


def multiple_coins(
    x: torch.Tensor,
    y: torch.Tensor,
    coin_lengths: list[int],
    n: int,
    n_splits: int,
    logger_: logger.Logger,
) -> list[list[int]]:
    """Generate random time points for multiple coins.

    For each coin, this slices the time dimension to `coin_lengths[i]`,
    checks basic validity, and calls `single_coin` to get a list of
    sorted time points.

    Args:
        x: 3D tensor [coins, features, total_time].
        y: 2D tensor [coins, total_time], matching x in time length.
        coin_lengths: List of time lengths for each coin.
        n: Number of points to generate per coin.
        n_splits: Number of segments for splitting the time range.
        logger_: Logger instance for error reporting.

    Returns:
        A list of lists, each sub-list containing the sorted time points
        for the corresponding coin.

    Raises:
        SystemExit:
            - If `n % n_splits != 0`.
            - If `n_splits` or `n` is too large for a coin’s length.
            - If any internal call to `single_coin` fails due to insufficient data.

    """
    if n % n_splits != 0:
        logger_.error("n must be divisible by n_splits.")

    results = []
    for i, coin_length in enumerate(coin_lengths):
        # Ensure we have enough time steps to split and sample from
        if coin_length < n_splits or coin_length < n:
            logger_.error(
                f"Coin {i} has only {coin_length} steps, but n_splits={n_splits} and n={n}.",
            )

        # Slice each coin’s data and call single_coin
        x_coin = x[i, :, :coin_length]
        y_coin = y[i, :coin_length]
        points_for_coin = single_coin(x_coin, y_coin, n, n_splits, logger_)
        results.append(points_for_coin)

    return results


def cut_generator(
    x: torch.Tensor,
    y: torch.Tensor,
    points_in_time: list[list[int]],
    logger_: logger.Logger,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Generate slices of tensors for each coin based on points in time.

    Args:
        x: 3D tensor [coins, features, time].
        y: 2D tensor [coins, time].
        points_in_time: List of lists, where each sublist contains cutoff
            points for slicing a coin's data.
        logger_: Logger instance for error reporting.

    Yields:
        Tuples of tensors `(x_cut, y_cut)`:
        - `x_cut`: 2D tensor [features, n], where `n` is the number of points.
        - `y_cut`: 1D tensor [n], matching the selected points.

    Raises:
        SystemExit: If any cutoff point is invalid (e.g., out of bounds for
        the tensor's dimensions), an error is logged, and processing stops.

    """
    for i, time_indices in enumerate(points_in_time):
        for cutoff in time_indices:
            if cutoff <= 0 or cutoff > x.size(2):  # Validate cutoff range
                logger_.error(
                    f"Invalid cutoff {cutoff} for coin {i} with time dimension {x.size(2)}.",
                )
                continue  # Skip invalid cutoffs and proceed to the next
            # Slice the tensors up to the cutoff point
            x_cut = x[i, :, :cutoff]
            y_cut = y[i, :cutoff]
            yield x_cut, y_cut


def cut(
    x: torch.Tensor,
    y: torch.Tensor,
    points_in_time: list[list[int]],
    logger_: logger.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create tensors for all cutoffs based on points in time.

    Args:
        x: 3D tensor [coins, features, time].
        y: 2D tensor [coins, time].
        points_in_time: List of lists, where each sublist contains cutoff
            points for slicing a coin's data.
        logger_: Logger instance for error reporting.

    Returns:
        A tuple `(x_cuts, y_cuts)`:
        - `x_cuts`: 4D tensor [coins, features, max_time, cutoff points].
        - `y_cuts`: 3D tensor [coins, max_time, cutoff points].

    Raises:
        SystemExit: If any cutoff point is invalid (e.g., out of bounds for
        the tensor's dimensions), an error is logged, and processing stops.

    """
    # Calculate the global maximum cutoff across all coins
    max_cutoff = max(max(indices) for indices in points_in_time)

    x_cuts = []
    y_cuts = []

    for i, time_indices in enumerate(points_in_time):
        x_slices = []
        y_slices = []

        for cutoff in time_indices:
            if cutoff <= 0 or cutoff > x.size(2):  # Validate cutoff range
                logger_.error(
                    f"Invalid cutoff {cutoff} for coin {i} with time dimension {x.size(2)}.",
                )
            # Slice the tensors up to the cutoff point
            x_slice = torch.zeros(
                x.size(1), max_cutoff
            )  # Pre-allocate with zeros
            y_slice = torch.zeros(max_cutoff)  # Pre-allocate with zeros

            x_slice[:, :cutoff] = x[i, :, :cutoff]  # Copy valid data
            y_slice[:cutoff] = y[i, :cutoff]  # Copy valid data

            x_slices.append(x_slice.unsqueeze(-1))  # Add cutoff as new dim
            y_slices.append(y_slice.unsqueeze(-1))  # Add cutoff as new dim

        # Concatenate slices along the new dimension (cutoff points)
        x_cuts.append(torch.cat(x_slices, dim=-1))
        y_cuts.append(torch.cat(y_slices, dim=-1))

    # Stack along the coin dimension
    x_cuts = torch.stack(
        x_cuts, dim=0
    )  # Shape: [coins, features, max_time, cutoff points]
    y_cuts = torch.stack(
        y_cuts, dim=0
    )  # Shape: [coins, max_time, cutoff points]

    return x_cuts, y_cuts
