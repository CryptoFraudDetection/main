"""Test points in time generation and cutting tensor up to them."""

import torch

from CryptoFraudDetection.utils import logger, points_in_time
from CryptoFraudDetection.utils.enums import LoggerMode

logger_ = logger.Logger(
    name=__name__,
    level=LoggerMode.DEBUG,
    log_dir="../logs",
)

torch.manual_seed(42)


def test_single_coin():
    """Test points in time for a single coin."""
    n_features = 3
    time_steps = 1000
    n = 100
    n_splits = 10
    x = torch.rand(n_features, time_steps)
    y = torch.rand(time_steps)
    points_in_time_: list[int] = points_in_time.single_coin(
        x,
        y,
        n,
        n_splits,
        logger_,
    )
    assert isinstance(points_in_time_, list)
    # Total number of points must match n
    assert len(points_in_time_) == n
    # Last cut off point must match total steps
    assert points_in_time_[-1] == time_steps
    # Points in first split (inclusive)
    assert len([i for i in points_in_time_ if i <= 100]) == 10
    # Points must be sorted
    assert points_in_time_ == sorted(points_in_time_)


def test_multiple_coins():
    """Test points in time for multiple coins."""
    n_coins = 2
    coin_lengths = [1000, 500]
    n_features = 3
    time_steps = 1000
    n = 100
    n_splits = 10
    x = torch.rand(n_coins, n_features, time_steps)
    y = torch.rand(n_coins, time_steps)
    points_in_time_: list[list[int]] = points_in_time.multiple_coins(
        x,
        y,
        coin_lengths,
        n,
        n_splits,
        logger_,
    )
    assert isinstance(points_in_time_, list)
    assert isinstance(points_in_time_[0], list)
    assert isinstance(points_in_time_[1], list)
    # Points must be sorted
    assert points_in_time_[0] == sorted(points_in_time_[0])
    assert points_in_time_[1] == sorted(points_in_time_[1])
    # Last point matches length
    assert points_in_time_[0][-1] == coin_lengths[0]
    assert points_in_time_[1][-1] == coin_lengths[1]
    # Total points
    assert len(points_in_time_[0]) == n
    assert len(points_in_time_[1]) == n
    # Points in first split
    assert len([i for i in points_in_time_[0] if i < 100]) == 10
    assert len([i for i in points_in_time_[1] if i < 50]) == 10


def test_cut():
    """Test cutting a tensor."""
    n_coins = 2
    coin_lengths = [1000, 500]
    n_features = 3
    time_steps = 1000
    n = 100
    n_splits = 10

    # Generate data and points in time
    x = torch.rand(n_coins, n_features, time_steps)
    y = torch.rand(n_coins, time_steps)
    points_in_time_ = points_in_time.multiple_coins(
        x,
        y,
        coin_lengths,
        n,
        n_splits,
        logger_,
    )

    tensors = []
    # Iterate over the generator
    for x_cut, y_cut in points_in_time.cut_generator(
        x,
        y,
        points_in_time_,
        logger_,
    ):
        assert isinstance(x_cut, torch.Tensor)
        assert isinstance(y_cut, torch.Tensor)
        tensors.append((x_cut, y_cut))
    # There should be n different tensors for each coin
    assert len(tensors) == n_coins * n


def test_cut():
    """Test tensor cutting with points in time."""
    n_coins = 2
    coin_lengths = [1000, 500]
    n_features = 3
    time_steps = 1000
    n = 100
    n_splits = 10

    # Generate data and points in time
    x = torch.rand(n_coins, n_features, time_steps)
    y = torch.rand(n_coins, time_steps)
    points_in_time_ = points_in_time.multiple_coins(
        x,
        y,
        coin_lengths,
        n,
        n_splits,
        logger_,
    )

    # Get the cut tensors
    x_cuts, y_cuts = points_in_time.cut(
        x,
        y,
        points_in_time_,
        logger_,
    )

    # Validate global max_cutoff
    global_max_cutoff = max(max(indices) for indices in points_in_time_)
    assert x_cuts.shape == (
        n_coins,
        n_features,
        global_max_cutoff,
        len(points_in_time_[0]),
    )
    assert y_cuts.shape == (
        n_coins,
        global_max_cutoff,
        len(points_in_time_[0]),
    )

    # Validate slices
    for i, time_indices in enumerate(points_in_time_):
        for j, cutoff in enumerate(time_indices):
            assert torch.equal(
                x_cuts[i, :, :cutoff, j], x[i, :, :cutoff],
            ), f"x_cuts slice at coin {i}, cutoff {cutoff} does not match"
            assert torch.equal(
                y_cuts[i, :cutoff, j], y[i, :cutoff],
            ), f"y_cuts slice at coin {i}, cutoff {cutoff} does not match"


if __name__ == "__main__":
    test_single_coin()
    test_multiple_coins()
    test_cut()
