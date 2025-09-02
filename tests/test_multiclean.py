import numpy as np
import pytest


# Import directly from module to avoid potential package __init__ import issues
from multiclean.multiclean import clean_array


def test_identity_when_no_smoothing_no_island_removal():
    # Arrange: small multiclass integer array
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 3, size=(7, 7), dtype=np.int32)

    # Act: no-op settings (1x1 kernel and min_island_size eliminating nothing)
    out = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=0,  # 1x1 structuring element => identity
        min_island_size=1,  # remove components with area < 1 (none)
        connectivity=4,
        max_workers=1,
    )

    # Assert: unchanged and dtype preserved
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


def test_removes_single_pixel_island_and_fills_with_nearest():
    # Arrange: mostly background (0) with a 1-pixel island (class 1)
    arr = np.zeros((7, 7), dtype=np.int32)
    arr[3, 3] = 1

    # Act: ensure island size threshold removes the single pixel
    out = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=0,  # avoid edge effects; focus on island removal
        min_island_size=2,  # single pixel island should be removed
        connectivity=4,
        max_workers=1,
    )

    # Assert: the lone 1 should be replaced by nearest valid (background 0)
    assert out.dtype == arr.dtype
    assert out[3, 3] == 0
    # All other pixels remain 0
    assert np.count_nonzero(out) == 0


def test_fills_nan_holes_with_nearest_valid_class():
    # Arrange: float array with two classes and a NaN hole
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[:, 3:] = 1.0  # right side class 1
    arr[2, 2] = np.nan  # a NaN hole in the middle

    # Act: light processing; should fill NaN by nearest (either 0.0 or 1.0)
    out = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=0,
        min_island_size=1,
        connectivity=4,
        max_workers=1,
    )

    # Assert: no NaNs and values limited to known classes
    assert not np.isnan(out).any()
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_invalid_connectivity_raises():
    arr = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        clean_array(arr, connectivity=6)
