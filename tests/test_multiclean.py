import numpy as np
import pytest

from multiclean import clean_array


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


def test_retains_nan_holes_in_float_arrays():
    # Arrange: float array with two classes and a NaN hole
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[:, 3:] = 1.0  # right side class 1
    arr[2, 2] = np.nan  # a NaN hole in the middle

    # Act: light processing; NaN should be retained (nodata)
    out = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=0,
        min_island_size=1,
        connectivity=4,
        max_workers=1,
    )

    # Assert: single NaN is retained; other values are valid classes
    assert np.isnan(out[2, 2])
    non_nan = out[~np.isnan(out)]
    assert set(np.unique(non_nan)).issubset({0.0, 1.0})


def test_invalid_connectivity_raises():
    arr = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        clean_array(arr, connectivity=6)


def test_smoothing_removes_single_pixel_when_enabled():
    # Single pixel of class 1 in background 0
    arr = np.zeros((5, 5), dtype=np.int32)
    arr[2, 2] = 1

    # With smoothing on and island removal effectively off (threshold < 1)
    out = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=2,
        min_island_size=1,  # remove components with area < 1 => none
        connectivity=4,
        max_workers=1,
    )

    # The single 1 is smoothed away and filled from nearest background (0)
    assert out.dtype == arr.dtype
    assert out[2, 2] == 0


def test_island_threshold_strictness_preserves_area_equal_to_threshold():
    # 2-pixel island (area = 2) should be preserved when min_island_size = 2
    arr = np.zeros((5, 5), dtype=np.int32)
    arr[2, 2] = 1
    arr[2, 3] = 1

    out = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=0,
        min_island_size=2,  # remove strictly < 2 => keep area 2
        connectivity=4,
        max_workers=1,
    )

    assert out[2, 2] == 1 and out[2, 3] == 1

    # Increasing threshold to 3 should remove the area-2 island
    out2 = clean_array(
        arr,
        class_values=None,
        smooth_edge_size=0,
        min_island_size=3,
        connectivity=4,
        max_workers=1,
    )
    assert out2[2, 2] == 0 and out2[2, 3] == 0


def test_connectivity_affects_island_merging():
    # Two diagonally touching pixels (1,1) and (2,2)
    arr = np.zeros((4, 4), dtype=np.int32)
    arr[1, 1] = 1
    arr[2, 2] = 1

    # With 4-connectivity and threshold 2, each area=1 island is removed
    out4 = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=2,
        connectivity=4,
        max_workers=1,
    )
    assert out4.sum() == 0

    # With 8-connectivity, the two pixels form a single area=2 component => kept
    out8 = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=2,
        connectivity=8,
        max_workers=1,
    )
    assert out8[1, 1] == 1 and out8[2, 2] == 1


def test_class_values_subset_limits_processing():
    # Build islands for classes 1 and 2
    arr = np.zeros((6, 6), dtype=np.int32)
    arr[2, 2] = 1  # tiny island of class 1
    arr[3, 3] = 2  # tiny island of class 2

    # Only process class 1; class 2 should be preserved
    out = clean_array(
        arr,
        class_values=[1],
        smooth_edge_size=0,
        min_island_size=2,  # remove single-pixel islands
        connectivity=4,
        max_workers=2,
    )

    # Class 1 pixel removed, class 2 pixel kept
    assert out[2, 2] == 0
    assert out[3, 3] == 2


def test_empty_class_values_means_identity():
    # If class_values is an empty list, treat everything as background (no-op)
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 3, size=(8, 8), dtype=np.int32)
    out = clean_array(
        arr,
        class_values=[],
        smooth_edge_size=2,
        min_island_size=100,
        connectivity=4,
        max_workers=2,
    )
    assert np.array_equal(out, arr)


def test_invalid_parameters_raise():
    arr = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        clean_array(arr, smooth_edge_size=-1)
    with pytest.raises(ValueError):
        clean_array(arr, min_island_size=-5)


def test_float_dtype_and_nan_retention_preserved():
    arr = np.zeros((3, 3), dtype=np.float32)
    arr[1, 1] = np.nan
    out = clean_array(arr, smooth_edge_size=0, min_island_size=1)
    assert out.dtype == np.float32
    assert np.isnan(out[1, 1])


def test_fill_nan_true_fills_single_nan_with_nearest():
    # Arrange: two classes with a NaN hole; nearest valid around the hole is 0
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[:, 3:] = 1.0  # right side class 1
    arr[2, 2] = np.nan

    # Act: enable NaN filling with no smoothing/island removal side effects
    out = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=1,  # remove components with area < 1 (none)
        connectivity=4,
        max_workers=1,
        fill_nan=True,
    )

    # Assert: NaN is replaced by nearest valid (0 in this layout)
    assert out.dtype == np.float32
    assert not np.isnan(out[2, 2])
    assert out[2, 2] == 0.0


def test_fill_nan_respects_island_removal_order():
    # Arrange: NaN adjacent to a single-pixel island (class 1) in background 0
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[2, 3] = 1.0  # single-pixel island to be removed
    arr[2, 2] = np.nan  # NaN hole next to the island

    # Act: remove islands of area < 2 and fill NaNs afterwards
    out = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=2,  # remove the single-pixel island
        connectivity=4,
        max_workers=1,
        fill_nan=True,
    )

    # Assert: the NaN fills from background (0), not the removed island (1)
    assert out[2, 2] == 0.0
    # The former island pixel is also filled from nearest valid (0)
    assert out[2, 3] == 0.0


def test_fill_nan_true_with_all_nan_returns_all_nan():
    # Arrange: all-NaN array has no valid source to fill from
    arr = np.full((4, 4), np.nan, dtype=np.float32)

    # Act: even with fill_nan=True, nothing to fill from
    out = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=1,
        connectivity=4,
        max_workers=1,
        fill_nan=True,
    )

    # Assert: still all NaN due to absence of any valid pixel
    assert np.isnan(out).all()
