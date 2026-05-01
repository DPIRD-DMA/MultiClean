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


def test_all_nan_fill_nan_true_is_deterministically_nan():
    # Regression: with no real classes the internal code-to-value lookup
    # has only its sentinel slot. That slot must be deterministically NaN
    # for float input, not whatever np.empty happened to leave there.
    arr = np.full((6, 6), np.nan, dtype=np.float32)
    for _ in range(3):
        out = clean_array(arr, smooth_edge_size=0, min_island_size=1, fill_nan=True)
        assert out.dtype == np.float32
        assert np.isnan(out).all()


def test_dtype_preserved_for_float64_and_large_ints():
    # float64 input must round-trip as float64 (the previous implementation
    # silently downcast through float32 and lost precision).
    arr_f64 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    out_f64 = clean_array(arr_f64, smooth_edge_size=0, min_island_size=1)
    assert out_f64.dtype == np.float64
    assert np.array_equal(out_f64, arr_f64)

    # int32 values past 2**24 cannot survive a float32 round-trip exactly --
    # this asserts they are preserved bit-exactly.
    arr_i32 = np.array([[2_147_483_600, 2_147_483_601]], dtype=np.int32)
    out_i32 = clean_array(arr_i32, smooth_edge_size=0, min_island_size=1)
    assert out_i32.dtype == np.int32
    assert np.array_equal(out_i32, arr_i32)

    # int64 with values past 2**53 likewise cannot round-trip via float64.
    big = (1 << 60) + 7
    arr_i64 = np.array([[big, big + 1], [big + 2, big + 3]], dtype=np.int64)
    out_i64 = clean_array(arr_i64, smooth_edge_size=0, min_island_size=1)
    assert out_i64.dtype == np.int64
    assert np.array_equal(out_i64, arr_i64)


def test_many_classes_exercises_uint16_code_path():
    # K > 254 forces the internal label codes onto the uint16 path. Use
    # >300 unique values to make sure the wider dtype is exercised.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 300, size=(120, 120), dtype=np.int32)
    assert len(np.unique(arr)) > 254

    out = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=1,  # remove components with area < 1 (none)
        connectivity=4,
    )
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


def test_subset_targets_leave_multiple_background_classes_untouched():
    # Build small islands for several classes; only ask to clean class 1.
    # Every other class must come back bit-identical, regardless of size.
    arr = np.zeros((6, 8), dtype=np.int32)
    arr[1, 1] = 1  # tiny class-1 island (target -- should be removed)
    arr[2, 5] = 2  # tiny class-2 island (background -- must be preserved)
    arr[4, 2] = 3  # tiny class-3 island (background -- must be preserved)
    arr[4, 6] = 4  # tiny class-4 island (background -- must be preserved)

    out = clean_array(
        arr,
        class_values=[1],
        smooth_edge_size=0,
        min_island_size=2,  # would remove every single-pixel island if processed
        connectivity=4,
    )

    assert out[1, 1] == 0  # class-1 island removed and filled from background
    # Background-class pixels are untouched even though they are tiny:
    assert out[2, 5] == 2
    assert out[4, 2] == 3
    assert out[4, 6] == 4
