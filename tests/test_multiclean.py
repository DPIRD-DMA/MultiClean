from pathlib import Path

import numpy as np
import pytest

from multiclean import clean_array
from multiclean.utils import create_circle_kernel, smooth_edges_to_codes

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
LANDSAT_INPUT = TEST_DATA_DIR / "Landsat cloud and cloud shadow.tif"
LANDSAT_EXPECTED = TEST_DATA_DIR / "landsat_expected.npz"


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


def _centred_blob(size: int = 24, half_width: int = 6) -> np.ndarray:
    """Solid square of class 1 centred in a background-0 array.

    The array is symmetric under a 180-degree rotation, which the smoothing
    kernels are too, so any correct opening must preserve that symmetry.
    """
    arr = np.zeros((size, size), dtype=np.uint8)
    centre = (size - 1) / 2
    lo, hi = int(centre - half_width + 1), int(centre + half_width + 1)
    arr[lo:hi, lo:hi] = 1
    return arr


@pytest.mark.parametrize("smooth_edge_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_smoothing_does_not_translate_blobs(smooth_edge_size):
    # Regression: cv2's MORPH_OPEN applies one anchor to both the erosion and
    # the dilation, which is only correct when the structuring element's centre
    # of symmetry lands on that anchor. For even kernel sizes it does not, and
    # the whole blob came back shifted one pixel down and right -- most obvious
    # at smooth_edge_size=2, where the opening should otherwise be a no-op on a
    # solid blob.
    arr = _centred_blob()
    out = clean_array(
        arr,
        class_values=[0, 1],
        smooth_edge_size=smooth_edge_size,
        min_island_size=0,
        connectivity=4,
        max_workers=1,
    )

    # The blob is far larger than any kernel here, so it must survive with its
    # bounding box unmoved (a circular opening rounds the corners off a square
    # but leaves the edge midpoints, so the extent is unchanged).
    ys, xs = np.where(out == 1)
    in_ys, in_xs = np.where(arr == 1)
    assert (ys.min(), ys.max()) == (in_ys.min(), in_ys.max())
    assert (xs.min(), xs.max()) == (in_xs.min(), in_xs.max())


@pytest.mark.parametrize("smooth_edge_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_smoothing_preserves_symmetry(smooth_edge_size):
    # Sharper form of the same regression: the kernels are all symmetric under
    # a 180-degree rotation, so a symmetric input must smooth to a symmetric
    # result. A one-pixel translation in either axis breaks this even when the
    # bounding box survives.
    #
    # Asserted against the smoothing stage rather than clean_array because the
    # nearest-neighbour fill that follows it breaks ties between equidistant
    # source pixels arbitrarily, which is asymmetric by design.
    arr = _centred_blob()
    codes, _ = smooth_edges_to_codes(
        arr,
        smooth_edge_size=smooth_edge_size,
        target_class_values=[1],
        background_class_values=[0],
        all_class_values=[0, 1],
        max_workers=1,
    )
    assert np.array_equal(codes, codes[::-1, ::-1])


@pytest.mark.parametrize("smooth_edge_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_smoothing_never_grows_a_class(smooth_edge_size):
    # An opening is anti-extensive: it can only remove pixels from a class,
    # never add them. The even-size anchor bug broke this -- the shifted blob
    # covered pixels that were background in the input. Checked against a
    # ragged multiclass array so it is not just the solid-blob case.
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[5:20, 5:20] = 1
    arr[22:36, 22:36] = 1
    arr[10:14, 24:30] = 2
    arr[30, 3] = 1  # thin spur that smoothing is expected to erase

    out = clean_array(
        arr,
        class_values=[1, 2],
        smooth_edge_size=smooth_edge_size,
        min_island_size=0,
        connectivity=4,
        max_workers=1,
    )

    # Every pixel that came out as a smoothed class must have held that class
    # on input; fill may reassign a pixel to another class, but smoothing must
    # not extend one beyond its original footprint.
    for cv in (1, 2):
        grew = (out == cv) & (arr != cv)
        # Fill can only draw from surviving neighbours, so any growth here is
        # the smoothing step inventing coverage.
        assert not grew.any(), f"class {cv} grew by {int(grew.sum())} pixels"


def _reference_opening(mask: np.ndarray, kernel: np.ndarray, anchor: int) -> np.ndarray:
    """Morphological opening straight from the definition, for small arrays.

    ``A opened by B`` is the union of every translate of ``B`` that fits
    entirely inside ``A``. Written out as an explicit slide so it shares no
    machinery with the cv2 erode/dilate pair under test.

    Two details model cv2's finite-image behaviour, so this is exact at the
    borders and not just in the interior:

    * Pixels outside the image read as foreground, matching the border value
      cv2 uses for erosion -- content is not eaten away merely because the
      image ends.
    * Translate positions are restricted to the image domain, because that is
      the domain cv2's intermediate erosion is defined on.

    The second point is why ``anchor`` has to be named: it fixes where a
    translate sits relative to the position that must stay in-domain. Away
    from the border the choice cannot matter (a translated structuring element
    is still the same set of pixels), and the interior test below asserts
    exactly that. Within one kernel width of the border it does matter, so the
    border test passes cv2's own anchor.
    """
    ks = kernel.shape[0]
    height, width = mask.shape
    offsets = [(i - anchor, j - anchor) for i, j in np.argwhere(kernel > 0)]

    # Foreground-padded view, wide enough that no translate can run off it.
    extended = np.ones((height + 2 * ks, width + 2 * ks), dtype=np.uint8)
    extended[ks : ks + height, ks : ks + width] = mask

    out = np.zeros_like(mask)
    for row in range(height):
        for col in range(width):
            if all(extended[row + ks + dr, col + ks + dc] for dr, dc in offsets):
                for dr, dc in offsets:
                    r, c = row + dr, col + dc
                    if 0 <= r < height and 0 <= c < width:
                        out[r, c] = 1
    return out


def _smoothed_mask(arr: np.ndarray, smooth_edge_size: int) -> np.ndarray:
    """Run the smoothing stage alone and return class 1's mask."""
    codes, code_to_value = smooth_edges_to_codes(
        arr,
        smooth_edge_size=smooth_edge_size,
        target_class_values=[1],
        background_class_values=[],
        all_class_values=[0, 1],
        max_workers=1,
    )
    return (code_to_value[codes] == 1).astype(np.uint8)


@pytest.mark.parametrize("smooth_edge_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_smoothing_matches_reference_opening(smooth_edge_size):
    # The other smoothing tests assert properties (no shift, symmetric,
    # anti-extensive). Properties alone cannot distinguish a correct opening
    # from a differently-wrong one -- silently rounding even kernel sizes up to
    # odd, for instance, satisfies every one of them while changing how much
    # smoothing the caller actually asked for. This pins the exact result
    # against an independent implementation of the definition instead.
    rng = np.random.default_rng(7)
    kernel = create_circle_kernel(smooth_edge_size)

    for _ in range(10):
        arr = np.zeros((34, 34), dtype=np.uint8)
        arr[9:25, 9:25] = rng.random((16, 16)) > 0.35  # wide zero margin

        expected = _reference_opening(arr, kernel, anchor=smooth_edge_size // 2)
        assert np.array_equal(_smoothed_mask(arr, smooth_edge_size), expected)

        # Away from the border the reference must not depend on how the
        # structuring element is anchored. This is the no-shift property
        # restated at the definition level, and it keeps the assertion above
        # from silently inheriting the implementation's anchor convention.
        for alt_anchor in (0, smooth_edge_size - 1):
            assert np.array_equal(
                _reference_opening(arr, kernel, anchor=alt_anchor), expected
            )


@pytest.mark.parametrize("smooth_edge_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_smoothing_matches_reference_opening_at_borders(smooth_edge_size):
    # Same equivalence, but with content running flush to all four edges, where
    # cv2's border convention decides the answer. Untested until now: the
    # interior test deliberately keeps a margin so border handling cannot
    # affect it, which left the edges of every real raster unpinned.
    rng = np.random.default_rng(11)
    kernel = create_circle_kernel(smooth_edge_size)

    for _ in range(6):
        arr = (rng.random((20, 20)) > 0.35).astype(np.uint8)
        expected = _reference_opening(arr, kernel, anchor=smooth_edge_size // 2)
        assert np.array_equal(_smoothed_mask(arr, smooth_edge_size), expected)


@pytest.mark.parametrize("smooth_edge_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_smoothing_does_not_erode_content_at_the_image_edge(smooth_edge_size):
    # The human-readable half of the border contract: a band running the full
    # width of the image, flush against the top, left and right edges, must
    # come through an opening completely intact. If the erosion treated
    # out-of-image pixels as background it would chew a kernel-wide bite out of
    # all three edges, which on tiled processing would show up as seams along
    # every tile boundary.
    #
    # A band rather than a square block: its only boundary is the straight
    # edge along the bottom, which an opening preserves exactly. A block would
    # additionally have an interior corner, and a circular kernel rounds those
    # off by design -- correct behaviour that has nothing to do with borders.
    arr = np.zeros((24, 24), dtype=np.uint8)
    arr[:12, :] = 1

    assert np.array_equal(_smoothed_mask(arr, smooth_edge_size), arr)


@pytest.mark.parametrize("smooth_edge_size", [0, 2, 3])
def test_output_is_independent_of_max_workers(smooth_edge_size):
    # Smoothing runs one class per thread and the opening now dilates back into
    # the erosion's own buffer. That is safe because the buffer is created per
    # call, but hoisting it out to avoid a per-class allocation would be an easy
    # and plausible "optimisation" -- and would corrupt results only under
    # concurrency, which every other test pins to a single worker count.
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 6, size=(64, 64), dtype=np.uint8)
    arr[10:30, 10:30] = 2  # solid regions so smoothing has real work to do
    arr[35:60, 35:60] = 4

    kwargs = dict(
        class_values=[1, 2, 3, 4, 5],
        smooth_edge_size=smooth_edge_size,
        min_island_size=6,
        connectivity=4,
    )
    baseline = clean_array(arr, max_workers=1, **kwargs)
    for workers in (2, 4, 8):
        assert np.array_equal(clean_array(arr, max_workers=workers, **kwargs), baseline)


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


def test_class_values_absent_from_array_are_ignored():
    # A requested class that does not occur in the array is a no-op, not an
    # error, and must behave the same with and without edge smoothing.
    arr = np.full((64, 64), 254, dtype=np.uint8)
    arr[10:50, 10:50] = 1  # only class 1 (and background 254) present

    for smooth_edge_size in (0, 2, 3):
        kwargs = dict(
            smooth_edge_size=smooth_edge_size,
            min_island_size=10,
            connectivity=4,
            max_workers=2,
        )
        with_absent = clean_array(arr, class_values=[0, 1], **kwargs)
        without_absent = clean_array(arr, class_values=[1], **kwargs)

        assert with_absent.dtype == arr.dtype
        # Naming an absent class changes nothing about the result.
        assert np.array_equal(with_absent, without_absent)


# --- Parameter sweeps -------------------------------------------------------
#
# The absent-class KeyError was invisible to single-point tests because it only
# fired in one corner of the parameter space (smoothing enabled). These sweeps
# assert invariants across the grid instead, so a regression that survives in
# any single branch still fails the suite.

SWEEP_DTYPES = [np.uint8, np.int16, np.int32, np.float32]
# Even kernel sizes are included deliberately: the anchor bug that translated
# smoothed blobs by one pixel only ever fired for even ``smooth_edge_size``,
# and the sweep's previous [0, 1, 3] never touched that half of the space.
SWEEP_SMOOTH = [0, 1, 2, 3, 4]
SWEEP_ISLAND = [0, 25]


def _sweep_array(dtype) -> np.ndarray:
    """Multiclass array holding classes {1, 2, 3, 7}, with a 1-pixel island."""
    arr = np.full((32, 32), 7, dtype=dtype)
    arr[4:20, 4:20] = 1
    arr[24:28, 24:28] = 2
    arr[0, 0] = 3  # single-pixel island, removable by min_island_size
    return arr


@pytest.mark.parametrize("dtype", SWEEP_DTYPES)
@pytest.mark.parametrize("smooth_edge_size", SWEEP_SMOOTH)
@pytest.mark.parametrize("min_island_size", SWEEP_ISLAND)
@pytest.mark.parametrize("absent", [[0], [0, 99], [99], [-1]])
def test_absent_class_values_are_inert(
    dtype, smooth_edge_size, min_island_size, absent
):
    # Naming classes that do not occur in the array must never raise and must
    # never change the result, at any point in the parameter grid. This is the
    # property tiled processing relies on: one fixed class list reused across
    # tiles whose contents vary.
    arr = _sweep_array(dtype)
    kwargs = dict(
        smooth_edge_size=smooth_edge_size,
        min_island_size=min_island_size,
        connectivity=4,
        max_workers=2,
    )

    baseline = clean_array(arr, class_values=[1, 2], **kwargs)
    padded = clean_array(arr, class_values=[1, 2] + absent, **kwargs)

    assert padded.dtype == baseline.dtype
    assert np.array_equal(padded, baseline)


@pytest.mark.parametrize("dtype", SWEEP_DTYPES)
@pytest.mark.parametrize("smooth_edge_size", SWEEP_SMOOTH)
def test_only_absent_class_values_is_identity(dtype, smooth_edge_size):
    # If every requested class is absent there is nothing to clean, so the
    # array must come back untouched rather than raising or being blanked.
    arr = _sweep_array(dtype)
    out = clean_array(
        arr,
        class_values=[99, 100],
        smooth_edge_size=smooth_edge_size,
        min_island_size=1000,
        connectivity=4,
        max_workers=2,
    )
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("dtype", SWEEP_DTYPES)
@pytest.mark.parametrize("smooth_edge_size", SWEEP_SMOOTH)
@pytest.mark.parametrize("min_island_size", SWEEP_ISLAND)
@pytest.mark.parametrize("class_values", [None, 1, [1], [1, 2], [], [0], [1, 99], [99]])
def test_output_invariants_hold_across_grid(
    dtype, smooth_edge_size, min_island_size, class_values
):
    # Structural guarantees that hold for every accepted input shape: no
    # exception, shape and dtype preserved, and no class value invented that
    # was not already in the input.
    arr = _sweep_array(dtype)
    out = clean_array(
        arr,
        class_values=class_values,
        smooth_edge_size=smooth_edge_size,
        min_island_size=min_island_size,
        connectivity=4,
        max_workers=2,
    )

    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    assert set(np.unique(out).tolist()) <= set(np.unique(arr).tolist())


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


@pytest.mark.skipif(
    not LANDSAT_INPUT.exists() or not LANDSAT_EXPECTED.exists(),
    reason="Landsat fixture missing -- run from a checkout that includes "
    "tests/data/Landsat cloud and cloud shadow.tif and "
    "tests/data/landsat_expected.npz",
)
def test_landsat_cloud_shadow_regression():
    # Pixel-exact regression test on the cleaned cloud/cloud-shadow mask
    # from the ``notebooks/Cloud example.ipynb`` example. The expected
    # output ``tests/data/landsat_expected.npz`` is generated by running
    # the current implementation with the same (smooth_edge_size=3,
    # min_island_size=5) settings the notebook uses; regenerate with::
    #
    #   python -c "import numpy as np, rasterio; from multiclean import \
    #     clean_array; arr = rasterio.open('tests/data/Landsat cloud and \
    #     cloud shadow.tif').read(1); np.savez_compressed('tests/data/\
    #     landsat_expected.npz', cleaned=clean_array(arr, smooth_edge_size=3, \
    #     min_island_size=5))"
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(LANDSAT_INPUT) as ds:
        arr = ds.read(1)

    out = clean_array(array=arr, smooth_edge_size=3, min_island_size=5)

    expected = np.load(LANDSAT_EXPECTED)["cleaned"]
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert np.array_equal(out, expected)


def test_fill_works_on_tiny_image_with_one_invalid_pixel():
    # cv2 has minimum-size requirements for some operations; make sure the
    # fill path (cv2.distanceTransformWithLabels under the hood) handles a
    # 4x4 input with a single invalid pixel without crashing or returning
    # garbage.
    arr = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 0, 4],  # the lone 0 is the only thing to fill
            [3, 3, 4, 4],
        ],
        dtype=np.int32,
    )
    out = clean_array(
        arr,
        class_values=[0],  # only target class 0 (the single invalid pixel)
        smooth_edge_size=0,
        min_island_size=2,  # remove the area-1 class-0 island
        connectivity=4,
    )

    # The (2, 2) pixel should be filled from a nearest valid neighbour.
    # Its 4-neighbours are 2 (above), 4 (right, below), 3 (left) -- any of
    # those is a valid choice; we only assert it's no longer 0 and that
    # all other pixels are unchanged.
    assert out.dtype == arr.dtype
    assert out[2, 2] != 0
    assert out[2, 2] in (2, 3, 4)
    untouched = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, out[2, 2], 4],
            [3, 3, 4, 4],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(out, untouched)


def test_single_valid_pixel_propagates_to_all_invalid():
    # Extreme sparsity: one valid source pixel surrounded by an entire grid
    # of invalid (NaN) pixels with fill_nan=True. Every invalid pixel must
    # be assigned back to that one source -- this also exercises the
    # ``int(valid_labels.max()) + 1`` lookup-size calculation in
    # fill_invalids when there is exactly one entry.
    arr = np.full((6, 6), np.nan, dtype=np.float32)
    arr[3, 3] = 7.0  # single valid pixel

    out = clean_array(
        arr,
        smooth_edge_size=0,
        min_island_size=1,
        connectivity=4,
        fill_nan=True,
    )

    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    # Every pixel must inherit the only valid source value.
    assert (out == 7.0).all()
