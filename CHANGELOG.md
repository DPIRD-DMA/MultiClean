# Changelog

All notable changes to MultiClean are documented here.

## [Unreleased]

### Changed
- **Performance.** `clean_array` is substantially faster on multi-class inputs.
  On a 15669×18633 / 147-class land-use raster, end-to-end runtime dropped
  from ~85 s to ~40 s. On the 8011×7901 / 4-class Landsat cloud-and-shadow
  example, runtime dropped from ~2.5 s to ~1.1 s. Wins came from:
  - Replacing the float32 smoothed-labels buffer with a `uint8`/`uint16`
    class-code array (selected automatically based on class count). The
    per-class equality scan is 2-4× cheaper in memory bandwidth.
  - OR-reducing per-class small-island masks in flight rather than holding
    a `Dict[class -> ndarray]` of K masks simultaneously and then reducing
    them in one shot. This avoids a `(K, H, W)` stacked-mask intermediate
    in the OR-reduce step. Peak RSS on the 4-class 8011×7901 Landsat
    example dropped from ~5.4 GB to ~2.8 GB; on the 147-class
    15669×18633 raster the saving was smaller in absolute terms because
    OS memory compression made the per-class bool masks cheap to hold.
  - Filling invalid pixels in place rather than allocating a copy.
  - Replacing `scipy.ndimage.distance_transform_edt` with
    `cv2.distanceTransformWithLabels` for the nearest-valid fill (~3.4×
    faster on the fill stage). Both algorithms produce mathematically
    equivalent output (the same minimum L2 distance); they differ only in
    which equidistant source pixel wins a tie.
- **dtype preservation.** The output now strictly matches the input dtype.
  Previously the pipeline routed everything through float32 internally,
  which silently downcast `float64` inputs and rounded `int32` values
  larger than 2²⁴ (and `int64` values larger than 2⁵³).

### Fixed
- All-NaN float input with `fill_nan=True` now deterministically returns
  an all-NaN array. The previous code relied on whatever value
  `np.empty` happened to leave in the sentinel slot.
- Large integer class values (`int32` > 2²⁴, `int64` > 2⁵³) are now
  preserved bit-exactly, instead of being silently rounded by the
  internal float32 round-trip.

### Removed
- Dropped the `scipy` runtime dependency. `cv2` (already a runtime
  dependency) now handles the distance-transform fill.

## [0.2.0] - 2025-09-03

### Added
- `fill_nan` option on `clean_array`: when `True`, NaN values in float
  input arrays are filled from the nearest valid pixel rather than
  preserved as nodata.

## [0.1.0] - 2025-09-02

### Added
- Initial public release.
- `clean_array` API for morphological cleaning of multi-class 2D arrays:
  per-class edge smoothing (morphological opening), per-class small-island
  removal (connected components), and gap filling using nearest-valid via
  Euclidean distance transform.
- Documentation: README, two example notebooks (land use, cloud
  shadow), and a Google Colab tutorial notebook.

[Unreleased]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DPIRD-DMA/MultiClean/releases/tag/v0.1.0
