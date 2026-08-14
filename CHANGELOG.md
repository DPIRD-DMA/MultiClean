# Changelog

All notable changes to MultiClean are documented here.

## [Unreleased]

### Fixed
- Edge smoothing no longer translates the output by one pixel down and to the
  right when `smooth_edge_size` is even. `cv2.morphologyEx(MORPH_OPEN)` applies
  a single anchor to both the erosion and the dilation, which is only correct
  when that anchor coincides with the structuring element's centre of symmetry
  — true for odd kernel sizes, false for even ones. The result was a shifted
  opening that could also add pixels to a class rather than only removing them.
  The erosion and dilation are now run separately with the dilation anchored at
  the reflection of the erosion's anchor, giving a true opening at every size.

  **This changes output for even `smooth_edge_size`, including the default of
  `2`.** Output for odd values is bit-identical to previous releases. If you
  have cached results produced with an even `smooth_edge_size`, regenerate
  them or expect a one-pixel offset against new output.

- The source distribution no longer ships the example notebooks, their sample
  rasters, or the README artwork. `setuptools-scm` hands the sdist every
  git-tracked file, so these were swept in automatically — 4.9 MB of a 7.2 MB
  tarball, none of it read by anything in the package. A `MANIFEST.in` now
  prunes them, taking the sdist to 3.9 MB. The notebooks remain on GitHub,
  where the README links them. Wheels were never affected and are unchanged.

## [0.4.0] - 2026-07-28

### Changed
- Switched the OpenCV runtime dependency from `opencv-python` to
  `opencv-python-headless`. MultiClean only uses `morphologyEx`,
  `connectedComponentsWithStats` and `distanceTransformWithLabels` — no GUI
  or video I/O — so the GUI build was pulling in shared libraries that are
  absent from slim container and CI images, where plain `opencv-python`
  fails at import with `libGL.so.1: cannot open shared object file`.
  Both distributions provide the same `cv2` module, so if your environment
  needs the full `opencv-python` build for other work, install it
  explicitly; do not rely on both being present at once.
- Raised the OpenCV floor to `>=4.10`. `opencv-python` builds before
  4.10.0.84 are compiled against NumPy 1.x and fail at import when paired
  with NumPy 2.x, which the previous `>=4.0` floor allowed a resolver to do.

### Added
- Declared `numpy>=1.21` as an explicit runtime dependency. It was always
  imported directly and appears in the public type signatures, but was only
  installed transitively via OpenCV.

## [0.3.1] - 2026-07-21

### Fixed
- `clean_array` no longer raises `KeyError` when `class_values` names a class
  that does not occur in the input array. Requested classes absent from the
  array are now ignored, matching the behaviour that already applied when
  `smooth_edge_size=0`. This mainly affected tiled processing, where a single
  fixed class list is reused across tiles whose contents vary — the failure
  depended on the data, so it surfaced intermittently.

## [0.3.0] - 2026-05-02

### Changed
- **Performance.** `clean_array` is substantially faster on multi-class inputs.
  On a 15669×18633 / 147-class land-use raster, end-to-end runtime dropped
  from ~85 s to ~40 s. On the 8011×7901 / 4-class Landsat cloud-and-shadow
  example, runtime dropped from ~2.5 s to ~1.1 s. Wins came from:
  - Replacing the float32 smoothed-labels buffer with a `uint8`/`uint16`
    class-code array (selected automatically based on class count). The
    per-class equality scan is 2-4× cheaper in memory bandwidth.
  - Combining per-class small-island masks in flight instead of
    accumulating all K of them first.
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

[Unreleased]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DPIRD-DMA/MultiClean/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DPIRD-DMA/MultiClean/releases/tag/v0.1.0
