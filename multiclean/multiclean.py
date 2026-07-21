from typing import List, Optional, Union

import numpy as np

from .utils import build_invalid_mask, fill_invalids, smooth_edges_to_codes


def clean_array(
    array: np.ndarray,
    class_values: Optional[Union[int, List[int]]] = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 4,
    max_workers: Optional[int] = None,
    fill_nan: bool = False,
) -> np.ndarray:
    """
    Clean classification arrays through edge smoothing and island removal.

    Applies morphological opening to smooth class boundaries, removes small isolated
    regions below size threshold, and fills gaps using nearest-neighbour interpolation.
    Can process specific target classes whilst preserving background classes unchanged.
    Uses parallel processing for efficiency on large arrays.

    Parameters:
    -----------
    array : np.ndarray
        Input classification 2D array with integer class labels
    class_values : Optional[Union[int, List[int]]]
        Specific classes to process. If None, processes all classes in array
    smooth_edge_size : int
        Size of circular kernel for edge smoothing operations
    min_island_size : int
        Minimum area threshold for connected components (pixels)
    connectivity : int
        Pixel connectivity for island detection (4 or 8)
    max_workers : Optional[int]
        Number of worker threads for parallel processing
    fill_nan : bool
        Whether to fill NaN values from the input array

    Returns:
    --------
    np.ndarray
        Cleaned classification array with same dtype as input
    """
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    if smooth_edge_size < 0:
        raise ValueError("smooth_edge_size must be non-negative")
    if min_island_size < 0:
        raise ValueError("min_island_size must be non-negative")
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    is_float = np.issubdtype(array.dtype, np.floating)

    all_class_values = np.unique(array).tolist()
    if is_float:
        all_class_values = [v for v in all_class_values if not np.isnan(v)]

    if class_values is None:
        target_class_values = list(all_class_values)
    elif isinstance(class_values, int):
        target_class_values = [class_values]
    else:
        target_class_values = list(class_values)

    # Requested classes that do not occur in this array have nothing to clean,
    # so drop them here rather than letting them reach the code lookups.
    present = set(all_class_values)
    target_class_values = [v for v in target_class_values if v in present]

    background_class_values = list(set(all_class_values) - set(target_class_values))

    if is_float and not fill_nan:
        nan_mask = np.isnan(array)
        if not nan_mask.any():
            nan_mask = None
    else:
        nan_mask = None

    codes, code_to_value = smooth_edges_to_codes(
        array=array,
        smooth_edge_size=smooth_edge_size,
        target_class_values=target_class_values,
        background_class_values=background_class_values,
        all_class_values=all_class_values,
        max_workers=max_workers,
    )

    # Find target codes (1..K) for the requested target classes.
    classes_sorted = sorted(all_class_values)
    value_to_code = {v: i + 1 for i, v in enumerate(classes_sorted)}
    target_codes = [value_to_code[v] for v in target_class_values if v in value_to_code]

    invalid_mask = build_invalid_mask(
        codes=codes,
        target_codes=target_codes,
        min_island_size=min_island_size,
        connectivity=connectivity,
        max_workers=max_workers,
    )

    codes = fill_invalids(codes, invalid_mask)

    # Decode codes back to class values via vectorised lookup. ``np.take``
    # uses ``out`` so we can write directly into a typed buffer.
    output = code_to_value[codes]

    if is_float and nan_mask is not None:
        # Restore original NaN positions when fill_nan=False (they were
        # included in invalid_mask only to keep them off the fill-source set).
        if not np.issubdtype(output.dtype, np.floating):
            output = output.astype(np.float64)
        output[nan_mask] = np.nan

    if np.issubdtype(array.dtype, np.integer):
        output = output.astype(array.dtype, copy=False)
    elif is_float and output.dtype != array.dtype:
        output = output.astype(array.dtype, copy=False)

    return output
