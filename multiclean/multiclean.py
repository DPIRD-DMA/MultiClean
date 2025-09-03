from typing import List, Optional, Union

import numpy as np

from .utils import (
    build_invalid_mask,
    fill_invalids,
    find_small_islands,
    smooth_edges,
)


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

    all_class_values = np.unique(array).tolist()
    # Remove NaN from class values if present
    if np.issubdtype(array.dtype, np.floating):
        all_class_values = [v for v in all_class_values if not np.isnan(v)]

    if class_values is None:
        target_class_values = all_class_values
    else:
        if isinstance(class_values, int):
            target_class_values = [class_values]
        else:
            target_class_values = list(class_values)

    background_class_values = list(set(all_class_values) - set(target_class_values))

    if np.issubdtype(array.dtype, np.floating) and not fill_nan:
        nan_mask = np.isnan(array)
        if nan_mask.any():
            background_class_values.append(np.nan)
    else:
        nan_mask = None

    smoothed_labels = smooth_edges(
        array=array,
        smooth_edge_size=smooth_edge_size,
        target_class_values=target_class_values,
        background_class_values=background_class_values,
        max_workers=max_workers,
    )

    small_islands_by_class = find_small_islands(
        smoothed_labels=smoothed_labels,
        target_class_values=target_class_values,
        min_island_size=min_island_size,
        connectivity=connectivity,
        max_workers=max_workers,
    )

    invalid_mask = build_invalid_mask(
        smoothed_labels=smoothed_labels,
        small_islands_by_class=small_islands_by_class,
    )

    if not invalid_mask.any():
        # Apply original NaN mask if present
        if nan_mask is not None and nan_mask.any():
            smoothed_labels[nan_mask] = np.nan
        if np.issubdtype(array.dtype, np.integer):
            return smoothed_labels.astype(array.dtype, copy=False)
        return smoothed_labels

    output = fill_invalids(
        smoothed_labels=smoothed_labels,
        invalid_mask=invalid_mask,
        all_class_values=all_class_values,
    )

    # Convert back to original dtype if integer
    if np.issubdtype(array.dtype, np.integer):
        return output.astype(array.dtype, copy=False)

    # Apply original NaN mask if present
    if nan_mask is not None and nan_mask.any():
        output[nan_mask] = np.nan
    return output
