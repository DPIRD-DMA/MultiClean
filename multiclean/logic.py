import numpy as np
import cv2
from typing import List, Union
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from typing import Optional


def _small_islands_mask_for_class(
    image: np.ndarray,
    class_value: int,
    min_size: int,
    connectivity: int,
) -> np.ndarray:
    """Return a boolean mask of small connected components for a single class."""
    # NaNs compare False, so they are excluded automatically
    class_mask_u8 = (image == class_value).astype(np.uint8, copy=False)
    if class_mask_u8.sum() == 0:
        return np.zeros_like(class_mask_u8, dtype=bool)

    # labels: 0 is background
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        class_mask_u8, connectivity=connectivity, ltype=cv2.CV_32S
    )
    areas = stats[:, cv2.CC_STAT_AREA]

    small_component_label = areas < int(min_size)
    small_component_label[0] = False  # keep background out

    return small_component_label[labels]


def clean_array(
    array: np.ndarray,
    class_values: Optional[Union[int, List[int]]] = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 4,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Smooth edges per class, remove small islands, and inpaint invalid pixels
    using nearest valid class — with NaN used as a safe sentinel.
    """
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")

    # Build the target class list
    if class_values is None:
        if np.issubdtype(array.dtype, np.floating):
            unique_vals = np.unique(array[~np.isnan(array)])
        else:
            unique_vals = np.unique(array)
        target_classes: List[int] = unique_vals.astype(np.int32).tolist()
    elif isinstance(class_values, int):
        target_classes = [class_values]
    else:
        target_classes = list(class_values)

    # Kernel for morphological opening
    ksz = smooth_edge_size * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    # Work in float with NaN as nodata
    smoothed_labels = np.full(array.shape, np.nan, dtype=np.float32)

    # Step 1: edge smoothing per class
    def _opened_mask_for_class(cv: int) -> Tuple[int, np.ndarray]:
        class_mask_u8 = (array == cv).astype(np.uint8, copy=False)
        opened_u8 = cv2.morphologyEx(
            class_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1
        )
        return cv, opened_u8.astype(bool, copy=False)

    opened_masks: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_opened_mask_for_class, cv): cv for cv in target_classes
        }
        for fut in as_completed(futures):
            cv, opened_mask = fut.result()
            opened_masks[cv] = opened_mask
            smoothed_labels[opened_mask] = float(cv)

    # Step 2: small island detection per class (on the smoothed map)
    small_islands_by_class: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _small_islands_mask_for_class,
                image=smoothed_labels,
                class_value=cv,
                min_size=min_island_size,
                connectivity=connectivity,
            ): cv
            for cv in target_classes
        }
        for fut in as_completed(futures):
            cv = futures[fut]
            small_islands_by_class[cv] = fut.result()

    # Step 3: build invalid mask
    # invalid if NaN (no class wrote here) or part of a small component
    invalid_mask = np.isnan(smoothed_labels)
    if small_islands_by_class:
        invalid_mask = np.logical_or.reduce(
            [invalid_mask]
            + [small_islands_by_class[cv] for cv in small_islands_by_class]
        )

    if not invalid_mask.any():
        # Cast back to the original dtype if it was integer-like
        if np.issubdtype(array.dtype, np.integer):
            return smoothed_labels.astype(array.dtype, copy=False)
        return smoothed_labels

    # Step 4: fill invalids from nearest valid pixels
    output = smoothed_labels.copy()
    valid_mask = ~invalid_mask & np.isin(smoothed_labels, target_classes)

    if valid_mask.any():
        _, nearest_idx = distance_transform_edt(~valid_mask, return_indices=True)  # type: ignore
        yy = nearest_idx[0, invalid_mask]
        xx = nearest_idx[1, invalid_mask]
        output[invalid_mask] = smoothed_labels[yy, xx]
    else:
        # If everything is invalid, just return what we’ve got post-smoothing
        output = smoothed_labels

    # Restore original dtype where appropriate
    if np.issubdtype(array.dtype, np.integer):
        return output.astype(array.dtype, copy=False)
    return output
