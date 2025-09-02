import numpy as np
import cv2
from typing import List
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from typing import Optional


def small_islands_mask_for_class(
    image: np.ndarray,
    class_value: int,
    min_size: int,
    connectivity: int,
) -> np.ndarray:
    """
    Identify small connected components for a single class using area analysis.

    Uses OpenCV's connected components analysis to find regions smaller than
    the minimum size threshold. NaN values are automatically excluded from
    analysis as they compare False in equality operations.

    Parameters:
    -----------
    image : np.ndarray
        Classification array (may contain NaN values)
    class_value : int
        Specific class value to analyse for small components
    min_size : int
        Minimum area threshold for connected components (pixels)
    connectivity : int
        Pixel connectivity for component analysis (4 or 8)

    Returns:
    --------
    np.ndarray
        Boolean mask indicating small connected components for the specified class
    """
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


def create_circle_kernel(kernel_size: int) -> np.ndarray:
    """
    Create a circular morphological kernel with proper radius scaling.

    For small kernels (size < 3), uses minimal radius adjustment to preserve
    shape. For larger kernels, uses more aggressive adjustment for better
    circular appearance.

    Parameters:
    -----------
    kernel_size : int
        Size of the square kernel (width and height)

    Returns:
    --------
    np.ndarray
        Binary circular kernel of shape (kernel_size, kernel_size)
    """
    kernel_center = (kernel_size - 1) / 2
    row_indices, col_indices = np.ogrid[:kernel_size, :kernel_size]

    # Calculate Euclidean distance from each pixel to kernel center
    distance_from_center = np.sqrt(
        (col_indices - kernel_center) ** 2 + (row_indices - kernel_center) ** 2
    )

    # Determine radius adjustment based on kernel size
    # Small kernels need minimal adjustment, larger ones need more for clean circles
    radius_adjustment = 0.1 if kernel_size < 3 else 0.4
    effective_radius = kernel_size / 2 - radius_adjustment

    # Create circular mask where distance <= radius
    circular_mask = distance_from_center <= effective_radius

    return circular_mask.astype(np.uint8)


def smooth_edges(
    array: np.ndarray,
    smooth_edge_size: int,
    target_class_values: List[int],
    background_class_values: List[int],
    max_workers: Optional[int],
) -> np.ndarray:
    """
    Apply morphological opening to smooth edges for specified target classes.

    Background classes are preserved without morphological processing and used
    to fill regions where target class smoothing creates gaps. Uses parallel
    processing for efficiency across multiple classes.

    Parameters:
    -----------
    array : np.ndarray
        Input classification array
    smooth_edge_size : int
        Size of circular kernel for morphological opening operations
    target_class_values : List[int]
        Classes to apply edge smoothing to
    background_class_values : List[int]
        Classes to preserve as-is for gap filling
    max_workers : Optional[int]
        Number of worker threads for parallel processing

    Returns:
    --------
    np.ndarray
        Float array with smoothed target classes and preserved background classes
    """
    if smooth_edge_size > 0:
        # Kernel for morphological opening
        kernel = create_circle_kernel(smooth_edge_size)

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
                executor.submit(_opened_mask_for_class, cv): cv
                for cv in target_class_values
            }
            for fut in as_completed(futures):
                cv, opened_mask = fut.result()
                opened_masks[cv] = opened_mask
                smoothed_labels[opened_mask] = float(cv)
        # fill any nan value with background class if possible
        if background_class_values:
            background_class_mask = np.isin(array, background_class_values) & np.isnan(
                smoothed_labels
            )
            if background_class_mask.any():
                smoothed_labels[background_class_mask] = array[
                    background_class_mask
                ].astype(np.float32)

    else:
        smoothed_labels = array.astype(np.float32, copy=True)
    return smoothed_labels


def find_small_islands(
    smoothed_labels: np.ndarray,
    target_class_values: List[int],
    min_island_size: int,
    connectivity: int,
    max_workers: Optional[int],
) -> Dict[int, np.ndarray]:
    """
    Detect small connected components (islands) for each target class in parallel.

    Identifies regions smaller than the minimum size threshold using OpenCV's
    connected components analysis. Only processes target classes, ignoring
    background classes that were preserved during edge smoothing.

    Parameters:
    -----------
    smoothed_labels : np.ndarray
        Array after edge smoothing (may contain NaN values)
    target_class_values : List[int]
        Classes to analyse for small islands
    min_island_size : int
        Minimum size threshold for connected components
    connectivity : int
        Connectivity for component analysis (4 or 8)
    max_workers : Optional[int]
        Number of worker threads for parallel processing

    Returns:
    --------
    Dict[int, np.ndarray]
        Dictionary mapping class values to boolean masks of small islands
    """
    small_islands_by_class: Dict[int, np.ndarray] = {}
    if min_island_size <= 0:
        return small_islands_by_class
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                small_islands_mask_for_class,
                image=smoothed_labels,
                class_value=cv,
                min_size=min_island_size,
                connectivity=connectivity,
            ): cv
            for cv in target_class_values
        }
        for fut in as_completed(futures):
            cv = futures[fut]
            small_islands_by_class[cv] = fut.result()
    return small_islands_by_class


def build_invalid_mask(
    smoothed_labels: np.ndarray, small_islands_by_class: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Create combined mask of all pixels requiring gap filling.

    Combines NaN pixels from edge smoothing with small island pixels identified
    during island detection. The resulting mask indicates all regions that need
    to be filled using nearest-neighbour interpolation.

    Parameters:
    -----------
    smoothed_labels : np.ndarray
        Array after edge smoothing (may contain NaN values)
    small_islands_by_class : Dict[int, np.ndarray]
        Dictionary of small island masks for each class

    Returns:
    --------
    np.ndarray
        Boolean mask indicating all pixels requiring filling
    """
    invalid_mask = np.isnan(smoothed_labels)
    if small_islands_by_class:
        invalid_mask = np.logical_or.reduce(
            [invalid_mask]
            + [small_islands_by_class[cv] for cv in small_islands_by_class]
        )
    return invalid_mask


def fill_invalids(
    smoothed_labels: np.ndarray, invalid_mask: np.ndarray, all_class_values: List[int]
) -> np.ndarray:
    """
    Fill invalid pixels using nearest-neighbour interpolation from valid pixels.

    Uses Euclidean distance transform to find the closest valid pixel for each
    invalid region. Valid pixels include all classes present in the array,
    ensuring natural gap filling across class boundaries.

    Parameters:
    -----------
    smoothed_labels : np.ndarray
        Array with invalid regions marked for filling
    invalid_mask : np.ndarray
        Boolean mask indicating pixels requiring filling
    all_class_values : List[int]
        All valid class values that can serve as sources for filling

    Returns:
    --------
    np.ndarray
        Array with all invalid regions filled using nearest valid values
    """
    output = smoothed_labels.copy()
    valid_mask = ~invalid_mask & np.isin(smoothed_labels, all_class_values)

    if valid_mask.any():
        _, nearest_idx = distance_transform_edt(~valid_mask, return_indices=True)  # type: ignore
        yy = nearest_idx[0, invalid_mask]
        xx = nearest_idx[1, invalid_mask]
        output[invalid_mask] = smoothed_labels[yy, xx]
    else:
        # If everything is invalid, just return what we’ve got post-smoothing
        output = smoothed_labels
    return output
