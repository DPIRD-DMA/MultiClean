# MultiClean
[![image](https://img.shields.io/pypi/v/multiclean.svg)](https://pypi.python.org/pypi/multiclean)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tutorials](https://img.shields.io/badge/Tutorials-Learn-brightgreen)](https://github.com/DPIRD-DMA/MultiClean/tree/main/notebooks)

**MultiClean** is a Python library for morphological cleaning of multiclass 2D numpy arrays (segmentation masks and classification rasters). It provides efficient tools for edge smoothing and small-island removal across multiple classes, then fills gaps using the nearest valid class.

## Visual Example

Below: Land Use before/after cleaning (smoothed edges, small-island removal, nearest-class gap fill).

<img src="https://raw.githubusercontent.com/DPIRD-DMA/MultiClean/main/assets/land_use_before_after.png" alt="Land Use before/after" width="900"/>

## Installation

```bash
pip install multiclean
```
or
```bash
uv add multiclean
```

## Quick Start

```python
import numpy as np
from multiclean import clean_array

# Create a sample classification array with classes 0, 1, 2, 3
array = np.random.randint(0, 4, (1000, 1000), dtype=np.int32)

# Clean with default parameters
cleaned = clean_array(array)

# Custom parameters
cleaned = clean_array(
    array,
    class_values=[0, 1, 2, 3],
    smooth_edge_size=2,     # kernel width, larger value increases smoothness
    min_island_size=100,    # remove components with area < 100
    connectivity=8,         # 4 or 8
    max_workers=4,
    fill_nan=False          # enable/disable the filling of nan values in input array
)
```

## Use Cases

MultiClean is designed for cleaning segmentation outputs from:

- **Remote sensing**: Land cover classification, crop mapping
- **Computer vision**: Semantic segmentation post-processing  
- **Geospatial analysis**: Raster classification cleaning
- **Machine learning**: Neural network output refinement

## Key Features

- **Multi-class processing**: Clean all classes in one pass
- **Edge smoothing**: Morphological opening to reduce jagged boundaries
- **Island removal**: Remove small connected components per class
- **Gap filling**: Fill invalids via nearest valid class (distance transform)
- **Fast**: NumPy + OpenCV + SciPy with parallelism


## How It Works

MultiClean uses morphological operations to clean classification arrays:

1. **Edge smoothing (per class)**: Morphological opening with an elliptical kernel.
2. **Island removal (per class)**: Find connected components (OpenCV) and mark components with area `< min_island_size` as invalid.
3. **Gap filling**: Compute a distance transform to copy the nearest valid class into invalid pixels.

Classes are processed together and the result maintains a valid label at every pixel.

## API Reference

### `clean_array`

```python
from multiclean import clean_array

out = clean_array(
    array: np.ndarray,
    class_values: int | list[int] | None = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 4,
    max_workers: int | None = None,
    fill_nan: bool = False
)
```

- `array`: 2D numpy array of class labels (int or float). For float arrays, `NaN` is treated as nodata and will remain `NaN` unless `fill_nan` is set to `True`.
- `class_values`: Classes to consider. If `None`, inferred from `array` (ignores `NaN` for floats). An int restricts cleaning to a single class.
- `smooth_edge_size`: Kernel size (pixels) for morphological opening. Use `0` to disable.
- `min_island_size`: Remove components with area strictly `< min_island_size`. Use `1` to keep single pixels.
- `connectivity`: Pixel connectivity for components, `4` or `8`.
- `max_workers`: Parallelism for per-class operations (None lets the executor choose).
- `fill_nan`: If True will fill NAN values from input array with nearest valid value.

Returns a numpy array matching the input shape. Integer inputs return integer outputs. Float arrays with `NaN` are supported and can be filled or remain as NAN.

## Examples

### Cleaning Land Cover Data

```python
from multiclean import clean_array
import rasterio

# Read land cover classification
with rasterio.open('landcover.tif') as src:
    landcover = src.read(1)

# Clean with appropriate parameters for satellite data
cleaned = clean_array(
    landcover,
    class_values=[0, 1, 2, 3, 4],  # forest, water, urban, crop, other
    smooth_edge_size=1,
    min_island_size=25,
    connectivity=8,
    fill_nan=False
)
```

### Cleaning Neural Network Segmentation Output

```python
from multiclean import clean_array

# Model produces logits; convert to class predictions
np_pred = np_model_logits.argmax(axis=0)  # shape: (H, W)

# Clean the segmentation
cleaned = clean_array(
    np_pred,
    smooth_edge_size=2,
    min_island_size=100,
    connectivity=4,
)
```
## Notebooks

See the notebooks folder for end-to-end examples:
- [Land Use Example Notebook](https://github.com/DPIRD-DMA/MultiClean/blob/main/notebooks/Land%20use%20example.ipynb): land use classification cleaning
- [Cloud Example Notebook](https://github.com/DPIRD-DMA/MultiClean/blob/main/notebooks/Cloud%20example.ipynb): cloud/shadow classification cleaning

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/github/DPIRD-DMA/MultiClean/blob/main/notebooks/Land%20use%20example%20(Colab).ipynb 'Try MultiClean In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
