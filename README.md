# MultiClean

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MultiClean** is a Python library for morphological cleaning of multiclass 2D numpy arrays (segmentation masks and classification rasters). It provides efficient tools for edge smoothing and small-island removal across multiple classes, then fills gaps using the nearest valid class.

## Key Features

- **Multi-class processing**: Clean all classes in one pass
- **Edge smoothing**: Morphological opening to reduce jagged boundaries
- **Island removal**: Remove small connected components per class
- **Gap filling**: Fill invalids via nearest valid class (distance transform)
- **Fast + memory‑aware**: NumPy + OpenCV + SciPy with optional parallelism

## Installation

```bash
pip install multiclean
```

### Development Installation

```bash
git clone https://github.com/DPIRD-DMA/MultiClean.git
cd MultiClean
pip install -e .[dev]
```

## Quick Start

```python
import numpy as np
from multiclean.logic import clean_array  # or: from multiclean import clean_array

# Create a sample classification array with classes 0, 1, 2, 3
array = np.random.randint(0, 4, (1000, 1000), dtype=np.int32)

# Clean with sensible defaults
cleaned = clean_array(array)

# Custom parameters + parallel island checks
cleaned = clean_array(
    array,
    class_values=[0, 1, 2, 3],
    smooth_edge_size=2,     # kernel radius; kernel = 2*r + 1
    min_island_size=100,    # remove components with area < 100
    connectivity=8,         # 4 or 8
    max_workers=4,
)
```

## Use Cases

MultiClean is designed for cleaning segmentation outputs from:

- **Remote sensing**: Land cover classification, crop mapping
- **Computer vision**: Semantic segmentation post-processing  
- **Medical imaging**: Tissue segmentation, organ delineation
- **Geospatial analysis**: Raster classification cleaning
- **Machine learning**: Neural network output refinement

## How It Works

MultiClean uses morphological operations to clean classification arrays:

1. **Edge smoothing (per class)**: Morphological opening with an elliptical kernel of size `(2*smooth_edge_size + 1)`.
2. **Island removal (per class)**: Find connected components (OpenCV) and mark components with area `< min_island_size` as invalid.
3. **Gap filling**: Compute a distance transform to copy the nearest valid class into invalid pixels.

Classes are processed together and the result maintains a valid label at every pixel.

## API

### `clean_array`

```python
from multiclean.logic import clean_array  # or: from multiclean import clean_array

out = clean_array(
    array: np.ndarray,
    class_values: int | list[int] | None = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 4,
    max_workers: int | None = None,
)
```

- `array`: 2D numpy array of class labels (int or float). For float arrays, `NaN` is treated as nodata and will be filled.
- `class_values`: Classes to consider. If `None`, inferred from `array` (ignores `NaN` for floats). An int restricts cleaning to a single class.
- `smooth_edge_size`: Radius for morphological opening; kernel size is `2*radius + 1`. Use `0` to disable.
- `min_island_size`: Remove components with area strictly `< min_island_size`. Use `1` to keep single pixels.
- `connectivity`: Pixel connectivity for components, `4` or `8`.
- `max_workers`: Parallelism for per-class operations (None lets the executor choose).

Returns a numpy array matching the input shape. Integer inputs return integer outputs; float inputs return float with no `NaN`s left unless everything was invalid.


## Performance

MultiClean is optimised for large arrays:

- **Vectorised operations** using NumPy, OpenCV, and SciPy
- **Parallel processing** for island detection across classes
- **Memory efficient** algorithms that minimise array copying
- **Fast distance transforms** for gap filling

Typical performance on a 1000×1000 array with ~4 classes: ~100–500 ms depending on content and parameters.

## Requirements

- Python ≥ 3.9
- OpenCV ≥ 4.0 (`opencv-python`)
- SciPy ≥ 1.0 (brings NumPy)

## Examples

### Cleaning Satellite Land Cover Data

```python
from multiclean.logic import clean_array
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
)
```

### Cleaning Neural Network Segmentation Output

```python
from multiclean.logic import clean_array

# Model produces logits; convert to class predictions
pred = model_logits.argmax(axis=0)  # shape: (H, W)

# Clean the segmentation
cleaned = clean_array(
    pred,
    class_values=list(range(num_classes)),
    smooth_edge_size=2,
    min_island_size=100,
    connectivity=8,
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MultiClean in your research, please cite:

```bibtex
@software{multiclean,
  author = {Wright, Nick},
  title = {MultiClean: Morphological cleaning for multiclass segmentation},
  url = {https://github.com/DPIRD-DMA/MultiClean},
  year = {2024}
}
```

## Contact

**Nick Wright** - nicholas.wright@dpird.wa.gov.au

Project Link: [https://github.com/DPIRD-DMA/MultiClean](https://github.com/DPIRD-DMA/MultiClean)
