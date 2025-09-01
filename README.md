# MultiClean

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MultiClean** is a Python library for morphological cleaning of multiclass 2D segmentation masks and classification arrays. It provides efficient tools for edge smoothing and small island removal across multiple classes simultaneously.

## Key Features

- **Multi-class processing**: Clean all classes in a single operation
- **Edge smoothing**: Remove jagged edges and noise from segmentation boundaries  
- **Island removal**: Eliminate small isolated regions below a size threshold
- **Integrated workflow**: Combined edge cleaning and island removal with gap filling
- **Memory efficient**: Vectorised operations using OpenCV and SciPy

## Installation

```bash
pip install multiclean
```

```bash
pip add multiclean
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
import multiclean as mc

# Create a sample classification array with classes 0, 1, 2, 3
array = np.random.randint(0, 4, (1000, 1000), dtype=np.int32)

# Clean the segmentation with default settings
cleaned = mc.clean_classification_array(
    array, 
    edge_size=2,           # Smooth edges by 2 pixels
    min_island_size=50     # Remove islands smaller than 50 pixels
)

# Advanced usage with parallel processing
cleaned = mc.integrated_edge_and_island_cleaning_parallel(
    array,
    class_values=[0, 1, 2, 3],
    smooth_edge_size=3,
    min_island_size=100,
    connectivity=8,
    max_workers=4
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

1. **Edge Cleaning**: Applies erosion followed by dilation to smooth class boundaries and remove noise
2. **Island Removal**: Identifies small connected components using OpenCV's connected components analysis
3. **Gap Filling**: Uses distance transforms to fill removed regions with the nearest valid class values

The key advantage is that all classes are processed together, maintaining spatial relationships and ensuring no gaps or overlaps in the final output.

## API Overview

### Core Functions

```python
# Simple integrated cleaning
multiclean.clean_classification_array(array, edge_size=2, min_island_size=50)

# Advanced integrated cleaning  
multiclean.integrated_edge_and_island_cleaning(array, class_values, ...)

# Parallel version for large arrays
multiclean.integrated_edge_and_island_cleaning_parallel(array, ..., max_workers=4)
```

### Individual Operations

```python
# Edge smoothing only
multiclean.smooth_edges(array, edge_size=2, class_values=[0,1,2,3])

# Island removal only  
multiclean.remove_small_islands(array, min_size=50, class_values=[0,1,2,3])

# Fill gaps with nearest neighbours
multiclean.fill_with_nearest(array, invalid_mask)
```

## Performance

MultiClean is optimised for large arrays:

- **Vectorised operations** using NumPy, OpenCV, and SciPy
- **Parallel processing** for island detection across classes
- **Memory efficient** algorithms that minimise array copying
- **Fast distance transforms** for gap filling

Typical performance on a 1000×1000 array with 4 classes: ~100-500ms depending on complexity.

## Requirements

- Python ≥ 3.9
- OpenCV ≥ 4.0 (`opencv-python`)
- SciPy ≥ 1.0
- NumPy (installed with SciPy)

## Examples

### Cleaning Satellite Land Cover Data

```python
import multiclean as mc
import rasterio

# Read land cover classification
with rasterio.open('landcover.tif') as src:
    landcover = src.read(1)

# Clean with appropriate parameters for satellite data
cleaned = mc.clean_classification_array(
    landcover,
    edge_size=1,           # Gentle smoothing
    min_island_size=25,    # Remove small misclassified areas
    class_values=[0, 1, 2, 3, 4]  # Forest, water, urban, crop, other
)
```

### Cleaning Neural Network Segmentation Output

```python
import multiclean as mc

# Your model produces logits, convert to class predictions
predictions = model_logits.argmax(axis=0)  # Shape: (H, W)

# Clean the segmentation
cleaned_predictions = mc.integrated_edge_and_island_cleaning(
    predictions,
    class_values=list(range(num_classes)),
    smooth_edge_size=2,
    min_island_size=100,
    connectivity=8
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