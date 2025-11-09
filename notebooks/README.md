# Jupyter Notebooks (Optional)

## Included Notebooks

### 1. camera_demo.ipynb
Demonstrates real-time camera usage in Jupyter

### 2. prediction_examples.ipynb
Test predictions on sample images

### 3. model_exploration.ipynb
Understand model architecture and weights

## How to Use

```bash
pip install jupyter ipywidgets
jupyter notebook
```

Then open any .ipynb file and run cells.

## Camera in Jupyter

Use Python bindings to access camera:
```python
from PIL import Image
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img = Image.fromarray(frame)
```

## Examples

See `../backend/` for Python API examples.
