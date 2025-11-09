# Dataset Directory

## Structure

```
data/flowers/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

## Where to Get Data

### Kaggle Flowers Recognition Dataset
https://www.kaggle.com/alxmamaev/flowers-recognition

### Steps:
1. Download from Kaggle
2. Extract to `data/flowers/`
3. Run: `python backend/train.py`

## Quick Test Dataset

For testing without full dataset:
1. Create directories as shown above
2. Add 5-10 images per category
3. Run training (will work but low accuracy)

## Using Live Camera Instead

No dataset needed! Just:
1. Start camera: Click "Start Camera"
2. Point at flowers
3. Click "Capture Frame"
4. Get instant predictions

The pre-trained model works with your camera directly.
