# Pre-trained Models

## flower_model.pth

The neural network model file.

### Download (Optional)

If flower_model.pth doesn't exist:
1. The system creates untrained model on first run (for testing)
2. Or train your own: `python backend/train.py`

### Pre-trained Weights

To use pre-trained weights:
1. Download from Kaggle or other source
2. Place in this directory
3. Rename to `flower_model.pth`
4. Restart the application

### Training Your Own

```bash
cd backend
python train.py
```

Requires dataset in: `../data/flowers/`
