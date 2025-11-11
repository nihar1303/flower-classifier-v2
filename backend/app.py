"""
Flower Detector - Working Backend
"""
import os
import sys
import json
from datetime import datetime
from io import BytesIO
import base64

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ==================== PATH SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== FLASK SETUP ====================
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

FLOWER_CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# ==================== MODEL ====================
class FlowerCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

print("🌼 Initializing model...")
model = FlowerCNN(num_classes=5)
MODEL_PATH = os.path.join(BASE_DIR, "models", "flower_model.pth")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    print(f"✅ Loaded trained weights from {MODEL_PATH}")
else:
    print(f"⚠️ Trained model not found at {MODEL_PATH}. Using untrained weights.")
model.eval()

# ==================== TRANSFORMS ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== HELPERS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_data):
    """Process image for prediction"""
    try:
        if isinstance(image_data, str):
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))
        else:
            img = Image.open(image_data)

        img = img.convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        raise

def make_prediction(img_tensor):
    """Make prediction"""
    try:
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probabilities, 1)
            top_probs, top_indices = torch.topk(probabilities, 5, dim=1)

            return {
                'flower': FLOWER_CLASSES[pred_idx.item()],
                'confidence': float(conf.item()),
                'top_5': [
                    {
                        'flower': FLOWER_CLASSES[top_indices[0, i].item()],
                        'probability': float(top_probs[0, i].item())
                    }
                    for i in range(5)
                ]
            }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise

# ==================== ROUTES ====================
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0',
        'model': 'FlowerCNN',
        'classes': FLOWER_CLASSES
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        start_time = datetime.utcnow()

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file', 'success': False}), 400
            img_tensor = process_image(file)

        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            img_tensor = process_image(image_data)

        else:
            return jsonify({'error': 'No image provided', 'success': False}), 400

        prediction = make_prediction(img_tensor)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = {
            'success': True,
            'prediction': {
                'id': f'pred_{int(datetime.utcnow().timestamp())}',
                'timestamp': datetime.utcnow().isoformat(),
                'flower': prediction['flower'],
                'confidence': prediction['confidence'],
                'processing_time': int(processing_time),
                'top_5': prediction['top_5']
            }
        }

        print(f"✅ Prediction: {prediction['flower']} ({prediction['confidence']:.2%})")
        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/predict-batch', methods=['POST', 'OPTIONS'])
def predict_batch():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        files = request.files.getlist('files')
        results = []

        for file in files:
            if not allowed_file(file.filename):
                continue
            try:
                img_tensor = process_image(file)
                prediction = make_prediction(img_tensor)
                results.append({
                    'filename': file.filename,
                    'flower': prediction['flower'],
                    'confidence': prediction['confidence'],
                    'top_5': prediction['top_5']
                })
            except Exception as e:
                print(f"⚠️ Error processing {file.filename}: {e}")
                continue

        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 60)
    print("🌸 Flower Detector Pro - Starting Server")
    print("=" * 60)
    print("✅ Model initialized")
    print("✅ CORS enabled")
    print("✅ Routes configured")
    print("=" * 60)
    print("🚀 Server running at: http://localhost:5002")
    print("📱 Open in browser: http://localhost:5002")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5002,       # Changed from 5000 → 5002
        debug=True,
        threaded=True
    )
