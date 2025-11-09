"""
Flask API for Flower Classification v2.0
Supports both file upload and real-time processing
"""

import os
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MODEL_PATH, FLOWER_CLASSES
from model import FlowerClassifier

# ==============================================
# Path Configuration
# ==============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FOLDER = os.path.join(BASE_DIR, '..', 'frontend')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FILE_PATH = os.path.join(BASE_DIR, MODEL_PATH)

# ==============================================
# Flask App Initialization
# ==============================================
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app)

# ==============================================
# Upload Config
# ==============================================
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ==============================================
# Model Initialization
# ==============================================
try:
    classifier = FlowerClassifier(model_path=MODEL_FILE_PATH)
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    classifier = FlowerClassifier()

# ==============================================
# Helper Functions
# ==============================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================================
# Routes
# ==============================================
@app.route('/')
def index():
    """Serve the frontend interface"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'flower_classifier_v2.0',
        'classes': FLOWER_CLASSES,
        'features': ['live_camera', 'image_upload', 'real_time_detection']
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = classifier.predict(filepath)

        try:
            os.remove(filepath)
        except Exception:
            pass

        return jsonify({
            'success': True,
            'prediction': prediction,
            'message': f"Predicted: {prediction['flower']} ({prediction['confidence']*100:.1f}%)"
        }), 200

    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': FLOWER_CLASSES}), 200

# ==============================================
# Run Server
# ==============================================
if __name__ == '__main__':
    print("🚀 Flower Classification API v2.0 Starting...")
    print(f"📱 Features: Live Camera, Image Upload, Real-time Detection")
    print(f"🌸 Classes: {', '.join(FLOWER_CLASSES)}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5002, debug=True)
