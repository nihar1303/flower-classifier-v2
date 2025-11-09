"""
Configuration file for Flower Classification Project v2.0
"""

FLOWER_CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
MODEL_PATH = 'models/flower_model.pth'
DEVICE = 'cpu'
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_FILE_SIZE = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
