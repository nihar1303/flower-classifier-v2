#!/bin/bash

echo "🌸 Flower Classifier v2.0 - Setup Script"
echo "========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# Create directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p models

echo "✅ Setup complete!"
echo ""
echo "To start:"
echo "1. source venv/bin/activate"
echo "2. python backend/app.py"
echo "3. Open http://localhost:5000"
