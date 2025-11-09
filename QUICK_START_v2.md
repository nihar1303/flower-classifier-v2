# 🚀 Quick Start - Flower Classifier v2.0

## 30 Seconds to Live Detection

### Step 1: Extract
```bash
unzip flower-classifier-live.zip && cd flower-classifier-live
```

### Step 2: Install
```bash
python3 -m venv venv
source venv/bin/activate  # Or: venv\Scripts\activate (Windows)
cd backend && pip install -r requirements.txt && cd ..
```

### Step 3: Run
```bash
python backend/app.py
```

You'll see:
```
* Running on http://0.0.0.0:5000
```

### Step 4: Open Browser
```
http://localhost:5000
```

### Step 5: Start Detecting

**LIVE CAMERA MODE:**
1. Click "LIVE CAMERA" tab
2. Click "Start Camera"
3. Allow camera access
4. Click "Capture Frame" for instant predictions
5. Or click "Auto Detect: ON" for continuous detection

**UPLOAD IMAGE MODE:**
1. Click "UPLOAD IMAGE" tab
2. Drag image or click to select
3. Click "Execute Prediction"

## That's It! 🎉

Enjoy real-time flower detection with your hacker-themed terminal interface!

---

**Tip:** For best results:
- Use good lighting for camera
- Position flower clearly in frame
- Adjust confidence threshold if needed
- Try auto-detect with 1s interval
