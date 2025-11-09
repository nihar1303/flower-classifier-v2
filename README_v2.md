# 🌸 Flower Classifier v2.0 - Live Camera Edition

## What's New in v2.0

✨ **LIVE CAMERA DETECTION**
- Real-time flower identification from your webcam
- Auto-detection with configurable intervals (500ms - 3s)
- Manual capture and instant predictions
- FPS monitoring

✨ **ENHANCED FEATURES**
- Dual mode: Live camera + Image upload
- Adjustable detection interval
- Confidence threshold filtering
- Auto-detect toggle
- Frame capture from video stream
- Real-time FPS counter

✨ **IMPROVED INTERFACE**
- Tab-based navigation (Camera / Upload)
- Live video preview with grid overlay
- Camera status indicator (recording dot)
- Detection settings panel
- Processing metrics

## Quick Start (3 Steps)

### 1. Setup (2 minutes)
```bash
unzip flower-classifier-live.zip
cd flower-classifier-live
python3 -m venv venv
source venv/bin/activate  # Mac/Linux or venv\Scripts\activate Windows
cd backend && pip install -r requirements.txt && cd ..
```

### 2. Run (30 seconds)
```bash
python backend/app.py
```

### 3. Open (10 seconds)
```
http://localhost:5000
```

## Using Live Camera

1. **Start Camera**
   - Click "LIVE CAMERA" tab
   - Click "Start Camera" button
   - Grant camera permission in browser

2. **Manual Capture**
   - Click "Capture Frame" button
   - Instant prediction displayed
   - See results with confidence

3. **Auto Detection**
   - Click "Auto Detect: OFF" to enable
   - Camera frames processed every 1 second (configurable)
   - Only shows predictions above confidence threshold
   - FPS displayed in footer

4. **Adjust Settings**
   - Detection Interval: 500ms - 3000ms
   - Confidence Threshold: 30% - 95%
   - Real-time updates

## Using Image Upload

1. Click "UPLOAD IMAGE" tab
2. Drag-drop or click to select image
3. Click "Execute Prediction"
4. View results

## Camera Browser Requirements

Works on:
- ✅ Chrome/Chromium
- ✅ Firefox
- ✅ Safari (iOS 11+)
- ✅ Edge

## Features

### Live Detection Modes

**Manual Mode:**
- User clicks "Capture Frame"
- Single image processed
- Results displayed immediately

**Auto Mode:**
- Continuous camera feed
- Auto-process every N milliseconds
- Only display high-confidence results
- FPS counter shows detection rate

### Adjustable Parameters

**Detection Interval:** 500ms - 3000ms
- Faster: More predictions, higher CPU usage
- Slower: Less processing, smoother UX

**Confidence Threshold:** 30% - 95%
- Higher: Only very confident predictions
- Lower: More predictions, including uncertain

### Real-time Monitoring

- Processing time (milliseconds)
- Detection mode (Auto/Manual)
- FPS rate (auto mode only)
- Flower identification with confidence
- Top 3 predictions always shown

## API Improvements

Same API as v1, works with both:
- Static image files
- Real-time camera frames (JPEG from canvas)
- Batch processing ready

## Performance

- **Camera Resolution:** 1280×720 (optimized)
- **Processing:** 500-1000ms (CPU)
- **Auto-detect:** Up to 2 FPS on CPU
- **GPU-enabled:** Up to 10 FPS (with CUDA)

## Troubleshooting

### Camera Won't Start
- Grant browser permission for camera
- Check if another app using camera
- Try different browser

### Low FPS
- Increase detection interval
- Disable auto-detect
- Use GPU if available

### High CPU Usage
- Increase detection interval (slower)
- Lower image resolution
- Reduce number of auto-detections

## Extension Ideas

- Add recording functionality
- Multi-object detection
- Batch process images
- Export predictions to CSV
- Plant care recommendations
- Confidence graphs over time

## Hardware Requirements

**Minimum:**
- Webcam or built-in camera
- Modern browser
- ~300MB RAM
- 2GHz processor

**Recommended:**
- USB camera (better quality)
- NVIDIA GPU with CUDA
- 4GB+ RAM
- Multi-core processor

## For Your University Class

Perfect for:
- Real-time AI demonstrations
- Live debugging neural networks
- Performance comparisons (CPU vs GPU)
- User interaction patterns
- IoT/Edge computing discussions

---

**Version:** 2.0 | **Camera Support:** ✅ | **Status:** Ready for University Presentation

🚀 **Welcome to Live Flower Detection!**
