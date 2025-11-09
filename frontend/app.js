/**
 * FLOWER CLASSIFIER v2.0 - Live Camera + Upload
 * Real-time Detection and Predictions
 */

const API_BASE_URL = 'http://localhost:5000';
let selectedFile = null;
let processingTime = 0;
let cameraStream = null;
let autoDetectEnabled = false;
let autoDetectInterval = null;
let detectionFrameCount = 0;

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const cameraVideo = document.getElementById('cameraVideo');
const captureCanvas = document.getElementById('captureCanvas');
const commandLog = document.getElementById('commandLog');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const autoDetectToggle = document.getElementById('autoDetectToggle');
const captureFrameBtn = document.getElementById('captureFrameBtn');
const detectionInterval = document.getElementById('detectionInterval');
const confidenceThreshold = document.getElementById('confidenceThreshold');
const intervalValue = document.getElementById('intervalValue');
const thresholdValue = document.getElementById('thresholdValue');
const predictionFlower = document.getElementById('predictionFlower');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const predictionsBody = document.getElementById('predictionsBody');
const processingTimeElement = document.getElementById('processingTime');
const detectionModeElement = document.getElementById('detectionMode');
const footerFPS = document.getElementById('footerFPS');

// ======================== TAB NAVIGATION ========================

tabButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => {
        // Remove active from all
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        // Add active to clicked
        btn.classList.add('active');
        const tabName = btn.getAttribute('data-tab');
        document.getElementById(`${tabName}-tab`).classList.add('active');

        if (tabName === 'upload') {
            stopCamera();
        }
    });
});

// ======================== SETTINGS ========================

detectionInterval.addEventListener('input', (e) => {
    const ms = parseInt(e.target.value);
    intervalValue.textContent = (ms / 1000) + 's';
});

confidenceThreshold.addEventListener('input', (e) => {
    const percent = Math.round(parseFloat(e.target.value) * 100);
    thresholdValue.textContent = percent + '%';
});

// ======================== CAMERA FUNCTIONS ========================

async function startCamera() {
    try {
        stopCamera(); // Stop existing stream first

        const constraints = {
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;

        startCameraBtn.style.display = 'none';
        stopCameraBtn.style.display = 'flex';
        captureFrameBtn.style.display = 'flex';

        addLog('camera', 'Camera initialized successfully', 'success');

        // Setup canvas
        cameraVideo.onloadedmetadata = () => {
            captureCanvas.width = cameraVideo.videoWidth;
            captureCanvas.height = cameraVideo.videoHeight;
        };

    } catch (error) {
        showError(`Camera access denied: ${error.message}`);
        addLog('camera', 'Failed to access camera', 'error');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        autoDetectEnabled = false;
        autoDetectToggle.textContent = '🔄 Auto Detect: OFF';
        clearInterval(autoDetectInterval);
    }

    startCameraBtn.style.display = 'flex';
    stopCameraBtn.style.display = 'none';
    captureFrameBtn.style.display = 'none';
}

// ======================== AUTO DETECT ========================

autoDetectToggle.addEventListener('click', () => {
    if (!cameraStream) {
        showError('Camera not active. Start camera first.');
        return;
    }

    autoDetectEnabled = !autoDetectEnabled;
    autoDetectToggle.textContent = autoDetectEnabled ? '🔄 Auto Detect: ON' : '🔄 Auto Detect: OFF';

    if (autoDetectEnabled) {
        addLog('auto-detect', 'Auto detection enabled', 'success');
        startAutoDetection();
    } else {
        clearInterval(autoDetectInterval);
        addLog('auto-detect', 'Auto detection disabled', 'info');
    }
});

function startAutoDetection() {
    const interval = parseInt(detectionInterval.value);

    autoDetectInterval = setInterval(async () => {
        if (cameraVideo.readyState === cameraVideo.HAVE_ENOUGH_DATA) {
            captureAndPredict('auto');
            detectionFrameCount++;
            const fps = (1000 / interval).toFixed(1);
            footerFPS.textContent = fps;
        }
    }, interval);
}

// ======================== CAPTURE & PREDICT ========================

captureFrameBtn.addEventListener('click', () => {
    captureAndPredict('manual');
});

async function captureAndPredict(mode) {
    try {
        // Draw to canvas
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(cameraVideo, 0, 0);

        // Convert to blob
        captureCanvas.toBlob(async (blob) => {
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            await predictFromFile(file, mode);
        }, 'image/jpeg', 0.9);

    } catch (error) {
        showError(`Capture failed: ${error.message}`);
    }
}

// ======================== UPLOAD FUNCTIONS ========================

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    if (files.length === 0) return;

    const file = files[0];

    if (!file.type.startsWith('image/')) {
        showError('Invalid file type. Please select an image.');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;
    predictBtn.style.display = 'flex';

    uploadArea.innerHTML = `
        <div class="upload-icon">✓</div>
        <p class="upload-text">${file.name}</p>
        <p class="upload-hint">Ready for prediction</p>
    `;

    addLog('upload', `File selected: ${file.name}`, 'success');
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    await predictFromFile(selectedFile, 'manual');
});

// ======================== PREDICTION ========================

async function predictFromFile(file, mode) {
    try {
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
        loadingContainer.style.display = 'flex';

        const threshold = parseFloat(confidenceThreshold.value);
        const startTime = Date.now();
        const formData = new FormData();
        formData.append('file', file);

        addLog('prediction', `Executing neural network (${mode})...`, 'command');

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        processingTime = Date.now() - startTime;

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }

        // Check confidence threshold
        if (result.prediction.confidence < threshold && mode === 'auto') {
            loadingContainer.style.display = 'none';
            return; // Skip low confidence auto detections
        }

        displayResults(result.prediction, mode);
        addLog('prediction', `Classification complete - ${result.prediction.flower}`, 'success');

    } catch (error) {
        showError(error.message || 'Prediction failed');
        addLog('error', error.message, 'error');
    } finally {
        loadingContainer.style.display = 'none';
    }
}

// ======================== DISPLAY RESULTS ========================

function displayResults(prediction, mode) {
    detectionModeElement.textContent = mode === 'auto' ? 'Auto' : 'Manual';

    predictionFlower.textContent = prediction.flower.toUpperCase();

    const confidence = Math.round(prediction.confidence * 100);
    confidenceValue.textContent = confidence + '%';
    confidenceFill.style.width = confidence + '%';

    predictionsBody.innerHTML = '';
    prediction.top_3.forEach((pred, index) => {
        const percentage = Math.round(pred.probability * 100);
        const row = `
            <tr>
                <td>${index + 1}</td>
                <td>${pred.flower.toUpperCase()}</td>
                <td>${percentage}%</td>
            </tr>
        `;
        predictionsBody.innerHTML += row;
    });

    processingTimeElement.textContent = processingTime + 'ms';
    resultsSection.style.display = 'flex';

    confidenceFill.style.width = '0%';
    setTimeout(() => {
        confidenceFill.style.width = confidence + '%';
    }, 100);
}

// ======================== UI UTILITIES ========================

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
    loadingContainer.style.display = 'none';
}

function addLog(prefix, text, type = 'info') {
    const logLine = document.createElement('div');
    logLine.className = 'log-line';

    const prefixSpan = document.createElement('span');
    prefixSpan.className = 'prefix';
    prefixSpan.textContent = prefix + ':';

    const textSpan = document.createElement('span');
    textSpan.className = 'text';
    if (type === 'success') textSpan.classList.add('success');
    textSpan.textContent = text;

    logLine.appendChild(prefixSpan);
    logLine.appendChild(textSpan);

    commandLog.appendChild(logLine);
    commandLog.scrollTop = commandLog.scrollHeight;

    const lines = commandLog.querySelectorAll('.log-line');
    if (lines.length > 6) {
        lines[0].remove();
    }
}

clearBtn.addEventListener('click', () => {
    uploadArea.classList.remove('dragover');
    uploadArea.innerHTML = `
        <div class="upload-icon">📤</div>
        <p class="upload-text">Drag image here or click to upload</p>
        <p class="upload-hint">Supported: JPG, PNG, GIF, BMP (Max 16MB)</p>
    `;

    fileInput.value = '';
    selectedFile = null;
    predictBtn.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingContainer.style.display = 'none';

    commandLog.innerHTML = `
        <div class="log-line">
            <span class="prefix">system:</span>
            <span class="text">System reset. Ready for new input...</span>
        </div>
    `;

    addLog('system', 'Interface cleared', 'success');
});

// ======================== INITIALIZATION ========================

document.addEventListener('DOMContentLoaded', () => {
    addLog('system', 'Flower Classification System v2.0 Initialized');
    addLog('system', 'Select LIVE CAMERA or UPLOAD IMAGE mode...', 'success');
    checkAPIHealth();
});

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            addLog('health', 'API connection established', 'success');
        }
    } catch (error) {
        addLog('health', `Cannot connect to API. Make sure Flask server is running.`);
    }
}

// Prevent keyboard shortcuts from interfering
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && autoDetectEnabled) {
        autoDetectToggle.click();
    }
});
