# Traffic Maadi - Demo

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
The app will open at http://localhost:8501

## Features

- **Violation Detection**: Upload traffic images → get annotated results with detected violations
- **Analytics Dashboard**: Real-time statistics, charts, hotspot mapping
- **System Architecture**: Visual explanation of the full pipeline
- **About**: Problem statement, solution overview, impact metrics

## Detection Modes

- **Real Mode**: If `ultralytics` and `easyocr` are installed, uses actual YOLOv8 + EasyOCR
- **Simulation Mode**: Falls back to realistic mock detections if models aren't available

Deployed app: `https://gridlock-3peasinapod.streamlit.app/`

## Sample Images for Testing

Search for "Indian traffic" images or use:
- Bengaluru traffic junction photos
- Two-wheeler/motorcycle on Indian roads
- Traffic signal/intersection images

## Tech Stack

- Streamlit (Frontend)
- YOLOv8 / Ultralytics (Object Detection)
- EasyOCR (License Plate Reading)
- OpenCV (Image Processing)
- Plotly (Charts & Analytics)
