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

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo
4. Set `app.py` as the main file
5. Deploy! You'll get a public URL like `https://yourapp.streamlit.app`

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
