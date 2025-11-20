StrayWatch – AI-Based Stray Dog Monitoring & Surveillance System

StrayWatch is an AI-driven website designed to detect and monitor stray dog behavior in real-time for enhanced public safety and urban animal management. It classifies their emotional state
(angry / happy / sad / relaxed), estimate distance, log detections with GPS coordinates, and visualize them on an interactive heatmap surveillance dashboard.

It uses:
- YOLOv8 → Dog detection
- MobileNetV2 → Emotion classification
- FastAPI → Backend
- SQLite → Local detection database
- HTML/JS → Web dashboard (camera + upload + map)

🚀 Features:

📷 Live Camera Detection
- Runs YOLO + Emotion classifier on webcam feed
- Logs every detection with GPS coordinates
- Streams results to the frontend

🗺️ Surveillance Dashboard
- Leaflet Mpas API based interactive heatmap
- Shows all logged detections
- Filter by emotion: angry / happy / sad / relaxed
- Click markers to view thumbnail, confidence, distance, timestamp

📤 Upload Image/Video:
- Upload any photo/video
- AI detects dogs & classifies emotions
- Large popup-style result UI

🗃️ SQLite Database Logging-
Stores latitude, longitude, emotion, confidence, distance, thumbnail path, timestamp

📦 ML Models & Files-
Large files are NOT stored in the repo (GitHub limits big files).
Instead, download them from the Releases section.
Link: GitHub → Releases → v1.0.0 (StrayWatch ML Model Releases)
Included in Release-
- yolov8n.pt: YOLOv8 model for dog detection
- mobilenetv2_final.h5: Final trained emotion classifier
- mobilenetv2_final.onnx: ONNX version of MobileNet for deployment
- best_head.h5: Trained classifier head (before fine-tuning)
- best_finetuned.h5: Fine-tuned MobileNet model
- surveillance.db: Sample detection database with entries
