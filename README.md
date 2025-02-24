﻿# Roboflow Supervision FastAPI
This FastAPI application provides two main endpoints:
1. **Root Endpoint ("/"):** A GET endpoint that returns an HTML response indicating the server is ready, with a centered bold message "Server is ready 😋".
2. **Detect Objects Endpoint ("/detect_objects"):** A POST endpoint that accepts JSON data containing image dimensions (width, height) and pixel data. It processes the image using a YOLOv8 model for object detection, converts the detections to a JSON-serializable format, and returns the results. The endpoint supports CORS for cross-origin requests and includes error handling for invalid inputs.
