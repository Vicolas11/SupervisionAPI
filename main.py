from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from inference import get_model
import supervision as sv
from typing import Dict
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = get_model("yolov8s-640")

def callback(image_slice: np.ndarray) -> sv.Detections:
    results = model.infer(image_slice)[0]
    return sv.Detections.from_inference(results)

slicer = sv.InferenceSlicer(
    callback=callback,
    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION,
)

@app.post("/detect_objects")
async def detect_objects(request: Request) -> Dict:
    try:
        data = await request.json()
        width = data['width']
        height = data['height']
        image_data = np.array(data['data'], dtype=np.uint8).reshape((height, width, 4))  # Assuming RGBA format

        # Convert RGBA to RGB if needed
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)

        # Perform inference to get detections
        detections = slicer(image_data)
        class_name = detections.data.get('class_name')

        # Convert detections to a JSON-serializable format
        detections_dict = {
            'xyxy': detections.xyxy.tolist(),
            'confidence': detections.confidence.tolist(),
            'class_id': detections.class_id.tolist(),
            'class_name': class_name.tolist() if class_name is not None else []
        }

        return JSONResponse(content=detections_dict)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root() -> str:
    html_content = """
    <html>
    <head>
        <title>Server Status</title>
    </head>
    <body>
        <h1><center><b>Server is ready 😋</b></center></h1>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
