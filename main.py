import logging
from typing import Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import torch
from PIL import Image
from torchvision import transforms
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model (add error handling if the file doesn't exist)
local_download_path = './mode_full_download.pt'
try:
    model = torch.load(local_download_path, map_location=torch.device('cpu'))
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Failed to load the model: %s", e)
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

# Image transformations (consider using a more suitable image size for your model)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI()

# Class names (replace with your actual class labels)
class_names = ['2', '3', '4', '5', '6', '7']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            logging.warning("Invalid file type: %s", file.content_type)
            raise HTTPException(status_code=400, detail="File must be an image")

        img = Image.open(file.file)
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = model(batch_t)

        _, predicted_idx = torch.max(out, 1)
        logging.info("Prediction successful id: %s", predicted_idx)
        if predicted_idx.item() > len(class_names) - 1:
            logging.error("Prediction index out of bounds: %s", predicted_idx.item())
            predicted_class = "Unknown"
        else:
            predicted_class = class_names[predicted_idx.item()]
            logging.info("Prediction successful class: %s", predicted_class)

        return JSONResponse({"predicted_class": predicted_class})

    except Exception as e:
        logging.error("Error predicting image: %s", e)
        raise HTTPException(status_code=500, detail=f"Error predicting image: {e}")


@app.get("/")
def read_root():
    logging.info("Root endpoint called.")
    return {"Hello": "World"}
