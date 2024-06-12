from typing import Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import torch
from PIL import Image
from torchvision import transforms

# Load model (you may want to add error handling if the file doesn't exist)
# model = torch.load('mode_full.pt', map_location=torch.device('cpu'))
print("Loading model")
url = "https://media.githubusercontent.com/media/BMukhtar/bananaApi/main/mode_full.pt"  # Replace with the actual URL
torch.hub.download_url_to_file('https://media.githubusercontent.com/media/BMukhtar/bananaApi/main/mode_full.pt', './mode_full_download.pt')

model = torch.load('./mode_full_download.pt', map_location=torch.device('cpu'))
model.eval()
print("Loading model done")

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
            raise HTTPException(status_code=400, detail="File must be an image")

        img = Image.open(file.file)
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = model(batch_t)

        _, predicted_idx = torch.max(out, 1)  
        predicted_class = class_names[predicted_idx.item()]

        return JSONResponse({"predicted_class": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting image: {e}")


@app.get("/")
def read_root():
    return {"Hello": "World"}

    