# load torch model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


# laod image
from PIL import Image

img = Image.open('0-3_17.jpg')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_t = transform(img)

# predict

loaded_model = torch.load('mode_full.pt', map_location=torch.device('cpu'))
loaded_model.eval()

batch_t = torch.unsqueeze(img_t, 0)
out = loaded_model(batch_t)
_, predicted = torch.max(out.data, 1)
print(predicted)
