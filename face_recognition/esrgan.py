import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('./models_ia/esrgan.pth', download=True)

path_to_image = './SegmentacionDocumento.jpg'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('./sr_cedula.jpg')
