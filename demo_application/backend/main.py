from __future__ import annotations
import sys, types
from torchvision.transforms.functional import rgb_to_grayscale

m = types.ModuleType("torchvision.transforms.functional_tensor")
m.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = m

import io
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from backend_model.generator import Generator
from backend_model import globals

print("Initializing generator...")
G_EMA = Generator().to(globals.DEVICE)
checkpoint = torch.load("backend_model/weights/ada_stylegan_64_more_channels.pth", map_location=torch.device('cpu'))
G_EMA.load_state_dict(checkpoint["G_EMA_state_dict"])
G_EMA.fade_in(64)
G_EMA.set_layer_opacity(1.0)
print("Generator initialization complete!")

print("Initializing upsampler")
UPSAMPLER: RealESRGANer | None = None
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
UPSAMPLER = RealESRGANer(scale=4, model_path="backend_model/weights/RealESRGAN_x4plus.pth", model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
print("Upsampler initialization complete")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_UPSAMPLED_NP_IMG = None

@app.post("/generate")
def generate():
    global G_EMA
    global LAST_UPSAMPLED_NP_IMG

    z = torch.randn(1, globals.Z_DIM, device=globals.DEVICE)
    with torch.no_grad():
        img = G_EMA(z)[0]

    img = (img + 1) / 2.0
    img_np = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    LAST_UPSAMPLED_NP_IMG = img_np
    pil_img = Image.fromarray(img_np, mode="RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return Response(content=png_bytes, media_type="image/png")

@app.post("/upsample")
def upsample():
    global LAST_UPSAMPLED_NP_IMG, UPSAMPLER

    img_bgr = cv2.cvtColor(LAST_UPSAMPLED_NP_IMG, cv2.COLOR_RGB2BGR)

    out_bgr, _ = UPSAMPLER.enhance(img_bgr, outscale=4)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    
    LAST_UPSAMPLED_NP_IMG = out_rgb

    pil_img = Image.fromarray(out_rgb, mode="RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

    