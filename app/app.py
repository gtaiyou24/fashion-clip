import os
import pickle
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from domain.model.clip import ClipModel
from port.adapter.resource.ping import ping_resource
from port.adapter.resource.inference import inference_resource


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.clip = ClipModel(os.getenv('CLIP_MODEL_PATH', 'sonoisa/clip-vit-b-32-japanese-v1'), device=device)
    yield
    app.clip = None


app = FastAPI(title="Fashion CLIP", openapi_prefix=os.getenv('OPENAPI_PREFIX'), lifespan=lifespan)

app.include_router(ping_resource.router)
app.include_router(inference_resource.router)
