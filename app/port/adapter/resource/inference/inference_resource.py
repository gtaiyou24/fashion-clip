import fastapi
import torch
from PIL import Image
from fastapi import APIRouter

from domain.model.clip import ClipModel, normalize_text
from port.adapter.resource.inference.request import InvocationsRequest, Mode

router = APIRouter(prefix='/invocations', tags=['推論'])


@router.post('', name='推論エンドポイント')
def invocations(invocations: InvocationsRequest, request: fastapi.Request):
    clip: ClipModel = request.app.clip
    if invocations.is_(Mode.ImageEmbeddings):
        images = [Image.open(image) for image in invocations.image_bytes_list()]
        return clip.vision_model.encode_image(images).tolist()
    elif invocations.is_(Mode.TextEmbeddings):
        texts = [normalize_text(text) for text in invocations.texts]
        return clip.text_model.encode_text(texts).tolist()
    elif invocations.is_(Mode.ImageClassification):
        logits_per_image, logits_per_text = clip.encode(
            [Image.open(image) for image in invocations.image_bytes_list()],
            [normalize_text(text) for text in invocations.texts]
        )

        response = []
        for logit in torch.softmax(logits_per_image, dim=1):
            scores = {}
            for sim, text in zip(logit, invocations.texts):
                scores[text] = float(sim)
            response.append(scores)
        return response
