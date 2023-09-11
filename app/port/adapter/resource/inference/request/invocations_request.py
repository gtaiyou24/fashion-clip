import base64
from enum import Enum
from io import BytesIO
from typing import List

import requests
from pydantic import BaseModel, Field


class Mode(str, Enum):
    ImageClassification = 'image_classification'
    ImageEmbeddings = 'image_embeddings'
    TextEmbeddings = 'text_embeddings'


class InvocationsRequest(BaseModel):
    mode: Mode = Mode.ImageEmbeddings
    texts: List[str] = Field(title='テキストの一覧')
    images: List[str] = Field(title='画像データもしくは画像URLの一覧')

    def is_(self, mode: Mode) -> bool:
        return self.mode.value == mode.value

    def image_bytes_list(self) -> List[str]:
        data = []
        for image in self.images:
            if image[:8] == 'https://':
                image = requests.get(image, stream=True).raw
            else:
                image = BytesIO(base64.b64decode(image.encode('utf-8')))
            data.append(image)
        return data
