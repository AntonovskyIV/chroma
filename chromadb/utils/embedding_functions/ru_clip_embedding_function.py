import importlib
import logging
from typing import Optional, Union, cast

from chromadb.api.types import (Document, Documents, Embedding,
                                EmbeddingFunction, Embeddings, Image, Images,
                                is_document, is_image)

logger = logging.getLogger(__name__)


class RuCLIPEmbeddingFunction(EmbeddingFunction[Union[Documents, Images]]):
    def __init__(
        self,
        model_name: str = "ruclip-vit-base-patch32-384",
        device: Optional[str] = "cuda",
    ) -> None:
        try:
            import ruclip
        except ImportError:
            raise ValueError(
                "The ruclip python package is not installed. Please install it with `pip install ruclip`"
            )
        try:
            self._torch = importlib.import_module("torch")
        except ImportError:
            raise ValueError(
                "The torch python package is not installed. Please install it with `pip install torch`"
            )

        try:
            self._PILImage = importlib.import_module("PIL.Image")
        except ImportError:
            raise ValueError(
                "The PIL python package is not installed. Please install it with `pip install pillow`"
            )

        clip, processor = ruclip.load(model_name, device=device)
        self.processor = processor
        self.predictor = ruclip.Predictor(clip, processor, device, bs=8)

    def _encode_image(self, image: Image) -> Embedding:
        pil_image = self._PILImage.fromarray(image)
        return self.predictor.get_image_latents([pil_image]).tolist()[0]

    def _encode_text(self, text: Document) -> Embedding:
        return self.predictor.get_text_latents([text]).tolist()[0]

    def __call__(self, input: Union[Documents, Images]) -> Embeddings:
        embeddings: Embeddings = []
        for item in input:
            if is_image(item):
                embeddings.append(self._encode_image(cast(Image, item)))
            elif is_document(item):
                embeddings.append(self._encode_text(cast(Document, item)))
        return embeddings
