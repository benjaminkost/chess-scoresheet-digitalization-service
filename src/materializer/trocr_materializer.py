from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Tuple, Type
from pathlib import Path

from zenml.materializers.base_materializer import BaseMaterializer


class TrOCRMaterializer(BaseMaterializer):
    def save(self, obj: Tuple[TrOCRProcessor, VisionEncoderDecoderModel]) -> None:
        processor, model = obj
        path = Path(self.uri)
        processor.save_pretrained(path/"processor")
        model.save_pretrained(path/"model")

    def load(self, data_type: Type) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        path = Path(self.uri)
        processor = TrOCRProcessor.from_pretrained(path/"processor")
        model = VisionEncoderDecoderModel.from_pretrained(path/"model")
        return processor, model