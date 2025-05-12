from typing import Tuple
from zenml import step
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.materializer.trocr_materializer import TrOCRMaterializer


@step(output_materializers={"model": TrOCRMaterializer,
                            "processor": TrOCRMaterializer
                            })
def model_and_processor_loader(hf_owner: str, hf_model_name: str) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """
    Loads the current production model pipeline from Hugging Face.

    :param hf_owner: owner of the model
    :param hf_model_name: name of the model
    :return: loaded a model and processor
    """

    # Load model
    model = VisionEncoderDecoderModel.from_pretrained(f"{hf_owner}/{hf_model_name}")
    processor = TrOCRProcessor.from_pretrained(f"{hf_owner}/{hf_model_name}")

    return model, processor