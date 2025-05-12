from PIL import Image
from zenml import step

@step
def dynamic_importer(file_path: str) -> Image:
    """
    Should load the image file and return the image as PIL.Image.

    :param file_path: directory for a image file that was uploaded by a user
    :return: image as PIL.Image
    """
    if not file_path:
        raise ValueError("No file path provided.")
    if not file_path.endswith(".png"):
        raise ValueError("File path must point to a .png file.")

    return Image.open(file_path)