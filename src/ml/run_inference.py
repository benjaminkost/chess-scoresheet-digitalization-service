from PIL import Image
import os
from src.ml.pipelines.inference_pipeline import inference_pipeline

def run_main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uploads_folder = os.path.abspath(os.path.join(dir_path, os.pardir))

    # Define file path
    # TODO: correct file fetch
    upload_files = os.listdir(uploads_folder)
    file_path = upload_files[-1]

    # Define model name
    model_name = "trocr-base-handwritten-with-pre-and-post-processing"

    # get prediction pgn string
    pgn_str = inference_pipeline(file_path, model_name)

    # Show image and pgn prediction
    file = Image.open(file_path)
    file.show()
    print(pgn_str)

if __name__ == "__main__":
    run_main()
