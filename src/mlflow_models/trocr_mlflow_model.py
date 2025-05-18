import mlflow
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from src.classes_for_steps.postprocessing_strategy import PostprocessingStrategy
from src.classes_for_steps.preprocessing_strategy import HuggingFacePreprocessingStrategy, ThresholdMethod

class TrOCRBaseHandwrittenModelWrapper(mlflow.pyfunc.PythonModel):
    def predict(self, np_img):
        # Convert np_array to PIL
        input_image = Image.fromarray(np_img)

        # Preprocess image
        list_of_move_boxes = self.preprocess_image(input_image)

        # Initialize Model
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        # Run the prediction
        list_of_predictions = []
        for move_box in list_of_move_boxes:
            # convert images to RGB
            move_box = move_box.convert("RGB")

            # Generate pixel_values with processor
            pixel_values = processor(move_box, return_tensors="pt").pixel_values

            # Generate ids from model
            generated_ids = model.generate(pixel_values)

            # Decode the result from the model
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Add it to the list of predictions for the move boxes
            list_of_predictions.append(prediction)

        # Post-process prediction list
        pgn_str = self.post_process_prediction_list(list_of_predictions)

        return {"pgn_str": pgn_str}

    def preprocess_image(self, image) -> list:
        # load preprocessing strategy
        ## Best parameters tested with tuning: src/tuning/preprocessing/preprocessing_hyperparameter_tuning.py
        kernelsize_gaussianBlur = (5, 5)
        sigmaX = 0
        threshold_method = ThresholdMethod.OTSU
        maxValue_threshold = 255
        block_size = 9
        c_value = 1
        horizontal_kernel_divisor = 30
        vertical_kernel_divisor = 20
        erosion_iterations = 1
        dilation_iterations = 1

        preprocessing_strategy = HuggingFacePreprocessingStrategy(
            kernelsize_gaussianBlur=kernelsize_gaussianBlur,
            sigmaX=sigmaX,
            threshold_method=threshold_method,
            maxValue_threshold=maxValue_threshold,
            block_size=block_size,
            c_value=c_value,
            horizontal_kernel_divisor=horizontal_kernel_divisor,
            vertical_kernel_divisor=vertical_kernel_divisor,
            erosion_iterations=erosion_iterations,
            dilation_iterations=dilation_iterations
        )

        # Image into gray scale
        image_gray_scaled = image.convert("L")

        # Convert gray-scale image to binary image with otsu's method
        image_binary = preprocessing_strategy.process_image_gray_scaled_to_binary_with_threshold(image_gray_scaled)

        ## Generate image containing only grid lines
        image_only_grid_lines = preprocessing_strategy.process_binary_image_to_grid_lines(image_binary)

        ## Find contours in image with only grid lines
        list_of_contour_for_image = preprocessing_strategy.generate_binary_grid_image_to_list_of_contours(image_only_grid_lines)

        ## Cut out boxes with padding
        list_cut_out_move_boxes = []
        for contour in list_of_contour_for_image:
            cut_out_image = preprocessing_strategy.generate_from_four_contour_points_and_image_a_cut_out_image(contour, image)
            list_cut_out_move_boxes.append(cut_out_image)

        return list_cut_out_move_boxes

    def post_process_prediction_list(self, list_of_predictions) -> str:
        postprocessing_strategy = PostprocessingStrategy()

        pgn_string = postprocessing_strategy.turn_list_of_text_into_pgn(list_of_predictions)

        return pgn_string