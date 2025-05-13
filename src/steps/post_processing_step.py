from zenml import step

from src.classes_for_steps.postprocessing_strategy import PostprocessingStrategy

@step
def post_process_prediction_list(list_of_predictions: list) -> str:
    postprocessing_strategy = PostprocessingStrategy()

    pgn_string = postprocessing_strategy.turn_list_of_text_into_pgn(list_of_predictions)

    return pgn_string