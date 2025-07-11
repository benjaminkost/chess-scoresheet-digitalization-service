from src.ml.scripts_for_steps.postprocessing_strategy import PostprocessingStrategy

def postprocessing_prediction(list_of_predictions: list) -> list:
    postprocessing_strategy = PostprocessingStrategy()

    pgn_string = postprocessing_strategy.turn_list_of_text_into_pgn(list_of_predictions)

    return pgn_string