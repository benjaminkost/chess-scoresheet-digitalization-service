import chess.pgn


class PostprocessingStrategy:
    def turn_list_of_text_into_pgn(self, list_of_text) -> chess.pgn.Game:
        chess_game = chess.pgn.Game()

        return chess_game.add_line(list_of_text)