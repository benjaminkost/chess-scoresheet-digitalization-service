class PostprocessingStrategy:
    def turn_list_of_text_into_pgn(self, list_of_text):
        pgn_text = "[Event \"?\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?^\"]\n[Result \"*\"]\n\n"

        move_count = 1

        for index, move in enumerate(list_of_text):
            if index % 2 == 0:
                pgn_text+= f"{move_count}. " + f"{move} "
                move_count+=1
            else:
                pgn_text += f"{move} "

        return pgn_text

postprocessing = PostprocessingStrategy()

list_text = ["e4", "e5", "d4", "Sc6", "Sc3"]

pgn_text = postprocessing.turn_list_of_text_into_pgn(list_text)

print(pgn_text)