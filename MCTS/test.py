import importlib.util

spec = importlib.util.spec_from_file_location("game", "../game_simulation/game.py")
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)

gb = game.GameBoard()
gb.print_board()
