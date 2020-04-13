import sys
sys.path.insert(1, '../game_simulation')
from GameBoard import GameBoard
filepath = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/TD_learning/initboard"
gameboard = GameBoard(filename = filepath)
gameboard.print_board()
