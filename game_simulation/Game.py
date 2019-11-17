from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import random
import pygame
import time
from pygame.locals import *
from sys import exit
import sys
import sys
sys.path.insert(1, '../MCTS')

from GameBoard import GameBoard
from MCTS import State,Node,monte_carlo_tree_search


cell_size = 30
colors = [
(0,0,255),
(255,255,0),
(0, 128, 0),
(255, 128, 0),
(128, 0, 128),
(220,220,220),
(255,0,0),
(0, 0, 0),
]

ASK_BOARD = "b"
ASK_ROUND = "r"
ASK_CURSOR = "c"
FPS = 30
input_string = ""

if len(sys.argv) != 2:
    print("enter your mode:")
    print("human or ai")
    exit(0)
mode = sys.argv[1]
if mode != "human" and mode != "ai":
    print("Undefined mode!!")
    exit(0)

class Game:

    def __init__(self):
        self.stable_mode = True
        self.gameboard = GameBoard()
        pygame.init()
        self.testFont = pygame.font.Font('freesansbold.ttf', 20)
        self.screen = pygame.display.set_mode(((self.gameboard.column_dim+5)*cell_size, self.gameboard.row_dim*cell_size), 0, 32)
        ## round means how many bottom rows are added
        self.round = 0

    def draw_screen(self,board,offset,cursor_pos):
        self.screen.fill(pygame.Color(0,0,0))

        for y, row in enumerate(board):
            for x, val in enumerate(row):
                pygame.draw.rect(self.screen,colors[val],pygame.Rect(x *cell_size,+ y *cell_size-offset,cell_size,cell_size), 0)

        for x in range(self.gameboard.column_dim):
            pygame.draw.rect(self.screen, colors[self.gameboard.new_row[x]],pygame.Rect(x * cell_size,self.gameboard.row_dim*cell_size-offset, cell_size, offset), 0)

        pygame.draw.rect(self.screen, pygame.Color(255,255,255),pygame.Rect(cursor_pos[1] * cell_size, + cursor_pos[0] * cell_size - offset, cell_size*2, cell_size), 5)

        scoreObj = self.testFont.render("Score: %d" % self.gameboard.score, False, (255, 255, 255))
        self.screen.blit(scoreObj,((self.gameboard.column_dim + 1) * cell_size, self.gameboard.row_dim * cell_size / 2))


    def print_board(self):
        for row in range(self.gameboard.row_dim):
            for col in range(self.gameboard.column_dim):
                print(self.gameboard.board[row][col], end=",")
            print(".", end="")
        print()

    def insert_row_to_board(self):
        self.round += 1
        self.gameboard.board.remove(self.gameboard.board[0])
        self.gameboard.board.append(self.gameboard.new_row)
        self.gameboard.cursor_pos[0] -= 1
        self.gameboard.height += 1

    def offset_increment(self, offset, score, frame_index):
        # if score < 20:
        #     ## It take 15 second to add the whole row
        #     new_row_time = 15
        # elif score < 30:
        #     new_row_time = 10
        # elif score < 50:
        #     new_row_time = 7
        # else:
        #     new_row_time = 5
        new_row_time = 3

        frame_dist = int (new_row_time * FPS / cell_size)
        if frame_index % frame_dist == 0:
            return (offset + 1) % cell_size
        else:
            return offset % cell_size

    def get_cursor_pos(self,cursor_pos,key):
        #pygame.time.wait(70)
        if(key == "w" and cursor_pos[0] != 0):
            cursor_pos[0] = cursor_pos[0] - 1
        elif(key == "s" and cursor_pos[0] != self.gameboard.row_dim-1):
            cursor_pos[0] = cursor_pos[0] + 1
        elif(key == "d" and cursor_pos[1] != self.gameboard.column_dim-2):
            cursor_pos[1] = cursor_pos[1] + 1
        elif (key == "a" and cursor_pos[1] != 0):
            cursor_pos[1] = cursor_pos[1] - 1

        return cursor_pos

    def run(self):
        #stable means the board does not move

        #gameboard.new_row=self.get_new_row(self.gameboard.board)
        swap_count  = 0
        offset = 0
        gameover = False
        frame_index = 0
        clock = pygame.time.Clock()
        global input_string
        choice = None
        while not gameover:
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit(0)
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        exit(0)

                    if mode == "ai":
                        if event.unicode == ".":
                            choice = input_string.split(",")
                            input_string = ""
                        elif event.unicode == ASK_BOARD:
                            self.print_board()
                        elif event.unicode == ASK_ROUND:
                            print(self.round)
                        elif event.unicode == ASK_CURSOR:
                            print(self.gameboard.cursor_pos[0], self.gameboard.cursor_pos[1])
                        else:
                            input_string += event.unicode

                        if choice != None:
                            self.gameboard.proceed_next_state(int(choice[0]), int(choice[1]))
                            choice = None




                    #pressed_keys = pygame.key.get_pressed()
                    pressed_key = event.unicode
                    self.get_cursor_pos(self.gameboard.cursor_pos,pressed_key)
                    if(pressed_key == " "):
                        self.gameboard.proceed_next_state(self.gameboard.cursor_pos[0], self.gameboard.cursor_pos[1])
                        swap_count += 1
                        #pygame.time.wait(200)


            if self.gameboard.score > 10 and self.stable_mode == True:
                self.stable_mode = False
                offset += 1


            if not self.stable_mode:
                offset = self.offset_increment(offset, self.gameboard.score, frame_index)
                #offset = (offset + 1) % cell_size
                if offset == 0:
                    self.insert_row_to_board()
                    # BUG: fixing the 3 color but not eliminate
                    self.gameboard.proceed_next_state()
                    self.gameboard.get_new_row()
                    offset += 1




                ## read the action queue

                ## pass the action to proceed_next_state



            self.draw_screen(self.gameboard.board, offset,self.gameboard.cursor_pos)


            if(self.gameboard.board[0]!=self.gameboard.empty_row):
                gameover_text = self.testFont.render("Game Over", False, (255, 255, 255))
                self.screen.blit(gameover_text, ((self.gameboard.column_dim + 1) * cell_size, (self.gameboard.row_dim * cell_size+1) / 2))
                gameover = True

            pygame.display.update()
            if gameover:
                pygame.time.wait(2000)
                print("performance: %.2f" % (self.gameboard.score / swap_count))


            clock.tick(FPS)
            frame_index = (frame_index + 1) % (FPS * 10)


if __name__ == "__main__":
    game = Game()
    game.run()
