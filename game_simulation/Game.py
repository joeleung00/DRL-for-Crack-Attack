import random
import pygame
import time
from pygame.locals import *
from sys import exit

from GameBoard import GameBoard


cell_size = 30
colors = [
(0,0,255),
(255,255,0),
(0, 128, 0),
(255, 128, 0),
(128, 0, 128),
(0, 0, 0)
]

FPS = 30

class Game:

    def __init__(self):
        self.stable_mode = True
        self.gameboard = GameBoard()
        pygame.init()
        self.testFont = pygame.font.Font('freesansbold.ttf', 20)
        self.screen = pygame.display.set_mode(((self.gameboard.column_dim+5)*cell_size, self.gameboard.row_dim*cell_size), 0, 32)

    def draw_screen(self,board,offset):
        self.screen.fill(pygame.Color(0,0,0))

        for y, row in enumerate(board):
            for x, val in enumerate(row):
                pygame.draw.rect(self.screen,colors[val],pygame.Rect(x *cell_size,+ y *cell_size-offset,cell_size,cell_size), 0)

        for x in range(self.gameboard.column_dim):
            pygame.draw.rect(self.screen, colors[self.gameboard.new_row[x]],pygame.Rect(x * cell_size,self.gameboard.row_dim*cell_size-offset, cell_size, offset), 0)

        scoreObj = self.testFont.render("Score: %d" % self.gameboard.score, False, (255, 255, 255))
        self.screen.blit(scoreObj, (200, 200))

    def insert_row_to_board(self):

        self.gameboard.board.remove(self.gameboard.board[0])
        self.gameboard.board.append(self.gameboard.new_row)

    def offset_increment(self, offset, score, frame_index):
        if score < 20:
            ## It take 15 second to add the whole row
            new_row_time = 15
        elif score < 30:
            new_row_time = 10
        elif score < 50:
            new_row_time = 7
        else:
            new_row_time = 5

        frame_dist = int (new_row_time * FPS / cell_size)
        if frame_index % frame_dist == 0:
            return (offset + 1) % cell_size
        else:
            return offset % cell_size


    def run(self):
        #stable means the board does not move

        #gameboard.new_row=self.get_new_row(self.gameboard.board)
        offset = 0
        gameover = False
        frame_index = 0
        clock = pygame.time.Clock()
        while not gameover:
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        exit()

            if self.gameboard.score > 10:
                self.stable_mode = False


            if not self.stable_mode:
                offset = self.offset_increment(offset, self.gameboard.score, frame_index)
                #offset = (offset + 1) % cell_size
                if offset == 0:
                    self.insert_row_to_board()
                    # BUG: fixing the 3 color but not eliminate
                    self.gameboard.proceed_next_state()
                    self.gameboard.get_new_row()
                    offset += 1

            pressed = pygame.mouse.get_pressed()

            self.draw_screen(self.gameboard.board, offset)

            if(pressed[0]):
                mouse_x,mouse_y=pygame.mouse.get_pos()
                #pygame.display.set_caption(str(mouse_x)+" "+str(mouse_y)+" "+str(int(mouse_x/30))+" "+str(int(mouse_y/30)))
                self.gameboard.proceed_next_state(int((mouse_y + offset) / cell_size),int(mouse_x/cell_size))
                pygame.time.wait(200)

            if(self.gameboard.board[0]!=self.gameboard.empty_row):
                gameover_text = self.testFont.render("Game Over", False, (255, 255, 255))
                self.screen.blit(gameover_text, (200, 250))
                gameover = True

            pygame.display.update()
            if gameover:
                pygame.time.wait(2000)

            clock.tick(FPS)
            frame_index = (frame_index + 1) % (FPS * 10)


if __name__ == "__main__":
    game = Game()
    game.run()


