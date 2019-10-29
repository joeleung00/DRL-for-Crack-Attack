import random
import pygame
import time
from pygame.locals import *
from sys import exit

pygame.init()

EMPTY = -1
BLUE = 0
YELLOW = 1
GREEN = 2
ORANGE = 3
PURPLE = 4
MARKED = 12
cell_size = 30
colors = [
(0,0,255),
(255,255,0),
(0, 128, 0),
(255, 128, 0),
(128, 0, 128),
(0, 0, 0)
]

#class Game:

#    def __init__(self):
#        self.gameboard = GameBoard()
#        self.gameboard.init_board()

#    def wait_input(self):
#        self.gameboard.print_board()
#        while True:
#            x = input()
#            y = input()
#            #self.gameboard.proceed_next_state(int(x), int(y))
#            self.gameboard.print_board()
#            pygame.display.update()

class Game:
    def __init__(self):
        self.score = 0
        self.row_dim = 12
        self.column_dim = 6
        self.blocks_init_height = 8
        self.num_of_color = 5
        self.board = [[-1 for i in range(self.column_dim)] for j in range(self.row_dim)]
        self.new_row=[None]*self.column_dim
        self.init_board()
        self.testFont = pygame.font.Font('freesansbold.ttf', 20)
        self.screen = pygame.display.set_mode(((self.column_dim+5)*cell_size, self.row_dim*cell_size), 0, 32)

    def print_board(self):
        for i in range(self.row_dim):
            for j in range(self.column_dim):
                print(self.board[i][j], end=" ")
            print()

    def get_random_block(self, row, col):
        result = None
        if (row == self.row_dim - 1 and col == 0):
            result = random.randint(0, self.num_of_color - 1)
        elif (row == self.row_dim - 1):
            tmp = random.randint(0, self.num_of_color - 1)
            if (tmp == self.board[row][col - 1]):
                result = (tmp + 1) % self.num_of_color
            else:
                result = tmp
        elif (col == 0):
            tmp = random.randint(0, self.num_of_color - 1)
            if (tmp == self.board[row + 1][col]):
                result = (tmp + 1) % self.num_of_color
            else:
                result = tmp
        else:
            while True:
                tmp = random.randint(0, self.num_of_color - 1)
                if ((tmp != self.board[row + 1][col]) and (tmp != self.board[row][col - 1])):
                    result = tmp
                    break
        return result

    def init_board(self):
        for row in range(self.row_dim - 1, self.row_dim - self.blocks_init_height, -1):
            for col in range(self.column_dim):
                self.board[row][col] = self.get_random_block(row, col)

    def swap(self, row, col):
        # Out of bound:
        if row < 0 or row >= self.row_dim or col < 0 or col > (self.column_dim - 2):
            return False
        # Empty position:
        #if self.board[row][col] == -1:
        #    return False
        self.board[row][col + 1], self.board[row][col] = self.board[row][col], self.board[row][col + 1]

        return True

    def gravity(self):
        for row in range(self.row_dim - 2, -1, -1):
            for col in range(self.column_dim):
                block = self.board[row][col]
                cur_row = row
                cur_col = col
                if block != EMPTY:
                    while cur_row < self.row_dim - 1 and self.board[cur_row + 1][cur_col] == EMPTY:
                         self.board[cur_row][cur_col] = EMPTY
                         self.board[cur_row + 1][cur_col] = block
                         cur_row += 1

    def mark_sub(self, i, start, end, marked_board, row_major):

        if row_major:
            if self.board[i][start] == EMPTY:
                return
            for j in range(start, end + 1):
                marked_board[i][j] = MARKED
        else:
            if self.board[start][i] == EMPTY:
                return
            for j in range(start, end + 1):
                marked_board[j][i] = MARKED

    def mark(self, marked_board):
        # row major scan:
        for row in range(self.row_dim):
            count = 1
            cur_color = self.board[row][0]
            for col in range(self.column_dim):
                if col == self.column_dim - 1:
                    if count >= 3:
                        self.mark_sub(row, col - count + 1, col, marked_board, True)
                else:
                    if self.board[row][col + 1] == cur_color:
                        count += 1
                    else:
                        if count >= 3:
                            self.mark_sub(row, col - count + 1, col, marked_board, True)
                        count = 1
                        cur_color = self.board[row][col + 1]

        # column major scan:
        for col in range(self.column_dim):
            count = 1
            cur_color = self.board[0][col]
            for row in range(self.row_dim):
                if row == self.row_dim - 1:
                    if count >= 3:
                        self.mark_sub(col, row - count + 1, row, marked_board, False)
                else:
                    if self.board[row + 1][col] == cur_color:
                        count += 1
                    else:
                        if count >= 3:
                            self.mark_sub(col, row - count + 1, row, marked_board, False)
                        count = 1
                        cur_color = self.board[row + 1][col]

    def elimination(self):
        count = 0
        marked_board = [[0 for i in range(self.column_dim)] for j in range(self.row_dim)]
        self.mark(marked_board)

        for row in range(self.row_dim):
            for col in range(self.column_dim):
                if marked_board[row][col] == MARKED:
                    self.board[row][col] = EMPTY
                    count += 1
        self.score=self.score+10*count
        return count

    def proceed_next_state(self, row, col):
        self.swap(row, col)
        self.gravity()
        while (True):
            num = self.elimination()
            self.gravity()
            if (num == 0):
                break

    def get_new_row(self,board):
        row=[None]*self.column_dim
        for i in range(self.column_dim):
            row[i] = random.randint(0, self.num_of_color - 1)
            while(board[self.row_dim - 1][i]==row[i] and board[self.row_dim - 2][i]==row[i]):
                row[i] = random.randint(0, self.num_of_color - 1)
            if(i>1):
                while(row[i]==row[i-1] and row[i-1]==row[i-2]):
                    row[i] = random.randint(0, self.num_of_color - 1)
        return row


        #self.board.remove(self.board[0])
        #self.board.append(row)

    def draw_screen(self,board,offset,while_count,done):
        self.screen.fill(pygame.Color(0,0,0))

        if(offset==0 and not(done)):
            if (while_count!=0):
                self.board.remove(self.board[0])
                self.board.append(self.new_row)
            self.new_row=self.get_new_row(self.board)

        for y, row in enumerate(board):
            for x, val in enumerate(row):
                pygame.draw.rect(self.screen,colors[val],pygame.Rect(x *cell_size,+ y *cell_size-offset,cell_size,cell_size), 0)

        for x in range(self.column_dim):
            pygame.draw.rect(self.screen, colors[self.new_row[x]],pygame.Rect(x * cell_size,self.row_dim*cell_size-offset, cell_size, offset), 0)

        scoreObj = self.testFont.render("Score: %d" % self.score, False, (255, 255, 255))
        self.screen.blit(scoreObj, (200, 200))

    def run(self):
        done = 0
        over = 0
        while_count=0
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        exit()

            pressed = pygame.mouse.get_pressed()

            if(int(while_count/cell_size)%cell_size-1==0):
                done=0

            self.draw_screen(self.board,int(while_count/cell_size)%cell_size,while_count,done)

            if (int(while_count / cell_size) % cell_size == 0):
                done = 1

            if(over):
                pygame.time.wait(1000)
                break

            if(pressed[0]):
                mouse_x,mouse_y=pygame.mouse.get_pos()
                #pygame.display.set_caption(str(mouse_x)+" "+str(mouse_y)+" "+str(int(mouse_x/30))+" "+str(int(mouse_y/30)))
                self.proceed_next_state(int((mouse_y+int(while_count/cell_size)%cell_size)/cell_size),int(mouse_x/cell_size))
                pygame.time.wait(150)

            if(self.board[0]!=[-1,-1,-1,-1,-1,-1]):
                over = self.testFont.render("Game Over", False, (255, 255, 255))
                self.screen.blit(over, (200, 250))
                over=1

            while_count=while_count+1
            pygame.display.update()

game = Game()
game.run()