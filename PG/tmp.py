import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, '../MCTS')
from GameCLI import Game
from MCTS import *
import threading

NUM_OF_TEST_CASE = 10
ROUND_PER_EPISODE = 20
PENALTY_PER_STEP = 1
NUM_OF_THREAD = 4

lock = threading.Lock()
total_score = 0
total_test_case = 0

def play():
    global total_test_case
    global total_score
    thread_total_score = 0
    thread_test_case = NUM_OF_TEST_CASE // NUM_OF_THREAD
    for i in range(thread_test_case):
        game = Game(show=False)
        for j in range(ROUND_PER_EPISODE):
            ## pick an action from stategy
            if j % 3 == 0:
                gameboard = GameBoard(game.gameboard.board)
                num_available_choices = len(gameboard.get_available_choices())
                init_state = State(gameboard.board, 0, [], num_available_choices)
                root_node = Node(state=init_state)
                current_node = root_node

            current_node = monte_carlo_tree_search(current_node)
            x, y = current_node.state.get_choice()

            ##########################
            game.input_pos(x, y)
        thread_total_score += game.gameboard.score - ROUND_PER_EPISODE * PENALTY_PER_STEP

    lock.acquire()
    total_test_case += thread_test_case
    total_score += thread_total_score
    lock.release()

threads = []
for i in range(NUM_OF_THREAD):
    threads.append(threading.Thread(target = play))
    threads[i].start()

for i in range(NUM_OF_THREAD):
    threads[i].join()

print("Average score per test: " + str(total_score/total_test_case))
