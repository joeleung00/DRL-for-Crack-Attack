import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, '../MCTS')
from GameCLI import Game
from MCTS import *
from multiprocessing import Process, Lock, Value

NUM_OF_TEST_CASE = 100
ROUND_PER_EPISODE = 20
PENALTY_PER_STEP = 1
NUM_OF_THREAD = 2

lock = Lock()
total_score = Value('d', 0.0)
total_test_case = Value('d', 0.0)

def play(total_score, total_test_case):
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
    total_test_case.value += thread_test_case
    total_score.value += thread_total_score
    lock.release()

processes = []
for i in range(NUM_OF_THREAD):
    processes.append(Process(target = play, args=(total_score, total_test_case)))
    processes[i].start()

for i in range(NUM_OF_THREAD):
    processes[i].join()

print("Average score per test: " + str(total_score.value / total_test_case.value))
