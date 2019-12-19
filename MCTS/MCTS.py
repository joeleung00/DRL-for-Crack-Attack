import random
import sys
import math
from copy import copy, deepcopy
import sys
sys.path.insert(1, '../game_simulation')
from GameBoard import GameBoard
from GameCLI import Game
from Inferencer import Inferencer

#small_inferencer = Inferencer("/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/cnn/network/network4.pth", True)
#big_inferencer = Inferencer("/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/cnn/network/network3.pth", False)
#MAX_ROUND_NUMBER = 8
MAX_ROUND_NUMBER = 4
MAX_ROLLOUT_ROUND_NUMBER = 3
class Node:
    def __init__(self, state, parent = None):
        self.parent = parent
        self.child = []
        self.visited_times = 0
        self.quality_value = 0.0
        self.state = state


    def is_all_expanded(self):
        return len(self.child) == self.state.num_available_choices

    def add_child(self, node):
        self.child.append(node)


class State:
    def __init__(self, board, round_index, cumulative_choices, num_available_choices = 0, action_reward = 0):
        self.current_board = deepcopy(board)
        self.current_round_index = round_index
        self.cumulative_choices = deepcopy(cumulative_choices)
        self.num_available_choices = num_available_choices
        if len(cumulative_choices) != 0:
            self.choice_id = self.get_choice_id(cumulative_choices[-1][0], cumulative_choices[-1][1])
        else:
            self.choice_id = None
        self.action_reward = action_reward

    def is_terminal(self, rollout=False):
        ## Add one more case - check is there any possible move?
        if not rollout:
            max_round = MAX_ROUND_NUMBER
        else:
            max_round = MAX_ROLLOUT_ROUND_NUMBER

        if self.current_round_index == max_round:
            return True
        elif self.num_available_choices == 0:
            return True
        else:
            return False

    def get_choice_id(self, row, col):
        return row * 10 + col

    def compute_reward(self, simulation_board):
        return simulation_board.score

    # def set_num_available_choices(self):
    #     gameboard.board = self.board
    #     available_choices = gameboard.get_available_choices()
    #     self.num_available_choices = len(available_choices)

    def get_next_state_with_random_choice(self, simulation_board, exclude=None):
        ## AVAILABLE_CHOICES is a double integer tupple list
        available_choices = simulation_board.get_available_choices()
        random_choice = random.choice(available_choices)

        if exclude != None:
            while self.get_choice_id(random_choice[0], random_choice[1]) in exclude:
                random_choice = random.choice(available_choices)

        action_reward = simulation_board.proceed_next_state(random_choice[0], random_choice[1])
        available_choices = simulation_board.get_available_choices()
        next_state =  State(simulation_board.board, self.current_round_index + 1,
            self.cumulative_choices + [random_choice], len(available_choices), action_reward)

        return next_state

    def get_next_best_state(self, simulation_board):
        available_choices = simulation_board.get_available_choices()
        best_choice = available_choices[0]
        best_score = 0
        for choice in available_choices:
            tmp_board = GameBoard(simulation_board.board, simulation = True)
            score = tmp_board.proceed_next_state(choice[0], choice[1])
            if score > best_score:
                best_score = score
                best_choice = choice
        action_reward = simulation_board.proceed_next_state(best_choice[0], best_choice[1])
        available_choices = simulation_board.get_available_choices()
        next_state =  State(simulation_board.board, self.current_round_index + 1,
            self.cumulative_choices + [best_choice], len(available_choices), action_reward)

        return next_state

    def get_choice(self):
        return self.cumulative_choices[-1]


def tree_policy(node):
    # Check if the current node is the leaf node
    while node.state.is_terminal() == False:

        if node.is_all_expanded():
            node = best_child(node, True)
        else:
            # Return the new sub node
            sub_node = expand(node)
            #print(node.state.current_board)
            return sub_node

    # Return the leaf node
    return node


def default_policy(node):

    # Get the state of the game
    current_state = node.state
    #print(current_state.current_board)

    #############
    ## Then add the inferencing score
    # big_score = big_inferencer.inference(current_state.current_board)
    # small_score = small_inferencer.inference(current_state.current_board)
    # return big_score + small_score
    #############

    ############
    simulation_board = GameBoard(current_state.current_board, simulation = True)


    while current_state.is_terminal() == False:

        # Pick one random action to play and get next state
        current_state = current_state.get_next_state_with_random_choice(simulation_board)

    final_state_reward = current_state.compute_reward(simulation_board)


    return final_state_reward #+ additional_score
    ############

    # simulation_board = GameBoard(current_state.current_board, simulation = True)
    # while current_state.is_terminal() == False:
    #
    #     # Pick one random action to play and get next state
    #     current_state = current_state.get_next_best_state(simulation_board)
    #
    # final_state_reward = current_state.compute_reward(simulation_board)
    #
    # return final_state_reward





def expand(node):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """
    child_node_state_set = set()
    for sub_node in node.child:
        child_node_state_set.add(sub_node.state.choice_id)

    # tried_sub_node_states = [
    #   sub_node.state for sub_node in node.child
    # ]
    simulation_board = GameBoard(node.state.current_board, simulation = True)

    new_state = node.state.get_next_state_with_random_choice(simulation_board, exclude=child_node_state_set)

    # Check until get the new state which has the different action from others
    # while new_state in tried_sub_node_states:
    #     new_state = node.state.get_next_state_with_random_choice()

    sub_node = Node(parent=node, state=new_state)
    node.add_child(sub_node)

    return sub_node


def best_child(node, is_exploration):
  """
  使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
  """

  # TODO: Use the min float value
  best_score = -sys.maxsize
  best_sub_node = None

  # Travel all sub nodes to find the best one
  for sub_node in node.child:

    # Ignore exploration for inference
    if is_exploration:
      C = 1 / math.sqrt(2.0)
      # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
      left = sub_node.quality_value / sub_node.visited_times
      right = 2.0 * math.log(node.visited_times) / sub_node.visited_times
      score = left + C * math.sqrt(right)
    else:
      score = sub_node.visited_times
      #score = sub_node.quality_value

    if score > best_score:
      best_sub_node = sub_node
      best_score = score

  return best_sub_node


def backup(node, reward):

    # Update util the root node
    while node != None:
        #  Update the visit times
        node.visited_times += 1

        # Update the quality value
        total_reward = reward + node.state.action_reward
        if total_reward > node.quality_value:
            node.quality_value = total_reward

        reward = node.quality_value
        # Change the node to the parent node
        node = node.parent



def monte_carlo_tree_search(node):

    computation_budget = 150

    # Run as much as possible under the computation budget
    for i in range(computation_budget):

        #print(node.state.current_board)
        # 1. Find the best node to expand
        expand_node = tree_policy(node)
        # 2. Random run to add node and get reward
        reward = default_policy(expand_node)
        # 3. Update all passing nodes with reward

        backup(expand_node, reward)

    # N. Get the best next node
    best_next_node = best_child(node, False)
    print("my quality_value :" + str(node.quality_value))

    return best_next_node

def get_best_child(node):
    best_quality_value = 0
    best_child = None
    for child in node.child:
        if child.quality_value > best_quality_value:
            best_quality_value = child.quality_value
            best_child = child
    return best_child

if __name__ == "__main__":
    gameplay = Game()

    num_available_choices = len(gameplay.gameboard.get_available_choices())
    init_state = State(gameplay.gameboard.board, 0, [], num_available_choices)
    root_node = Node(state=init_state)
    current_node = root_node

    gameplay.gameboard.print_board()

    for i in range(30):
        if i % MAX_ROUND_NUMBER == 0:
            num_available_choices = len(gameplay.gameboard.get_available_choices())
            init_state = State(gameplay.gameboard.board, 0, [], num_available_choices)
            root_node = Node(state=init_state)
            current_node = root_node

        previous_node = current_node
        current_node = monte_carlo_tree_search(current_node)
        if current_node == None:
            print("No good move after round:", str(i))
            break
        choice = current_node.state.get_choice()
        print("You have choosen : " + str(choice[0]) + " " + str(choice[1]))
        gameplay.input_pos(choice[0], choice[1])
        if gameplay.gameboard.score >= 30 and gameplay.gameboard.score < 40:
            print("Score per round: %f" % (gameplay.gameboard.score / i))

        if gameplay.gameboard.score >= 40 and gameplay.gameboard.score < 50:
            print("Score per round: %f" % (gameplay.gameboard.score / i))

        if gameplay.gameboard.score >= 50 and gameplay.gameboard.score < 60:
            print("Score per round: %f" % (gameplay.gameboard.score / i))

        if gameplay.gameboard.score >= 60 or i == 69:
            print("Score per round: %f" % (gameplay.gameboard.score / i))
            break
