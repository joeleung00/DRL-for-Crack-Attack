import random
import sys
import math
from copy import copy, deepcopy
import sys
sys.path.insert(1, '../game_simulation')
from GameBoard import GameBoard
from GameCLI import Game


class Node:
    def __init__(self, state, parent = None):
        self.parent = parent
        self.child = []
        self.visited_times = 0
        ## quality value is from the perspective of the parent
        self.quality_value = 0.0
        self.state = state


    def is_all_expanded(self):
        return len(self.child) == self.state.num_available_choices

    def add_child(self, node):
        self.child.append(node)


class State:
    def __init__(self, board, choice, num_available_choices = 0, action_reward = 0):
        self.current_board = deepcopy(board)
        self.choice = choice ## choice means what action you take in order to get into this child state from parent state
        self.num_available_choices = num_available_choices
        self.action_reward = action_reward

    def is_terminal(self):
        return self.num_available_choices == 0
    
    def get_choice(self):
        return self.choice
    
    def get_next_state_with_random_choice(self, simulation_board, exclude=None):
        ## exclude is a list of actions e.g. [(1,2), (3,4), (5,3)]
        available_choices = simulation_board.get_available_choices()
        if exclude != None:
            for action in exclude:
                index = available_choices.index(action)
                del available_choices[index]
        
        random_choice = random.choice(available_choices)

        action_reward = simulation_board.proceed_next_state(random_choice[0], random_choice[1])
        available_choices = simulation_board.get_available_choices()
        next_state =  State(simulation_board.board, random_choice, len(available_choices), action_reward)

        return next_state


class MCTS:
    def __init__(self, init_gameboard):
        self.MAX_SEARCH_DEPTH = 15
        self.MAX_ROLLOUT_DEPTH = 3
        self.root_node = self.init_root_node(init_gameboard)
        self.current_node = self.root_node
        self.COMPUTATION_BUDGET = 60
        self.C = 1 / math.sqrt(2.0)
        self.gamma = 0.6
        
    def init_root_node(self, gameboard):
        num_available_choices = len(gameboard.get_available_choices())
        init_state = State(gameboard.board, None, num_available_choices)
        root_node = Node(state=init_state)
        return root_node
    

    def tree_policy(self, node):
        depth = 0
        while not node.state.is_terminal() and depth < self.MAX_SEARCH_DEPTH:
            if node.is_all_expanded():
                node = self.best_child(node, True)
                depth += 1
            else:
                sub_node = self.expand(node)

                return sub_node
        return node


    def default_policy(self, node):

        current_state = node.state

#         simulation_board = GameBoard(current_state.current_board, simulation = True)

#         depth = 0
#         while not current_state.is_terminal() and depth < self.MAX_ROLLOUT_DEPTH:

#             # Pick one random action to play and get next state
#             current_state = current_state.get_next_state_with_random_choice(simulation_board)
#             depth += 1

#         final_state_reward = simulation_board.score
        
#         return final_state_reward

        simulation_board = GameBoard(current_state.current_board, simulation = True)
        total_score = 0
        for i in range(self.MAX_ROLLOUT_DEPTH):
            board = simulation_board.board
            available_choice = simulation_board.get_available_choices()
            choice = random.choice(available_choice)
            score = simulation_board.proceed_next_state(choice[0], choice[1])
            total_score += self.gamma**i * score
        return total_score
        


    def expand(self, node):

        child_node_state_action_list = []
        for sub_node in node.child:
            child_node_state_action_list.append(sub_node.state.choice)


        simulation_board = GameBoard(node.state.current_board, simulation = True)

        new_state = node.state.get_next_state_with_random_choice(simulation_board, exclude=child_node_state_action_list)


        sub_node = Node(parent=node, state=new_state)
        node.add_child(sub_node)

        return sub_node


    def best_child(self, node, is_exploration):
        best_score = -sys.maxsize
        best_sub_node = None

        # Travel all sub nodes to find the best one
        for sub_node in node.child:

            # Ignore exploration for inference
            if is_exploration:
                C = self.C
                # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
                left = sub_node.quality_value #/ sub_node.visited_times
                right = 2.0 * math.log(node.visited_times) / sub_node.visited_times
                right = C * math.sqrt(right)
                score = left + right

            else:
                #score = sub_node.visited_times
                score = sub_node.quality_value

            if score > best_score:
                best_sub_node = sub_node
                best_score = score

        return best_sub_node



    def backup(self, node, reward):
        while node != None:
            node.visited_times += 1
            new_quality_value = self.gamma * reward + node.state.action_reward
            node.quality_value += 1.0/node.visited_times * (new_quality_value - node.quality_value) ## cumulative averaging
            reward = node.quality_value 
            node = node.parent


    def monte_carlo_tree_search(self, node):
        for i in range(self.COMPUTATION_BUDGET):


            expand_node = self.tree_policy(node)
            reward = self.default_policy(expand_node)
            self.backup(expand_node, reward)

        best_next_node = self.best_child(node, False)
        #print("quality value:", best_next_node.quality_value)
        return best_next_node
    
    def get_next_node(self):
        self.current_node = self.monte_carlo_tree_search(self.current_node)
        return self.current_node

    
if __name__ == "__main__":
    total_score = 0
    total_game = 1000
    score_list = []
    for game in range(total_game):
        gameplay = Game(show=False)
        mcts = MCTS(gameplay.gameboard)

        for i in range(15):
            current_node = mcts.get_next_node()
            choice = current_node.state.get_choice()
            #print("You have choosen : " + str(choice[0]) + " " + str(choice[1]))
            gameplay.input_pos(choice[0], choice[1])
        
        score_list.append(gameplay.gameboard.score)
        total_score += gameplay.gameboard.score
    ave_score = total_score/total_game
    print(ave_score)
    print(ave_score/15)
    
    with open("./plot/mcts_alone.txt", "w") as fd:
        for value in score_list:
            fd.write(str(value) + "\n")
    
