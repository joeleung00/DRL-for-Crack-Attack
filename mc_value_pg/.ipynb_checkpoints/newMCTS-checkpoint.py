import MCTS as mcts
import sys
import math
import sys
sys.path.insert(1, "../game_simulation")
from GameBoard import GameBoard 
class Node(mcts.Node):
    def __init__(self, state, parent = None):
        super().__init__(state, parent)
        self.policy_probi = None

class MCTS(mcts.MCTS):
    def __init__(self, init_gameboard, agent):
        super().__init__(init_gameboard)
        self.agent = agent
        policy_probi, quality_value = agent.inference(self.root_node.state.current_board)
        #self.root_node.quality_value = quality_value
        self.root_node.policy_probi = policy_probi
    
    def best_child(self, node, is_exploration):
        best_score = -sys.maxsize
        best_sub_node = None
        # Travel all sub nodes to find the best one
        for sub_node in node.child:

            # Ignore exploration for inference
            if is_exploration:
                C = self.C
                # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
                left = sub_node.quality_value 
                right = 2.0 * math.log(node.visited_times) / sub_node.visited_times
                row, col = sub_node.state.choice
                right = C * math.sqrt(right) * node.policy_probi[row][col]
                score = left + right
                
                
            else:
                #score = sub_node.visited_times
                score = sub_node.quality_value

            if score > best_score:
                best_sub_node = sub_node
                best_score = score
        #if is_exploration:
         #   print(left, right)
        return best_sub_node
    
#     def default_policy(self, node):
#         policy_probi, quality_value = self.agent.inference(node.state.current_board)
#         node.policy_probi = policy_probi
#         print(quality_value)
#         return quality_value

    def default_policy(self, node):
        policy_probi, values = self.agent.inference(node.state.current_board)
        node.policy_probi = policy_probi
        current_state = node.state

        simulation_board = GameBoard(current_state.current_board, simulation = True)
        total_score = 0
        for i in range(3):
            board = simulation_board.board
            choice = self.agent.best_move(board)
            score = simulation_board.proceed_next_state(choice[0], choice[1])
            total_score += self.gamma**i * score
        #print(total_score)
        return total_score
        
        
        
if __name__ == "__main__":
    pass