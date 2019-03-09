from sample_players import DataPlayer
import random
import math
import hashlib
import logging
import argparse

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')
SCALAR = 1 / math.sqrt(2.0)
"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  Some features
of the example by design are that moves do not commute and early mistakes are more costly.
In particular there are two models of best child that one can use
"""


# class State():
#     NUM_TURNS = 10
#     GOAL = 0
#     MOVES = [2, -2, 3, -3]
#     MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
#     num_moves = len(MOVES)

#     def __init__(self, value=0, moves=[], turn=NUM_TURNS):
#         self.value = value
#         self.turn = turn
#         self.moves = moves

#         def next_state(self):
#             nextmove = random.choice([x * self.turn for x in self.MOVES])
#             next = State(self.value + nextmove, self.moves +
#                          [nextmove], self.turn - 1)
#             return next

#         def terminal(self):
#             if self.turn == 0:
#                 return True
#             return False

#         def reward(self):
#             r = 1.0 - (abs(self.value-self.GOAL) / self.MAX_VALUE)
#             return r

#         def __hash__(self):
#             return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

#         def __eq__(self, other):
#             if hash(self) == hash(other):
#                 return True
#             return False

#         def __repr__(self):
#             s = "Value: %d; Moves: %s" % (self.value, self.moves)
#             return s


class Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.prev_action = None
        self.parent = parent

    def add_child(self, child_state, prev_action):
        assert(prev_action != None)
        child = Node(child_state, self)
        child.prev_action = prev_action
        # print("adding new child with parent", child.parent)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        return not self.state._has_liberties(self.state.player())

    def next_state(self):  # pick random untried action
        random_action = random.choice(self.state.actions())
        # print("total actions", self.state.actions())
        # print("random action from node", random_action)
        return [self.state.result(random_action), random_action]

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f; state:%s prev_action:%s" % (
            len(self.children), self.visits, self.reward, self.state, self.prev_action)
        return s


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    SCALAR = 1 / math.sqrt(2.0)

    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))

    # run mcts and return action
    def mcts(self, state):

        # return state that opponent have put us in
        def getStateNode(state):
            node = self.context
            if node is None:
                return Node(state)
            # print('CCCCCCCCCC CONTEXT IS', self.context)
            # print('children ', node.children)
            # print('state', state)
            # child = None
            for child in node.children:
                if child.state == state:
                    return child
            return Node(state)
            # if child is not None:
            #     return child
            # else:
            #     Node(state)

        def tree_policy(node):  # selection & expansion
                # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
                #             logger.info("treepolicy start")
                # print('NODE IS', node)
            while node.state.terminal_test() == False:
                if len(node.children) == 0:
                    return expand(node)
                elif random.uniform(0, 1) < .5:
                    node = best_child(node, SCALAR)
                else:
                    if node.fully_expanded() == False:
                        return expand(node)
                    else:
                        node = best_child(node, SCALAR)
                # print('NODE IS ', node)
            return node

        def expand(node):  # expand node, try a untried action and return new node
            [next_state, next_action] = node.next_state()
    #             assert(node.state.locs[0] != None xor node.state.locs[1] != None)
            assert(next_action != None)
    #             print("node is ", node)
    #             print("chosen action is ", next_action, " among ", node.state.actions())
            new_state = node.state.result(next_action)
    #             print("new state locs", new_state.locs[0], " ", new_state.locs[1])
            assert(new_state.locs[0] != None)
            assert(new_state.locs[1] != None)
    #             print("new_state from expand is ", new_state, "with action", next_action)
            i = 0
    #             while new_state in tried_children:
    #                 i += 1
    #                 print("new state from expansion", new_state, "i:", i)
    #                 next_action = random.choice(node.state.actions())
    #                 new_state = node.state.result(next_action)
            node.add_child(new_state, next_action)
    #             print("return new state that should be unexistant in tried states", node.children[-1])
            return node.children[-1]

        # current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
        # pick one child of input node with the best score
        def best_child(node, scalar=SCALAR):
            #             logger.info("bestchild start")
            #             logger.warn("warning bestchild start")
            bestscore = 0.0
            bestchildren = []
            bestactions = []
            for c in node.children:
                exploit = c.reward / c.visits
                explore = math.sqrt(
                    2.0 * math.log(node.visits) / float(c.visits))
                score = exploit + scalar * explore
                if score == bestscore:
                    #                     logger.info("children has best score")
                    bestchildren.append(c)
                    bestactions.append(c.prev_action)
                if score > bestscore:
                    bestchildren = [c]
                    bestactions = [c.prev_action]
                    bestscore = score
            if len(bestchildren) == 0:
                bestchild = random.choice(node.children)
                randomaction = random.choice(node.state.actions())
    #                 print("bad sign returning random action", randomaction)
                return bestchild
            # else:
            #     print("best child found")
            bestchild = random.choice(bestchildren)
            bestaction = random.choice(bestactions)
    #             print("best child returned", bestchild)
    #             print("best action returned", bestaction)
    #             print("best actions", bestactions)
    #             print("best children", bestchildren)
            return bestchild

        # return reward from terminal node starting from input state
        def default_policy(state):
            #             logger.info("default policy start")
            depth = 0
            while state.terminal_test() == False:
                depth += 1
                state = state.result(random.choice(state.actions()))
    #             print("digged ", depth)
            reward = score(state) / 16 + 0.5
    #             print("returned reward", reward)
            return reward

        def backup(node, reward):  # backpropagate
            depth = 0
            while node != None:
                depth += 1
                node.visits += 1
                node.reward += reward
    #                 print("backpropagating node ", node)
    #                 print("backpropagated reward from", node.reward - reward, "to node.reward", node.reward)
                node = node.parent
    #             print("depth is ", depth)
            return

        def score(state):
            own_loc = state.locs[self.player_id]
            opp_loc = state.locs[1 - self.player_id]
            own_liberties = state.liberties(own_loc)
            opp_liberties = state.liberties(opp_loc)
            return len(own_liberties) - len(opp_liberties)

        root = Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        iter_limit = 100
        for _ in range(iter_limit):
            child = tree_policy(root)
            if not child:
                continue
            reward = default_policy(child.state)
            backup(child, reward)

        idx = root.children.index(best_child(root))
        return root.children[idx].prev_action


# class CustomPlayer(DataPlayer):

#     def get_action(self, state):
#         #         print("play count:", state.ply_count)
#         best_move = None
#         if state.ply_count < 2:
#             #             print("chosing random")
#             best_move = random.choice(state.actions())
#         else:
#             for depth in range(1, 5):
#                 #                 print("d:", depth)
#                 best_move = self.alpha_beta_search(state, depth)
# #         print("player ", self.player_id, " choses ", best_move)
#         self.queue.put(best_move)

#     def alpha_beta_search(self, state, depth):

#         def min_value(state, alpha, beta, depth):
#             if state.terminal_test():
#                 return state.utility(self.player_id)
#             if depth <= 0:
#                 return self.score(state)
#             value = float("inf")
#             for action in state.actions():
#                 value = min(value, max_value(
#                     state.result(action), alpha, beta, depth - 1))
#                 if value <= alpha:
#                     return value
#                 beta = min(value, beta)
#             return value

#         def max_value(state, alpha, beta, depth):
#             if state.terminal_test():
#                 return state.utility(self.player_id)
#             if depth <= 0:
#                 return self.score(state)
#             value = float("-inf")
#             for action in state.actions():
#                 value = max(value, min_value(
#                     state.result(action), alpha, beta, depth - 1))
#                 if value >= beta:
#                     return value
#                 alpha = max(alpha, value)
#             return value

# #         print("state is ", state)
#         alpha = float("-inf")
#         beta = float("inf")
#         best_score = float("-inf")
#         best_move = None
#         for a in state.actions():
#             v = min_value(state.result(a), alpha, beta, depth - 1)
#             if v > best_score:
#                 best_score = v
#                 best_move = a
#         return best_move

#     def score(self, state):
#         own_loc = state.locs[self.player_id]
#         opp_loc = state.locs[1 - self.player_id]
#         own_liberties = state.liberties(own_loc)
#         opp_liberties = state.liberties(opp_loc)
#         return len(own_liberties) - len(opp_liberties)
