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
            for child in node.children:
                if child.state == state:
                    return child
            return Node(state)

        def tree_policy(node):  # selection & expansion
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
            return node

        def expand(node):  # expand node, try a untried action and return new node
            [next_state, next_action] = node.next_state()
            assert(node.state.locs[0] != None xor node.state.locs[1] != None)
            assert(next_action != None)
            new_state = node.state.result(next_action)
            assert(new_state.locs[0] != None)
            assert(new_state.locs[1] != None)
            i = 0
            node.add_child(new_state, next_action)
            return node.children[-1]

        # current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
        # pick one child of input node with the best score
        def best_child(node, scalar=SCALAR):
            bestscore = 0.0
            bestchildren = []
            bestactions = []
            for c in node.children:
                exploit = c.reward / c.visits
                explore = math.sqrt(
                    2.0 * math.log(node.visits) / float(c.visits))
                score = exploit + scalar * explore
                if score == bestscore:
                    bestchildren.append(c)
                    bestactions.append(c.prev_action)
                if score > bestscore:
                    bestchildren = [c]
                    bestactions = [c.prev_action]
                    bestscore = score
            if len(bestchildren) == 0:
                bestchild = random.choice(node.children)
                randomaction = random.choice(node.state.actions())
                return bestchild
            bestchild = random.choice(bestchildren)
            bestaction = random.choice(bestactions)
            return bestchild

        # return reward from terminal node starting from input state
        def default_policy(state):
            depth = 0
            while state.terminal_test() == False:
                depth += 1
                state = state.result(random.choice(state.actions()))
            reward = score(state) / 16 + 0.5
            return reward

        def backup(node, reward):  # backpropagate
            depth = 0
            while node != None:
                depth += 1
                node.visits += 1
                node.reward += reward
                node = node.parent
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
