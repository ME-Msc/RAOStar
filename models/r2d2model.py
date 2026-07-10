#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# r2d2 model as simple model to test rao star

import numpy as np
from belief import BeliefState


class R2D2Model(object):
    # noted there are 7 blocks total, A through G
    # G is the goal, F is fire
    DEFAULT_ACTION_COST = {
        "down": 1,
        "right": 1,
        "up": 1,
        "left": 1,
    }

    def __init__(self, length=1, DetermObs=True, obsProb=0.6,
                 icy_move_forward_prob=0.8, icy_blocks=None,
                 fire_blocks=None, blocked_blocks=None, allow_left=False,
                 action_cost=DEFAULT_ACTION_COST):
        if length < 1:
            raise ValueError('length must be positive')
        if not 0.0 <= obsProb <= 1.0:
            raise ValueError('obsProb must be between 0 and 1')
        if not 0.0 <= icy_move_forward_prob <= 1.0:
            raise ValueError('icy_move_forward_prob must be between 0 and 1')
        if not isinstance(action_cost, dict):
            raise TypeError('action_cost must be a dictionary')
        required_actions = {'down', 'right', 'up'}
        if allow_left:
            required_actions.add('left')
        missing_actions = required_actions - set(action_cost)
        if missing_actions:
            raise ValueError('missing action costs: ' + str(sorted(missing_actions)))
        if any(action_cost[name] <= 0.0 for name in required_actions):
            raise ValueError('action costs must be positive')

        # icy blocks are defined blocks that are icy
        if icy_blocks is None:
            self.icy_blocks = [(1, i) for i in range(length + 1)]
        else:
            self.icy_blocks = [tuple(block) for block in icy_blocks]
        self.icy_blocks_lookup = {block: 1 for block in self.icy_blocks}

        self.icy_move_forward_prob = icy_move_forward_prob
        self.DetermObs = DetermObs
        self.obsProb = obsProb
        self.action_cost = action_cost.copy()
        # environment will be represented as a 3 x 3 grid, with (2,0) and (2,2) blocked
        # top left corner of grid is (0,0) and first index is row
        self.length = length
        self.env = np.zeros([3, 2 + self.length])

        # setup bottom left and right corners as impassible
        if blocked_blocks is None:
            self.blocked_blocks = [(2, 0), (2, self.length + 1)]
        else:
            self.blocked_blocks = [tuple(block) for block in blocked_blocks]
        for block in self.blocked_blocks:
            self.env[block] = 1

        # fire located at self.env[2,1]: terminal
        if fire_blocks is None:
            self.fires = [(2, i + 1) for i in range(self.length)]
        else:
            self.fires = [tuple(block) for block in fire_blocks]

        # goal position
        self.goal = (1, self.length + 1)
        if self.goal in self.icy_blocks + self.fires + self.blocked_blocks:
            raise ValueError('goal cannot be icy, on fire, or blocked')
        if (1, 0) in self.fires + self.blocked_blocks:
            raise ValueError('start cannot be on fire or blocked')
        if set(self.icy_blocks) & (set(self.fires) | set(self.blocked_blocks)):
            raise ValueError('icy blocks cannot overlap fire or blocked blocks')
        if set(self.fires) & set(self.blocked_blocks):
            raise ValueError('fire blocks cannot overlap blocked blocks')

        self.optimization = 'minimize'  # want to minimize the steps to goal
        self.action_symbols = {
            "right": "->",
            "left": "<-",
            "up": "^",
            "down": "v",
        }
        self.action_list = [(1, 0, "down"), (0, 1, "right"),
                            (-1, 0, "up")]
        if allow_left:
            self.action_list.append((0, -1, "left"))

    def initial_belief(self):
        return {(1, 0, 0): 1.0}

    def state_valid(self, state):  # check if a particular state is valid
        if state[0] < 0 or state[1] < 0:
            return False
        try:
            return self.env[state[0], state[1]] == 0
        except IndexError:
            return False

    def in_a_fire(self, state):
        # print(state)
        # print(self.fires)

        for fire in self.fires:
            if state[0] == fire[0] and state[1] == fire[1]:
                # print('   risk!!   ')
                return True
        return False

    def actions(self, state):
        # print('actions for: ' + str.(state))
        validActions = []
        for act in self.action_list:
            newx = state[0] + act[0]
            newy = state[1] + act[1]
            if self.state_valid((newx, newy)):
                validActions.append(act)
        if (state[0], state[1]) == self.goal:
            return []
        if self.in_a_fire(state):
            return []
        # if state[0] == 1 and state[1] == 1:
        #     print('got to center cell')
        #     return []
        return validActions

    def is_terminal(self, state):
        # For some reason we get a BeliefState here when deadend state found
        if isinstance(state, BeliefState):
            state = next(iter(state.belief))
        # Added fire state to terminal to differentiate it from deadends
        return state[0] == self.goal[0] and state[1] == self.goal[1] or self.in_a_fire(state)

    def state_transitions(self, state, action):
        newstates = []
        # intended_new_state = (state[0] + action[0],
        #                       state[1] + action[1])
        # added depth to the state
        intended_new_state = (state[0] + action[0],
                              state[1] + action[1], state[2] + 1)
        if not self.state_valid(intended_new_state):
            # Keep probability mass when this action came from another particle
            # in the same belief state.
            return [[(state[0], state[1], state[2] + 1), 1.0]]

        if (state[0], state[1]) in self.icy_blocks and "right" in action:
            # print('got right action!')
            newstates.append([intended_new_state, self.icy_move_forward_prob])
            for slip in [-1, 1]:
                slipped = [(action[i] + slip) % 2 * slip for i in range(2)]
                # slipped_state = (state[0] + slipped[0],
                #                  state[1] + slipped[1])
                # added depth to the state
                slipped_state = (state[0] + slipped[0],
                                 state[1] + slipped[1], state[2] + 1)
                if self.state_valid(slipped_state):
                    newstates.append(
                        [slipped_state, (1 - self.icy_move_forward_prob) / 2])
        else:
            newstates.append([intended_new_state, 1.0])

        # Need to normalize probabilities for cases where slip only goes to one
        # cell, not two possible cells
        sum_probs = sum(n for _, n in newstates)
        for child in newstates:
            child[1] = child[1] / sum_probs

        return newstates

    def observations(self, state):
        if self.DetermObs:
            return [(state, 1.0)]

        # Terminal outcomes are observed exactly so violating and safe paths do
        # not become mixed in a single belief state.
        if self.is_terminal(state):
            return [(state, 1.0)]

        position = (state[0], state[1])
        neighboring_cells = []
        for row_delta, col_delta in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            neighbor = (position[0] + row_delta, position[1] + col_delta)
            if self.state_valid(neighbor) and not self.is_terminal(neighbor):
                neighboring_cells.append(neighbor)

        if not neighboring_cells:
            return [(position, 1.0)]

        wrong_prob = (1.0 - self.obsProb) / len(neighboring_cells)
        return [(position, self.obsProb)] + [
            (neighbor, wrong_prob) for neighbor in neighboring_cells
        ]

    def state_risk(self, state):
        # For some reason we get a BeliefState here when deadend state found
        if isinstance(state, BeliefState):
            state = next(iter(state.belief))
        if self.in_a_fire(state):
            return 1.0
        return 0.0

    def costs(self, action):
        return self.action_cost[action[2]]

    def values(self, state, action):
        # return value (heuristic + cost)
        # print('state here', state)
        # if state[0] == 0:
            # return 2.0
        # return 1.0
        return self.costs(action)
        # return self.costs(action) + self.heuristic(state)

    def heuristic(self, state):
        # Euclidean distance times the cheapest action cost is an optimistic
        # estimate of the remaining path cost.
        min_action_cost = min(self.costs(action) for action in self.action_list)
        return min_action_cost * np.sqrt(
            sum([(self.goal[i] - state[i])**2 for i in range(2)]))

    def execution_risk_heuristic(self, state):
        # sqaure of euclidean distance to fire as heuristic
        return 0
        # return sum([(self.fire[i] - state[i])**2 for i in range(2)])

    def print_model(self):
        height, width = self.env.shape
        print(" ")
        print("    ** Model environment **")
        for j in range(height):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(width):
                if self.env[j][i]:
                    row_str += " [-----] "
                    continue
                row_str += " [" + str(j) + "," + str(i)
                if self.goal == (j, i):
                    row_str += " g] "
                elif self.state_risk((j, i)):
                    row_str += " f] "
                elif (j, i) in self.icy_blocks_lookup:
                    row_str += " i] "
                else:
                    row_str += "  ] "
            print(row_str)

    def print_policy(self, policy):
        height, width = self.env.shape
        policy_map = np.full([height, width], "", dtype=object)
        depth_found = {}

        for key in policy:
            belief_name = key[0] if isinstance(key, tuple) else key
            coords = belief_name.split(":")[0].split("(")[1].split(")")[0]
            col = int(coords.split(",")[0])
            row = int(coords.split(",")[1])
            depth = key[2] if isinstance(key, tuple) else int(coords.split(",")[2])
            col_row_str = str(col) + ',' + str(row)
            action_string = policy[key]

            for action_name, action_symbol in self.action_symbols.items():
                if action_name in action_string:
                    if col_row_str not in depth_found:  # first depth found
                        depth_found[col_row_str] = depth
                        policy_map[col][row] = action_symbol
                        break
                    elif depth < depth_found[col_row_str]:
                        depth_found[col_row_str] = depth
                        policy_map[col][row] = action_symbol
                        break
        print(" ")
        print("         ** Policy **")
        for j in range(height):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(width):
                if self.goal == (j, i):
                    row_str += " [goal] "
                elif self.state_risk((j, i)):
                    row_str += " [fire] "
                elif self.env[j][i]:
                    row_str += " [----] "
                else:
                    row_str += " [ " + policy_map[j][i].center(2) + " ] "
            print(row_str)
