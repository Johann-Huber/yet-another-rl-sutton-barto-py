import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.ion()

# -----------------------------------------------


GRID_SIDE_SIZE = 4
VIS_RATIO = 10
FIGURE_SIZE = (16,8)

DISCOUNT_FACTOR = 1. #0.9

POS_A = (0,1) #(1,4)
POS_A_next = (4,1) #(1,0)
POS_B = (0,3) #(3,4)
POS_B_next = (2,3) #(3,2)

GOAL_POS = [(0,0), (3,3)]
STEP_MAX_THRESHOLD = 0.001

# -----------------------------------------------

class Grid :
    def __init__(self, grid_side_size = GRID_SIDE_SIZE, vis_ratio = VIS_RATIO, fig_size = FIGURE_SIZE,
                 pos_a=POS_A, pos_a_next=POS_A_next, pos_b=POS_B, pos_b_next=POS_B_next):

        self._side_size = grid_side_size

        self._vis_ratio = vis_ratio
        self._fig_size = fig_size

        self._pos_a = pos_a
        self._pos_a_next = pos_a_next
        self._pos_b = pos_b
        self._pos_b_next = pos_b_next

        self.actions_to_move_ij = {'up': (-1,0), 'left': (0,-1), 'down': (1,0), 'right': (0,1)}

        # Init plot
        self.fig, self.ax = plt.subplots(1, 2, figsize=self._fig_size)


    def plot(self, values=None, k=None, policy_states=None, plot_time=0.01):

        # Left : state values

        for i in range(self._side_size):
            self.ax[0].plot([0, self._side_size], [i , i], color='k')
        for j in range(self._side_size):
            self.ax[0].plot([j , j], [0, self._side_size], color='k')


        # Value function
        if values is not None:

            for i in range(self._side_size):
                for j in range(self._side_size):
                    x,y = self.cvt_ij2xy((i,j))
                    self.ax[0].text(x + 0.5, y + 0.5, f'{values[i,j]:.1f}', fontsize=16, ha='center', va='center')

        self.ax[0].set_xlim(0, self._side_size)
        self.ax[0].set_ylim(0, self._side_size)
        self.ax[0].set_title('Value(state)')


        # Right : policy
        for i in range(self._side_size):
            self.ax[1].plot([0, self._side_size], [i , i], color='k')
        for j in range(self._side_size):
            self.ax[1].plot([j , j], [0, self._side_size], color='k')

        terminal_tl = patches.Rectangle((0, 3), 1, 1, linewidth=1, edgecolor='none', facecolor='c')
        terminal_br = patches.Rectangle((3, 0), 1, 1, linewidth=1, edgecolor='none', facecolor='c')
        self.ax[1].add_patch(terminal_tl)
        self.ax[1].add_patch(terminal_br)


        # Arrows : greedy policy w.r.t. values
        for i in range(self._side_size):
            for j in range(self._side_size):

                if (i,j) in GOAL_POS:
                    continue

                x, y = self.cvt_ij2xy((i, j))

                ind_state = i * self._side_size + j
                if policy_states[ind_state]['up']:
                    text_up = self.ax[1].text(x + 0.5, y + 0.6, '↑', fontsize=20, ha='center', va='center')
                if policy_states[ind_state]['left']:
                    text_left = self.ax[1].text(x + 0.4, y + 0.5, '←', fontsize=20, ha='center', va='center')
                if policy_states[ind_state]['down']:
                    text_down = self.ax[1].text(x + 0.5, y + 0.4, '↓', fontsize=20, ha='center', va='center')
                if policy_states[ind_state]['right']:
                    text_right = self.ax[1].text(x + 0.6, y + 0.5, '→', fontsize=20, ha='center', va='center')



        self.ax[1].set_xlim(0, self._side_size)
        self.ax[1].set_ylim(0, self._side_size)
        self.ax[1].set_title('Policy')


        plt.suptitle(f"Iterative policy evaluation | k = {k}", fontsize=18)

        plt.pause(plot_time)
        self.ax[0].clear()
        self.ax[1].clear()



    def cvt_ij2xy(self, pos_ij):
        return pos_ij[1], self._side_size - 1 - pos_ij[0]


    def cvt_xy2ij(self, pos_xy):
        return self._side_size - 1 - pos_xy[1], pos_xy[0]


    def move(self, state_ij, action):

        if state_ij in GOAL_POS:
            reward = 0
            return state_ij, reward

        next_state_ij = tuple(map(operator.add, state_ij, self.actions_to_move_ij[action]))

        is_off_borders = (next_state_ij[0] < 0) or (next_state_ij[0] >= self._side_size) or \
                         (next_state_ij[1] < 0) or (next_state_ij[1] >= self._side_size)
        if is_off_borders:
            next_state_ij = state_ij
            reward = -1
            return next_state_ij, reward

        reward = -1
        return next_state_ij, reward



class Agent:
    def __init__(self, gamma = DISCOUNT_FACTOR, grid_side_size = GRID_SIDE_SIZE):
        # Current pos on the grid
        self._pos = (0,0)
        # Value function of state (i,j)
        self._v = np.zeros((grid_side_size,grid_side_size), dtype=np.float32)

        # Discount factor
        self._gamma = gamma

        self.actions_to_prob = {'up': 0.25, 'left': 0.25, 'down': 0.25, 'right': 0.25}

        # Policy evaluation increment
        self._k = 0

    def get_current_evaluation(self):
        return self._v, self._k


    def get_greedy_policy(self, grid_side_size = GRID_SIDE_SIZE):

        policy_states = {}

        for i in range(grid_side_size):
            for j in range(grid_side_size):

                up_val = self._v[i-1, j] if i > 0 else self._v[i,j]
                left_val = self._v[i, j - 1] if j > 0 else self._v[i, j]
                down_val = self._v[i+1, j] if i < grid_side_size - 1 else self._v[i,j]
                right_val = self._v[i, j+1] if j < grid_side_size - 1 else self._v[i, j]

                all_values = np.array([up_val, left_val, down_val, right_val])
                args_max = np.argwhere(all_values == np.max(all_values)).flatten()

                action_probs = np.zeros((4,)) # up, left, down, right
                for i_max in args_max:
                    action_probs[i_max] = 1/len(args_max)

                ind = i * grid_side_size + j
                policy_states[ind] = {'up': action_probs[0],
                                     'left': action_probs[1],
                                     'down': action_probs[2],
                                     'right': action_probs[3]}

        return policy_states


    def get_current_policy(self, state, grid_side_size = GRID_SIDE_SIZE):

        i,j = state

        up_val = self._v[i-1, j] if i > 1 else self._v[i,j]
        left_val = self._v[i, j - 1] if j > 1 else self._v[i, j]
        down_val = self._v[i+1, j] if i < grid_side_size - 1 else self._v[i,j]
        right_val = self._v[i, j+1] if j < grid_side_size - 1 else self._v[i, j]

        all_values = np.array([up_val, left_val, down_val, right_val])
        args_max = np.argwhere(all_values == np.max(all_values)).flatten()

        action_probs = np.zeros((4,)) # up, left, down, right
        for i_max in args_max:
            action_probs[i_max] = 1/len(args_max)

        return {'up': action_probs[0], 'left': action_probs[1], 'down': action_probs[2], 'right': action_probs[3]}

    def iterate_policy_eval(self, grid, grid_side_size = GRID_SIDE_SIZE):

        step_max = 0
        moves = ['up', 'left', 'down', 'right']

        v_prev = self._v.copy()

        for i in range(grid_side_size):
            for j in range(grid_side_size):

                state = (i, j)
                prev_value = v_prev[i,j]
                value = 0

                #policy_s_a = self.get_current_policy(state) # greedy approach

                for action in moves:
                    next_state, reward = grid.move(state, action)

                    i_next, j_next = next_state
                    value += self.actions_to_prob[action]*(reward + self._gamma*v_prev[i_next,j_next])

                self._v[i, j] = value

                step_max = max(step_max, abs(value-prev_value))

        self._k += 1

        return step_max

    def get_value_state(self):
        values, k = self.get_current_evaluation()
        policy_states = self.get_greedy_policy()
        return values, k, policy_states


def policy_evaluation_dynamic_plot():
    """Computes policy evaluation and plot results dynamically.
    Reproduces figure p77."""

    grid = Grid()
    agent = Agent()

    values, k, policy_states = agent.get_value_state()
    grid.plot(values, k, policy_states)

    while (True):
        step_max = agent.iterate_policy_eval(grid)

        values, k, policy_states = agent.get_value_state()
        grid.plot(values, k, policy_states)

        if (step_max < STEP_MAX_THRESHOLD):
            break

    values, k, policy_states = agent.get_value_state()
    grid.plot(values, k, policy_states, plot_time=10)

if __name__ == '__main__':
    policy_evaluation_dynamic_plot()
