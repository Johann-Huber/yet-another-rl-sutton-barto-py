import numpy as np
import operator
import matplotlib.pyplot as plt


# -----------------------------------------------

GRID_SIDE_SIZE = 5
VIS_RATIO = 10
FIGURE_SIZE = (8,8)

DISCOUNT_FACTOR = 0.9

POS_A = (0,1)
POS_A_next = (4,1)
POS_B = (0,3)
POS_B_next = (2,3)

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


    def plot(self, values = None):
        fig, ax = plt.subplots(1,1, figsize=self._fig_size)

        for i in range(self._side_size):
            ax.plot([0, self._side_size], [i , i], color='k')

        for j in range(self._side_size):
            ax.plot([j , j], [0, self._side_size], color='k')

        a_xy = self.cvt_ij2xy(self._pos_a)
        a_n_xy = self.cvt_ij2xy(self._pos_a_next)
        b_xy = self.cvt_ij2xy(self._pos_b)
        b_n_xy = self.cvt_ij2xy(self._pos_b_next)

        ax.text(a_xy[0] + 0.5, a_xy[1] + 0.5, 'A', fontsize=20, ha='center', va='center')
        ax.text(a_n_xy[0] + 0.5, a_n_xy[1] + 0.5, "A'", fontsize=20, ha='center', va='center')
        ax.text(b_xy[0] + 0.5, b_xy[1] + 0.5, 'B', fontsize=20, ha='center', va='center')
        ax.text(b_n_xy[0] + 0.5, b_n_xy[1] + 0.5, "B'", fontsize=20, ha='center', va='center')

        # Value function
        if values is not None:

            for i in range(self._side_size):
                for j in range(self._side_size):
                    x,y = self.cvt_ij2xy((i,j))
                    ax.text(x + 0.95, y + 0.95, f'{values[i,j]:.1f}', fontsize=16, ha='right', va='top')

        ax.set_xlim(0, self._side_size)
        ax.set_ylim(0, self._side_size)

        plt.waitforbuttonpress()


    def cvt_ij2xy(self, pos_ij):
        return pos_ij[1], self._side_size - 1 - pos_ij[0]


    def cvt_xy2ij(self, pos_xy):
        return self._side_size - 1 - pos_xy[1], pos_xy[0]


    def move(self, state_ij, action):

        # Independent from action

        if state_ij == POS_A:
            next_state_ij = POS_A_next
            reward = 10
            return next_state_ij, reward

        if state_ij == POS_B:
            next_state_ij = POS_B_next
            reward = 5
            return next_state_ij, reward


        # Apply the attempted move

        next_state_ij = tuple(map(operator.add, state_ij, self.actions_to_move_ij[action]))

        is_off_borders = (next_state_ij[0] < 0) or (next_state_ij[0] >= self._side_size) or \
                         (next_state_ij[1] < 0) or (next_state_ij[1] >= self._side_size)
        if is_off_borders:
            next_state_ij = state_ij
            reward = -1
            return next_state_ij, reward
        else:
            reward = 0
            return next_state_ij, reward





class Agent:
    def __init__(self, gamma = DISCOUNT_FACTOR, grid_side_size = GRID_SIDE_SIZE):
        # Current pos on the grid
        self._pos = (0,0)

        # Value function of state (i,j)
        self._v = np.zeros((grid_side_size,grid_side_size))

        # Discount factor
        self._gamma = gamma

        # Policy
        self.actions_to_prob = {'up': 0.25, 'left': 0.25, 'down': 0.25, 'right': 0.25}

    def compute_values(self, grid, grid_side_size = GRID_SIDE_SIZE):

        # todo : compute_values

        moves = ['up', 'left', 'down', 'right']

        #  V(s) = sum_over_actions_and_next_state [ prob(s,a,s')*r + prob(s,a,s')*gamma*V(S') ]
        #
        # <=>  prob(s,a,s_1')*gamma*V(S_1') + ... + prob(s,a,s_n')*gamma*V(S_n') - V(s)
        #                              + ( prob(s,a,s_1')*r_1  + ... +  prob(s,a,s_1')*r_n ) = 0
        # n : grid_side_size
        #
        # <=>  prob(s,a,s_1')*gamma*V(S_1') + ... + prob(s,a,s_n')*gamma*V(S_n') - V(s)
        #                              = - ( prob(s,a,s_1')*r_1  + ... +  prob(s,a,s_1')*r_n ) =
        #
        # => state_value_coefficients * x = state_value_rewards
        #
        # ax = b, linear matrix equation with :
        #   state_value_coefficients (a) : coeff associated to each state_value
        #   x : state values to compute
        #   state_value_rewards (b) : reward term of each linear equation (must be computed negatively)

        state_value_coefficients = -1*np.eye(grid_side_size*grid_side_size)
        state_value_rewards = np.zeros(grid_side_size*grid_side_size)


        for i in range(grid_side_size):
            for j in range(grid_side_size):

                state = (i,j)
                ind_state = i * grid_side_size + j

                for action in moves:

                    next_state, reward = grid.move(state, action)
                    ind_state_next = next_state[0] * grid_side_size + next_state[1]

                    state_value_coefficients[ind_state, ind_state_next] += self.actions_to_prob[action] * self._gamma
                    state_value_rewards[ind_state] -= reward * self.actions_to_prob[action]

        values = np.linalg.solve(state_value_coefficients, state_value_rewards)
        values = values.reshape((grid_side_size,grid_side_size))

        return values




if __name__ == '__main__':

    grid = Grid()
    agent = Agent()

    values = agent.compute_values(grid)

    grid.plot(values)