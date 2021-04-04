
###########################
# NOTES :
###########################
#
## ON IMPLEMENTATION :
#
# - Behavior policy randomly choose between the 9 available action with equal probability.
# - I've added a reward penalty when the agent takes a useless acceleration. It doesn't
# change anything beside avoiding noise when plotting the trajectory.
# - The approach for detection boundary collision is ok, but not perfect. This degree of complexity
# is enough regarding the purpose of this exercise.
#
#
## ON EXECUTION :
#
# As mentioned in the book, the training process is slow :
#
# - Each episode takes a long time to compute.
#       It is quite hard for the agent to reach the finish line by following a pure random exploration, as the it restarts from the beginning
#	as soon as an obstacle is hit.
#
# - The Q-values updating process stops as soon as the target policy changed within an episode.
#       This problem is raised in the book (p.111) : "[...] this method learns only from the tails of episodes,
#       when all the remaining actions in the episode are greedy." That is why large maps can't be quickly solved using this approach.
#
#
###########################

import numpy as np
from enum import Enum
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_theme()

#--------------


# Parameters related to map

MapDims = namedtuple('MapDims', ['grid_h', 'grid_w'])

MAP_TYPES = ['DEBUG', 'NARROW', 'LARGE']

MAP_DIMS = {'DEBUG' : MapDims(grid_h=10, grid_w=5),
            'NARROW': MapDims(grid_h=33, grid_w=18),
            'LARGE': MapDims(grid_h=31, grid_w=33)}

START_LINE = {'DEBUG' : [(9, j) for j in range(1,3)],
              'NARROW': [(32, j) for j in range(4,11)],
              'LARGE': [(30, j) for j in range(1,24)]}

FINISH_LINE = {'DEBUG' : [(i, 4) for i in range(1,3)],
               'NARROW': [(i, 17) for i in range(1,7)],
               'LARGE': [(i, 32) for i in range(1,10)]}

N_EPISODS = {'DEBUG' : 2000,
             'NARROW': 30000,
             'LARGE' : 30000,}


# Parameters related to the problem (from book)
ALL_ACTIONS = [(acc_i, acc_j) for acc_i in range(-1,2) for acc_j in range(-1,2)]
N_ACTIONS = len(ALL_ACTIONS)

VEL_RANGE = list(range(5))
N_VEL = len(VEL_RANGE)

DISCOUNT_FACTOR = 1

NO_VEL_INCREMENT_THRESHOLD = 0.1


# Visualization
FIGURE_SIZE = (8,10)


# Utils
MAX_STEPS_THRESH = 200

USELESS_ACC_RWD_PENALTY = -100

#--------------

# Utils
class CellType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    FINISH = 3

Trajectory = namedtuple('Trajectory', ['s_a_r_hist', 'finish_pos'])


#--------------


class Grid:
    def __init__(self, map_type, fig_size=FIGURE_SIZE, start_line=START_LINE,
                 finish_line=FINISH_LINE, all_map_types=MAP_TYPES, map_dims=MAP_DIMS):

        assert map_type in all_map_types, 'Invalid map type.'
        self._map_type = map_type

        self._grid_h = map_dims[map_type].grid_h
        self._grid_w = map_dims[map_type].grid_w

        self._start_line = start_line[map_type]
        self._finish_line = finish_line[map_type]

        if map_type == 'DEBUG':
            self._data = self.create_debug_map()
        elif map_type == 'NARROW':
            self._data = self.create_narrow_map()
        elif map_type == 'LARGE':
            self._data = self.create_large_map()

        self._fig_size = fig_size

        self._celltype_num2color = {0:'w', 1:'k', 2:'r', 3:'g'}

    @property
    def width(self):
        return self._grid_w

    @property
    def height(self):
        return self._grid_h

    @property
    def start_line(self):
        return self._start_line

    @property
    def finish_line(self):
        return self._finish_line

    @property
    def data(self):
        return self._data

    def create_debug_map(self):
        map = np.zeros((self._grid_h, self._grid_w), dtype=np.int32)
        map[:, 0] = 1
        map[0, :] = 1
        map[3:, 3:] = 1

        for (i,j) in self._start_line:
            map[i, j] = 2
        for (i,j) in self._finish_line:
            map[i, j] = 3

        return map

    def create_large_map(self):
        """Hard coded narrow map to get the exact same track as the one proposed in the book (right figure p.112)."""
        map = np.zeros((self._grid_h, self._grid_w), dtype=np.int32)

        # TL
        map[:, 0] = 1
        map[0, :] = 1
        map[1, :17] = 1
        map[2, :14] = 1
        map[3, :13] = 1
        map[4:8, :12] = 1
        map[8, :13] = 1
        map[9, :14] = 1
        map[10:15, :15] = 1
        for inc in range(13):
            map[-4-inc, :(2+inc)] = 1

        # BR
        map[14:, 24:] = 1
        map[13:, 25:] = 1
        map[12:, 27:] = 1
        map[11:, 28:] = 1
        map[10:, 31:] = 1

        for (i,j) in self._start_line:
            map[i, j] = 2
        for (i,j) in self._finish_line:
            map[i, j] = 3

        return map


    def create_narrow_map(self):
        """Hard coded narrow map to get the exact same track as the one proposed in the book (left figure p.112)."""
        map = np.zeros((self._grid_h, self._grid_w), dtype=np.int32)

        # Top left corner
        map[:, 0] = 1
        map[0, :] = 1
        map[1, :4] = 1
        map[2:4, :3] = 1
        map[4, :2] = 1

        # Bottom left corner
        map[15:, 1] = 1
        map[23:, 2] = 1
        map[-3:, 3] = 1
        map[-1, :4] = 1

        # Bottom right corner
        map[8:, 11:] = 1
        map[7:, 12:] = 1


        for (i,j) in self._start_line:
            map[i, j] = 2

        for (i,j) in self._finish_line:
            map[i, j] = 3


        return map

    def cvt_ij2xy(self, i, j):
        x = j
        y = self._grid_h - 1 - i
        return x, y

    def draw(self, trajectory=None, export=False):

        fig, ax = plt.subplots(1, 1, figsize=self._fig_size)

        # Draw racetrack
        for i in range(self._grid_h):
            for j in range(self._grid_w):
                x,y = self.cvt_ij2xy(i,j)
                cell_val = self._data[i,j]

                cell_rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none',\
                                              facecolor=self._celltype_num2color[cell_val])
                ax.add_patch(cell_rect)


        # Draw trajectory
        if trajectory:

            for (state, action, _) in trajectory.s_a_r_hist:
                i, j, v_i, v_j = state
                acc_i, acc_j = action
                x, y = self.cvt_ij2xy(i, j)

                # Car position in blue
                car_pos = plt.Circle((x+0.5, y+0.5), 0.1, color='b')
                ax.add_patch(car_pos)

                # Car speed in cyan
                if (v_i, v_j) != (0,0):
                    ax.arrow(x + 0.5, y + 0.5, v_j, v_i, head_width=0.05, head_length=0.1,
                             fc='tab:cyan', ec='tab:cyan')

                # Attempted increment in orange
                if (acc_i, acc_j) != (0,0):
                    ax.arrow(x + 0.5, y + 0.5, acc_j, acc_i, ls=':' ,head_width=0.05, head_length=0.1,
                             fc='tab:orange', ec='tab:orange')

            # Finish position
            i, j = trajectory.finish_pos
            x, y = self.cvt_ij2xy(i, j)
            car_pos = plt.Circle((x+0.5, y+0.5), 0.1, color='b')
            ax.add_patch(car_pos)


        ax.set_xlim(0, self._grid_w)
        ax.set_ylim(0, self._grid_h)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

        if export:
            plt.savefig(f'racetrack_{self._map_type}.png'.lower())

        plt.waitforbuttonpress()




class Environment:
    def __init__(self, grid, all_actions=ALL_ACTIONS, max_step_tresh=MAX_STEPS_THRESH, max_vel=N_VEL-1,
                 useless_acc_reward_penalty=USELESS_ACC_RWD_PENALTY, no_vel_inc_thresh=NO_VEL_INCREMENT_THRESHOLD):
        self._grid = grid
        self._all_actions = all_actions
        self._max_step_tresh = max_step_tresh
        self._max_vel = max_vel
        self._useless_acc_reward_penalty = useless_acc_reward_penalty
        self._no_vel_inc_thresh = no_vel_inc_thresh

    def get_random_init_pos(self):
        """Returns a tuple (i,j) with corresponding to a randomly picked position of the starting line."""
        return self._grid.start_line[np.random.choice(len(self._grid.start_line))]


    def get_starting_state(self):
        """Returns a randomly picked initial state."""
        i, j = self.get_random_init_pos()
        v_i, v_j = (0, 0)
        return i, j, v_i, v_j


    def is_state_valid(self, state):
        i, j, v_i, v_j = state

        assert (i >= 0) and (i < self._grid.height), 'Invalid i position.'
        assert (j >= 0) and (j < self._grid.width), 'Invalid j position.'
        assert (v_i >= 0) and (v_i <= self._max_vel), 'Invalid i velocity.'
        assert (v_j >= 0) and (v_j <= self._max_vel), 'Invalid i velocity.'

        return True


    def is_hurting_obstacle(self, pos_from, pos_to):
        """Return true if the car is hurting an obstacle by moving from pos_from to pos_to in a single step."""

        if pos_from == pos_to:
            return False

        p1_i, p1_j = pos_from
        p2_i, p2_j = pos_to

        sampling = int(np.sqrt( (p1_i-p2_i)**2 + (p2_i-p2_j)**2 ))

        di, dj = 0, 0

        for it in range(sampling):
            di += abs(p2_i - p1_i) / sampling
            dj += abs(p2_j - p1_j) / sampling

            if self._grid.data[p1_i - int(np.round(di)), p1_j + int(np.round(dj))] == CellType.OBSTACLE.value:
                return True

        if self._grid.data[p2_i, p2_j] == CellType.OBSTACLE.value:
            return True

        return False

    def get_reward_penalty(self, state, action):
        """ Get additional penalty if a noisy action is taken.

        Noisy actions includes :
        - negative speed increment while having zero speed on that component ;
        - positive speed increment while having max speed on that component.

        Not specified in the book. Meant to avoid noisy arrows in optimal trajectory plot."""

        assert self.is_state_valid(state), 'Invalid state.'
        assert action in self._all_actions, 'Invalid action.'

        i, j, v_i, v_j = state
        acc_i, acc_j = action

        add_penalty = ((acc_i > 0) and (v_i == self._max_vel)) or \
                      ((acc_i < 0) and (v_i == 0)) or \
                      ((acc_j > 0) and (v_j == self._max_vel)) or \
                      ((acc_j < 0) and (v_j == 0))
        if add_penalty:
            return self._useless_acc_reward_penalty

        return 0


    def step(self, state, action):
        """ Apply action from state.

        :param state: State before the step. Tuple of shape (4,) containing (i, j, v_i, v_j), the
        car's position and speed.
        :param action: Action chosen by the agent. Tuple of shape (2,) containing (acc_i, acc_j) the attempted speed
        increment w.r.t. i and j. Ex : acc_i = 1 should imply v_i += 1.
        :return: (next_state, reward) tuple containing the next state (next_i, next_j, next_v_i, next_v_j).
        """
        assert self.is_state_valid(state), 'Invalid state.'
        assert action in self._all_actions, 'Invalid action.'


        # State : Car pos, Speed
        i, j, v_i, v_j = state
        acc_i, acc_j = action

        dr = self.get_reward_penalty(state, action)

        if np.random.random() < self._no_vel_inc_thresh:
            # Speed increment randomly set to 0
            next_v_i = v_i
            next_v_j = v_j
        else:
            next_v_i = v_i + acc_i
            next_v_j = v_j + acc_j
            next_v_i = np.clip(next_v_i, a_min=0, a_max=4)
            next_v_j = np.clip(next_v_j, a_min=0, a_max=4)

        next_i = (i - next_v_i) # i positive speed -> going up -> decrease i
        next_j = (j + next_v_j) # j positive speed -> going right -> increase j
        next_i = np.clip(next_i, a_min=0, a_max=self._grid.height-1)
        next_j = np.clip(next_j, a_min=0, a_max=self._grid.width-1)

        if self.is_hurting_obstacle(pos_from=(i, j), pos_to=(next_i, next_j)):
            return self.get_starting_state(), -1 + dr

        next_state = next_i, next_j, next_v_i, next_v_j

        if self._grid.data[next_i, next_j] == CellType.FINISH.value:
            return next_state, 0 + dr

        return next_state, -1 + dr


    def run_episode(self, policy, get_finish=False):
        """Run episode until the agent reached the terminal state."""

        n_steps = 0
        s_a_r_hist = [] # [(state_t, action_t, reward_t+1), ...]

        state = self.get_starting_state()
        action = policy(state)

        is_running = True

        while is_running:
            next_state, reward = self.step(state, action)

            s_a_r_hist.append((state, action, reward))
            n_steps += 1

            next_pos = next_state[0], next_state[1]
            if next_pos in self._grid.finish_line:
                is_running = False
            else:
                state = next_state
                action = policy(state)

            if get_finish :
                assert n_steps < self._max_step_tresh, 'Invalid optimal policy : too much steps to reach finish line. Try longer training.'


        if get_finish:
            finish_pos = next_pos
            return n_steps, s_a_r_hist, finish_pos

        return n_steps, s_a_r_hist



class Agent:

    def __init__(self, grid, n_vel=N_VEL, n_actions=N_ACTIONS,
                 gamma=DISCOUNT_FACTOR, all_actions=ALL_ACTIONS):

        # Parameters defining n_states
        grid_h, grid_w = grid.height, grid.width
        n_vel_i, n_vel_j = n_vel, n_vel

        # State actions values
        self._Q = np.full((grid_h, grid_w, n_vel_i, n_vel_j, n_actions), -100)  # n_states, n_actions

        # Cumulative sum of importance sampling ratios
        self._C = np.zeros((grid_h, grid_w, n_vel_i, n_vel_j, n_actions)) # n_states, n_actions

        # State to action policy
        self._policy = np.zeros((grid_h, grid_w, n_vel_i, n_vel_j), dtype=np.int8) # n_states

        # Discount factor
        self._gamma = gamma

        # All available actions
        self._all_actions = all_actions

        # Grid dimensions
        self._grid_h = grid_h
        self._grid_w = grid_w

        # Maximal velocity
        self._max_vel = n_vel_i -1


    @property
    def state_action_values(self):
        return self._Q

    def save_trained_parameters(self, verbose=True):

        np.save('Q_trained', self._Q)
        np.save('C_trained', self._C)
        np.save('policy_trained', self._policy)

        if verbose:
            print("Parameters successfully saved.")

    def load_trained_parameters(self, verbose=True):

        self._Q = np.load('Q_trained.npy')
        self._C = np.load('C_trained.npy')
        self._policy = np.load('policy_trained.npy')

        if verbose:
            print("Parameters successfully loaded.")

    def is_state_valid(self, state):
        i, j, v_i, v_j = state

        assert (i >= 0) and (i < self._grid_h ), 'Invalid i position.'
        assert (j >= 0) and (j < self._grid_w), 'Invalid j position.'
        assert (v_i >= 0) and (v_i <= self._max_vel), 'Invalid i velocity.'
        assert (v_j >= 0) and (v_j <= self._max_vel), 'Invalid i velocity.'

        return True

    def target_policy(self, state):
        """Follow target policy from state. Returns the corresponding action."""
        assert self.is_state_valid(state), 'Invalid state.'
        i, j, v_i, v_j = state

        ind_action = self._policy[i, j, v_i, v_j]
        action = self.cvt_ind2action(ind_action)

        return action


    def behavior_policy(self, state):
        """Return an action from state w.r.t. the behavior policy."""
        assert self.is_state_valid(state), 'Invalid state.'
        return self._all_actions[np.random.choice(len(self._all_actions))]


    def behavior_policy_prob(self, state, action):
        """Return probability to choose action from state by following behavior policy."""
        assert self.is_state_valid(state), 'Invalid state.'
        assert action in self._all_actions, 'Invalid action.'

        return 1/len(self._all_actions)


    def cvt_action2ind(self, action):
        """Return index of action in self._all_actions."""

        assert action in self._all_actions
        for ind, acc_ij in enumerate(self._all_actions):
            if acc_ij == action:
                return ind
        return -1


    def cvt_ind2action(self, ind_action):
        """Return the action that corresponds to ind_action in self._all_actions."""

        assert (ind_action >= 0) and (ind_action < len(self._all_actions)), 'Invalid action index.'
        return self._all_actions[ind_action]



    def run_episode_off_policy_mc_control(self, env, verbose=False):

        # Generate an episode
        T, s_a_r_hist = env.run_episode(self.behavior_policy)

        # Update state action values
        cumu_return = 0 # G
        importance_sampling_ratio = 1  # W

        for t in range(T-1, -1, -1): # [T-1, T-2, ..., 0]


            state, action, reward = s_a_r_hist[t] # (s_t, a_t, r_t+1)
            ind_action = self.cvt_action2ind(action)
            state_action = state + (ind_action,) # (i, j, v_i, v_j, acc_ind)

            cumu_return = self._gamma * cumu_return + reward

            self._C[state_action] += importance_sampling_ratio

            self._Q[state_action] += (importance_sampling_ratio / self._C[state_action]) * \
                                                   (cumu_return - self._Q[state_action])

            self._policy[state] = self._Q[state].argmax()

            if self._policy[state] != ind_action:
                # Move on to the next episode
                break;

            importance_sampling_ratio *= 1/self.behavior_policy_prob(state, action)


    def get_rand_init_optimal_trajectory(self, env, get_reward=False, verbose=True):

        n_steps, s_a_r_hist, finish_pos = env.run_episode(self.target_policy, get_finish=True)

        optimal_trajectory = Trajectory(s_a_r_hist, finish_pos)

        if verbose:
            print('Optimal trajectory :')
            for i_step, (state, action, reward) in enumerate(s_a_r_hist):
                i, j, v_i, v_j = state
                acc_i, acc_j = action
                print(f'[{i_step}] pos=({i},{j}), vel=({v_i},{v_j}) | action=({acc_i},{acc_j}) | reward={reward}')

            print(f'Finish pos : ({finish_pos[0]},{finish_pos[1]})')

        if get_reward:
            cumu_reward = 0
            for _, _, reward in s_a_r_hist:
                cumu_reward += reward

            return optimal_trajectory, cumu_reward

        return optimal_trajectory



def run_off_policy_mc_control(map_type, load_params=False):
    """ Apply off policy MC control method to learn the optimal trajectory on a given map.

    :param map_type: Name (str) of the map to train on.
    :return: None
     """

    grid = Grid(map_type=map_type)

    env = Environment(grid)
    agent = Agent(grid)

    if load_params:
        agent.load_trained_parameters()


    n_episodes = N_EPISODS[map_type]

    for e in tqdm(range(n_episodes)):
        agent.run_episode_off_policy_mc_control(env)

    # Save trained parameters
    agent.save_trained_parameters()

    # Get results
    optimal_trajectory = agent.get_rand_init_optimal_trajectory(env)

    grid.draw(trajectory=optimal_trajectory, export=True)




if __name__ == '__main__':

    #run_off_policy_mc_control(map_type='DEBUG', load_params=True) # ~ 1 minute

    #run_off_policy_mc_control(map_type='LARGE') # ~ 30 minutes

    run_off_policy_mc_control(map_type='NARROW') # Several hours


