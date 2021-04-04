import numpy as np
import pandas as pd
import copy
from collections import deque
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()

# ---------------

GRID_DIMS = (6, 9)

POS_START = (2, 0)
POS_GOAL = (0, 8)

POS_OBSTACLES = [(1,2), (2,2), (3,2), (4,5), (0,7), (1,7), (2,7)]

ALL_4_ACTIONS = [(i,j) for i in range(-1,2) for j in range(-1,2) if abs(i) != abs(j)]

# TD step size
ALPHA = 0.5

# Discount factor
GAMMA = 0.95

# Exploration ratio
EPSILON = 0.1

PRIORITY_THRESHOLD = 1e-4


FIGURE_SIZE = (12,8)

RANDOM_SEED = 2

# ---------------

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


class Grid:
    def __init__(self, dims=GRID_DIMS, pos_start=POS_START, pos_goal=POS_GOAL,
                 pos_obstacles=POS_OBSTACLES, fig_size=FIGURE_SIZE):
        self._h, self._w = dims

        self._pos_start = pos_start
        self._pos_goal = (0,self._w-1)
        self._pos_obstacles = pos_obstacles

        self._fig_size = fig_size

    @property
    def height(self):
        return self._h

    @property
    def width(self):
        return self._w

    @property
    def wind(self):
        return self._wind

    @property
    def pos_start(self):
        return self._pos_start

    @property
    def pos_goal(self):
        return self._pos_goal

    def is_valid_state(self, state):
        i, j = state
        return True if (i >= 0) and (i <= self._h-1) and (j >= 0) and (j <= self._w-1) else False

    def cvt_ij2xy(self, pos_ij):
        return pos_ij[1], self._h - 1 - pos_ij[0]

    def get_action_char(self, action):
        assert action in ALL_4_ACTIONS

        if action == (-1, 0):
            return '↑'
        if action == (0, -1):
            return '←'
        if action == (1, 0):
            return '↓'
        if action == (0, 1):
            return '→'

        return -1

    def draw(self, agent=None):
        fig, ax = plt.subplots(1, 1, figsize=self._fig_size)

        # cells
        for i in range(self._h):
            ax.plot([0, self._w], [i, i], color='k')
        for j in range(self._w):
            ax.plot([j, j], [0, self._h], color='k')

        # start & goal
        start_xy = self.cvt_ij2xy(self._pos_start)
        goal_xy = self.cvt_ij2xy(self._pos_goal)

        ax.text(start_xy[0] + 0.5, start_xy[1] + 0.5, 'S', fontsize=20, ha='center', va='center')
        ax.text(goal_xy[0] + 0.5, goal_xy[1] + 0.5, 'G', fontsize=20, ha='center', va='center')

        # obstacles
        for id, obs_ij in enumerate(self._pos_obstacles):
            x, y = self.cvt_ij2xy((obs_ij[0], obs_ij[1]))
            obs_rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor='tab:gray')
            ax.add_patch(obs_rect)

        # greedy actions
        if agent is not None:
            Q_indices = agent.get_greedy_policy_found()

            for i in range(self._h):
                for j in range(self._w):
                    if Q_indices[i,j] != -1:
                        action = agent.ind2action(Q_indices[i,j])
                        x, y = self.cvt_ij2xy((i, j))
                        arrow_char = self.get_action_char(action)
                        ax.text(x + 0.5, y + 0.5, arrow_char, fontsize=20, ha='center', va='center', color='b')

        ax.set_xlim(0, self._w)
        ax.set_ylim(0, self._h)

        plt.waitforbuttonpress()


class Environment:
    def __init__(self, grid, all_actions=ALL_4_ACTIONS):
        self._grid = grid
        self._all_actions = all_actions

    def step(self, state, action):

        assert self._grid.is_valid_state(state), 'Invalid state'
        assert action in self._all_actions, 'Invalid action'

        next_state = np.array(state) + np.array(action)

        next_state = (np.clip(next_state[0], a_min=0, a_max=self._grid.height-1),
                      np.clip(next_state[1], a_min=0, a_max=self._grid.width-1))

        if next_state in self._grid._pos_obstacles:
            next_state = state

        if next_state == self._grid.pos_goal:
            return next_state, 1.

        return next_state, 0.


class Agent:
    def __init__(self, grid, all_actions=ALL_4_ACTIONS,
                 alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, seed=RANDOM_SEED,
                 priority_thresh=PRIORITY_THRESHOLD, training_mode='Dyna-Q'):

        self._Q = np.zeros((grid.height, grid.width, len(all_actions))) # states, actions

        self.model = {}

        self._all_actions = all_actions

        self._grid = grid

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon

        self._rng = np.random.RandomState(seed)

        self._n_updates = 0

        assert training_mode in ['Dyna-Q', 'Prioritized_sweeping']
        self._training_mode = training_mode

        if training_mode == 'Prioritized_sweeping':
            # PQueue
            self._pq_idx2priority = {}  # {idx1: P1, ... }
            self._pq_idx2sa = {}  # {idx1: (state1, action1), ... }
            self._curr_idx = 0

            self._priority_thresh = priority_thresh


    @property
    def n_updates(self):
        return self._n_updates

    @property
    def step_episode_list(self):
        return self._step_episode_list

    @property
    def action_values(self):
        return self._Q

    def get_greedy_policy_found(self):
        """Returns a (grid.height, grid.width) matrix containing index of the greedy action follwing the
        found policy. Index is -1 when there is no maximum action value for the state."""

        Q_indices = np.zeros((self._grid.height, self._grid.width), dtype=np.int32) # states, actions

        for i in range(grid.height):
            for j in range(grid.width):
                greedy_action_inds = np.where(self._Q[i, j] == self._Q[i, j].max())[0]

                if len(greedy_action_inds) > 1:
                    Q_indices[i,j] = -1
                else:
                    Q_indices[i, j] = greedy_action_inds[0]

        return Q_indices


    def policy(self, state, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        assert self._grid.is_valid_state(state), 'Invalid state'

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = self._all_actions[self._rng.choice(range(len(self._all_actions)))]
            return action

        i, j = state

        greedy_action_inds = np.where(self._Q[i,j] == self._Q[i,j].max())[0]
        ind_action = self._rng.choice(greedy_action_inds)

        action = self._all_actions[ind_action]
        return action

    def get_start_pos(self):
        return self._grid.pos_start

    def is_terminal_state(self, state):
        return True if state == self._grid.pos_goal else False

    def action2ind(self, action):
        assert action in self._all_actions, 'Invalid action.'
        for ind, a in enumerate(self._all_actions):
            if action == a:
                return ind
        return -1

    def ind2action(self, ind):
        assert (ind >= 0) and ind <= len(self._all_actions)-1
        return self._all_actions[ind]


    def q_learning_update(self, env, state, action, planning=False):
        """Apply a q_learning update."""

        if planning:
            next_state, reward = self.model[(state, action)]
        else:
            next_state, reward = env.step(state, action)

        next_action = self.policy(next_state, explore_flg=False)  # pure greedy

        SA = state + (self.action2ind(action),)
        SA_next = next_state + (self.action2ind(next_action),)

        self._Q[SA] += self._α * reward + self._α * (self._γ * self._Q[SA_next] - self._Q[SA])

        self._n_updates += 1

        return next_state, reward


    def update_model(self, state, action, next_state, reward):
        """Update the model with (S,A) -> (S',R), assuming the environment is deterministic."""

        # update model
        state_cp = copy.deepcopy(state)
        action_cp = copy.deepcopy(action)
        next_state_cp = copy.deepcopy(next_state)
        reward_cp = copy.deepcopy(reward)

        self.model[(state_cp, action_cp)] = (next_state_cp, reward_cp)

    def is_optimal_policy_found(self, last_3_n_steps):
        if len(last_3_n_steps) < 3:
            # At least 3 runs to check convergence
            return False
        if (last_3_n_steps[0] == last_3_n_steps[1]) and (last_3_n_steps[1] == last_3_n_steps[2]) :
            return True

        return False

    def get_prev_SA_from_model(self, dest_state):
        """Return a list containing all (prev_state, prev_action) pairs that lead to the state dest_state,
        according to the constructed model."""

        prev_pairs = []
        for (state, action), (next_state, reward) in self.model.items():
            if next_state == dest_state:
                # (state, action) leads to the dest_state
                prev_pairs.append((state, action))

        return prev_pairs


    def get_priority(self, state, action, next_state, reward):

        next_action = self.policy(next_state, explore_flg=False)  # pure greedy
        SA = state + (self.action2ind(action),)
        SA_next = next_state + (self.action2ind(next_action),)

        P = abs(self._α * reward + self._α * (self._γ * self._Q[SA_next] - self._Q[SA]))
        return P, SA


    def add_to_PQueue(self, priority, SA_id):

        self._pq_idx2priority[self._curr_idx] = priority
        self._pq_idx2sa[self._curr_idx] = SA_id
        self._curr_idx += 1

    def remove_from_PQueue(self, idx):
        len_p, len_sa = len(self._pq_idx2priority), len(self._pq_idx2sa)
        self._pq_idx2priority.pop(idx)
        self._pq_idx2sa.pop(idx)
        assert (len_p == len(self._pq_idx2priority)+1) , (len_sa == len(self._pq_idx2sa)+1)

    def is_PQueue_empty(self):
        assert len(self._pq_idx2priority) == len(self._pq_idx2sa)
        return len(self._pq_idx2priority) == 0

    def run_PQueue_iteration(self):
        """Runs an iteration of training from Prioritized sweeping queue's head.
        1) Get the (state,action) pair corresponding to the maximal priority (largest expected modification
        of Q value) and update its Q value. Removes this pair from the PQueue ;
        2) Iterates over all (prev_state, prev_action) pairs that lead to state, and
        are therefore expected to be affected by the large Q value modification of (state, action)'s QValue.
        Compute their priority and add them to PQueue if the expected Q value modification is high enough."""

        ## 1 #

        # Get (state, action) with the highest P
        head_queue_idx = max(self._pq_idx2priority, key=self._pq_idx2priority.get)
        SA = self._pq_idx2sa[head_queue_idx]
        state, action = (SA[0], SA[1]), self.ind2action(SA[2])

        # Updates its Q value
        next_state, reward = self.model[(state, action)]
        next_action = self.policy(next_state, explore_flg=False)  # pure greedy
        SA_next = next_state + (self.action2ind(next_action),)
        self._Q[SA] += self._α * reward + self._α * (self._γ * self._Q[SA_next] - self._Q[SA])

        self._n_updates += 1

        # Remove it from the PQueue
        self.remove_from_PQueue(idx=head_queue_idx)


        ## 2 #

        # Get all (S_prev, A_prev) pairs that lead to S
        prev_pairs = self.get_prev_SA_from_model(state)
        for (state_prev, action_prev) in prev_pairs:

            updated_state, reward = self.model[(state_prev, action_prev)]
            assert updated_state == state

            P_prev, SA_prev = self.get_priority(state=state_prev, action=action_prev, next_state=updated_state, reward=reward)
            if P_prev > self._priority_thresh:
                self.add_to_PQueue(priority=P_prev, SA_id=SA_prev)



    def run_prioritized_sweeping(self, env, n_episodes_max, n_step_ps):
        """Runs prioritized sweeping learning. Returns True if the optimal policy has been found,
        False otherwise."""

        assert self._training_mode == 'Prioritized_sweeping'

        last_3_n_steps = deque(maxlen=3)

        for e in tqdm(range(n_episodes_max)):
            # New episode
            curr_state = self.get_start_pos()

            n_steps = 0

            running = True
            while (running):

                # Experience
                state = curr_state
                action = self.policy(state)
                next_state, reward = env.step(state, action)
                self.update_model(state, action, next_state, reward)

                P, SA = self.get_priority(state=state, action=action, next_state=next_state, reward=reward)
                if P > self._priority_thresh:
                    self.add_to_PQueue(priority=P, SA_id=SA)

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    curr_state = next_state

                for _ in range(n_step_ps):
                    if self.is_PQueue_empty():
                        break

                    self.run_PQueue_iteration()

                n_steps += 1

            # End of episode
            last_3_n_steps.append(n_steps)

            if self.is_optimal_policy_found(last_3_n_steps):
                return True # converged

        return False # not converged


    def run_tabular_dyna_q(self, env, n_episodes_max, n_planning_steps):
        """Runs dyna_q learning. Returns True if the optimal policy has been found, False otherwise."""

        assert self._training_mode == 'Dyna-Q'

        last_3_n_steps = deque(maxlen=3)

        for e in tqdm(range(n_episodes_max)):
            curr_state = self.get_start_pos()

            n_steps = 0

            running = True
            while (running):

                # Experience
                state = curr_state
                action = self.policy(state)

                next_state, reward = self.q_learning_update(env, state, action)

                self.update_model(state, action, next_state, reward)

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    curr_state = next_state

                # Planning
                for _ in range(n_planning_steps):

                    state, action = random.choice(list(self.model.keys()))

                    self.q_learning_update(env, state, action, planning=True)

                n_steps += 1

            last_3_n_steps.append(n_steps)

            if self.is_optimal_policy_found(last_3_n_steps):
                return True # converged

        return False # not converged


def run_prioritized_sweeping_experiment():
    """Reproduces experiment proposed p.170 of the book."""

    n_episodes_max = 500
    n_planning_steps = 5
    n_runs = 10
    n_dim_max = 4  # to limit computation time

    # Generate a range a gridworld resolution
    all_dims = [(3, 4)] + [(i * 3, i * 5) for i in range(2, 22, 2)]

    all_n_update_lists = {}

    for training_mode in ['Dyna-Q', 'Prioritized_sweeping']:

        all_n_updates = []  # [[resolution1,n_update_1], ... ]

        for dims in all_dims[:n_dim_max]:
            print('dims=', dims)

            grid = Grid(dims=dims)
            env = Environment(grid)

            all_n_update_grid_config = []
            for i_run in range(n_runs):

                assert training_mode in ['Dyna-Q', 'Prioritized_sweeping']

                if training_mode == 'Dyna-Q':
                    agent = Agent(grid, training_mode='Dyna-Q')
                    is_converged = agent.run_tabular_dyna_q(env, n_episodes_max, n_planning_steps=n_planning_steps)
                else:
                    agent = Agent(grid, training_mode='Prioritized_sweeping')
                    is_converged = agent.run_prioritized_sweeping(env, n_episodes_max, n_step_ps=5)

                assert is_converged, 'Unable to find optimal policy. Try longer training.'
                print(f'Dims {dims} | Run {i_run} | Convergence : OK')
                all_n_update_grid_config.append(agent.n_updates)

            # Avg over all runs
            all_n_update_grid_config = np.array(all_n_update_grid_config).mean()

            # Stores results corresponding to the current grid resolution
            all_n_updates.append([dims[0] * dims[1], all_n_update_grid_config])

        # Stores results corresponding to the current training mode
        all_n_update_lists[training_mode] = np.array(all_n_updates)

    # Plot output
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=all_n_update_lists['Dyna-Q'][:, 0],
                 y=all_n_update_lists['Dyna-Q'][:, 1], ax=ax, label='Dyna-Q')
    sns.lineplot(x=all_n_update_lists['Prioritized_sweeping'][:, 0],
                 y=all_n_update_lists['Prioritized_sweeping'][:, 1], ax=ax, label='Prioritized_sweeping')
    ax.set_xlabel('Gridworld size (#states)')
    ax.set_ylabel('Updates unit optimal solution')
    ax.legend()
    plt.title('Prioritized Sweeping on Mazes', fontsize=18)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    run_prioritized_sweeping_experiment()