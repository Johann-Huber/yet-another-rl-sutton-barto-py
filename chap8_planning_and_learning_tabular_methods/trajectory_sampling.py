
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()

# ----------------

DEBUG = False

N_STATES = 10 if DEBUG else 1000
TERMINAL_ID = 10 if DEBUG else 1000
N_ACTIONS = 2

# TD step size
ALPHA = 0.1 #0.5

# Discount factor
GAMMA = 1.

# Exploration ratio
EPSILON = 0.1

FIGURE_SIZE = (16,12)

# ----------------

class Agent:
    def __init__(self, branching_factor, n_actions=N_ACTIONS, n_states=N_STATES, terminal_id=TERMINAL_ID,
                 alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):

        # Note : Some of those parameters could be associated to an environment object. In this experiment, we
        # only work with simulated transition from an expected model ; that is why it makes sense that Agent
        # object does have those parameters.

        self._Q = np.zeros((n_states + 1, n_actions)) # + terminal state

        self._b = branching_factor
        self._n_actions = n_actions
        self._n_states = n_states
        self._terminal_id = terminal_id

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon


        # {(s0,a0): [list_of_all_next_states], (s0,a1): [...], (s1,a0): [...], ...}
        self._transitions = {(state, action) : list(np.random.choice(range(n_states), self._b))
                             for state in range(n_states)
                             for action in range(n_actions)}

        self._rewards = {(state, action) : list(np.random.normal(0., 1., self._b))
                             for state in range(n_states)
                             for action in range(n_actions)}


        self._n_updates = 0


    @property
    def action_values(self):
        return self._Q

    def model(self, state, action):
        """Simulated model."""

        if state == self._terminal_id:
            # Already at terminal state
            return state, 0.

        if np.random.rand() < 0.1:
            # Terminal state
            next_state = self._terminal_id
            return next_state, 0.
        else :
            i_next_state = np.random.choice(range(self._b))
            next_state =  self._transitions[(state, action)][i_next_state]
            reward = self._rewards[(state, action)][i_next_state]

            return next_state, reward

        #reward = np.random.normal(0.,1.) # (i) once at the beginning ? Or each time like this ?


    def get_start_pos(self):
        return 0


    def q_learning_update(self, state, action, next_state, reward):
        """Apply a q_learning update."""

        next_action = self.policy(next_state, explore_flg=False)  # pure greedy

        SA = (state,) + (action,)
        SA_next = (next_state,) + (next_action,)

        self._Q[SA] += self._α * reward + self._α * (self._γ * self._Q[SA_next] - self._Q[SA])


    def policy(self, state, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        assert (state >= 0) and (state >= 0 <= self._n_states), 'Invalid state'

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = np.random.choice(range(self._n_actions))
            return action

        greedy_action_inds = np.where(self._Q[state] == self._Q[state].max())[0]
        action = np.random.choice(greedy_action_inds)
        return action


    def get_true_value_init_state(self):
        curr_state = self.get_start_pos()

        V = 0
        running = True
        while (running):

            state = curr_state
            action = self.policy(state, explore_flg=False)

            next_state, reward = self.model(state, action)
            V += reward

            if next_state == self._terminal_id:
                running = False
            else:
                curr_state = next_state

        return V


    def run_uniform_training(self, max_n_update, step_get_v0):
        """Randomly pick from all the state-action space and update the Q value in-place."""

        v0_hist = [(0, self.get_true_value_init_state())]

        for _ in range(max_n_update):

            state = np.random.choice(range(self._n_states))
            action = np.random.choice(range(self._n_actions))

            next_state, reward = self.model(state, action)

            self.q_learning_update(state, action, next_state, reward)
            self._n_updates += 1

            if (self._n_updates % step_get_v0) == 0:
                v0_hist.append((self._n_updates, self.get_true_value_init_state()))

        return np.array(v0_hist)


    def run_on_policy_training(self, max_n_update, step_get_v0, n_episodes_max=500):
        """Train through trajectory sampling ; simulates episodes, updates Q values that correspond to
        each selected (state, action) pair by following epsilon-greedy policy."""

        v0_hist = [(0, self.get_true_value_init_state())]


        for e in range(n_episodes_max):
            curr_state = self.get_start_pos()

            running = True
            while (running):

                # Experience
                state = curr_state
                action = self.policy(state)

                next_state, reward = self.model(state, action)

                self.q_learning_update(state, action, next_state, reward)
                self._n_updates += 1

                if (self._n_updates % step_get_v0) == 0:
                    v0_hist.append((self._n_updates, self.get_true_value_init_state()))

                if next_state == self._terminal_id:
                    running = False
                else:
                    curr_state = next_state

                if self._n_updates == max_n_update:
                    return np.array(v0_hist)

        assert False, 'Trajectory sampling over while max_n_update is not reached. Try longer training.'



def reproduce_upper_figure_p_176():
    """Reproduces the upper part of the figure p.176"""
    n_runs = 200  # 1000
    max_n_update = 20000
    step_get_v0 = 5  # 1000
    all_branching_factors = [1, 3, 10]

    all_v0_hists = {}

    for b in all_branching_factors:
        for method in ['uniform', 'on_policy']:
            curr_v0_hists = []
            for _ in tqdm(range(n_runs)):
                agent = Agent(branching_factor=b)
                if method == 'uniform':
                    v0_hist = agent.run_uniform_training(max_n_update=max_n_update,
                                                         step_get_v0=step_get_v0)
                else:
                    v0_hist = agent.run_on_policy_training(max_n_update=max_n_update,
                                                           step_get_v0=step_get_v0,
                                                           n_episodes_max=10000)

                curr_v0_hists.append(v0_hist)

            curr_v0_hists = np.array(curr_v0_hists).mean(0)

            all_v0_hists[method + f'_b{b}'] = curr_v0_hists

    # Plot output
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for b in all_branching_factors:
        for method in ['uniform', 'on_policy']:
            method_str = method + f'_b{b}'
            sns.lineplot(x=all_v0_hists[method_str][:, 0],
                         y=all_v0_hists[method_str][:, 1], ax=ax, label=method_str)
    ax.set_xlabel('Computation time, in expected updates')
    ax.set_ylabel('Value of the start state under greedy policy.')
    ax.legend()
    plt.title('Trajectory sampling', fontsize=18)
    plt.savefig('trajectory_sampling_8_8_up')
    plt.waitforbuttonpress()

def reproduce_lower_figure_p_176():
    """Reproduces the lower part of the figure p.176"""
    n_runs = 20 #200  # 1000
    max_n_update = 50000 #200000
    step_get_v0 = 5  # 1000
    all_branching_factors = [1]

    all_v0_hists = {}

    for b in all_branching_factors:
        for method in ['uniform', 'on_policy']:
            curr_v0_hists = []
            for _ in tqdm(range(n_runs)):
                agent = Agent(branching_factor=b, n_states=10000, terminal_id=10000)
                if method == 'uniform':
                    v0_hist = agent.run_uniform_training(max_n_update=max_n_update,
                                                         step_get_v0=step_get_v0)
                else:
                    v0_hist = agent.run_on_policy_training(max_n_update=max_n_update,
                                                           step_get_v0=step_get_v0,
                                                           n_episodes_max=10000)

                curr_v0_hists.append(v0_hist)

            curr_v0_hists = np.array(curr_v0_hists).mean(0)

            all_v0_hists[method + f'_b{b}'] = curr_v0_hists

    # Plot output
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for b in all_branching_factors:
        for method in ['uniform', 'on_policy']:
            method_str = method + f'_b{b}'
            sns.lineplot(x=all_v0_hists[method_str][:, 0],
                         y=all_v0_hists[method_str][:, 1], ax=ax, label=method_str)
    ax.set_xlabel('Computation time, in expected updates')
    ax.set_ylabel('Value of the start state under greedy policy.')
    ax.legend()
    plt.title('Trajectory sampling', fontsize=18)
    plt.savefig('trajectory_sampling_8_8_down')
    plt.waitforbuttonpress()


if __name__ == '__main__':

    #reproduce_upper_figure_p_176()

    reproduce_lower_figure_p_176()
