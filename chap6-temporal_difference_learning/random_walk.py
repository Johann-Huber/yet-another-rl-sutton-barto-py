import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

# ----------------

# Number of states
N_STATES = 7

# Id terminal states
ID_TERMINALS = [0,6]

# Id initial state
INIT_STATE = 3

# Default number of episodes
N_EPISODES = 100

# Constant step-size parameter
ALPHA = 0.1

# Discount factor
GAMMA = 1

# Targeted values
TARGET_VALUES = [i/6 for i in range(1,6)]

# Threshold for sum of increments during batch training.
# Under this threshold, we assume that state values have converged
INCREMENT_ERROR_TRESH = 1e-3

# Figure dimensions for plotting
FIGURE_SIZE = (12,12)

# ----------------

class Agent:

    def __init__(self, id_terminals=ID_TERMINALS, init_state=INIT_STATE, alpha=ALPHA, gamma=GAMMA,
                 target_values=TARGET_VALUES, err_increment_tresh=INCREMENT_ERROR_TRESH):

        self._values = np.full(N_STATES, 0.5)
        for i_term in id_terminals:
            # By convention : those terms we be skipped anyway
            self._values[i_term] = 0

        self._n_returns = np.zeros(N_STATES)

        self._init_state = init_state

        self._terminal_states = id_terminals

        self._all_actions = [-1, 1]

        self._α = alpha

        self._γ = gamma

        self._target_values = np.array(target_values)

        self._error_hist = []

        self._all_episodes = []

        self._err_inc_tresh = err_increment_tresh


    @property
    def values(self):
        return self._values[1:-1]


    @property
    def error_hist(self):
        return self._error_hist


    def run_episode_td(self, env):

        state = self._init_state

        running = True

        while running:
            action =  np.random.choice(self._all_actions) # Markov Random Process
            next_state, reward = env.step(state, action)

            self._values[state] += self._α * (reward + self._γ * self._values[next_state] - self._values[state])

            if next_state in self._terminal_states:
                running = False
            else:
                state = next_state


        # Compute RMS error
        rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)
        self._error_hist.append(rms_err)


    def run_episode_td_batch(self, env):

        hist_episode = []
        state = self._init_state

        # Compute episode

        running = True

        while running:
            action =  np.random.choice(self._all_actions) # Markov Random Process
            next_state, reward = env.step(state, action)

            hist_episode.append((state, action, next_state, reward))

            if next_state in self._terminal_states:
                running = False
            else:
                state = next_state

        self._all_episodes.append(hist_episode)


        # Batch update
        state_value_converged = False
        while not state_value_converged:
            cumu_increments = np.zeros(len(self._values))

            for hist_episode in self._all_episodes:
                for (state, action, next_state, reward) in hist_episode:
                    cumu_increments[state] += self._α * (reward + self._γ * self._values[next_state] - self._values[state])

            if np.absolute(cumu_increments).sum() < self._err_inc_tresh:
                state_value_converged = True
                continue

            for state in range(len(self._values)):
                self._values[state] += cumu_increments[state]


        # Compute RMS error
        rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)

        self._error_hist.append(rms_err)



    def run_episode_mc(self, env):

        T = 0
        s_a_r_hist = []

        state = self._init_state

        # Compute episode steps

        running = True

        while running:

            action = np.random.choice(self._all_actions)  # Markov Random Process
            next_state, reward = env.step(state, action)

            s_a_r_hist.append((state, action, reward))
            T += 1

            if next_state in self._terminal_states:
                running = False
            else:
                state = next_state

        # Update values
        cumu_return = 0  # G

        for t in range(T - 1, -1, -1):  # [T-1, T-2, ..., 0]
            state, action, reward = s_a_r_hist[t]
            cumu_return = self._γ * cumu_return + reward

            self._values[state] += self._α * (cumu_return - self._values[state])

        # Compute RMS error
        rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)
        self._error_hist.append(rms_err)


    def run_episode_mc_batch(self, env):

        hist_episode = []
        T = 0
        s_a_r_hist = []

        state = self._init_state

        # Compute episode steps

        running = True

        while running:

            action = np.random.choice(self._all_actions)  # Markov Random Process
            next_state, reward = env.step(state, action)

            hist_episode.append((state, action, next_state, reward))
            s_a_r_hist.append((state, action, reward))
            T += 1

            if next_state in self._terminal_states:
                running = False
            else:
                state = next_state

        self._all_episodes.append(hist_episode)


        # Batch update
        state_value_converged = False
        while not state_value_converged :
            cumu_increments = np.zeros(len(self._values))

            for hist_episode in self._all_episodes:

                T = len(hist_episode)
                cumu_return = 0

                for t in range(T - 1, -1, -1):  # [T-1, T-2, ..., 0]
                    state, action, next_state, reward = hist_episode[t]

                    cumu_return = self._γ * cumu_return + reward

                    cumu_increments[state] += self._α * (cumu_return - self._values[state])

            if np.absolute(cumu_increments).sum() < self._err_inc_tresh:
                state_value_converged = True
                continue

            for state in range(len(self._values)):
                self._values[state] += cumu_increments[state]
            

        # Compute RMS error
        rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)
        self._error_hist.append(rms_err)


    def run(self, env, method='TD', batch_training=False, n_episodes=N_EPISODES):

        assert method in ['TD', 'MC']
        assert batch_training in [True, False]

        if method == 'TD':
            if not batch_training:
                for e in range(n_episodes):
                    self.run_episode_td(env)
            else:
                for e in range(n_episodes):
                    self.run_episode_td_batch(env)
        elif method == 'MC':
            if not batch_training:
                for e in range(n_episodes):
                    self.run_episode_mc(env)
            else:
                for e in range(n_episodes):
                    self.run_episode_mc_batch(env)



class Environment:

    def __init__(self, id_terminals=ID_TERMINALS, n_states=N_STATES):
        self._rewards = {key:0 for key in range(n_states)}
        self._rewards[6] = 1

        self._terminal_states = id_terminals


    def step(self, state, action):

        assert state not in self._terminal_states
        assert action in [-1, 1]

        next_state = state + action
        reward = self._rewards[next_state]

        return next_state, reward



def td_learning_value_curves():
    """Reproduces left graph, p125 of the book."""

    # Apply TD learning

    env = Environment()

    estimated_values = {}

    n_episodes_list = [0, 1, 10, 100]
    for n_ep in n_episodes_list:
        agent = Agent()
        agent.run(env, method='TD', n_episodes=n_ep)

        estimated_values[n_ep] = agent.values


    # Plot output curves
    
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    for n_ep in n_episodes_list:
        sns.lineplot(x=range(len(estimated_values[n_ep])), y=estimated_values[n_ep],
                     marker='o', ax=ax, label=str(n_ep))

    sns.lineplot(x=range(len(TARGET_VALUES)), y=TARGET_VALUES,
                 marker='o', ax=ax, label='True values')
    ax.legend()
    ax.set_xlabel('States')
    ax.set_ylabel('Values')
    plt.xticks(np.arange(5), ('A', 'B', 'C', 'D', 'E'))

    plt.waitforbuttonpress()


def td_mc_rms_error_comparison():
    """Reproduces right graph, p125 of the book."""

    env = Environment()

    # Compute RMS error

    methods = ['TD', 'MC']
    n_runs = 100

    alphas = {'TD': [0.15, 0.1, 0.05],
              'MC': [0.01, 0.02, 0.03, 0.04]}

    rms_errs_all = {}

    for method in methods:
        for α in alphas[method]:

            all_errors = []

            for r in tqdm(range(n_runs)):
                agent = Agent(alpha=α)

                agent.run(env, method=method, n_episodes=100)
                all_errors.append(agent.error_hist)

            all_errors = np.array(all_errors)
            all_errors = all_errors.mean(axis=0)

            rms_errs_all['α_' + str(α) + '_' + method] = all_errors

    # Plots
    df_vis = pd.DataFrame(rms_errs_all)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, marker='o', ax=ax)
    ax.set_xlabel('Walks / Episodes')
    ax.set_ylabel('Empirical RMS error (averaged over states)')

    plt.waitforbuttonpress()


def td_mc_rms_error_comparison_batch_training():
    """Reproduces figure p127 of the book."""

    env = Environment()
    methods = ['TD', 'MC']
    n_runs = 100

    rms_errs_all = {}

    for method in methods:

        α = 0.001
        all_errors = []

        for r in tqdm(range(n_runs)):
            agent = Agent(alpha=α)

            agent.run(env, method=method, batch_training=True, n_episodes=100)
            all_errors.append(agent.error_hist)

        all_errors = np.array(all_errors)
        all_errors = all_errors.mean(axis=0)

        rms_errs_all['α_' + str(α) + '_' + method + '_batch'] = all_errors

    # Plots
    df_vis = pd.DataFrame(rms_errs_all)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('Walks / Episodes')
    ax.set_ylabel('Empirical RMS error (averaged over states)')
    ax.set_title('BATCH TRAINING', fontsize=18)
    plt.savefig('random_walk_batch_training')
    plt.waitforbuttonpress()


if __name__ == '__main__':

    # Reproduces left graph p125
    td_learning_value_curves()

    # Reproduces right graph p125
    td_mc_rms_error_comparison()

    # Reproduces figure p127
    td_mc_rms_error_comparison_batch_training()

