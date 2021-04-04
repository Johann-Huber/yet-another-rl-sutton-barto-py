import numpy as np
import pandas as pd

from collections import deque

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


# Todo : everything works, but the 5 states trial is not implemented in a clean way.

# ----------------

# Number of states
N_STATES = 21
#N_STATES = 7 # 5 states try

# Id terminal states
ID_TERMINALS = [0, N_STATES-1]

# Default number of episodes
N_EPISODES = 10

# Default constant step-size parameter
ALPHA = 0.1

# Discount factor
GAMMA = 1

TARGET_VALUES = [i/10 for i in range(-9,10)]
#TARGET_VALUES = [i/3 for i in range(-2,3)] # 5 states try



FIGURE_SIZE = (12,12)

# ----------------

class Agent:

    def __init__(self, id_terminals=ID_TERMINALS, alpha=ALPHA, gamma=GAMMA,
                 target_values=TARGET_VALUES, n_states=N_STATES):

        self._values = np.zeros(n_states)

        self._n_returns = np.zeros(n_states)


        self._init_state = int(n_states/2) + 1

        self._terminal_states = id_terminals

        self._all_actions = [-1, 1]

        self._α = alpha

        self._γ = gamma

        self._target_values = np.array(target_values)

        self._error_hist = []

        self._all_episodes = []


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


    def run_episode_n_step_td(self, env, n_steps):
        """ Simulates a random walk episode. Update state values with the n-step TD algorithm.

        :param env: Random walk environment the agent interacts with.
        :param n_steps: Number of states to take into account in the n-step TD algorithm.
        :return: None
        """

        state = self._init_state

        t = 0
        T = np.inf

        states_buff = deque(maxlen=(n_steps + 1)) # [Sτ, Sτ+1, ... Sτ+n]
        states_buff.append(state)
        rewards_buff = deque(maxlen=(n_steps)) # [Rτ+1, ... Rτ+n]

        running = True
        while(running):

            if t < T:
                # Terminal state not reached
                action = np.random.choice(self._all_actions)  # Markov Random Process
                next_state, reward = env.step(state, action)

                states_buff.append(next_state)
                rewards_buff.append(reward)

                if next_state in self._terminal_states:
                    T = t + 1

                state = next_state

            τ = t - n_steps + 1

            if τ >= 0:
                # State updatable
                G = np.array([self._γ**i * rwd for i,rwd in enumerate(rewards_buff)]).sum()

                if τ + n_steps < T :
                    # Rewards beyond Rτ+n must be approximated
                    Sτ_n = states_buff[-1]
                    G += self._γ**n_steps * self._values[Sτ_n]

                # Update value
                Sτ = states_buff[0]
                self._values[Sτ] += self._α * (G - self._values[Sτ])

                if T - τ <= n_steps :
                    # Last steps before termination
                    states_buff.popleft()
                    rewards_buff.popleft()

            if τ == T - 1:
                running = False
            else:
                t += 1

        # Compute RMS error
        rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)
        self._error_hist.append(rms_err)



    def run(self, env, n_steps, n_episodes=N_EPISODES):

        for e in range(n_episodes):
            self.run_episode_n_step_td(env, n_steps)

            # Compute RMS error
            rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)
            self._error_hist.append(rms_err)



class Environment:

    def __init__(self, id_terminals=ID_TERMINALS, n_states=N_STATES):
        self._rewards = {key:0 for key in range(n_states)}
        self._rewards[0] = -1
        self._rewards[n_states-1] = 1


        self._terminal_states = id_terminals


    def step(self, state, action):

        assert state not in self._terminal_states
        assert action in [-1, 1]



        next_state = state + action
        reward = self._rewards[next_state]

        return next_state, reward



def get_n_step_random_walk_benchmark_plot():
    """Reproduces figure p.145"""

    env = Environment()

    n_runs = 1000
    n_ep = 10

    α_range = np.linspace(0., 1., num=40)

    rms_errs_all = {}

    n_steps_range = [2 ** i for i in range(10)]

    for n_steps in n_steps_range:

        n_steps_err_hists_over_α = []

        for _ in tqdm(range(n_runs)):

            err_hists_over_α = []

            for α in α_range:
                agent = Agent(alpha=α)

                agent.run(env, n_steps=n_steps, n_episodes=n_ep)

                avg_err = np.array(agent.error_hist).mean()
                err_hists_over_α.append(avg_err)

            n_steps_err_hists_over_α.append(err_hists_over_α)

        n_steps_err_hists_over_α = np.array(n_steps_err_hists_over_α).mean(0)

        rms_errs_all[f'n = {n_steps}'] = n_steps_err_hists_over_α

    # Plot
    df_vis = pd.DataFrame(rms_errs_all)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('α')
    ax.set_ylabel('Average RMS error over 19 states and first 10 episodes')
    ax.set_ylim(0.20, 0.55)
    plt.waitforbuttonpress()


if __name__ == '__main__':

    get_n_step_random_walk_benchmark_plot()
