import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


# ----------------

# Number of states
N_STATES = 21 # [term=0] [1, ... , 19] [term=20]

# Id terminal states
ID_TERMINALS = [0, N_STATES-1]

# Default number of episodes
N_EPISODES = 10

# Default constant step-size parameter
ALPHA = 0.5

# Discount factor
GAMMA = 1

# Exponential weighting decrease parameter
LAMBDA = 0.8

TARGET_VALUES = [i/10 for i in range(-9,10)]

FIGURE_SIZE = (12,12)

# ----------------


class Environment:

    """Same task as the random walk from chapter 7."""

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



class Agent:

    def __init__(self, lmbda=LAMBDA, id_terminals=ID_TERMINALS, alpha=ALPHA, gamma=GAMMA,
                 target_values=TARGET_VALUES, n_states=N_STATES):

        self._w = np.zeros(n_states)

        self._z = np.zeros(n_states)

        self._init_state = int(n_states/2) + 1

        self._terminal_states = id_terminals
        self._n_states_max = n_states

        self._all_actions = [-1, 1]

        self._α = alpha
        self._γ = gamma
        self._λ = lmbda

        self._target_values = np.array(target_values)

        self._error_hist = []

        self._all_episodes = []


    @property
    def values(self):
        return self._values[1:-1]

    @property
    def error_hist(self):
        return self._error_hist

    def get_all_v_hat(self):
        all_v_hats = np.zeros(self._n_states_max)

        all_states = range(self._n_states_max)
        for s in all_states:
            all_v_hats[s] = self.v_hat(s)

        return all_v_hats[1:-1]


    def policy(self, state):
        """Action selection : uniform distribution."""
        assert (state >= 0) and (state <= self._n_states_max)
        return np.random.choice(self._all_actions)

    def v_hat(self, state):
        """Returns the approximated value for state, w.r.t. the weight vector."""

        if state in self._terminal_states:
            # By convention : R(S(T)) = 0
            return 0.

        value = self._w[state]
        return value



    def generate_episode(self, env):

        s_hist = [] # [S0, S1, ... S_T] -> len = T + 1
        r_hist = [] #     [R1, ... S_T] -> len = T

        curr_state = self._init_state
        s_hist.append(curr_state)

        running = True
        while running:

            state = curr_state
            action = self.policy(state)

            next_state, reward = env.step(state, action)

            s_hist.append(next_state)
            r_hist.append(reward)

            if next_state in self._terminal_states:
                running = False
            else:
                curr_state = next_state

        return s_hist, r_hist


    def run_offline_lambda_return_method(self, env, n_episodes):
        """Method described p288 of the book."""

        for n_ep in range(n_episodes):

            # 1) Run episode
            s_hist, r_hist = self.generate_episode(env)
            T = len(r_hist)

            for t in range(T):

                # 2) Get the n-steps updates associated to each horizons for t
                all_G = []

                for h in range(t+1, T+1):

                    curr_G_step = np.array([self._γ ** i * rwd for i, rwd in enumerate(r_hist[t:h])]).sum()
                    #= R(t+1) + γ*R(t+2) + ... + γ**(h-t-1)*R(h)

                    if h <= T-1 :
                        # Episode's end is not reached yet : following returns must be approximated
                        S_horizon = s_hist[h]
                        curr_G_step += self._γ ** (h-t) * self.v_hat(S_horizon)

                    all_G.append(curr_G_step)

                # all_G = [G(t:t+1), G(t:t+2), ..., G(t:t+T)] -> len = T - t - 1

                # 3) Update weights

                G_λ_t = [self._λ ** i * all_G[i] for i in range(0, T - t - 1)]
                G_λ_t = (1 - self._λ) * (np.array(G_λ_t).sum())
                G_λ_t += self._λ ** (T - t - 1) * all_G[T - t - 1]
                #= (1 - self._λ)(G(t:t+1) + λ*G(t:t+2) + ... + λ**(T-t-2)*G(t:T-1)) + λ**(T-t-1) * G(t)


                # Note : grad_v_hat(St) = 1 at the weight component corresponding to St, else 0
                St = s_hist[t]
                self._w[St] += self._α * (G_λ_t - self.v_hat(St))

            # Compute RMS error
            rms_err = mean_squared_error(self._target_values, self.get_all_v_hat(), squared=False)
            self._error_hist.append(rms_err)


    def run_semi_grad_td_lambda_return_method(self, env, n_episodes):
        """Method described p293 of the book."""

        for n_ep in range(n_episodes):

            curr_state = self._init_state
            self._z = np.zeros(self._n_states_max)

            running = True
            while running:

                state = curr_state
                action = self.policy(state)

                next_state, reward = env.step(state, action)

                # Traces vector
                grad_v_hat = np.zeros(len(self._z))
                grad_v_hat[state] = 1
                self._z = self._γ * self._λ * self._z + grad_v_hat

                # Moment-by-moment TD error
                δ = reward + self._γ * self.v_hat(next_state) - self.v_hat(state)

                # Weight vector update
                self._w += self._α * δ * self._z

                if next_state in self._terminal_states:
                    running = False
                else:
                    curr_state = next_state

            # Compute RMS error
            rms_err = mean_squared_error(self._target_values, self.get_all_v_hat(), squared=False)
            self._error_hist.append(rms_err)



    def run_online_lamba_return_method(self, env, n_episodes):
        """Method described p297 of the book.

        Note : it is computationally heavy (as expected from the theory)."""

        for n_ep in range(n_episodes):

            H = 0 # current horizon (= num steps)

            s_hist = []  # [S0, S1, ... S_H] -> len = H + 1
            r_hist = []  # [R1, ... S_H] -> len = H

            curr_state = self._init_state
            s_hist.append(curr_state)

            running = True
            while running:
                state = curr_state
                action = self.policy(state)

                # 1) Observe next state & reward
                next_state, reward = env.step(state, action)

                s_hist.append(next_state)
                r_hist.append(reward)
                H += 1

                for t in range(H):

                    # 2) Get the n-steps updates associated to each sub-horizons for t
                    all_G = []

                    for h in range(t + 1, H + 1):

                        curr_G_step = np.array([self._γ ** i * rwd for i, rwd in enumerate(r_hist[t:h])]).sum()
                        # = R(t+1) + γ*R(t+2) + ... reward+ γ**(h-t-1)*R(h)

                        if (h < H) or (next_state not in self._terminal_states):
                            # Episode's end is not reached yet : following returns must be approximated
                            S_horizon = s_hist[h]
                            curr_G_step += self._γ ** (h - t) * self.v_hat(S_horizon)

                        all_G.append(curr_G_step)

                    # all_G = [G(t:t+1), G(t:t+2), ..., G(t:t+H)] -> len = H - t - 1

                    # 3) Update weights

                    G_λ_t = [self._λ ** i * all_G[i] for i in range(0, H - t - 1)]
                    G_λ_t = (1 - self._λ) * (np.array(G_λ_t).sum())
                    G_λ_t += self._λ ** (H - t - 1) * all_G[H - t - 1]
                    # = (1 - self._λ)(G(t:t+1) + λ*G(t:t+2) + ... + λ**(H-t-2)*G(t:H-1)) + λ**(H-t-1) * G(t:H)

                    # Note : grad_v_hat(St) = 1 at the weight component corresponding to St, else 0
                    St = s_hist[t]
                    self._w[St] += self._α * (G_λ_t - self.v_hat(St))


                if next_state in self._terminal_states:
                    running = False
                else:
                    curr_state = next_state

            # Compute RMS error
            rms_err = mean_squared_error(self._target_values, self.get_all_v_hat(), squared=False)
            self._error_hist.append(rms_err)


    def run_true_online_td_lambda(self, env, n_episodes):
        """Method described p300 of the book."""

        for n_ep in range(n_episodes):

            curr_state = self._init_state
            self._z = np.zeros(self._n_states_max)
            V_old = 0

            running = True
            while running:
                state = curr_state
                action = self.policy(state)

                next_state, reward = env.step(state, action)

                # Moment-by-moment TD error
                δ = reward + self._γ * self.v_hat(next_state) - self.v_hat(state)

                # x(s) vector ( w.transpose() * x(s) = v_hat(s) )
                x_s = np.zeros(len(self._z))
                x_s[state] = 1

                # Dutch trace
                self._z = self._γ * self._λ * self._z + \
                          (1 - self._α * self._γ * self._λ * (self._z * x_s).sum()) * x_s

                # Weight vector update
                self._w += self._α * (δ + self.v_hat(state) - V_old) * self._z - \
                           self._α * (self.v_hat(state) - V_old) * x_s


                V_old = self.v_hat(next_state)

                if next_state in self._terminal_states:
                    running = False
                else:
                    curr_state = next_state

            # Compute RMS error

            rms_err = mean_squared_error(self._target_values, self.get_all_v_hat(), squared=False)
            self._error_hist.append(rms_err)




def get_offline_lambda_returns_error_curves():
    """Reproduces curves p.291 of the book."""

    env = Environment()

    α_range = np.linspace(0., 1., num=40)
    lmbda_range = [0., 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.]
    n_runs = 50

    rms_errs_all = {}

    for lmbda in lmbda_range:

        lambda_err_hists_over_α = []

        for n_run in tqdm(range(n_runs)):

            err_hists_over_α = []

            for α in tqdm(α_range):
                agent = Agent(alpha=α, lmbda=lmbda)

                agent.run_offline_lambda_return_method(env, n_episodes=10)

                avg_err = np.array(agent.error_hist).mean()
                err_hists_over_α.append(avg_err)

            lambda_err_hists_over_α.append(err_hists_over_α)

        lambda_err_hists_over_α = np.array(lambda_err_hists_over_α).mean(0)

        rms_errs_all[f'λ = {lmbda}'] = lambda_err_hists_over_α


    # Plot
    df_vis = pd.DataFrame(rms_errs_all)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('α')
    ax.set_ylabel('Average RMS error over 19 states and first 10 episodes')
    ax.set_ylim(0.20, 0.55)
    ax.set_title('Off-line λ-return algorithm', fontsize=18)
    plt.savefig('offline_lambda_return_random_walk')
    plt.waitforbuttonpress()

    df_vis.to_csv('out.zip', index=False)


def get_td_lambda_error_curves():
    """Reproduces curves p.295 of the book."""

    env = Environment()

    num_ticks = 20 #100

    all_α_ranges = {'λ = 0.0': np.linspace(0., 1., num=num_ticks),
                    'λ = 0.4': np.linspace(0., 1., num=num_ticks),
                    'λ = 0.8': np.linspace(0., 1., num=num_ticks),
                    'λ = 0.9': np.linspace(0., .6, num=num_ticks),
                    'λ = 0.95': np.linspace(0., .4, num=num_ticks),
                    'λ = 0.975': np.linspace(0., .2, num=num_ticks),
                    'λ = 0.99': np.linspace(0., .1, num=num_ticks),
                    'λ = 1.0': np.linspace(0., .1, num=num_ticks),
    }

    lmbda_range = [0., 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.]
    n_runs = 100

    rms_errs_all = {}

    for idx, lmbda in enumerate(tqdm(lmbda_range)):
        α_range = all_α_ranges[f'λ = {lmbda}']
        lambda_err_hists_over_α = []

        for n_run in tqdm(range(n_runs)):

            err_hists_over_α = []

            for α in α_range:
                agent = Agent(alpha=α, lmbda=lmbda)

                agent.run_semi_grad_td_lambda_return_method(env, n_episodes=10)

                avg_err = np.array(agent.error_hist).mean()
                err_hists_over_α.append(avg_err)

            lambda_err_hists_over_α.append(err_hists_over_α)

        lambda_err_hists_over_α = np.array(lambda_err_hists_over_α).mean(0)

        rms_errs_all[f'λ = {lmbda}'] = lambda_err_hists_over_α

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for key in rms_errs_all:
        sns.lineplot(x=all_α_ranges[key],
                     y=rms_errs_all[key], ax=ax, label=key)
    ax.set_xlabel('α')
    ax.set_ylabel('Average RMS error over 19 states and first 10 episodes')
    ax.set_ylim(0.20, 0.55)
    ax.set_title('T(λ)', fontsize=18)
    plt.savefig('td_lambda_random_walk')
    plt.waitforbuttonpress()


def get_online_lambda_returns_error_curves():
    env = Environment()

    num_ticks = 20
    all_α_ranges = [np.linspace(0., 1., num=num_ticks), np.linspace(0., 1., num=num_ticks),
                    np.linspace(0., 1., num=num_ticks), np.linspace(0., .6, num=num_ticks),
                    np.linspace(0., .4, num=num_ticks), np.linspace(0., .2, num=num_ticks),
                    np.linspace(0., .1, num=num_ticks), np.linspace(0., .05, num=num_ticks)]
    lmbda_range = [0., 0.4, 1.]
    n_runs = 20

    rms_errs_all = {}

    for idx, lmbda in enumerate(tqdm(lmbda_range)):
        α_range = all_α_ranges[idx]

        lambda_err_hists_over_α = []

        for n_run in tqdm(range(n_runs)):

            err_hists_over_α = []

            for α in α_range:
                agent = Agent(alpha=α, lmbda=lmbda)

                agent.run_online_lamba_return_method(env, n_episodes=10)

                avg_err = np.array(agent.error_hist).mean()
                err_hists_over_α.append(avg_err)

            lambda_err_hists_over_α.append(err_hists_over_α)

        lambda_err_hists_over_α = np.array(lambda_err_hists_over_α).mean(0)

        rms_errs_all[f'λ = {lmbda}'] = lambda_err_hists_over_α

    # Plot
    df_vis = pd.DataFrame(rms_errs_all)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('α')
    ax.set_ylabel('Average RMS error over 19 states and first 10 episodes')
    ax.set_ylim(0.20, 0.55)
    ax.set_title('On-line λ-return algorithm', fontsize=18)
    plt.savefig('online_lambda_return_random_walk')
    plt.waitforbuttonpress()



def get_true_online_td_lambda_error_curves():
    env = Environment()

    num_ticks = 40
    α_range = np.linspace(0., 1., num=num_ticks)
    lmbda_range = [0., 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.]
    n_runs = 100

    rms_errs_all = {}

    for idx, lmbda in enumerate(tqdm(lmbda_range)):



        lambda_err_hists_over_α = []

        for n_run in range(n_runs):

            err_hists_over_α = []

            for α in α_range:
                agent = Agent(alpha=α, lmbda=lmbda)

                agent.run_true_online_td_lambda(env, n_episodes=10)

                avg_err = np.array(agent.error_hist).mean()
                err_hists_over_α.append(avg_err)

            lambda_err_hists_over_α.append(err_hists_over_α)

        lambda_err_hists_over_α = np.array(lambda_err_hists_over_α).mean(0)

        rms_errs_all[f'λ = {lmbda}'] = lambda_err_hists_over_α

    print(rms_errs_all)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for key in rms_errs_all:
        sns.lineplot(x=np.linspace(0., 1., num=num_ticks),
                     y=rms_errs_all[key], ax=ax, label=key)
    ax.set_xlabel('α')
    ax.set_ylabel('Average RMS error over 19 states and first 10 episodes')
    ax.set_ylim(0.20, 0.55)
    ax.set_title('True on-line TD(λ) algorithm', fontsize=18)
    ax.legend()
    plt.savefig('true_online_td_lambda_random_walk_alpha_range')
    plt.waitforbuttonpress()



if __name__ == '__main__':

    # Off-line λ-return (curve p.291)
    get_offline_lambda_returns_error_curves()

    # Semi-gradient TD(λ) (curve p.295)
    #get_td_lambda_error_curves()

    # On-line λ-return (curve p.299)
    #get_online_lambda_returns_error_curves() # todo
    # requires very long run. todo : optimize computation time to get the same curve as True on-line TD(λ)

    # True on-line TD(λ)  (curve p.299)
    #get_true_online_td_lambda_error_curves():








