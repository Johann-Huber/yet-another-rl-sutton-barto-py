import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


# ----------------

# Number of states
N_STATES = 7

# Default number of training steps
N_STEPS = 10

# Default constant step-size parameter
ALPHA = 0.01

# Discount factor
GAMMA = 0.99

# Constant reward
CONST_REWARD = 0.

# Init weights (proposed p.262)
INIT_WEIGHTS = [1., 1., 1., 1., 1., 1., 10., 1.]

# Figure dimensions
FIGURE_SIZE = (12,12)

# ----------------



class Environment:

    def __init__(self, n_states=N_STATES, const_rwd=CONST_REWARD):
        self._n_states = n_states
        self._const_reward = const_rwd

    def step(self, state, action):

        assert state in range(self._n_states), "Invalid state."
        assert action in range(self._n_states), "Invalid action."

        next_state = action
        reward = self._const_reward

        return next_state, reward


class Agent:

    def __init__(self, alpha=ALPHA, gamma=GAMMA, n_states=N_STATES, init_weights=INIT_WEIGHTS):

        self._n_states = n_states

        self._values = np.zeros(n_states)

        self._w = np.array(init_weights)


        self._all_actions = range(n_states)

        self._α = alpha

        self._γ = gamma

        self._weights_hist = [self._w.copy()]



    @property
    def weights_hist(self):
        return np.array(self._weights_hist)

    def behavior_policy(self, state):
        return np.random.choice(self._all_actions)

    def transfrm2features(self, state):
        """s --> x(s)"""

        assert state in range(self._n_states), "Invalid state."

        x = np.zeros(self._n_states + 1)

        if state == self._n_states - 1:
            # Dest of solid actions
            x[state] = 1
            x[-1] = 2 # bias
        else :
            # Dest of dashed actions
            x[state] = 2
            x[-1] = 1  # bias

        return x


    def v_hat(self, state):

        assert state in range(self._n_states), "Invalid state."
        x_s = self.transfrm2features(state)

        assert len(x_s) == len(self._w)

        v = (self._w * x_s).sum()

        return v

    def grad_v_hat(self, state):

        assert state in range(self._n_states), "Invalid state."
        x_s = self.transfrm2features(state)

        assert len(x_s) == len(self._w)

        # V = w * x(s) => dV/dw = x(s)
        grad_v = x_s

        return grad_v


    def get_sampling_ratio(self, next_state):
        """ π(Ai|Si) / b(Ai|Si). Only depends on next_state on this task."""

        assert next_state in range(self._n_states), "Invalid state."

        target_prob = 1 if (next_state == self._n_states - 1) else 0

        behavior_prob = 1/self._n_states

        return target_prob/behavior_prob


    def run_semi_gradient_off_policy_TD(self, env, n_step_max):
        """ Semi-gradient off policy TD(0) (p.258) """

        state = np.random.choice(self._all_actions)

        for _ in tqdm(range(n_step_max)):

            action = self.behavior_policy(state)

            next_state, reward = env.step(state, action)

            # sampling ratio
            ρ = self.get_sampling_ratio(next_state)

            # td error
            δ = reward + self._γ * self.v_hat(next_state) - self.v_hat(state)

            self._w += self._α * ρ * δ * self.grad_v_hat(state)
            self._weights_hist.append(self._w.copy())


            state = next_state


    def run_semi_gradient_DP(self, n_step_max):
        """ Semi-gradient off policy DP (p.262)
        Simplifyed implementation to avoid many call to env with policy_prob = 0.
        todo : generic function."""


        for _ in tqdm(range(n_step_max)):


            for n_s in range(self._n_states):
                s6 = self._n_states-1

                sum_expect_err = np.array([self._γ * self.v_hat(s6) - self.v_hat(s) \
                                           for s in range(self._n_states)]).sum()

                self._w += (self._α / self._n_states) * sum_expect_err * self.grad_v_hat(n_s)

            self._weights_hist.append(self._w.copy())



def get_semi_grad_off_policy_td_curves():
    """Reproduces left figure p262."""
    n_step_max = 1000
    env = Environment()

    # Semi-grad off-policy TD

    agent = Agent()
    agent.run_semi_gradient_off_policy_TD(env, n_step_max=n_step_max)

    weights_hist = agent.weights_hist
    n_weights = weights_hist.shape[1]


    # Plot
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for i_w in range(n_weights):
        sns.lineplot(x=range(n_step_max + 1), y=weights_hist[:, i_w], ax=ax, label=f'w{i_w + 1}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Weight values')
    ax.set_title('Semi-gradient Off-policy TD', fontsize=18)
    plt.legend()
    plt.savefig('baird_counterexemple_divergent_weights_TD_method')
    plt.waitforbuttonpress()


def get_semi_grad_off_policy_dp_curves():
    """Reproduces right figure p262."""

    n_step_max = 1000

    # Semi-grad DP
    agent = Agent()
    agent.run_semi_gradient_DP(n_step_max=n_step_max)
    weights_hist = agent.weights_hist
    n_weights = weights_hist.shape[1]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for i_w in range(n_weights):
        sns.lineplot(x=range(n_step_max + 1), y=weights_hist[:, i_w], ax=ax, label=f'w{i_w + 1}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Weight values')
    ax.set_title('Semi-gradient Off-policy DP', fontsize=18)
    plt.legend()
    plt.savefig('baird_counterexemple_divergent_weights_DP_method')
    plt.waitforbuttonpress()



if __name__ == '__main__':

    # Semi-grad off-policy TD
    get_semi_grad_off_policy_td_curves()

    # Semi-grad DP
    get_semi_grad_off_policy_dp_curves()
