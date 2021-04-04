import numpy as np
import pandas as pd
import math
from collections import deque

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

# ----------------

# Number of states
N_STATES = 1000

# Id terminal states
ID_TERMINALS = [0, N_STATES+1]

# Starting state
INIT_STATE = 500

# Max reachable position when taking a action (left/negative, right/positive)
MOVE_RANGE = 100

# Number of state described by each component of the weight vector
AGG_GROUP_SIZE = 100

# Rewards
LEFT_REWARD = -1
RIGHT_REWARD = 1

# Default number of episodes
N_EPISODES = 10 # 100 000

# Default constant step-size parameter
ALPHA = 2e-5

# Discount factor
GAMMA = 1

# Available actions
ALL_ACTIONS = list(range(-100,0)) + list(range(1,101))

# Apply state aggregation as the default function approximation method
DEFAULT_APPROX_MODE = 'state_aggregation'

DEFAULT_POLYNOMIAL_ORDER = 5

FIGURE_SIZE = (12,12)

# ----------------

class Agent:

    def __init__(self, n_states=N_STATES, agg_group_size=AGG_GROUP_SIZE, init_state=INIT_STATE,
                 id_terminals=ID_TERMINALS, alpha=ALPHA, gamma=GAMMA, target_values=None,
                 all_actions=ALL_ACTIONS, func_approx_mode=DEFAULT_APPROX_MODE,
                 n_order=DEFAULT_POLYNOMIAL_ORDER):

        assert func_approx_mode in ['state_aggregation', 'polynomial'], 'Invalid function approximation mode.'
        self._func_approx_mode = func_approx_mode


        if self._func_approx_mode == 'state_aggregation' :
            # weight vector
            self._w = np.zeros(int(n_states/agg_group_size))

        # TODO - 1)
        elif self._func_approx_mode == 'polynomial' :
            # order of the polynomial basis
            self._n_order = n_order

            # weight vector dimension
            dimensional_space = 1 # random_walk task
            self.w_dim = (self._n_order + 1) ** dimensional_space

            # weight vector
            self._w = np.zeros(self.w_dim)


        self._init_state = init_state
        self._terminal_states = id_terminals
        self._all_actions = all_actions

        self._α = alpha
        self._γ = gamma

        self.n_states_max = n_states
        self.agg_group_size = agg_group_size

        self._target_values = target_values

        self._error_hist = []


    @property
    def error_hist(self):
        return self._error_hist


    def get_v_hat_all_states(self):
        all_states = list(range(self.n_states_max)) + np.array([1])

        all_predicted_values = np.zeros((len(all_states),2))
        for i, state in enumerate(all_states):
            all_predicted_values[i,:] = np.array([state, self.v_hat(state)])

        return all_predicted_values


    def policy(self, state):
        """Action selection : uniform distribution."""
        assert (state >= 0) and (state <= self.n_states_max)
        return np.random.choice(self._all_actions)


    def get_state_grp_id(self, state):
        """Used for state aggregation mode. Return the id corresponding to the state aggregation group."""
        id = math.ceil(state/self.agg_group_size) - 1
        assert (id >= 0) and (id < 10)
        return id


    def v_hat(self, state):
        """Returns the approximated value for state, w.r.t. the weight vector and the function
         approximation method."""

        if state in self._terminal_states:
            # By convention : R(S(T)) = 0
            return 0

        value = None

        if self._func_approx_mode == 'state_aggregation':
            id_s_grp = self.get_state_grp_id(state)
            value = self._w[id_s_grp]

        #TODO 2)
        if self._func_approx_mode == 'polynomial':
            x_s = np.array([state ** c for c in range(self._n_order + 1)])
            assert len(x_s) == len(self._w)

            # v = [w_1, ..., w_d] * [x_s_1, ..., x_s_d]
            value = (self._w*x_s).sum()


        return value


    def grad_v_hat(self, state, id_w):
        """Returns the gradient of the approximated value for state, w.r.t. the value of the weight vector
        at index id_w. The derivative depends on the the function approximation method."""

        if self._func_approx_mode == 'state_aggregation':
            id_s_grp = self.get_state_grp_id(state)
            return 1 if id_w == id_s_grp else 0

        # TODO 3)
        if self._func_approx_mode == 'polynomial':
            x_s = np.array([state ** c for c in range(self._n_order + 1)])
            assert len(x_s) == len(self._w)

            # v = (w_1 + x_s_1) + ... + (w_d * x_s_d)
            # => dv/dw_i = x_s_i
            return x_s[id_w]



    def generate_episode(self, env):
        n_steps = 0
        s_a_r_hist = []

        curr_state = self._init_state

        running = True
        while running:
            state = curr_state
            action = self.policy(state)

            next_state, reward = env.step(state, action)

            s_a_r_hist.append((state, action, reward)) # S(t), A(t), R(t+1)
            n_steps += 1

            if next_state in self._terminal_states:
                running = False
            else:
                curr_state = next_state

        return n_steps, s_a_r_hist


    def run_gradient_MC(self, env, n_episodes):
        """ Gradient Monte Carlo Algorithm (p.202) """

        for i_ep in tqdm(range(n_episodes)):

            n_steps, s_a_r_hist = self.generate_episode(env)

            # Update values
            Gt = 0  # cumulative_return

            for t in range(n_steps - 1, -1, -1):  # [T-1, T-2, ..., 0]
                state, action, reward = s_a_r_hist[t]
                Gt = self._γ * Gt + reward

                for id_w in range(len(self._w)):

                    sgd_step = self._α * (Gt - self.v_hat(state)) * self.grad_v_hat(state, id_w)
                    self._w[id_w] += sgd_step

            if i_ep%1000 == 0:
                print('self._w =', self._w)


    def run_semi_gradient_TD0(self, env, n_episodes):
        """ Semi-gradient TD(0) (p.203) """

        for i_ep in tqdm(range(n_episodes)):

            curr_state = self._init_state

            #print('Episode =', i_ep)

            running = True
            while running:
                state = curr_state
                action = self.policy(state)

                next_state, reward = env.step(state, action)

                for id_w in range(len(self._w)):

                    sgd_step = self._α * (reward + self._γ * self.v_hat(next_state) - self.v_hat(state)) * \
                        self.grad_v_hat(state, id_w)
                    #print('sgd_step=', sgd_step)
                    self._w[id_w] += sgd_step

                if next_state in self._terminal_states:
                    running = False
                else:
                    curr_state = next_state

            if i_ep%1000 == 0:
                print('self._w =', self._w)



    def run_n_step_semi_gradient_td(self, env, n_episodes, n_steps):
        """ N-step semi-gradient TD (p.209) """

        for _ in range(n_episodes):
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
                        G += self._γ ** n_steps * self.v_hat(Sτ_n)

                    # Update value
                    Sτ = states_buff[0]
                    i_s = self.get_state_grp_id(Sτ)
                    self._w[i_s] += self._α * (G - self.v_hat(Sτ))

                    if T - τ <= n_steps :
                        # Last steps before termination
                        states_buff.popleft()
                        rewards_buff.popleft()

                if τ == T - 1:
                    running = False
                else:
                    t += 1


            # Compute RMS error
            rms_err = mean_squared_error(self._target_values, self.get_v_hat_all_states()[:,1], squared=False)

            self._error_hist.append(rms_err)




class Environment:

    def __init__(self, n_states=N_STATES, left_reward=LEFT_REWARD, right_reward=RIGHT_REWARD,
                 id_terminals=ID_TERMINALS, all_actions=ALL_ACTIONS):
        self._rewards = {key:0 for key in range(n_states + 2)} # [0] [1, ..., 1000] [1001]
        self._rewards[0] = left_reward
        self._rewards[n_states+1] = right_reward

        self._terminal_states = id_terminals

        self._all_actions = all_actions

    def step(self, state, action):

        assert state not in self._terminal_states
        assert action in self._all_actions

        #if state in self._terminal_states: #early return ?
        next_state = state + action
        next_state = np.clip(next_state, a_min=self._terminal_states[0], a_max=self._terminal_states[1])

        reward = self._rewards[next_state]

        return next_state, reward


def get_target_values():
    """Apply DP for computing targetted values for each state.

    Method described p83 of the book."""

    err_thresh = 1e-1 #1e-2

    all_states = list(range(1000))+np.array(1) # [1 ... 1000]
    all_actions = ALL_ACTIONS # [-100, ..., -1, 1, ... 100]

    # Guessed value to initialize target_values for applying DP.
    # For convenience, len = 1002 (the 2 terminal states are included).
    target_values = np.array(range(-500, 502, 1)) / 500
    target_values[0] = 0
    target_values[-1] = 0
    # corresponding to states : [0=terminal_left] [1 ... 1000] [1001=terminal_right]

    n_it = 0

    running = True
    while(running):

        curr_err = 0

        for state in all_states:

            cum_state_action_value = 0

            for action in all_actions:
                prev_value = target_values[state]

                next_state = state + action
                next_state = np.clip(next_state, a_min=ID_TERMINALS[0], a_max=ID_TERMINALS[1])

                if next_state == ID_TERMINALS[0]:
                    reward = -1
                elif next_state == ID_TERMINALS[1]:
                    reward = 1
                else:
                    reward = 0

                sa_prob = (1 / len(all_actions))

                cum_state_action_value += sa_prob * (reward + GAMMA * target_values[next_state])


            target_values[state] = cum_state_action_value

            curr_err = max(curr_err, abs(target_values[state] - prev_value))

        n_it += 1
        print(f'[{n_it}] curr_err = ', curr_err)
        if curr_err < err_thresh:
            # converged
            running = False

    return target_values[1:-1] # discard terminal states


def get_pred_values_gradient_MC(func_approx_mode, alpha, n_order):
    n_episodes = 500000 # 100000
    env = Environment()
    agent = Agent(alpha=alpha, func_approx_mode=func_approx_mode, n_order=n_order)

    print(f'[*] Predicting values (gradient MC, {func_approx_mode})')
    agent.run_gradient_MC(env, n_episodes)

    pred_values = agent.get_v_hat_all_states()
    return pred_values


def get_pred_values_semi_grad_TD0(func_approx_mode):

    env = Environment()

    print(f'[*] Predicting values (semi-gradient TD(0), {func_approx_mode})')

    n_runs = 20
    pred_values = []
    for i_run in range(n_runs):
        agent = Agent(func_approx_mode=func_approx_mode)
        print(f'i_run = {i_run + 1}/{n_runs}')
        agent.run_semi_gradient_TD0(env, n_episodes=50000)
        pred_values.append(agent.get_v_hat_all_states())

    pred_values = np.array(pred_values).mean(0)
    return pred_values


def get_target_values_DP():

    print('[*] Computing true values (DP)')

    target_values = get_target_values()
    all_states = list(range(1000)) + np.array(1)

    print('Target values successfully computed.')
    return target_values, all_states


def reproduce_curves_gradient_MC_state_agg():

    # Target values
    target_values, all_states = get_target_values_DP()

    # Gradient MC
    func_approx_mode = 'state_aggregation'
    alpha = ALPHA
    n_order = None
    pred_values = get_pred_values_gradient_MC(func_approx_mode=func_approx_mode, alpha=alpha, n_order=n_order)

    # Plot output
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=pred_values[:,0], y=pred_values[:,1], ax=ax, label='pred (MC, state agg)')
    sns.lineplot(x=all_states, y=target_values, ax=ax, label='true')
    ax.set_xlabel('State')
    ax.set_ylabel('Value scale')
    plt.legend()
    plt.savefig('gradient_mc_state_aggreg_random_walk')
    #plt.waitforbuttonpress()


def reproduce_curves_semi_gradient_TD0():

    # Target values
    target_values, all_states = get_target_values_DP()

    # Semi-gradient TD0
    func_approx_mode = 'state_aggregation'
    pred_values = get_pred_values_semi_grad_TD0(func_approx_mode)

    # Plot output
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=pred_values[:, 0], y=pred_values[:, 1], ax=ax, label='pred (TD, state agg)')
    sns.lineplot(x=all_states, y=target_values, ax=ax, label='true')
    ax.set_xlabel('State')
    ax.set_ylabel('Value scale')
    plt.legend()
    plt.savefig('semi_gradient_td0_state_aggreg_random_walk_2')
    #plt.waitforbuttonpress()


def reproduce_curves_n_step_semi_gradient_TD():
    # Target values
    target_values, all_states = get_target_values_DP()

    # Estimated values
    print('[*] Predicting values (n-step semi-gradient TD)')

    n_steps_range = [2 ** i for i in range(10)]
    n_runs = 2  # 500  # 100
    α_range = np.linspace(0., 1., num=40)
    env = Environment()

    rms_errs_all = {}

    for n_steps in n_steps_range:

        n_steps_err_hists_over_α = []

        for _ in tqdm(range(n_runs), desc=f'n_steps={n_steps}'):

            err_hists_over_α = []

            for α in α_range:
                agent = Agent(alpha=α, target_values=target_values)

                agent.run_n_step_semi_gradient_td(env, n_episodes=10, n_steps=n_steps)

                avg_err = np.array(agent.error_hist).mean()
                err_hists_over_α.append(avg_err)

            n_steps_err_hists_over_α.append(err_hists_over_α)

        n_steps_err_hists_over_α = np.array(n_steps_err_hists_over_α).mean(0)

        rms_errs_all[f'n = {n_steps}'] = n_steps_err_hists_over_α

    # n-step semi-gradient TD

    # Plot output
    df_vis = pd.DataFrame(rms_errs_all)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('α')
    ax.set_ylabel('Average RMS error over 1000 states and first 10 episodes')
    ax.set_ylim(0.20, 0.55)
    plt.savefig('n_step_semi_gradient_td_state_aggreg_random_walk')
    #plt.waitforbuttonpress()




def reproduce_curves_gradient_MC_polynomial():

    # Target values

    target_values, all_states = get_target_values_DP()

    # Gradient MC
    func_approx_mode = 'polynomial'
    alpha = 2e-6
    n_order = 1
    pred_values = get_pred_values_gradient_MC(func_approx_mode=func_approx_mode, alpha=alpha, n_order=n_order)

    print('pred_values =', pred_values)


    # Plot output
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=pred_values[:,0], y=pred_values[:,1], ax=ax, label='pred (MC, polynomial)')
    sns.lineplot(x=all_states, y=target_values, ax=ax, label='true')
    ax.set_xlabel('State')
    ax.set_ylabel('Value scale')
    plt.legend()
    plt.savefig('gradient_mc_random_walk_polynom_order1')
    plt.waitforbuttonpress()



if __name__ == '__main__':

    #reproduce_curves_gradient_MC_state_agg()

    reproduce_curves_semi_gradient_TD0()

    # TODO : make it work with explicit grad
    #reproduce_curves_n_step_semi_gradient_TD()

    # Note : polynomial approx works, but gradient easily explode. Conditions proposed p214 of the book
    # (alpha = 1e-4, n_order= {5, 10, 20}) makes the training unstable.
    #reproduce_curves_gradient_MC_polynomial()
