from tiles3 import tiles, IHT # from Sutton website : http://www.incompleteideas.net/tiles/tiles3.html
from math import cos
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()
from matplotlib import cm
from collections import deque
#import matplotlib.gridspec as gridspec



ALL_ACTIONS = [-1, 0, 1]
POS_LIMITS = [-1.2, 0.5]
VEL_LIMITS = [-0.07, 0.07]


# Index Hash Table size
IHT_DEFAULT_SIZE = 4096

NUM_TILINGS = 8
TILING_RESOLUTION = [(POS_LIMITS[1]-POS_LIMITS[0])/NUM_TILINGS**2, (VEL_LIMITS[1]-VEL_LIMITS[0])/NUM_TILINGS**2]


TERMINAL_REWARD = 0
STEP_REWARD = -1


# Default constant step-size parameter
ALPHA = 1/(10*NUM_TILINGS)

# Discount factor
GAMMA = 1.

# Exploration rate
EPSILON = 0.

# Exponential weighting decrease parameter
LAMBDA = 0.8


FIGURE_SIZE = (16,12)

MAX_N_STEP = 4000

# ------------------


class IndexHashTable:

    def __init__(self, iht_size=IHT_DEFAULT_SIZE, num_tilings=NUM_TILINGS):
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings

    def get_tiles(self, state, action):
        """Get the encoded state_action using grid tiling.
        Ultimate resolution = 1/16.
        :param state: (x, x_dot)
        :param action: {-1, 0, 1}
        :return:
        """
        x, x_dot = state
        #return tiles(self.iht, self.num_tilings, [x * 8, x_dot * 8, action])

        return tiles(self.iht, self.num_tilings, [x * 8/(0.5 + 1.2), x_dot * 8/(0.07+0.07)], [action])

    # print(get_tiles(0, 0, 0))
    # print(get_tiles(0.0, 0.01562, 1))
    # print(get_tiles(0.0, 0.01563, -1))


class Environment:

    def __init__(self, all_actions=ALL_ACTIONS, pos_lims=POS_LIMITS, vel_lims=VEL_LIMITS,
                 term_rwd=TERMINAL_REWARD, step_rwd=STEP_REWARD):

        self._all_actions = all_actions
        self._pos_lims = pos_lims
        self._vel_lims = vel_lims
        self._pos_terminal = self._pos_lims[1]

        self._terminal_reward = term_rwd
        self.step_reward = step_rwd


    def step(self, state, action):

        x, x_dot = state
        assert (x >= self._pos_lims[0]) and (x <= self._pos_lims[1]), "Invalid position"
        assert (x_dot >= self._vel_lims[0]) and (x_dot <= self._vel_lims[1]), "Invalid speed"
        assert action in self._all_actions, "Invalid action"

        x_dot_next = x_dot + 0.001 * action - 0.0025 * cos(3 * x)
        x_dot_next = np.clip(x_dot_next, a_min=self._vel_lims[0], a_max=self._vel_lims[1])

        x_next = x + x_dot_next
        x_next = np.clip(x_next, a_min=self._pos_lims[0], a_max=self._pos_lims[1])

        if x_next == self._pos_lims[0]:
            # Left border : reset speed
            x_dot_next = 0.

        next_state = (x_next, x_dot_next)
        reward = self._terminal_reward if (x_next == self._pos_terminal) else self.step_reward

        return next_state, reward


class Agent:

    def __init__(self, iht=IndexHashTable(), weight_vec_size=IHT_DEFAULT_SIZE,
                 pos_lims=POS_LIMITS, vel_lims=VEL_LIMITS, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                 all_actions=ALL_ACTIONS, num_tilings=NUM_TILINGS, tiling_res=TILING_RESOLUTION,
                 lmbda=LAMBDA, max_n_step=MAX_N_STEP):

        # Index Hash Table for position encoding
        self._iht = iht

        # weight vector
        self._w = np.zeros(weight_vec_size)

        # trace vector
        self._z = np.zeros(weight_vec_size)

        self._λ = lmbda

        # Maximum number of step within an episode (avoid inf loop)
        self.max_n_step = max_n_step

        # Minimum cumulated reward (means that q values have diverged).
        self.default_min_reward = -4000

        self._all_actions = all_actions
        self._pos_lims = pos_lims
        self._vel_lims = vel_lims
        self._pos_terminal = self._pos_lims[1]

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon

        self._num_tilings = num_tilings
        self._tiling_res = tiling_res

        self._n_step_hist = []
        self._cumu_reward_hist = []

        # Dict all several cost-to-go function at some time of the training for 3D plotting (reproduces p245)
        self._all_cost2go_func_vis = {}

    @property
    def all_cost2go_func_vis(self):
        return self._all_cost2go_func_vis


    @property
    def n_step_hist(self):
        return self._n_step_hist

    @property
    def cumu_reward_hist(self):
        return self._cumu_reward_hist


    def get_cost2go_func(self):

        res = np.zeros((self._num_tilings**2, self._num_tilings**2))

        for i_pos in range(NUM_TILINGS ** 2):
            for i_vel in range(NUM_TILINGS ** 2):
                q_s_values = []
                for a in ALL_ACTIONS:
                    # pos = low_bound_pos + index_tile_pos * resolution_pos_tile
                    # velocity = low_bound_vel + index_tile_vel * resolution_vel_tile
                    s = (self._pos_lims[0] + i_pos * self._tiling_res[0],
                         self._vel_lims[0] + i_vel * self._tiling_res[1])
                    #print(f's={s} | a={a}')

                    q_s_values.append(self.q_hat(state=s, action=a))

                q_s_value = (-1)*(np.array(q_s_values).max())
                res[i_pos, i_vel] = q_s_value

        return res


    def get_init_state(self):
        """Get a random starting position in the interval [-0.6, -0.4)."""
        x = np.random.uniform(low=-0.6, high=-0.4)
        x_dot = 0.
        return x, x_dot


    def q_hat(self, state, action):
        """ Compute the q value for the current state-action pair.

        :param state: (x, x_dot) pair of the current state.
        :param action: current action.
        :return: The computed q value : float.
        """

        x, x_dot = state
        assert (x >= self._pos_lims[0]) and (x <= self._pos_lims[1]), "Invalid position"
        assert (x_dot >= self._vel_lims[0]) and (x_dot <= self._vel_lims[1]), "Invalid speed"
        assert action in self._all_actions, "Invalid action"

        x_s_a = self._iht.get_tiles(state, action)
        q = np.array([self._w[id_w] for id_w in x_s_a]).sum()
        return q


    def grad_q_hat(self, state, action):
        """ Compute the gradient of the q value w.r.t. weight vector.

        :param state: (x, x_dot pair) of the current state.
        :param action: current action.
        :return: a len(self._w) vector full of 0, except for indices corresponding the weights related
        to the (state, action) pair.
        """
        x, x_dot = state
        assert (x >= self._pos_lims[0]) and (x <= self._pos_lims[1]), "Invalid position"
        assert (x_dot >= self._vel_lims[0]) and (x_dot <= self._vel_lims[1]), "Invalid speed"
        assert action in self._all_actions, "Invalid action"

        x_s_a = self._iht.get_tiles(state, action)

        grad_q = np.zeros(len(self._w))

        for id_w in x_s_a:
            grad_q[id_w] = 1

        return grad_q

    def get_feature_vector(self, state, action):
        """Get x(s,a), such as : x.transpose() * x(s,a) = q_hat(s,a)."""
        None
        # already grad_q_hat


    def policy(self, state, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        x, x_dot = state
        assert (x >= self._pos_lims[0]) and (x <= self._pos_lims[1]), "Invalid position"
        assert (x_dot >= self._vel_lims[0]) and (x_dot <= self._vel_lims[1]), "Invalid speed"

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = self._all_actions[np.random.choice(range(len(self._all_actions)))]
            return action

        # 1) Compute Q(s,a,w) for each action
        q_sa_next = np.array([self.q_hat(state, a) for a in self._all_actions])
        # 2) Choose the highest Q-value
        greedy_action_inds = np.where(q_sa_next == q_sa_next.max())[0]
        ind_action = np.random.choice(greedy_action_inds)
        action = self._all_actions[ind_action]

        return action


    def run_episodic_semi_gradient_sarsa(self, env, n_episodes):
        """ Episodic semi-gradient Sarsa for estimating q_hat. (p.244) """

        self._all_cost2go_func_vis['ep0_s0'] = self.get_cost2go_func()

        for i_ep in range(n_episodes):

            n_step = 0

            curr_state = self.get_init_state()
            curr_action = self.policy(curr_state)

            running = True
            while running:
                state = curr_state
                action = curr_action

                next_state, reward = env.step(state, action)
                #print(f"s={state}, a={action} | s'={next_state}, r={reward}")
                n_step += 1

                # For 3D plot :
                if (i_ep == 0) and (n_step == 428) :
                    self._all_cost2go_func_vis[f'ep{i_ep}_s{n_step}'] = self.get_cost2go_func()

                if next_state[0] == self._pos_terminal:

                    increment = self._α * (reward - self.q_hat(state, action)) * self.grad_q_hat(state, action)
                    self._w += increment
                    running = False
                    continue # go to next episode


                next_action = self.policy(next_state)

                increment = self._α * (reward + self._γ * self.q_hat(next_state, next_action) - self.q_hat(state, action)) \
                           * self.grad_q_hat(state, action)
                self._w += increment

                curr_state = next_state
                curr_action = next_action

            self._n_step_hist.append(n_step)
            #print(f'n_ep {i_ep} complete | n_step = {n_step}')

            # For 3D plot :
            if i_ep in [12, 104, 1000, 9000]:
                self._all_cost2go_func_vis[f'ep{i_ep}_s{n_step}'] = self.get_cost2go_func()


    def run_episodic_semi_gradient_n_step_sarsa(self, env, n_episodes, n_steps):
        """ Episodic semi-gradient n-step Sarsa for estimating q_hat. (p.247) """

        for i_ep in range(n_episodes):

            n_it = 0

            #state = self._init_state
            state = self.get_init_state()
            action = self.policy(state)

            t = 0
            T = np.inf

            states_buff = deque(maxlen=(n_steps + 1)) # [Sτ, Sτ+1, ... Sτ+n]
            states_buff.append(state)
            actions_buff = deque(maxlen=(n_steps + 1))  # [Aτ, Aτ+1, ... Aτ+n]
            actions_buff.append(action)
            rewards_buff = deque(maxlen=(n_steps)) # [Rτ+1, ... Rτ+n]

            running = True
            while(running):

                if t < T:
                    # Terminal state not reached
                    #action = np.random.choice(self._all_actions)  # Markov Random Process
                    next_state, reward = env.step(state, action)

                    n_it += 1

                    states_buff.append(next_state)
                    rewards_buff.append(reward)

                    if next_state[0] == self._pos_terminal:
                        T = t + 1
                        next_action = None
                    else:
                        next_action = self.policy(next_state)
                        actions_buff.append(next_action)

                    state = next_state
                    action = next_action # here ? or in the else cond ?


                τ = t - n_steps + 1

                if τ >= 0:
                    # State updatable
                    G = np.array([self._γ**i * rwd for i,rwd in enumerate(rewards_buff)]).sum()

                    if τ + n_steps < T :
                        # Rewards beyond Rτ+n must be approximated
                        Sτ_n = states_buff[-1]
                        Aτ_n = actions_buff[-1]

                        G += self._γ ** n_steps * self.q_hat(Sτ_n, Aτ_n)

                    # Update value
                    Sτ = states_buff[0]
                    Aτ = actions_buff[0]

                    increment = self._α * (G - self.q_hat(Sτ, Aτ)) * self.grad_q_hat(Sτ, Aτ)
                    self._w += increment

                    if T - τ <= n_steps :
                        # Last steps before termination
                        states_buff.popleft()
                        actions_buff.popleft()
                        rewards_buff.popleft()

                if τ == T - 1:
                    running = False
                else:
                    t += 1

            # Compute RMS error
            #rms_err = mean_squared_error(self._target_values, self._values[1:-1], squared=False)
            #self._error_hist.append(rms_err)
            #print(f'i_ep={i_ep} | n_it={n_it}')
            self._n_step_hist.append(n_it)


    def update_trace_vector(self, z, state, action, method):
        """Update z with the given feature vector x(s,a), according to the given method.
        Return the updated vector."""

        x, x_dot = state
        assert (x >= self._pos_lims[0]) and (x <= self._pos_lims[1]), "Invalid position"
        assert (x_dot >= self._vel_lims[0]) and (x_dot <= self._vel_lims[1]), "Invalid speed"
        assert action in self._all_actions, "Invalid action"
        assert method in ['accumulating', 'replace', 'replace_with_clearing'], 'Invalid method arg.'


        if method == 'replace_with_clearing':
            for a in self._all_actions:
                # Clear the traces of other actions
                if a != action:
                    x_s_a2clear = self._iht.get_tiles(state, a)
                    for id_w in x_s_a2clear:
                        z[id_w] = 0

        # Update trace vector with current x(s,a)

        x_s_a = self._iht.get_tiles(state, action)

        for id_w in x_s_a:

            if (method == 'replace') or (method == 'replace_with_clearing'):
                #print('clearing z')
                z[id_w] = 1
            elif method == 'accumulating':
                z[id_w] += 1

        return z


    def run_sarsa_lambda_binary_feat_func_approx(self, env, n_episodes, method='replace'):
        """ Apply Sarsa(λ) algorithms (with optimization related to the use of binary features
        and linear function approximation) for estimating q_hat. (p.305).

        :param env: environment to interact with.
        :param n_episodes: number of episodes to train on.
        :param method: specify the Sarsa(λ) method :
                * 'accumulating' : With accumulating traces ;
                * 'replace' : With replacing traces ;
                * 'replace_with_clearing' : With replacing traces, and clearing the traces
                of other actions.
        :return:
        """

        assert method in ['accumulating', 'replace', 'replace_with_clearing'], 'Invalid method arg.'
        #print('METHOD =', method)


        overflow_flag = False

        for i_ep in range(n_episodes):

            if overflow_flag :
                # Training diverged : return default value until the end of training
                self._n_step_hist.append(self.max_n_step)
                self._cumu_reward_hist.append(self.default_min_reward)
                continue

            n_it = 0
            cumu_reward = 0

            state = self.get_init_state()
            action = self.policy(state)

            self._z = np.zeros(len(self._w))

            running = True
            while(running):

                # Prevent value overflow (alpha too large, or unstable method ('accumulating')
                try:
                    next_state, reward = env.step(state, action)
                    n_it += 1
                    cumu_reward += reward

                    δ = reward

                    # (Implicit) Loop for i in F(S,A)
                    δ -=  self.q_hat(state, action)
                    self._z = self.update_trace_vector(self._z, state, action, method)


                    if (next_state[0] == self._pos_terminal) or (n_it == self.max_n_step):
                        #if n_it == self.max_n_step:
                        #    print(f'Warning : max number of iteration reached. | λ={self._λ} , α*num_tile={self._α*8}')

                        self._w += self._α * δ * self._z.copy()

                        running = False
                        continue # go to next episode


                    next_action = self.policy(next_state)

                    # (Implicit) Loop for i in F(S',A')
                    δ += self._γ * self.q_hat(next_state, next_action)

                    self._w += self._α * δ * self._z.copy()
                    self._z *= self._γ * self._λ

                    state = next_state
                    action = next_action

                except ValueError:
                    # Value overflow : training diverged. Training data lists will be fed with default
                    # values for all the remaining epochs.
                    overflow_flag = True
                    running = False
                    continue


            if overflow_flag :
                n_it = self.max_n_step
                cumu_reward = self.default_min_reward

            self._n_step_hist.append(n_it)
            self._cumu_reward_hist.append(cumu_reward)



    def run_true_online_sarsa_lambda(self, env, n_episodes):
        """Apply true online Sarsa(λ) algorithm. (p.307)"""

        for i_ep in range(n_episodes):

            n_it = 0
            cumu_reward = 0

            state = self.get_init_state()
            action = self.policy(state)

            x_s_a = self.grad_q_hat(state, action)
            self._z = np.zeros(len(self._w))
            Q_old = 0

            running = True
            while (running):

                next_state, reward = env.step(state, action)
                n_it += 1
                cumu_reward += reward

                next_action = self.policy(next_state)

                x_s_a_next = self.grad_q_hat(next_state, next_action)

                δ = reward + self._γ * self.q_hat(next_state, next_action) - self.q_hat(state, action)

                self._z = self._γ * self._λ * self._z + \
                          (1 - self._α * self._γ * self._λ * (self._z * x_s_a).sum()) * x_s_a

                self._w += self._α * (δ + self.q_hat(state, action) - Q_old) * self._z - \
                           self._α * (self.q_hat(state, action) - Q_old) * x_s_a

                Q_old = self.q_hat(next_state, next_action)
                x_s_a = x_s_a_next

                if (next_state[0] == self._pos_terminal):
                    running = False
                else:
                    state = next_state
                    action = next_action

            self._n_step_hist.append(n_it)
            self._cumu_reward_hist.append(cumu_reward)





def get_steps_per_episode_semi_grad_sarsa_curves():
    """Reproduces curves p246."""

    env = Environment()

    n_episodes = 200
    n_runs = 20 #100
    all_n_steps_lists = {}

    for id_α in [0.1, 0.2, 0.5]:

        α = id_α / 8
        all_n_steps = []

        for _ in tqdm(range(n_runs), desc=f'alpha={id_α}/8'):
            agent = Agent(alpha=α)
            agent.run_episodic_semi_gradient_sarsa(env, n_episodes=n_episodes)

            all_n_steps.append(agent.n_step_hist)

        all_n_steps = np.array(all_n_steps).mean(0)

        all_n_steps_lists[f'alpha={id_α}/8'] = all_n_steps

    # Plot output
    df_vis = pd.DataFrame(all_n_steps_lists)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per episode (log scale)')
    ax.set_yticklabels(np.arange(0, 1100, step=100))
    ax.set_yscale('log')
    ax.set_title(f"Mountain Car", fontsize=18)

    #plt.savefig('mountain_car_semi_gradient_sarsa_nsteps')
    plt.waitforbuttonpress()


def get_steps_per_episode_semi_grad_n_step_sarsa_curves():
    """Reproduces plots p248."""

    env = Environment()

    n_episodes = 500
    n_runs = 100
    all_n_steps_lists = {}

    for n_steps, α in zip([1, 8], [0.5 / 8, 0.3 / 8]):

        all_n_steps = []

        for _ in tqdm(range(n_runs), desc=f'n={n_steps}'):
            agent = Agent(alpha=α)
            agent.run_episodic_semi_gradient_n_step_sarsa(env, n_episodes=n_episodes, n_steps=n_steps)

            all_n_steps.append(agent.n_step_hist)

        all_n_steps = np.array(all_n_steps).mean(0)

        all_n_steps_lists[f'n={n_steps}'] = all_n_steps

    # Plot output
    df_vis = pd.DataFrame(all_n_steps_lists)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per episode (log scale), averaged over 100 runs')
    ax.set_yscale('log')
    ax.set_title(f"MOUNTAIN CAR", fontsize=18)
    plt.savefig('mountain_car_semi_gradient_sarsa_n_steps')
    plt.waitforbuttonpress()


def get_sarsa_lambda_replacing_traces_n_step_curves():
    """Reproduces upper plot p306."""

    env = Environment()

    n_episodes = 50
    n_runs = 2 #100 # computation time ?
    all_n_steps_lists = {}

    α_range = np.linspace(0.2, 1.8, num=7) / 8
    lambda_range = [0., .68, .92, .96, 0.99]

    for lmbda in lambda_range:

        all_n_steps_over_α = []

        for α in tqdm(α_range):

            all_n_steps = []

            for _ in range(n_runs):
                agent = Agent(alpha=α, lmbda=lmbda)
                agent.run_sarsa_lambda_binary_feat_func_approx(env, n_episodes=n_episodes)

                all_n_steps.append(agent.n_step_hist)

            # Average over runs and episodes
            all_n_steps = np.array(all_n_steps).mean(0).mean(0)

            all_n_steps_over_α.append(all_n_steps)

        all_n_steps_lists[f'λ = {lmbda}'] = all_n_steps_over_α

    print('all_n_steps_lists= ', all_n_steps_lists)

    # Plot output

    df_vis = pd.DataFrame(all_n_steps_lists, index=α_range * 8)
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    sns.lineplot(data=df_vis, ax=ax)
    ax.set_xlabel('α * number of tilings (8)')
    ax.set_ylabel('Steps per episode, averaged over first 50 episodes and 100 runs')
    ax.set_title(f"MOUNTAIN CAR | Sarsa(λ) with replacing traces", fontsize=18)
    ax.set_ylim(160, 300)
    plt.savefig('mountain_car_sarsa_lambda_with_replacing_traces_nsteps')
    plt.waitforbuttonpress()


def get_sarsa_lambda_replacing_traces_rewards_curves():
    """Reproduces lower plot p306."""

    n_episodes = 20
    n_runs = 100
    all_reward_lists = {}

    env = Environment()

    α_range = np.linspace(0.2, 2.0, num=11) / 8
    lmbda = 0.92  # close to optimum

    all_methods = ['accumulating', 'replace', 'replace_with_clearing', 'true_online']
    method_desc = {'accumulating': 'Sarsa(λ) with accumulating trace',
                   'replace': 'Sarsa(λ) with replacing trace',
                   'replace_with_clearing': 'Sarsa(λ) with replacing trace and clearing the traces of other actions',
                   'true_online': 'True online Sarsa(λ)'}

    for method in all_methods:
        all_rewards_over_α = []

        for α in tqdm(α_range):

            all_reward_hists = []

            for _ in range(n_runs):
                agent = Agent(alpha=α, lmbda=lmbda)

                if method == 'true_online':
                    agent.run_true_online_sarsa_lambda(env, n_episodes=n_episodes)
                else:
                    agent.run_sarsa_lambda_binary_feat_func_approx(env, n_episodes=n_episodes, method=method)

                all_reward_hists.append(agent.cumu_reward_hist)

            # Average over runs and episodes
            all_reward_hists = np.array(all_reward_hists).mean(0).mean(0)

            all_rewards_over_α.append(all_reward_hists)

        all_reward_lists[f'{method_desc[method]}'] = all_rewards_over_α

    print('all_reward_lists= ', all_reward_lists)

    # Plot output

    df_vis = pd.DataFrame(all_reward_lists, index=α_range * 8)
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    sns.lineplot(data=df_vis, ax=ax, marker='o')
    ax.set_xlabel('α * number of tilings (8)')
    ax.set_ylabel('Reward per episode, averaged over first 20 episodes and 100 runs')
    ax.set_title(f"MOUNTAIN CAR | Summary comparison of Sarsa(λ) algorithms", fontsize=18)
    ax.set_ylim(-550, -150)
    plt.savefig('mountain_car_sarsa_lambda_with_replacing_traces_rewards')
    plt.waitforbuttonpress()


if __name__ == '__main__':

    # 1) Semi-gradient Sarsa
    #get_steps_per_episode_semi_grad_sarsa_curves()
    #get_cost2go_func_semi_grad_sarsa_3d_plots()
    # 2) Semi-gradient n-step Sarsa
    #get_steps_per_episode_semi_grad_n_step_sarsa_curves()
    #-------------------------------------------------------------to clean after plot computation



    # 3) Sarsa(λ) with replacing traces (upper plot p.306)
    get_sarsa_lambda_replacing_traces_n_step_curves()
    # todo: relancer les courbes en export

    # 4) Sarsa(λ) algorithms comparison (lower plot p.306)
    #get_sarsa_lambda_replacing_traces_rewards_curves()
    # todo : longer run on UC
    # ---------




