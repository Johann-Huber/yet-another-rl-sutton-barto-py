from tiles3 import tiles, IHT # from sutton's website : http://www.incompleteideas.net/tiles/tiles3.html
from math import cos
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()
from matplotlib import cm
from collections import deque



ALL_ACTIONS = [-1, 0, 1]
POS_LIMITS = [-1.2, 0.5]
VEL_LIMITS = [-0.07, 0.07]


# Index Hash Table size
IHT_DEFAULT_SIZE = 4096

NUM_TILINGS = 8
TILING_RESOLUTION = [(POS_LIMITS[1]-POS_LIMITS[0])/NUM_TILINGS**2, (VEL_LIMITS[1]-VEL_LIMITS[0])/NUM_TILINGS**2]



TERMINAL_REWARD = 0
STEP_REWARD = -1

ALPHA = 1/(10*NUM_TILINGS)
GAMMA = 1.
EPSILON = 0.

FIGURE_SIZE = (16,12)

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

        return tiles(self.iht, self.num_tilings, [x * 8/(0.5 + 1.2), x_dot * 8/(0.07+0.07)], [action])


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
                 all_actions=ALL_ACTIONS, num_tilings=NUM_TILINGS, tiling_res=TILING_RESOLUTION):

        # Index Hash Table for position encoding
        self._iht = iht

        # weight vector
        self._w = np.zeros(weight_vec_size)

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

        # Dict all several cost-to-go function at some time of the training for 3D plotting (reproduces p245)
        self._all_cost2go_func_vis = {}

    @property
    def all_cost2go_func_vis(self):
        return self._all_cost2go_func_vis


    @property
    def n_step_hist(self):
        return self._n_step_hist

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

            # For 3D plot :
            if i_ep in [12, 104, 1000, 9000]:
                self._all_cost2go_func_vis[f'ep{i_ep}_s{n_step}'] = self.get_cost2go_func()


    def run_episodic_semi_gradient_n_step_sarsa(self, env, n_episodes, n_steps):
        """ Episodic semi-gradient n-step Sarsa for estimating q_hat. (p.247) """

        for i_ep in range(n_episodes):

            n_it = 0

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
                    action = next_action


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
            self._n_step_hist.append(n_it)



def get_steps_per_episode_semi_grad_sarsa_curves():
    """Reproduces curves p246."""

    env = Environment()

    n_episodes = 200
    n_runs = 100
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
    ax.set_yscale('log')
    ax.set_title(f"Mountain Car", fontsize=18)
    plt.savefig('mountain_car_semi_gradient_sarsa_nsteps')
    plt.waitforbuttonpress()


def get_cost2go_func_semi_grad_sarsa_3d_plots():
    """Reproduces plots p245.

    TODO : Axis ticks & labels (axis x and y) must be corrected : [-1.2, 0.6] and [-.07,.07]."""

    env = Environment()
    α = 0.1 / 8
    agent = Agent(alpha=α)
    agent.run_episodic_semi_gradient_sarsa(env, n_episodes=9001)

    all_cost2go_func_vis_dict = agent.all_cost2go_func_vis

    # 3D plots
    fig = plt.figure(figsize=(20, 16))

    i = j = np.arange(NUM_TILINGS ** 2)
    jj, ii = np.meshgrid(i, j)

    plot1_name = 'Step 0'
    q_val_name = list(all_cost2go_func_vis_dict.keys())[0]
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(ii, jj, all_cost2go_func_vis_dict[q_val_name], cmap=cm.coolwarm)
    ax1.set_ylabel('Velocity')
    # Correct labels/ticks here -------------
    # ax1.set_yticks([y_ticks[0], y_ticks[int(len(y_ticks) / 2)], y_ticks[-1]])
    # ax1.set_yticklabels([-0.07, 0 ,0.07])
    # ax1.set_yticklabels(np.array(range()))
    # ax1.set_xticklabels(np.array(range(10)))
    ax1.set_xlabel('Position')
    ax1.set_title(plot1_name)

    plot2_name = 'Step 428'
    q_val_name = list(all_cost2go_func_vis_dict.keys())[1]
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(ii, jj, all_cost2go_func_vis_dict[q_val_name], cmap=cm.coolwarm)
    ax2.set_ylabel('Velocity')
    ax2.set_xlabel('Position')
    ax2.set_title(plot2_name)

    plot3_name = 'Episode 12'
    q_val_name = list(all_cost2go_func_vis_dict.keys())[2]
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot_surface(ii, jj, all_cost2go_func_vis_dict[q_val_name], cmap=cm.coolwarm)
    ax3.set_ylabel('Velocity')
    ax3.set_xlabel('Position')
    ax3.set_title(plot3_name)

    plot4_name = 'Episode 104'
    q_val_name = list(all_cost2go_func_vis_dict.keys())[3]
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.plot_surface(ii, jj, all_cost2go_func_vis_dict[q_val_name], cmap=cm.coolwarm)
    ax4.set_ylabel('Velocity')
    ax4.set_xlabel('Position')
    ax4.set_title(plot4_name)

    plot5_name = 'Episode 1000'
    q_val_name = list(all_cost2go_func_vis_dict.keys())[4]
    ax5 = fig.add_subplot(235, projection='3d')
    ax5.plot_surface(ii, jj, all_cost2go_func_vis_dict[q_val_name], cmap=cm.coolwarm)
    ax5.set_ylabel('Velocity')
    ax5.set_xlabel('Position')
    ax5.set_title(plot5_name)

    plot6_name = 'Episode 9000'
    q_val_name = list(all_cost2go_func_vis_dict.keys())[5]
    ax6 = fig.add_subplot(236, projection='3d')
    ax6.plot_surface(ii, jj, all_cost2go_func_vis_dict[q_val_name], cmap=cm.coolwarm)
    ax6.set_ylabel('Velocity')
    ax6.set_xlabel('Position')
    ax6.set_title(plot6_name)

    fig.suptitle('MOUNTAIN CAR - COST TO GO FUNCTION \n(x and y tick labels must be corrected)', fontsize=18)

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
            # agent.run_episodic_semi_gradient_sarsa(env, n_episodes=n_episodes)
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



if __name__ == '__main__':

    # 1) Semi-gradient Sarsa

    get_steps_per_episode_semi_grad_sarsa_curves()

    #get_cost2go_func_semi_grad_sarsa_3d_plots()


    # 2) Semi-gradient n-step Sarsa

    #get_steps_per_episode_semi_grad_n_step_sarsa_curves()

