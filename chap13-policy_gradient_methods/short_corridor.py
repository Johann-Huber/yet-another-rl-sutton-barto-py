import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


# ----------------

# Number of states
N_STATES = 4

# Available actions
ALL_ACTIONS = [-1, 1]

# Number of actions
N_ACTIONS = len(ALL_ACTIONS)

# Id initial states
ID_INIT = 0

# Id terminal states
ID_TERMINAL = 3

# Default number of episodes
N_EPISODES = 1000

# Default constant step-size parameter (policy)
ALPHA_THETA = 2e-4

# Default constant step-size parameter (value estimation)
ALPHA_W = 2e-4

# Discount factor
GAMMA = 1

# Plotting figure dimensions
FIGURE_SIZE = (12,12)

# ----------------


class Agent:
    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS, id_init=ID_INIT,
                 id_terminal=ID_TERMINAL, alpha_theta=ALPHA_THETA, alpha_w=ALPHA_W,
                 gamma=GAMMA):

        # Policy parameter
        self._θ = np.array([0.949, -1.996])
            # About θ : Figure p328 suggests that the parameters has been initialized so that the initial
            # stochastic policy corresponds to an epsilon-greedy left policy.
            # Therefore, (θ1,θ2) have been determined manually from the system of 2 equations corresponding to
            # the probability distribution for epsilon-greedy left policy.
            # That is : Pr(left | t=0) = 0.95, Pr(right | t=0) = 0.05
            # There is an infinite number of solutions to that system (integrating softmax gives a constant
            # that could be set arbitrarily).
            # Note that if we set θ = np.zeros(n_actions), as suggested by the book, the initial policy will be :
            # Pr(left | t=0) = Pr(right | t=0) = 0.5
            # Figure p323 (also reproduced bellow) shows that this policy isn't far from the optimal one ; REINFORCE
            # learning won't appears as clearly as showed on figure p328 and p330 of the book.

        # weight vector
        self._w = np.zeros(n_states)

        # list containing all available actions
        self._all_actions = ALL_ACTIONS

        # number of states
        self._n_states = n_states

        # number of actions
        self._n_actions = n_actions

        # initial state
        self._init_state = id_init

        # terminal state
        self._terminal_state = id_terminal

        # step-size (policy)
        self._α_θ = alpha_theta

        # step-size (value estimation)
        self._α_w = alpha_w

        # discounting parameter (=1 : no discounting on this task)
        self._γ = gamma

        # list where will be stored the cumulated reward for each episode
        self._reward_hist = []

    @property
    def reward_hist(self):
        return self._reward_hist


    def action2ind(self, action):
        """ -1 -> 0 (left), 1 -> 1 (right) """
        assert action in self._all_actions
        return 0 if (action == -1) else 1


    def ind2action(self, ind_action):
        """ 0 -> -1 (left), 1 -> 1 (right) """
        assert ind_action in [0, 1]
        return -1 if (ind_action == 0) else 1


    def get_feature_vector(self, action):
        """Get x(a) : one weight per action."""
        id_a = self.action2ind(action)

        x = np.zeros(self._n_actions)
        x[id_a] = 1

        # output shape = (2,)
        return x


    def get_action_preferences(self, action):
        """Get h(s,a,θ)."""

        x = self.get_feature_vector(action)
        θ = self._θ

        return (θ * x).sum()


    def get_exp_softmax_distribution(self):
        π = np.zeros(self._n_actions)

        for a in self._all_actions:
            id_a = self.action2ind(a)
            π[id_a] = np.exp(self.get_action_preferences(a))

        sum_preferences = π.sum()
        π /= sum_preferences

        π = np.clip(π, a_min=0.05, a_max=0.95) # at worse epsilon-greedy : avoid deterministic policies

        # output shape = (2,)
        return π


    def policy(self, state):
        """ Get an action from current state by applying exponential soft-max distribution.
        Book's notation : π(At|St,θ)

        Note : On this task, the policy is independent from the state : Our goal is to find the optimal
        stochastic policy without distinguishing states.

        :param state: Current state.
        :return: Chosen action.
        """

        π = self.get_exp_softmax_distribution()

        id_a = np.random.choice(range(self._n_actions), p=π)
        action = self.ind2action(id_a)

        # output shape = 0 (scalar)
        return action


    def policy_prob(self, state, action):
        """Get probability associated to (state, action) pair, w.r.t. exponential soft-max distribution.

        Note : On this task, the policy is independent from the state. State is given as argument for
        theoretical consistency."""

        id_a = self.action2ind(action)
        π = self.get_exp_softmax_distribution()

        prob = π[id_a]

        # output shape = () (scalar)
        return prob


    def grad_log_policy_prob(self, state, action):
        """ Get the gradient of the probability associated to (state,action) pair, w.r.t.
        exponential soft-max distribution."""

        x_sa = self.get_feature_vector(action)  # len = (2,)

        sum_terms = [self.get_feature_vector(a) * self.policy_prob(state, a) for a in self._all_actions]
        sum_terms = np.array(sum_terms).sum(-1)
        grad_log_prob = x_sa - sum_terms

        return grad_log_prob

    def generate_episode(self, env):

        s_hist = [] # [S0, S1, ... S_T-1, S_T] -> len = T + 1
        a_hist = [] # [A0, A1, ... A_T-1]      -> len = T
        r_hist = [] #     [R1, ... R_T-1, S_T] -> len = T

        curr_state = self._init_state
        curr_action = self.policy(curr_state)
        s_hist.append(curr_state)
        a_hist.append(curr_action)


        running = True
        while running:

            state = curr_state
            action = curr_action

            next_state, reward = env.step(state, action)

            s_hist.append(next_state)
            r_hist.append(reward)

            if next_state == self._terminal_state:
                running = False
            else:
                curr_state = next_state

                curr_action = self.policy(curr_state)
                a_hist.append(curr_action)

        return s_hist, a_hist, r_hist


    def run_REINFORCE_MC(self, env, n_episodes):

        for n_ep in range(n_episodes):

            cumu_reward = 0

            s_hist, a_hist, r_hist = self.generate_episode(env)
            T = len(r_hist)

            G = 0
            for t in range(T-1,-1,-1):

                R_k_next = r_hist[t]
                cumu_reward += R_k_next

                G += self._γ * R_k_next

                S_k = s_hist[t]
                A_k = a_hist[t]

                eligibility_vec = self.grad_log_policy_prob(S_k, A_k)
                increment = self._α_θ * (self._γ ** t) * G * eligibility_vec

                self._θ += increment

            self._reward_hist.append(cumu_reward)
            #print('θ =', self._θ, '| cum_rewards =', cumu_reward)


    def v_hat(self, state):
        """Returns the approximated value for the current state."""

        if state == self._terminal_state:
            # By convention : R(S(T)) = 0
            return 0

        value = self._w[state]

        # output shape = () (scalar)
        return value


    def grad_v_hat(self, state):
        """Returns the gradient of the approximated value for the current state."""

        assert state in range(self._n_states)

        # v(s) = w * x(s) => grad_v(s) = x(s)
        grad_v = np.zeros(self._n_states)
        grad_v[state] = 1

        # output shape = (2,)
        return grad_v


    def run_REINFORCE_MC_with_baseline(self, env, n_episodes):

        for n_ep in range(n_episodes):

            cumu_reward = 0

            s_hist, a_hist, r_hist = self.generate_episode(env)
            T = len(r_hist)

            G = 0
            for t in range(T-1,-1,-1):

                R_k_next = r_hist[t]
                S_k = s_hist[t]
                A_k = a_hist[t]

                cumu_reward += R_k_next

                # Return
                G += self._γ * R_k_next

                # Value estimation
                δ = G - self.v_hat(S_k)
                increment_w = self._α_w * δ * self.grad_v_hat(S_k)
                self._w += increment_w

                # Policy update
                eligibility_vec = self.grad_log_policy_prob(S_k, A_k)
                increment_theta = self._α_θ * (self._γ ** t) * δ * eligibility_vec
                self._θ += increment_theta

            self._reward_hist.append(cumu_reward)
            #print('θ =', self._θ, '| w =', self._w, '| cum_rewards =', cumu_reward)




class Environment:

    def __init__(self, id_terminal=ID_TERMINAL):

        self._reward = -1

        # self._transitions[state][action] -> next_state
        self._transitions = {0 : {-1: 0, 1: 1},
                             1 : {-1: 2, 1: 0},
                             2 : {-1: 1, 1: 3}}

        self._terminal_state = id_terminal


    def step(self, state, action):

        assert state in [0, 1, 2]
        assert action in [-1, 1]

        next_state = self._transitions[state][action]

        assert next_state in [0, 1, 2, 3]

        reward = self._reward

        return next_state, reward



# ----------------

def value_init_state(p):
    """Expected reward from S0 position (initial state-value).

    Found by developing Bellman state-value equations, assuming :
    p : probability of right action ;
    (1-p) : probability of left action.

    Reward is always -1, until the terminal state.

    We finally got : Vπ(S0) = (4-2p)/(p(p-1)) """

    value = (4 - 2 * p)/(p * (p - 1))
    return value


def get_state_values_distrib():
    """Reproduces figure p323."""

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    # State value distribution
    p_right_range = np.arange(0.01, 1., 0.01)
    state_value_distrib = [value_init_state(p=p) for p in p_right_range]
    sns.lineplot(x=p_right_range, y=state_value_distrib, ax=ax)

    # ε-greedy makers
    ε = 0.1
    v_epsilon_left, v_epsilon_right = value_init_state(p=ε / 2), value_init_state(p=1 - ε / 2)

    ax.plot(ε / 2, v_epsilon_left, marker='o', markersize=6, color='k')
    ax.text(ε / 2 + 0.03, v_epsilon_left, 'ε-greedy left', fontsize=15, va='center')

    ax.plot(1 - ε / 2, v_epsilon_right, marker='o', markersize=6, color='k')
    ax.text(1 - ε / 2 - 0.2, v_epsilon_right, 'ε-greedy right', fontsize=15, va='center')

    # optimal stochastic policy
    optim_state_value = np.max(state_value_distrib)
    p_optim_state_value = p_right_range[np.argmax(state_value_distrib)]
    ax.plot(p_optim_state_value, optim_state_value, marker='o', markersize=6, color='k')
    ax.text(p_optim_state_value, optim_state_value - 3, 'optimal stochastic policy', fontsize=15,
            va='top', ha='center')

    ax.set_ylim(-100, 0)
    ax.set_ylabel('vπ(S0)')
    ax.set_xlabel('probability of right action')
    ax.set_title('Short corridor with switched actions', fontsize=18)

    plt.savefig('short_corridor_switched_actions_state_values_distrib')
    plt.waitforbuttonpress()



def get_reinforce_reward_curves():
    """Reproduces figure p328."""

    all_rewards = {}
    all_alphas = [2e-3, 2e-4, 2e-5]
    n_runs = 100
    n_episodes = 1000
    env = Environment()

    ## Training

    for α in all_alphas:

        α_reward_list = []
        for _ in tqdm(range(n_runs), desc=f'α={α}'):
            agent = Agent(alpha_theta=α)
            agent.run_REINFORCE_MC(env, n_episodes=n_episodes)

            α_reward_list.append(agent.reward_hist)

        all_rewards[f'α = {α}'] = np.array(α_reward_list).mean(0)


    ## Plotting

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    # Optimal state value
    p_right_range = np.arange(0.01, 1., 0.01)
    state_value_distrib = [value_init_state(p=p) for p in p_right_range]
    optim_state_value = np.max(state_value_distrib)
    sns.lineplot(x=range(n_episodes),
                 y=np.repeat(optim_state_value, n_episodes), ax=ax, label='optimal value',
                 color='k', dashes=[(2, 2)])

    # Learning curves
    for key in all_rewards:
        sns.lineplot(x=range(len(all_rewards[key])),
                     y=all_rewards[key], ax=ax, label=key)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward on episode')
    ax.legend()
    plt.savefig('reinforce_mc_reward_curves')
    plt.waitforbuttonpress()




def get_reinforce_baseline_reward_curves():
    """Reproduces figure p328."""

    all_rewards = {}
    all_alpha_theta = [2e-3]
    n_runs = 100
    n_episodes = 1000
    env = Environment()

    ## Training

    # REINFORCE with baseline
    all_alpha_w = [2e-1]

    for α_w in all_alpha_w:
        for α_theta in all_alpha_theta:

            α_reward_list = []
            for _ in tqdm(range(n_runs), desc=f'REINFORCE with baseline | α_w={α_w} α_theta={α_theta}'):
                agent = Agent(alpha_theta=α_theta, alpha_w=α_w)
                agent.run_REINFORCE_MC_with_baseline(env, n_episodes=1000)

                α_reward_list.append(agent.reward_hist)

            all_rewards[f'REINFORCE with baseline | α_θ = {α_theta} α_w = {α_w}'] = np.array(α_reward_list).mean(0)

    # REINFORCE
    for α_theta in [2e-4]:
        α_reward_list = []
        for _ in tqdm(range(n_runs), desc=f'REINFORCE'):
            agent = Agent(alpha_theta=α_theta)
            agent.run_REINFORCE_MC(env, n_episodes=n_episodes)
            α_reward_list.append(agent.reward_hist)

        all_rewards[f'REINFORCE | α_θ = {α_theta}'] = np.array(α_reward_list).mean(0)


    ## Plotting

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    # Optimal state value
    p_right_range = np.arange(0.01, 1., 0.01)
    state_value_distrib = [value_init_state(p=p) for p in p_right_range]
    optim_state_value = np.max(state_value_distrib)
    sns.lineplot(x=range(n_episodes),
                 y=np.repeat(optim_state_value, n_episodes), ax=ax, label='optimal value',
                 color='k', dashes=[(2, 2)])

    # Learning curves
    for key in all_rewards:
        sns.lineplot(x=range(len(all_rewards[key])),
                     y=all_rewards[key], ax=ax, label=key)  # , marker='o'

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward on episode')
    ax.legend()
    plt.savefig('reinforce_mc_baseline_reward_curves')
    plt.waitforbuttonpress()



if __name__ == '__main__':

    # Get figure p323
    #get_state_values_distrib()

    # Get figure p328
    #get_reinforce_reward_curves()

    # Get figure p330
    get_reinforce_baseline_reward_curves()
