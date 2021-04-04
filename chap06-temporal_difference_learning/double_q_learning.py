import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")

# ---------------

#        B       A
# 0  |   1   |   2   |  3
# TER<-9  1-> <-1  1-> TER


N_ACTIONS_A = 2
N_ACTIONS_B = 9 + 1

N_STATES = 4


# TD step size
ALPHA = 0.1
# Discount factor
GAMMA = 1
# Exploration ratio
EPSILON = 0.1

FIGURE_SIZE = (10,4)

# ---------------

class Environment:
    def __init__(self, n_actions_a=N_ACTIONS_A, n_actions_b=N_ACTIONS_B):
        self._n_actions_a = n_actions_a
        self._n_actions_b = n_actions_b

        self.action2transition_a = {0: -1, 1: 1}

        self.action2transition_b = {key: -1 for key in range(n_actions_b)}


    def is_state_action_valid(self, state, action):
        assert state in [1, 2] , 'Invalid state'

        is_state_A = state == 2
        if (is_state_A):
            assert action in range(self._n_actions_a), 'Invalid action'

        is_state_B = state == 1
        if is_state_B:
            assert action in range(self._n_actions_b), 'Invalid action'

        return True

    def get_transition(self, state, action):

        is_state_A = state == 2
        if is_state_A:
            return self.action2transition_a[action]
        else:
            return self.action2transition_b[action]

    def get_reward(self, state, action):

        is_state_B = state == 1
        if is_state_B and (action in range(self._n_actions_b)):
            return np.random.normal(-0.1, 1)
        else:
            return 0.

    def step(self, state, action):

        assert self.is_state_action_valid(state, action)

        transition = self.get_transition(state, action)

        next_state = state + transition

        reward = self.get_reward(state, action)

        return next_state, reward



class Agent:
    def __init__(self, n_actions_a=N_ACTIONS_A, n_actions_b=N_ACTIONS_B,
                 n_states=N_STATES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                 double_q_learning=False):

        self._Q = [np.zeros(1), np.zeros(n_actions_b), np.zeros(n_actions_a), np.zeros(1)] # states, actions

        if double_q_learning:
            self._Q2 = [np.zeros(1), np.zeros(n_actions_b), np.zeros(n_actions_a), np.zeros(1)]  # states, actions
        self._double_q_learning = double_q_learning

        self._n_actions_a = n_actions_a
        self._n_actions_b = n_actions_b

        self._available_actions = [np.zeros(1, dtype=np.uint8),
                                   range(n_actions_b),
                                   range(n_actions_a),
                                   np.zeros(1,dtype=np.uint8)]

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon

        self._left_actions_ratio_hist = []


    @property
    def action_values(self):
        return self._Q


    @property
    def left_actions_ratio_hist(self):
        return self._left_actions_ratio_hist

    def action2ind(self, action):
        assert action in [-1 ,1], 'Invalid action'
        return 0 if (action==-1) else 1

    def ind2action(self, ind):
        assert ind in [0, 1], 'Invalid action'
        return -1 if (ind==0) else 1


    def policy(self, state, Q, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        assert state in range(4), 'Invalid state'

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = np.random.choice(self._available_actions[state])
            return action

        greedy_actions = np.where(Q[state] == Q[state].max())[0]
        action = np.random.choice(greedy_actions)
        return action

    def get_start_pos(self):
        return 2

    def is_terminal_state(self, state):
        return True if state in [0, 3] else False


    def run_q_learning(self, env, n_episodes):
        """Apply Q-learning for estimating Q."""

        for _ in range(n_episodes):

            state = self.get_start_pos()

            left_actions_a_cnt = 0
            actions_a_cnt = 0

            running = True
            while (running):

                if self._double_q_learning:
                    if np.random.random() > 0.5:
                        Q = self._Q
                        Q_estim = self._Q2
                    else:
                        Q = self._Q2
                        Q_estim = self._Q
                else :
                    Q = self._Q
                    Q_estim = self._Q


                action = self.policy(state, Q)

                left_actions_a_cnt += int((state == 2) and (action == 0))
                actions_a_cnt += int((state == 2))

                next_state, reward = env.step(state, action)

                next_action = self.policy(next_state, Q, explore_flg=False) # Always greedy

                increment = self._α * reward + self._α * (self._γ * Q_estim[next_state][next_action] - Q[state][action])

                Q[state][action] += increment

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    state = next_state

            left_actions_ratio = left_actions_a_cnt/actions_a_cnt if (actions_a_cnt != 0) else 0
            self._left_actions_ratio_hist.append(left_actions_ratio*100)


def get_maximization_bias_double_learning_curves():
    """Reproduces figure p135."""

    env = Environment()

    all_left_actions_ratio_hists = {'Q-learning': [], 'Double Q-learning': []}

    n_runs = 10000

    for _ in tqdm(range(n_runs)):
        agent = Agent()
        agent.run_q_learning(env, n_episodes=300)
        all_left_actions_ratio_hists['Q-learning'].append(agent.left_actions_ratio_hist)

    for _ in tqdm(range(n_runs)):
        agent = Agent(double_q_learning=True)
        agent.run_q_learning(env, n_episodes=300)
        all_left_actions_ratio_hists['Double Q-learning'].append(agent.left_actions_ratio_hist)

    all_left_actions_ratio_hists['Q-learning'] = np.array(all_left_actions_ratio_hists['Q-learning']).mean(0)
    all_left_actions_ratio_hists['Double Q-learning'] = np.array(
        all_left_actions_ratio_hists['Double Q-learning']).mean(0)

    # Plot left action ratio
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=range(len(all_left_actions_ratio_hists['Q-learning'])),
                 y=all_left_actions_ratio_hists['Q-learning'],
                 ax=ax, label='Q-learning')
    sns.lineplot(x=range(len(all_left_actions_ratio_hists['Double Q-learning'])),
                 y=all_left_actions_ratio_hists['Double Q-learning'],
                 ax=ax, label='Double Q-learning')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('% left actions from A')
    plt.legend()
    plt.waitforbuttonpress()

if __name__ == '__main__':
    get_maximization_bias_double_learning_curves()

