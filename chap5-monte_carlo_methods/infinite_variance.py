import numpy as np
import pandas as pd
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

#-------------

DISCOUNT_FACTOR = 1

REWARDED_TERMINAL_THRESHOLD = 0.1

#-------------


class Action(Enum):
    LEFT = 0
    RIGHT = 1

class State(Enum):
    S = 0
    TERMINAL = 1


#--------------


class Environment:
    def __init__(self):
        None

    def step(self, state, action):
        """
        Apply a step from state when choosing action.

        :param state: Initial state.
        :param action: Initial action.
        :return: (next_state, reward) tuple, as describe in the example 5.5 in the book.
        """

        assert action in [Action.LEFT, Action.RIGHT], 'Invalid action.'
        assert state is State.S, 'Invalid or already terminal state.'

        if action is Action.RIGHT:
            return State.TERMINAL, 0
        else:
            if np.random.random() < REWARDED_TERMINAL_THRESHOLD:
                return State.TERMINAL, 1
            else:
                return State.S, 0


    def run_episode(self, state_0, action_0, behavior_policy):
        """Run episode until the agent reached the terminal state."""

        n_steps = 0
        state_action_hist = []

        state = state_0
        action = action_0

        is_running = True

        while is_running:
            state_action_hist.append((state, action))
            n_steps += 1

            next_state, reward = self.step(state, action)

            if next_state is State.TERMINAL:
                is_running = False
            else:
                state = next_state
                action = behavior_policy(state)


        return n_steps, state_action_hist, reward





class Agent:
    def __init__(self, gamma=DISCOUNT_FACTOR):

        # Discount factor
        self._gamma = gamma

        # Init state
        self._init_state = State.S

        # Numerator of off-policy state-value computation
        self.sum_scaled_returned = 0

        # Denominator of off-policy state-value computation (weighted importance sampling)
        self.sum_importance_sampling_ratio = 0

        # Denominator of off-policy state-value computation (ordinary importance sampling)
        self.s_visit_cnt = 0

        # State value history (ordinary importance sampling)
        self.V_s_ordinary_hist = []

        # State value history (weighted importance sampling)
        self.V_s_weighted_hist = []



    def behavior_policy(self, state=None):
        """Randomly pick action regardless of the state, with equal probability."""
        return np.random.choice([Action.RIGHT, Action.LEFT])

    # left = 1
    def target_policy_prob(self, state, action):
        """Get the probability of taking action from state by following target policy (defined in the book)."""
        assert state is State.S
        assert action in [Action.LEFT, Action.RIGHT]
        return 1 if action is Action.LEFT else 0

    def get_importance_sampling_ratio(self, state_action_hist):

        target_policy_term = []
        behavior_policy_term = []

        for (state, action) in state_action_hist:
            target_policy_term.append(self.target_policy_prob(state, action))
            behavior_policy_term.append(0.5)

        return np.prod(target_policy_term) / np.prod(behavior_policy_term)


    def get_state_value_ordinary_method(self):
        return self.sum_scaled_returned/self.s_visit_cnt if (self.s_visit_cnt != 0) else 0


    def get_state_value_weighted_method(self):
        return self.sum_scaled_returned/self.sum_importance_sampling_ratio if (self.sum_importance_sampling_ratio != 0) else 0


    def state2inds3d(self, state):
        p_sum, d_card, p_has_ace = state

        assert (p_sum >= 12) and (p_sum <= 21)
        assert (d_card >= 1) and (d_card <= 10)
        assert p_has_ace in [0, 1]

        i_p_sum = p_sum - 12
        i_d_card = d_card - 1
        i_p_ace = p_has_ace

        return i_p_sum, i_d_card, i_p_ace


    def run_episode_off_policy(self, env, verbose=True):

        # Fixed inital state
        state_0 = self._init_state
        action_0 = self.behavior_policy()

        # Generate an episode
        n_steps, state_action_hist, reward = env.run_episode(state_0=state_0,
                                                             action_0=action_0,
                                                             behavior_policy=self.behavior_policy)
        # Update state values
        last2first_steps = np.arange(n_steps, 0, -1) - 1  # [T-1, T-2, ..., 0]

        cumu_return = 0
        for t in last2first_steps:
            state, action = state_action_hist[t]

            is_last_step = (t == last2first_steps[0]) # t==T-1
            step_rwd = reward if is_last_step else 0

            cumu_return = self._gamma * cumu_return + step_rwd

            if (state, action) not in state_action_hist[:t]:

                is_first_state = t == 0
                if is_first_state:
                    importance_sampling_ratio = self.get_importance_sampling_ratio(state_action_hist)

                    self.sum_scaled_returned += importance_sampling_ratio * cumu_return
                    self.sum_importance_sampling_ratio += importance_sampling_ratio
                    self.s_visit_cnt += 1

                    V_s_ordinary = np.around(self.get_state_value_ordinary_method(), 5)
                    V_s_weighted = np.around(self.get_state_value_weighted_method(), 5)

                    self.V_s_ordinary_hist.append(V_s_ordinary)
                    self.V_s_weighted_hist.append(V_s_weighted)

                    if verbose:
                        print(f'V_s_ordinary = {V_s_ordinary:.5f}, V_s_weighted = {V_s_weighted:.5f}')
                        print('-')


    def get_predicted_state_value(self):
        return self.V_s_ordinary_hist, self.V_s_weighted_hist



def plot_output_curves(all_V_s_ordinary_hist):
    data_state_value = pd.DataFrame(all_V_s_ordinary_hist)

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    sns.lineplot(data=data_state_value, ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel('V(s)')
    ax.get_legend().remove()
    fig.suptitle('MC estimate of V(s) with ordinary importance sampling (10 runs)', fontsize=18)

    plt.waitforbuttonpress()


def run_ordinary_importance_sampling_infinite_variance():
    """Reproduces figure p107."""

    env = Environment()

    n_episodes = 1000000
    n_runs = 10

    all_V_s_ordinary_hist = {}

    for r in tqdm(range(n_runs)):
        agent = Agent()

        for e in range(n_episodes):
            # Running an episode
            agent.run_episode_off_policy(env, verbose=False)

        V_s_ordinary_hist, _ = agent.get_predicted_state_value()
        all_V_s_ordinary_hist[f'run{r}'] = V_s_ordinary_hist

    # Plot outputs
    plot_output_curves(all_V_s_ordinary_hist)


if __name__ == '__main__':

    run_ordinary_importance_sampling_infinite_variance()

