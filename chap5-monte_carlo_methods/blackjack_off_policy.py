import numpy as np
from enum import Enum
from tqdm import tqdm
import pandas as pd
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

#-------------

N_EPISODES = 1

N_STATES = 200 # (Note : this param is know in this problem)
N_ACTIONS = 2 # Stick or hit

DISCOUNT_FACTOR = 1


# State we want to estimate the value
INIT_P_SUM = 13
INIT_D_CARD = 2
INIT_P_HAS_ACE = 1

# True associated state value
TRUE_STATE_VALUE = -0.27726
#-------------


class Move(Enum):
    HIT = 0
    STICK = 1

class GameResult(Enum):
    PLAYER_WIN = 0
    PLAYER_LOSE = 1
    DRAW = 2

State = namedtuple('State', ['p_sum', 'd_card', 'p_has_ace'])
# p_sum: sum score of player's cards (in {12, ..., 21})
# d_card: card showed by the dealer (in {1, ..., 10})
# p_has_ace: does the player have usable ace (in {0, 1})


#--------------


class Blackjack:
    def __init__(self):
        None

    def is_natural_hand(self, cards):
        """Returns True if cards (= np.array([card_1, card_2]) is a natural (i.e. auto-win). """
        assert len(cards) == 2, 'cards must be a [card_1, card_2] ndarray.'
        return (cards == np.array([10, 1])).all() or (cards == np.array([1, 10])).all()

    def has_usable_ace(self, cards_p):
         return 1 in cards_p

    def dealer_policy(self, d_sum):
        assert (d_sum <= 21) and (d_sum >= 12)
        if d_sum >= 17 :
            return Move.STICK
        else:
            return Move.HIT


    def play_game(self, state_0, action_0, policy, verbose=True):

        n_steps = 0
        state_action_hist = []

        p_sum, d_card, p_has_ace = state_0


        # ------------------------
        # 1) Get dealer's cards
        #------------------------

        cards_d = np.array([d_card, np.clip(np.random.randint(1, 14), a_min=1, a_max=10)])  # set picked heads to 10

        d_has_ace = int(1 in cards_d)
        d_sum = (cards_d.sum() + 10) if d_has_ace else cards_d.sum()

        while d_sum < 12:
            new_card = np.clip(np.random.randint(1, 14), a_min=1, a_max=10)

            if (new_card == 1) and (d_sum+11 <= 21):
                d_sum += 11
                assert d_has_ace == 0
                d_has_ace = 1
            else :
                d_sum += new_card

        assert (p_sum >= 12) and (p_sum <= 21) and (d_sum >= 12) and (d_sum <= 21)

        # ------------------------
        # 2) Turn Player
        # ------------------------

        while True:

            # Get next move from current state
            is_first_move = n_steps == 0
            if is_first_move:
                action = action_0
                state = state_0
            else:
                action = policy()

            # Play the move
            n_steps += 1
            state_action_hist.append((state, action))

            if action is Move.STICK:
                break
            else:
                # Pick another card
                ace_count = int(p_has_ace)
                new_card = np.clip(np.random.randint(1, 14), a_min=1, a_max=10)

                ace_count += int(new_card==1)
                p_sum += new_card if (new_card != 1) else 11

                # Use aces if necessary
                while p_sum > 21 and ace_count > 0:

                    p_sum -= 10
                    ace_count -= 1

                p_has_ace = int(ace_count > 0)


                # Check busts
                if p_sum > 21:
                    return n_steps, state_action_hist, -1

                # Update for next iteration
                state = State(p_sum, d_card, p_has_ace)

                assert ace_count <= 1



        # Turn Dealer
        while True:

            action = self.dealer_policy(d_sum)

            if action is Move.STICK:
                break
            else:
                # Pick another card
                ace_count = int(d_has_ace)
                new_card = np.clip(np.random.randint(1, 14), a_min=1, a_max=10)

                ace_count += int(new_card == 1)
                d_sum += new_card if (new_card != 1) else 11


                # Use aces if necessary
                while d_sum > 21 and ace_count > 0:
                    d_sum -= 10
                    ace_count -= 1

                d_has_ace = int(ace_count > 0)


                # Check busts
                if d_sum > 21:
                    return n_steps, state_action_hist, 1

                assert ace_count <= 1


        assert (p_sum >= 12) and (p_sum <= 21) and (d_sum >= 12) and (d_sum <= 21)


        # Game over
        if (p_sum > d_sum):
            return n_steps, state_action_hist, 1
        elif (p_sum < d_sum):
            return n_steps, state_action_hist, -1
        else:
            assert p_sum == d_sum
            return n_steps, state_action_hist, 0


class Player:
    def __init__(self, gamma=DISCOUNT_FACTOR):
        # State actions values
        self._q = np.zeros((10, 10, 2, N_ACTIONS))

        self._q_sum = np.zeros((10, 10, 2, N_ACTIONS))

        # State action counts
        self._q_cnt = np.zeros((10, 10, 2,N_ACTIONS))

        # Policy
        self._policy = self.init_policy() # shape : (10,10,2) = (n_sum_p,n_card_d,n_has_ace)

        # Discount factor
        self._gamma = gamma

        # Init state (given by the book)
        self._init_state = State(p_sum=INIT_P_SUM, d_card=INIT_D_CARD, p_has_ace=INIT_P_HAS_ACE)

        # Numerator of off-policy state-value computation
        self.sum_scaled_returned = 0

        # Denominator of off-policy state-value computation (weighted importance sampling)
        self.sum_importance_sampling_ratio = 0

        # Denominator of off-policy state-value computation (ordinary importance sampling)
        self.first_visit_cnt = 0

        # Error on state-value prediction (weighted importance sampling)
        self.mse_weighted_hist = []

        # Error on state-value prediction (ordinary importance sampling)
        self.mse_ordinary_hist = []



    def init_policy(self):
        """Init policy : stick if sum >= 20, else hits."""

        policy = np.zeros((10,10,2))

        for i_p_sum in range(10):
            for i_d_card in range(10):
                for i_p_has_ace in range(2):
                    policy[i_p_sum, i_d_card, i_p_has_ace] = np.random.choice([0,1])

        return policy


    def get_state_action_value(self):
        return self._q

    def get_state_action_value_cnt(self):
        return self._q_cnt

    def get_current_policy(self):
        return self._policy

    def get_reward(self, outcome):

        if outcome is GameResult.PLAYER_WIN:
            return 1
        elif outcome is GameResult.PLAYER_LOSE:
            return -1
        else :
            return 0


    def follow_policy(self, state):
        i_p_sum, i_d_card, i_p_ace = self.state2inds3d(state)

        assert self._policy[i_p_sum, i_d_card, i_p_ace] in [0, 1]
        return Move.HIT if (self._policy[i_p_sum, i_d_card, i_p_ace] == 0) else Move.STICK


    def behavior_policy(self):
        """Randomly pick action regardless of the state, with equal probability."""
        return np.random.choice([Move.HIT, Move.STICK])

    def target_policy_prob(self, state, action):
        """Get the probability of taking action from state by following target policy (defined in the book)."""
        p_sum, _, _ = state
        assert (p_sum <= 21) and (p_sum >= 1)

        if (p_sum >= 20):
            return 1 if (action==Move.STICK) else 0
        else :
            return 1 if (action == Move.HIT) else 0


    def get_importance_sampling_ratio(self, state_action_hist):

        target_policy_term = []
        behavior_policy_term = []

        for (state, action) in state_action_hist:
            target_policy_term.append(self.target_policy_prob(state, action))
            behavior_policy_term.append(0.5)

        return np.prod(target_policy_term) / np.prod(behavior_policy_term)


    def get_state_value_ordinary_method(self):
        return self.sum_scaled_returned/self.first_visit_cnt if (self.first_visit_cnt != 0) else 0


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


    def run_episode_off_policy(self, blackjack, verbose=True):

        # Fixed inital state
        state_0 = self._init_state
        action_0 = self.behavior_policy()

        # Generate an episode
        n_steps, state_action_hist, reward = blackjack.play_game(state_0=state_0,
                                                                 action_0=action_0,
                                                                 policy=self.behavior_policy,
                                                                 verbose=False)


        # Update state action values
        last2first_steps = np.arange(n_steps, 0, -1) - 1  # [T-1, T-2, ..., 0]

        cumu_return = 0
        for t in last2first_steps:
            state, action = state_action_hist[t]

            is_last_step = (t == last2first_steps[0]) # t==T-1
            step_rwd = reward if is_last_step else 0

            cumu_return = self._gamma * cumu_return + step_rwd

            if (state, action) not in state_action_hist[:t]:
                # first visit to state, action
                i_p_sum, i_d_card, i_p_ace = self.state2inds3d(state)

                i_action = 0 if (action == Move.HIT) else 1

                self._q_cnt[i_p_sum, i_d_card, i_p_ace, i_action] += 1
                self._q_sum[i_p_sum, i_d_card, i_p_ace, i_action] += cumu_return

                self._q[i_p_sum, i_d_card, i_p_ace, i_action] = self._q_sum[i_p_sum, i_d_card, i_p_ace, i_action]/self._q_cnt[i_p_sum, i_d_card, i_p_ace, i_action]

                self._policy[i_p_sum, i_d_card, i_p_ace] = self._q[i_p_sum, i_d_card, i_p_ace, :].argmax()

                if state is state_0:
                    importance_sampling_ratio = self.get_importance_sampling_ratio(state_action_hist)

                    self.sum_scaled_returned += importance_sampling_ratio * cumu_return
                    self.sum_importance_sampling_ratio += importance_sampling_ratio
                    self.first_visit_cnt += 1

                    V_s_ordinary = np.around(self.get_state_value_ordinary_method(), 5)
                    V_s_weighted = np.around(self.get_state_value_weighted_method(), 5)

                    mse_ordinary = np.around(mean_squared_error([V_s_ordinary],[TRUE_STATE_VALUE]), 5)
                    mse_weighted = np.around(mean_squared_error([V_s_weighted],[TRUE_STATE_VALUE]), 5)

                    self.mse_ordinary_hist.append(mse_ordinary)
                    self.mse_weighted_hist.append(mse_weighted)


                    if verbose:
                        print(f'V_s_ordinary = {V_s_ordinary:.5f}, V_s_weighted = {V_s_weighted:.5f}, V_s_true = {TRUE_STATE_VALUE}')
                        print(f'mse_ordinary = {mse_ordinary}, mse_weighted = {mse_weighted}')
                        print('-')


    def get_state_value_errors(self):
        return self.mse_ordinary_hist, self.mse_weighted_hist


def plot_output_curves(values_10k , values_100k):

    v_no_ace_10k = values_10k[:100].reshape(10, 10)
    v_with_ace_10k = values_10k[100:].reshape(10, 10)

    v_no_ace_100k = values_100k[:100].reshape(10, 10)
    v_with_ace_100k = values_100k[100:].reshape(10, 10)

    # -----------------

    fig = plt.figure(figsize=(18, 12))

    i = np.arange(10) + 12  # Player's sum score
    j = np.arange(10) + 1  # Dealer's card
    jj, ii = np.meshgrid(i, j)

    plot1_name = '10 000 episodes, with usable ace'
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(ii, jj, v_with_ace_10k , cmap=cm.coolwarm)
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    ax1.set_zlabel('V')
    ax1.set_title(plot1_name)

    plot2_name = '10 000 episodes, no usable ace'
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.plot_surface(ii, jj, v_no_ace_10k, cmap=cm.coolwarm)
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    ax2.set_zlabel('V')
    ax2.set_title(plot2_name)

    plot3_name = '100 000 episodes, with usable ace'
    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(ii, jj, v_with_ace_100k, cmap=cm.coolwarm)
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    ax1.set_zlabel('V')
    ax1.set_title(plot3_name)

    plot4_name = '100 000 episodes, no usable ace'
    ax2 = fig.add_subplot(224, projection='3d')
    ax2.plot_surface(ii, jj, v_no_ace_100k, cmap=cm.coolwarm)
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    ax2.set_zlabel('V')
    ax2.set_title(plot4_name)


    fig.suptitle('First-visit MC prediction (blackjack problem)', fontsize=18)

    plt.waitforbuttonpress()


def plot_output_heatmap(values_10k , values_100k):


    v_no_ace_10k = values_10k[:100].reshape(10, 10)
    v_with_ace_10k = values_10k[100:].reshape(10, 10)

    v_no_ace_100k = values_100k[:100].reshape(10, 10)
    v_with_ace_100k = values_100k[100:].reshape(10, 10)

    # -----------------

    fig = plt.figure(figsize=(18, 12))

    i = np.arange(10) + 12  # Player's sum score
    j = np.arange(10) + 1  # Dealer's card
    jj, ii = np.meshgrid(i, j)

    plot1_name = '10 000 episodes, with usable ace'
    ax1 = fig.add_subplot(221)
    ax1 = sns.heatmap(v_with_ace_10k, ax=ax1)
    ax1.set_xlabel('Player sum')
    ax1.set_ylabel('Dealer showing')
    ax1.set_title(plot1_name)


    plot2_name = '10 000 episodes, no usable ace'
    ax2 = fig.add_subplot(223)
    ax2 = sns.heatmap(v_with_ace_10k, ax=ax2)
    ax2.set_xlabel('Player sum')
    ax2.set_ylabel('Dealer showing')
    ax2.set_title(plot2_name)


    plot3_name = '100 000 episodes, with usable ace'
    ax3 = fig.add_subplot(222)
    ax3 = sns.heatmap(v_with_ace_100k, ax=ax3)
    ax3.set_xlabel('Player sum')
    ax3.set_ylabel('Dealer showing')
    ax3.set_title(plot3_name)

    plot4_name = '100 000 episodes, no usable ace'
    ax4 = fig.add_subplot(224)
    ax4 = sns.heatmap(v_with_ace_100k, ax=ax4)
    ax4.set_xlabel('Player sum')
    ax4.set_ylabel('Dealer showing')
    ax4.set_title(plot4_name)

    fig.suptitle('First-visit MC prediction (blackjack problem)', fontsize=18)

    plt.waitforbuttonpress()


def plot_output_curves(player):
    q = player.get_state_action_value()
    policy = player.get_current_policy()

    policy_no_ace = policy[:, :, 0]
    policy_with_ace = policy[:, :, 1]

    q_no_ace = q[:, :, 0, :]
    q_with_ace = q[:, :, 1, :]

    v_no_ace = q_no_ace.max(axis=-1)
    v_with_ace = q_with_ace.max(axis=-1)

    ##

    # todo : Those state action values aren't similar to the book's one. As all others results are similar, there seem to be a slight difference of implementation.
    ''' 
    pairs2checks = [(12,6),(12,4),(13,2),(13,3),(16,10)]

    print('-'*4)
    for i_p_sum in range(10):
        for i_d_card in range(10):
            p_sum = i_p_sum+12
            d_card = i_d_card+1

            if (p_sum,d_card) in pairs2checks:
                print(f'({p_sum},{d_card}) : q = {q_no_ace[i_p_sum,i_d_card]}')
    print('-' * 4)
    ##
    '''

    fig = plt.figure(figsize=(18, 12))

    plot1_name = '(policy) Usable ace'
    ax1 = fig.add_subplot(221)
    ax1 = sns.heatmap(policy_with_ace, ax=ax1)
    ax1.set_ylabel('Player sum')
    ax1.set_yticklabels(range(12, 22))
    ax1.set_xlabel('Dealer showing')
    ax1.set_xticklabels(range(1, 11))
    ax1.set_title(plot1_name)
    ax1.invert_yaxis()

    plot2_name = '(policy) No usable ace'
    ax2 = fig.add_subplot(223)
    ax2 = sns.heatmap(policy_no_ace, ax=ax2)
    ax2.set_ylabel('Player sum')
    ax2.set_yticklabels(range(12, 22))
    ax2.set_xlabel('Dealer showing')
    ax2.set_xticklabels(range(1, 11))
    ax2.set_title(plot2_name)
    ax2.invert_yaxis()

    # 3D plots
    i = j = np.arange(10)
    jj, ii = np.meshgrid(i, j)

    plot3_name = '(values) Usable ace'
    ax3 = fig.add_subplot(222, projection='3d')
    ax3.plot_surface(ii, jj, v_with_ace.transpose(), cmap=cm.coolwarm)
    ax3.set_ylabel('Player sum')
    ax3.set_yticklabels(range(12, 22))
    ax3.set_xlabel('Dealer showing')
    ax3.set_xticklabels(range(1, 11))
    ax3.set_zlabel('V*')
    ax3.set_title(plot3_name)

    plot4_name = '(values) No usable ace'
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_surface(ii, jj, v_no_ace.transpose(), cmap=cm.coolwarm)
    ax4.set_ylabel('Player sum')
    ax4.set_yticklabels(range(12, 22))
    ax4.set_xlabel('Dealer showing')
    ax4.set_xticklabels(range(1, 11))
    ax4.set_zlabel('V*')
    ax4.set_title(plot4_name)

    plt.waitforbuttonpress()


def run_off_policy_state_value_estimation_blackjack():
    """Reproduces figure p106."""

    blackjack = Blackjack()

    all_mse_ordinary_hist = []
    all_mse_weighted_hist = []

    n_runs = 100
    n_episodes = 10000

    for run in tqdm(range(n_runs)):

        # Train each run from scratch
        player = Player()

        for e in range(n_episodes):
            # Running an episode
            player.run_episode_off_policy(blackjack, verbose=False)

            # Get errors
            mse_ordinary_hist, mse_weighted_hist = player.get_state_value_errors()

        all_mse_ordinary_hist.append(mse_ordinary_hist)
        all_mse_weighted_hist.append(mse_weighted_hist)

    all_mse_ordinary_hist = np.array(all_mse_ordinary_hist)
    all_mse_weighted_hist = np.array(all_mse_weighted_hist)

    avg_mse_ordinary_hist = all_mse_ordinary_hist.mean(axis=0)
    avg_mse_weighted_hist = all_mse_weighted_hist.mean(axis=0)

    data_mse = pd.DataFrame({'Ordinary I.S.': avg_mse_ordinary_hist, 'Weighted I.S.': avg_mse_weighted_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    sns.lineplot(data=data_mse, ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel('MSE (avg over 100 runs)')
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 5)
    fig.suptitle('Off-policy prediction via Importance sampling', fontsize=18)

    plt.waitforbuttonpress()


if __name__ == '__main__':

    run_off_policy_state_value_estimation_blackjack()



