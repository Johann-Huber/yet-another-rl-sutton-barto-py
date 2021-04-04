import numpy as np
from enum import Enum
from tqdm import tqdm
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

#-------------

N_EPISODES = 1

N_STATES = 200 # (Note : this param is know in this problem)
N_ACTIONS = 2 # Stick or hit

DISCOUNT_FACTOR = 1

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
                action = policy(state)

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

        # State action counts
        self._q_cnt = np.zeros((10, 10, 2,N_ACTIONS))

        # Policy
        self._policy = self.init_policy() # shape : (10,10,2) = (n_sum_p,n_card_d,n_has_ace)

        # Discount factor
        self._gamma = gamma


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

    def generate_exporation_start(self):

        # Iterate until a correct state is found
        init_state_found = False
        while (not init_state_found):

            # Pick 2 cards
            cards_p = np.clip(np.random.randint(1, 14, 2), a_min=1, a_max=10)

            # Include usable ace
            p_has_ace = 1 if (1 in cards_p) else 0
            p_sum = (cards_p.sum() + 11 - 1) if (1 in cards_p) else cards_p.sum()

            # Pick cards until reaching sum >= 12
            while p_sum < 12:
                rand_picked = np.clip(np.random.randint(1, 11), a_min=1, a_max=10)
                if (rand_picked == 1) and (p_sum + 11 <= 21):
                    p_sum += 11
                    p_has_ace = 1
                else:
                    p_sum += rand_picked

            init_state_found = p_sum <= 21

        d_card = np.clip(np.random.randint(1, 14), a_min=1, a_max=10)

        assert (p_sum >= 12) and (p_sum <= 21)
        assert (d_card >= 1) and (d_card <= 10)
        assert p_has_ace in [0, 1]

        return State(p_sum, d_card, p_has_ace)

    def state2inds3d(self, state):
        p_sum, d_card, p_has_ace = state

        assert (p_sum >= 12) and (p_sum <= 21)
        assert (d_card >= 1) and (d_card <= 10)
        assert p_has_ace in [0, 1]

        i_p_sum = p_sum - 12
        i_d_card = d_card - 1
        i_p_ace = p_has_ace

        return i_p_sum, i_d_card, i_p_ace

    def run_episode_MC_exploring_starts(self, blackjack):

        # Randomly choose state & action
        state_0 = State(p_sum=np.random.choice(range(12,22)),
                        d_card=np.random.choice(range(1,11)),
                        p_has_ace=np.random.choice([0,1]))
        action_0 = np.random.choice([Move.HIT, Move.STICK])

        # Generate an episode
        n_steps, state_action_hist, reward = blackjack.play_game(state_0=state_0,
                                                                 action_0=action_0,
                                                                 policy=self.follow_policy,
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

                n = self._q_cnt[i_p_sum, i_d_card, i_p_ace, i_action]
                self._q[i_p_sum, i_d_card, i_p_ace, i_action] += (1/n)*(cumu_return-self._q[i_p_sum, i_d_card, i_p_ace, i_action])

                self._policy[i_p_sum, i_d_card, i_p_ace] = self._q[i_p_sum, i_d_card, i_p_ace, :].argmax()


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
    # todo : Those state action values are very close but different from the book's one.
    #  As all others results are similar, there seem to be a slight difference of implementation.
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
    ''';
    ##


    fig = plt.figure(figsize=(18, 12))

    plot1_name = '(policy) Usable ace [0=HIT, 1=STICK]'
    ax1 = fig.add_subplot(221)
    ax1 = sns.heatmap(policy_with_ace, ax=ax1)
    ax1.set_ylabel('Player sum')
    ax1.set_yticklabels(range(12, 22))
    ax1.set_xlabel('Dealer showing')
    ax1.set_xticklabels(range(1, 11))
    ax1.set_title(plot1_name)
    ax1.invert_yaxis()

    plot2_name = '(policy) No usable ace [0=HIT, 1=STICK]'
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


def run_blackjack_explorating_starts():
    """Runs algorithm presented p99. Reproduces figure p100."""

    blackjack = Blackjack()
    player = Player()

    n_episodes = 5000000
    for e in tqdm(range(n_episodes)):
        # Running an episode from random initial state
        player.run_episode_MC_exploring_starts(blackjack)

    plot_output_curves(player)


if __name__ == '__main__':

    run_blackjack_explorating_starts()


