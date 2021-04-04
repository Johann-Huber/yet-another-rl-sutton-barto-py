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

DISCOUNT_FACTOR = 1

#-------------


class Move(Enum):
    HIT = 0
    STICK = 1

class GameResult(Enum):
    PLAYER_WIN = 0
    PLAYER_LOSE = 1
    DRAW = 2

State = namedtuple('State', ['p_sum', 'd_card','p_has_ace'])
# p_sum: sum score of player's cards (in {12, ..., 21})
# d_card: card showed by the dealer (in {1, ..., 10})
# p_has_ace: card showed by the dealer (in {0, 1})


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


    def play_game(self, player, verbose=True):

        if verbose:
            print("Game starting.")

        # 1) Each player has 2 cards
        cards_p = np.clip(np.random.randint(1, 14, 2), a_min=0, a_max=10) # set picked heads to 10
        p_has_ace = self.has_usable_ace(cards_p)

        cards_d = np.clip(np.random.randint(1, 14, 2), a_min=0, a_max=10) # set picked heads to 10
        d_card = np.random.choice(cards_d)

        n_steps = 0
        state_hist = []

        # 2) Player pick cards to get sum >= 12

        # Include usable ace
        p_sum = (cards_p.sum() + 11 - 1) if (1 in cards_p) else cards_p.sum()
        d_sum = (cards_d.sum() + 11 - 1) if (1 in cards_d) else cards_d.sum()


        while p_sum < 12:
            rand_picked = np.random.randint(1, 11)
            if (rand_picked == 1) and (p_sum+11 <=21):
                p_has_ace = True
                p_sum += 11
            else :
                p_sum += rand_picked

        while d_sum < 12:
            rand_picked = np.random.randint(1, 11)
            if (rand_picked == 1) and (d_sum+11 <=21):
                d_sum += 11
            else :
                d_sum += rand_picked

        assert (p_sum >= 12) and (p_sum <= 21) and (d_sum >= 12) and (d_sum <= 21)

        #p_sum = np.random.randint(12, 21+1)
        #d_sum = np.random.randint(12, 21+1)

        if verbose:
            print('-')
            print("2) Picking cards")
            print(f"p_sum={p_sum} | d_sum={d_sum}")

        state_hist.append(State(p_sum, d_card, p_has_ace)) # init state (step = 0)

        # 3) Player's turn
        if verbose:
            print('-')
            print("3) Player's turn ")

        is_p_hitting = True
        while (is_p_hitting):
            n_steps += 1
            action = player.fixed_policy(State(p_sum, d_card, p_has_ace))

            if verbose:
                print("-")
                print(f"n_steps={n_steps}, state={State(p_sum, d_card, p_has_ace)}, action={action}")

            if action is Move.HIT:
                picked_card = np.random.randint(1, 11)

                if verbose:
                    print(f"picked_card={picked_card}")

                p_sum += picked_card

                state_hist.append(State(p_sum, d_card, p_has_ace))

                if p_sum > 21:
                    if verbose:
                        print(f"Player sum is over 21.")
                    outcome = GameResult.PLAYER_LOSE
                    return n_steps, state_hist, outcome

            else:
                if verbose:
                    print(f"p sticking | final_state = {State(p_sum, d_card, p_has_ace)}")

                state_hist.append(State(p_sum, d_card, p_has_ace))
                is_p_hitting = False

        # 4) Dealer's turn

        if verbose:
            print('--')
            print("4) Dealer's turn ")

        is_d_hitting = True
        while (is_d_hitting):

            if d_sum < 17:
                picked_card = np.random.randint(1, 11)

                if verbose:
                    print(f"d_sum={d_sum}, picked_card={picked_card}")

                d_sum += picked_card


                if d_sum > 21:

                    if verbose:
                        print(f"Dealer sum is over 21.")

                    outcome = GameResult.PLAYER_WIN
                    return n_steps, state_hist, outcome

            else:
                if verbose:
                    print(f"d sticking | d_sum = {d_sum}")

                is_d_hitting = False

        # 5) Game over : check outcome

        if verbose:
            print('--')
            print("5) Game over")

        assert d_sum <= 21 and p_sum <= 21, 'One of the players exceed 21 score.'

        if p_sum > d_sum:
            outcome = GameResult.PLAYER_WIN
        elif p_sum < d_sum:
            outcome = GameResult.PLAYER_LOSE
        else:
            outcome = GameResult.DRAW

        if verbose:
            print(f"n_steps={n_steps}")
            print(f"p_sum={p_sum}, d_sum={d_sum}, outcome={outcome}")
            print('-')
            print(f"state_hist={state_hist}")
            print('-'*25)

        return n_steps, state_hist, outcome


class Player:
    def __init__(self, gamma=DISCOUNT_FACTOR):
        # Expected reward from each state
        self._values = np.zeros(N_STATES,)
        # Dict containing the list of obtained returns for each already met states
        #self._returns = {k:[] for k in range(N_STATES)}
        self._returns_values = {k: [] for k in range(N_STATES)}

        # Discount factor
        self._gamma = gamma

    def fixed_policy(self, state):

        p_sum, _, _ = state

        if p_sum >= 20:
            return Move.STICK
        else:
            return Move.HIT

    def get_values(self):
        return self._values


    def get_reward(self, outcome):

        if outcome is GameResult.PLAYER_WIN:
            return 1
        elif outcome is GameResult.PLAYER_LOSE:
            return -1
        else :
            return 0

    def state2ind(self, state):
        p_sum, d_card, p_has_ace = state

        a = p_sum-12
        b = d_card-1
        c = p_has_ace

        return a + b * 10 + c * 100


    def first_visit_MC_prediction(self, n_steps, state_hist, outcome):
        """Compute value state estimation from the previously experienced episodes."""

        out_reward = self.get_reward(outcome)
        #print('out_reward=', out_reward)

        last2first_steps = np.arange(n_steps, 0, -1) - 1  # [T-1, T-2, ..., 0]
        #print('last2first_steps=', last2first_steps)

        cumu_return = 0 # rename explicitly
        for t in last2first_steps:
            state = state_hist[t]

            is_last_step = (t==last2first_steps[0])
            step_rwd = out_reward if is_last_step else 0

            cumu_return = self._gamma * cumu_return + step_rwd # R_t+1 ?

            if state not in state_hist[:t]:
                ind_state = self.state2ind(state)

                # first visit to state
                self._returns_values[ind_state].append(cumu_return)
                self._values[ind_state] = np.mean(self._returns_values[ind_state])


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


def run_first_visit_MC_blackjack():
    """Runs algorithm presented p92. Reproduces figure p94."""

    blackjack = Blackjack()
    player = Player()

    '''
    n_steps, state_hist, outcome = blackjack.play_game(player, verbose=False)
    print('output :')
    print(f"n_steps={n_steps}, state_hist={state_hist}, outcome={outcome}")
    ''';

    n_episodes = 10000
    for e in tqdm(range(n_episodes)):
        # Generates episode following policy :
        n_steps, state_hist, outcome = blackjack.play_game(player, verbose=False)
        # Update states value
        player.first_visit_MC_prediction(n_steps, state_hist, outcome)
    # Get results
    values_10k = player.get_values()

    player = Player()
    n_episodes = 100000
    for e in tqdm(range(n_episodes)):
        n_steps, state_hist, outcome = blackjack.play_game(player, verbose=False)
        player.first_visit_MC_prediction(n_steps, state_hist, outcome)
    values_100k = player.get_values()

    # Plot curves
    plot_output_curves(values_10k, values_100k)
    # plot_output_heatmap(values_10k, values_100k)

if __name__ == '__main__':

    run_first_visit_MC_blackjack()