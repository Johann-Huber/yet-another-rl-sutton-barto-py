
import numpy as np
import pandas as pd
import time
import itertools
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#--------------------------------------------------------------------
# PARAMETERS

# Display rate for real-time update
DISP_REFRESH_RATE = 0.1

# Debug flag
DEBUG_FLG = 0


board_code_2_char = {0: ' ',
                     1: 'x',
                     2: 'o'}

BOARD_NUM_ROWS = 3
BOARD_NUM_COLS = 3

# Required number of aligned markers to win a game
NUM2ALIGN = 3

# Required number of aligned markers to win a game
WIN_CONFIG_DEFAULT_VALUE = 1.
LOSE_CONFIG_DEFAULT_VALUE = 0.
NEUTRAL_CONFIG_DEFAULT_VALUE = 0.5

EXPLORATION_RATE = 0.1
LEARNING_STEP_SIZE = 0.001
DO_DECAY = True

MAX_NUM_GAMES = 1000
RESULTS_PLOT_NAME = 'reward_over_time'

VERBOSE = False

FIGURE_SIZE = (16,12)

#--------------------------------------------------------------------


def print_board(board):
    num_rows,num_cols = board.shape()

    for i in range(num_rows+2):
        for j in range(num_cols+2):

            is_horiz_border = (i==0) or (i==num_rows+1)
            is_verti_border = (j==0) or (j==num_cols+1)

            if is_verti_border:
                char2print = '|'
            elif is_horiz_border:
                char2print = '-'
            else:
                char2print = board_code_2_char[board[i,j]]

            print(char2print)


def init_board(num_rows,num_cols):
    return np.zeros((num_rows,num_cols))


class Board:
    def __init__(self, num_rows_=BOARD_NUM_ROWS, num_cols_=BOARD_NUM_COLS, num2align_=NUM2ALIGN):
        self._state =  np.zeros((num_rows_,num_cols_), dtype=np.uint8)

        if DEBUG_FLG:
            self._state = np.array(range(num_rows_ * num_cols_)).reshape(num_rows_, num_cols_)

        self._shape = (num_rows_, num_cols_)
        self.num2align = num2align_


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    def reset_board(self, num_rows_=BOARD_NUM_ROWS, num_cols_=BOARD_NUM_COLS):
        self._state = np.zeros((num_rows_, num_cols_), dtype=np.uint8)

    def add_move_board(self, i, j, agent_num):
        current_state = self.state
        assert agent_num in [1,2], 'agent_num must be in [1,2].'
        assert current_state[i,j] == 0, 'Non-empty cell selected.'
        current_state[i,j] = agent_num

        self.state = current_state


    def draw_horiz_border(self):
        _, num_cols = self.shape
        horiz_border_len = num_cols + 2

        for j in range(horiz_border_len):
            is_last_col = (j == horiz_border_len-1)
            if is_last_col : print('-')
            else : print('-', end='')


    def draw_board_body(self):
        num_rows, num_cols = self.state.shape

        for i in range(num_rows):
            print('|', end='')
            for j in range(num_cols):
                is_border = (j == 0) or (j == num_cols + 1)

                if DEBUG_FLG:
                    print(str(self.state[i, j]), end='')
                else:
                    print(board_code_2_char[self.state[i, j]], end='')

            print('|')


    def display(self):
        self.draw_horiz_border()
        self.draw_board_body()
        self.draw_horiz_border()


    def get_alignment_subset(self, subset):
        """

        :param subset: slice from the board to examine.
        :return: 0 : Not alignment in the given subset;
                 1 : Agent 1 got an alignment;
                 2 : Agent 2 got an alignment.
        """
        if (subset == np.repeat(1, self.num2align)).all():
            return 1 # Player 1 wins
        if (subset == np.repeat(2, self.num2align)).all():
            return 2 # Player 2 wins

        return 0

    def check_alignments(self, board_state=None):
        """ Check whether there is a aligmnent on the board.

        :param board_state: Current state of the board. None : check alignment on the
                            current board config. Give a specific board config as parameter
                            to check alignments on simulated boards.
        :return: -1 : No winner yet ;
                 0 : Draw ;
                 1 : Agent 1 wins ;
                 2 : Agent 2 wins ;
        """

        if board_state is None:
            board_state = self.state

        num_rows, num_cols = self.shape

        # Player wins ?
        for i in range(num_rows):
            for j in range(num_cols):

                # Horizontal alignment
                if j < (num_cols - (self.num2align - 1)):
                    horiz_subset = board_state[i, j:j + self.num2align]
                    horiz_alignment_outcome = self.get_alignment_subset(horiz_subset)
                    if horiz_alignment_outcome != 0:
                        return horiz_alignment_outcome

                # Vertical alignment
                if i < (num_rows - (self.num2align - 1)):
                    vert_subset = board_state[i:i + self.num2align, j]
                    vert_alignment_outcome = self.get_alignment_subset(vert_subset)
                    if vert_alignment_outcome != 0:
                        return vert_alignment_outcome

                # Top-left to bottom-right diagonal alignment
                if (i < (num_rows - (self.num2align - 1))) and (j < (num_cols - (self.num2align - 1))):
                    tl_br_diag_subset = board_state[i + np.array(range(self.num2align)),
                                                    j + np.array(range(self.num2align))]
                    tl_br_diag_alignment_outcome = self.get_alignment_subset(tl_br_diag_subset)
                    if tl_br_diag_alignment_outcome != 0:
                        return tl_br_diag_alignment_outcome

                # Top-right to bottom-left diagonal alignment
                if (i < (num_rows - (self.num2align - 1))) and (j >= self.num2align - 1):
                    tr_bl_diag_subset = board_state[i + np.array(range(self.num2align)),
                                                    j - (self.num2align - 1) + np.array(range(self.num2align))[::-1]]
                    tr_bl_diag_alignment_outcome = self.get_alignment_subset(tr_bl_diag_subset)
                    if tr_bl_diag_alignment_outcome != 0:
                        return tr_bl_diag_alignment_outcome

        # Draw ?
        if 0 not in board_state:
            return 0

        # Game not over yet
        return -1


    def check_draw(self):
        board_state = self.state

        if 0 not in board_state:
            return True
        else :
            return False



class Game:
    def __init__(self, board_, agent1_, agent2_, num2align_=NUM2ALIGN, max_num_games_=MAX_NUM_GAMES,
                 results_plot_name=RESULTS_PLOT_NAME, fig_size=FIGURE_SIZE):
        self.board = board_
        self.agent1 = agent1_
        self.agent2 = agent2_
        self.num2align = num2align_

        self.scores = {1: {'num_win': 0, 'num_loss': 0, 'num_draw': 0},
                       2: {'num_win': 0, 'num_loss': 0, 'num_draw': 0}}

        self.max_num_games = max_num_games_

        self.running_avg_score = []

        self.results_plot_name = results_plot_name

        self.fig_size = fig_size

    def get_alignment_subset(self, subset):
        """

        :param subset: slice from the board to examine.
        :return: 0 : Not alignment in the given subset;
                 1 : Agent 1 got an alignment;
                 2 : Agent 2 got an alignment.
        """
        if (subset == np.repeat(1, self.num2align)).all():
            return 1 # Player 1 wins
        if (subset == np.repeat(2, self.num2align)).all():
            return 2 # Player 2 wins

        return 0

    def get_current_outcome(self, board_state):
        """ Gets the current outcome of the game.

        :param board_state: Current state of the board.
        :return: -1 : No winner yet ;
                 0 : Draw ;
                 1 : Agent 1 wins ;
                 2 : Agent 2 wins ;
        """
        num_rows, num_cols = self.board.shape

        # Player wins ?
        alignmt = self.board.check_alignments()
        if alignmt in [1,2]:
            return alignmt

        # Draw ?
        is_draw = self.board.check_draw()
        if is_draw:
            return 0

        # Game not over yet
        return -1


    def is_game_over(self):
        board_state = self.board.state
        outcome = self.get_current_outcome(board_state)

        if outcome == -1:
            return False
        else :
            return True


    def play_game(self, verbose=True, disp_mode='never'):
        """ Simulate a game of tic tac toe.

        :param agent1: First agent to play.
        :param agent2: Second agent to play.
        :return: The number of the agent that won the game. Returns 0 for a draw.
        """

        assert disp_mode in ['end', 'real_time', 'never']

        for id_game in tqdm(range(self.max_num_games), desc='Simulating games'):

            if verbose:
                print('='*5 + f'GAME {id_game+1}/{self.max_num_games}' + '='*5)

            self.board.reset_board()
            running_game = True

            while(running_game):

                # Agent 1 turn
                i,j = self.agent1.play_move(self.board)
                self.board.add_move_board(i,j,1)

                if disp_mode == 'real_time':
                    self.board.display()
                    time.sleep(DISP_REFRESH_RATE)

                if self.is_game_over() :

                   is_agent1_winner = self.get_result(verbose=False) == 1
                   if is_agent1_winner :
                       # Back up defeat
                       self.agent2.call_back_up_prev_state(curr_state_value=LOSE_CONFIG_DEFAULT_VALUE)

                   running_game = False
                   break

                # Agent 2 turn
                i, j = self.agent2.play_move(self.board)
                self.board.add_move_board(i, j, 2)

                if disp_mode == 'real_time':
                    self.board.display()
                    time.sleep(DISP_REFRESH_RATE)

                if self.is_game_over() :

                    is_agent2_winner = self.get_result(verbose=False) == 2
                    if is_agent2_winner:
                        # Back up defeat
                        self.agent1.call_back_up_prev_state(curr_state_value=LOSE_CONFIG_DEFAULT_VALUE)

                    running_game = False

            # Game over
            if disp_mode == 'end':
                self.board.display()
                time.sleep(DISP_REFRESH_RATE)

            self.get_result(verbose=verbose)
            self.update_scores()
            self.update_running_avg_score()

            self.update_agents_game_ending_stats()

        # Session over
        self.disp_scores()



    def update_agents_game_ending_stats(self):
        self.agent1.update_game_stats()
        self.agent2.update_game_stats()


    def get_result(self, verbose=True):
        board_state = self.board.state
        outcome = self.get_current_outcome(board_state)

        assert outcome in [0,1,2], 'Game not over yet.'

        if verbose :
            if outcome == 1:
                print('Agent 1 wins.')
            elif outcome == 2:
                print('Agent 2 wins.')
            else:
                print('Draw.')

        return outcome

    def update_scores(self):
        board_state = self.board.state
        outcome = self.get_current_outcome(board_state)

        assert outcome in [0, 1, 2], 'Game not over yet.'

        if outcome == 1:
            self.scores[1]['num_win'] += 1
            self.scores[2]['num_loss'] += 1
        elif outcome == 2:
            self.scores[2]['num_win'] += 1
            self.scores[1]['num_loss'] += 1
        else:
            self.scores[1]['num_draw'] += 1
            self.scores[2]['num_draw'] += 1


    def disp_scores(self):
        wins1 = self.scores[1]['num_win']
        wins2 = self.scores[2]['num_win']
        draws = self.scores[1]['num_draw']
        num_games = self.scores[1]['num_win'] + self.scores[1]['num_loss'] + self.scores[1]['num_draw']
        print(f'Agent1 : {wins1} , Agent2 : {wins2} , Draw : {draws} , Num of games : {num_games}')


    def update_running_avg_score(self):
        # 1 point : win, 0.5 point : draw, 0 point : lose

        sum_score = self.scores[1]['num_win'] + 0.5*self.scores[1]['num_draw']
        num_games = self.scores[1]['num_win'] + self.scores[1]['num_loss'] + self.scores[1]['num_draw']
        avg_score = sum_score/num_games


        self.running_avg_score.append(avg_score)


    def plot_running_avg(self, title='', output_path=None, export=False):

        records = pd.DataFrame({'num_explor': self.agent1.num_explor_games, # instead : num_of_exploration_per_game
                                'avg_score': self.running_avg_score,
                                'step': list(range(len(self.running_avg_score))),
                               })

        fig, ax = plt.subplots(1, 1, figsize=self.fig_size)
        sns.scatterplot(data=records,
                        x='step',
                        y='avg_score',
                        hue='num_explor',
                        ax=ax)
        ax.set_title(title, fontsize=18)
        plt.waitforbuttonpress()

        if export:
            if output_path:
                output_file_name = f'{output_path}.png'
            else:
                output_file_name = f'{self.results_plot_name}.png'
            plt.savefig(output_file_name)
            print(f'Plot exported as : {output_file_name}')



    def get_records_df(self):
        return pd.DataFrame({'num_explor': self.agent1.num_explor_games,  # instead : num_of_exploration_per_game
                             'avg_score': self.running_avg_score,
                             'step': list(range(len(self.running_avg_score))),
                            })



class Agent:

    def __init__(self, policy, agent_num, board, lr_step_size=LEARNING_STEP_SIZE, do_decay=DO_DECAY,
                 exploration_rate = EXPLORATION_RATE):

        assert policy in ['random', 'temporal_learning']
        self._policy = policy
        self._agent_num = agent_num
        self._value_function = ValueFunction(board, agent_num, lr_step_size=lr_step_size, do_decay=do_decay)

        self._was_prev_step_explor = False

        self.num_explor_games = []
        self.running_num_explor = 0
        self.num_learning_steps = 0

        self.exploration_rate = exploration_rate

    def get_available_moves(self, board_state):
        """

        :param board_state: Current boad state.
        :return: Returns a ndarray of size (num_available_moves,2)
        """

        i_avlbl, j_avlbl = np.where(board_state == 0)

        available_moves = np.array(list(zip(i_avlbl, j_avlbl)))
        assert len(available_moves) > 0, 'No available move.'

        return available_moves



    def play_random_move(self, board):
        available_moves = self.get_available_moves(board.state)

        move_id = np.random.choice(len(available_moves))
        i,j = available_moves[move_id]

        return i,j

    def update_running_num_explor(self, is_exploration_step):
        if is_exploration_step:
            self.running_num_explor += 1

    def update_game_stats(self):
        # Call after each games.

        self.num_explor_games.append(self.running_num_explor)
        self.running_num_explor = 0


    def call_back_up_prev_state(self, curr_state_value):
        if self._policy == 'temporal_learning':
            num_steps = self.num_learning_steps
            self._value_function.update_prev_state_value(curr_state_value, num_steps)


    def play_temporal_learning_move(self, board):
        available_moves = self.get_available_moves(board.state)

        # Get score for each of the next move
        next_state_vals = []

        for i, j in available_moves:

            simulated_board = copy.deepcopy(board)
            simulated_board.add_move_board(i, j, self._agent_num)

            value = self._value_function.get_state_value(simulated_board.state.ravel())
            next_state_vals.append(value)

        # Extract the most promising one
        id_best_move = np.argmax(np.array(next_state_vals))

        # TEMPORAL TRAINING : update prev state value
        was_prev_step_exploit = not self._was_prev_step_explor
        if was_prev_step_exploit:
            best_move_value = np.max(np.array(next_state_vals))
            self.call_back_up_prev_state(best_move_value)

        # Exploration or exploitation
        is_exploration_step = (random.random() < self.exploration_rate) and len(next_state_vals) > 1
        self._was_prev_step_explor = is_exploration_step

        # Learn when exploiting :
        #self.update_running_num_explor(is_exploration_step)

        # Always learn :
        is_there_at_least_one_step = len(next_state_vals) > 1
        self.update_running_num_explor(is_there_at_least_one_step)

        if is_exploration_step:
            available_moves = np.delete(np.array(available_moves), id_best_move, 0)
            move_id = np.random.choice(len(available_moves))

            i_explor, j_explor = available_moves[move_id]
            i, j = i_explor, j_explor
        else:
            i_best, j_best = available_moves[id_best_move]

            i,j = i_best, j_best

        # Update previous state value for training
        simulated_board = copy.deepcopy(board)
        simulated_board.add_move_board(i, j, self._agent_num)
        self._value_function.save_prev_state(simulated_board.state.ravel())

        self.num_learning_steps += 1
        return i,j


    def play_move(self, board):

        assert self._policy in ['random', 'temporal_learning']

        if self._policy == 'random':
            return self.play_random_move(board)

        elif self._policy == 'temporal_learning':
            return self.play_temporal_learning_move(board)



class ValueFunction:
    def __init__(self, board, agent_num, lr_step_size = LEARNING_STEP_SIZE, do_decay=DO_DECAY):
        self._values_dict = {}
        self._agent_num = agent_num

        self._num_rows, self._num_cols = board.state.shape

        self._id_prev_state = -1
        self._prev_state_value = -1

        self._lr_step_size = lr_step_size
        self._do_decay = do_decay

        state_id = 0
        for board_state in itertools.product([0, 1, 2], repeat=self._num_rows*self._num_cols):
            board_state = np.array(board_state).astype(np.int8)

            num_x = np.where(board_state == 1)[0].shape[0]
            num_o = np.where(board_state == 2)[0].shape[0]

            is_plausible_state = (num_x == num_o) or (num_x == num_o + 1)
            # Note : Should also discard states where both players have alignments, or two alignments which
            # cannot appear together (ex : opposite side of the board). Those states will never be called in
            # the game. Less RAM could be used, but this optimization is not necessary for our experiments.
            if is_plausible_state:
                self._values_dict[state_id] = {'state': board_state,
                                               'value': -1}
                state_id += 1

        self.init_values(board)




    def init_values(self, board):

        agent_num = self._agent_num
        opp_num = agent_num % 2 + 1
        rows, cols = self._num_rows, self._num_cols

        for id_state in self._values_dict:

            s = self._values_dict[id_state]['state'].reshape(rows,cols)

            alignmt = board.check_alignments(board_state=s)

            is_a_win = alignmt == agent_num
            is_a_lose = alignmt == opp_num

            if is_a_win:
                self._values_dict[id_state]['value'] = WIN_CONFIG_DEFAULT_VALUE
            elif is_a_lose:
                self._values_dict[id_state]['value'] = LOSE_CONFIG_DEFAULT_VALUE
            else:
                self._values_dict[id_state]['value'] = NEUTRAL_CONFIG_DEFAULT_VALUE

            v = self._values_dict[id_state]['value']



    def get_state_value(self, board_state):

        is_state_found = False
        for id_state in self._values_dict:
            s = self._values_dict[id_state]['state']

            is_state_found = (s==board_state).all()
            if is_state_found:
                return self._values_dict[id_state]['value']

        assert is_state_found, 'State not found.'
        return -1

    def save_prev_state(self, board_state):

        is_state_found = False
        for id_state in self._values_dict:
            s = self._values_dict[id_state]['state']

            is_state_found = (s==board_state).all()
            if is_state_found:
                self._id_prev_state = id_state
                self._prev_state_value = self._values_dict[id_state]['value']
                return True

        assert is_state_found, 'State not found.'
        return False


    def update_prev_state_value(self, best_move_value, num_steps):

        no_exploit_step_yet = self._id_prev_state == -1
        if no_exploit_step_yet:
            # At least one exploitation step is required to do temporal learning
            return

        V_S = best_move_value
        V_S_last = self._prev_state_value
        alpha = self._lr_step_size

        if self._do_decay:
            decay_coef = 1./(1+alpha*100*num_steps)
        else :
            decay_coef = 1.

        # TEMPORAL LEARNING
        V_S_last_updated = V_S_last + decay_coef * alpha * (V_S - V_S_last)

        self._values_dict[self._id_prev_state]['value'] = V_S_last_updated



def run_experiment_1():
    """Experiment 1 : Temporal learning agent against only-random-actions agent. (p.10)"""

    board = Board()

    avg_scores = []

    num_runs = 1  # 5
    for i_run in range(num_runs):
        agent1 = Agent(policy='temporal_learning', agent_num=1, board=board, do_decay=True)
        agent2 = Agent(policy='random', agent_num=2, board=board)

        game = Game(board, agent1, agent2)
        game.play_game(verbose=VERBOSE)

        game.plot_running_avg(title="Learning agent's win rate over time", export=True,
                              output_path='play_against_random_opponent_win_rate')
        exp1_df = game.get_records_df()

        print(f'[run{i_run}] Avg score = ', exp1_df["avg_score"].iloc[-1])
        avg_scores.append(exp1_df["avg_score"].iloc[-1])

    print('=' * 10)
    avg_scores = np.array(avg_scores)
    print(f'Avg score = {avg_scores.mean():.2f}, std = {avg_scores.std():.2f} on {num_runs} runs.')


def run_experiment_2():
    """Experiment 2 : Two agents learning to play against the other one. (p.12)"""
    
    board = Board()

    agent1 = Agent(policy='temporal_learning', agent_num=1, board=board)
    agent2 = Agent(policy='temporal_learning', agent_num=2, board=board)
    game = Game(board, agent1, agent2)
    game.play_game(verbose=VERBOSE)
    game.plot_running_avg(title='Self-play win rate over time', export=True,
                          output_path='self_play_win_rate_3')
    exp2_df = game.get_records_df()
    print(exp2_df)


if __name__ == '__main__':

    run_experiment_1()

    #run_experiment_2()