import numpy as np
import operator
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.ion()

# -----------------------------------------------

# JACK'S CAR RENTAL PARAMETERS
NUM_MAX_CARS = 20
NUM_MAX_MOVES = 5

RENT_REWARD = 10
MOVE_COST = 2
FREE_PARKING_LOT_LIMIT = 10
ADDITIONAL_PARKING_LOT_COST = 4

LAMBDA_REQUEST_1 = 3
LAMBDA_REQUEST_2 = 4
LAMBDA_RETURN_1 = 3
LAMBDA_RETURN_2 = 2

MAX_RANGE_REQ_RET = 11 # k>12 : P(X=k) < 0.001

# POLICY ITERATION PARAMETERS

DISCOUNT_FACTOR = 0.9

STEP_MAX_THRESHOLD = 0.01


# UTILS

FIGURE_SIZE = (16,8)


# -----------------------------------------------


class Policy:
    def __init__(self, env, num_max_cars=NUM_MAX_CARS, num_max_moves=NUM_MAX_MOVES, gamma=DISCOUNT_FACTOR,
                 max_range_req_ret=MAX_RANGE_REQ_RET, fix_returned=True):

        # Car rental environment
        self._env = env

        # Number of states per parking lot : (0,1,...,20) car(s) available
        self._num_states = num_max_cars + 1

        # Policy's action for each state s = (i,j) = (num_cars_1,num_cars_2)
        # -> Init at a=0 for all states.
        self._actions = np.zeros((self._num_states,self._num_states))

        # Estimated value associated to each state s
        self._values = np.zeros((self._num_states, self._num_states))

        # Discount factor
        self._gamma = gamma

        # Max number of car per site
        self._num_max_cars = num_max_cars

        # Max number of car move per night
        self._num_max_moves = num_max_moves

        # Available action from each state s
        self._available_moves =  np.array(range(-NUM_MAX_MOVES, NUM_MAX_MOVES+1,1))

        # Max number of expected rental requests (or returns) per day
        self._max_range_req_ret = max_range_req_ret

        # Trick : fix returns to expected values to reduce computation
        self._fixed_return = fix_returned

        # Nested dict containing probability for each (num_req_1, num_ret_1, num_req_2, num_ret_2)
        self.ret_req_probs = self.init_probs()


    def get_current_policy(self):
        return self._actions

    def get_current_values(self):
        return self._values



    def init_probs(self):
        """ Compute combined probabilities for all event combination w.r.t. Poisson's law.

        :return: Intricated dicts containing the probability associated to each event. The dict follows
        the key pattern : [num_req_1][num_ret_1][num_req_2][num_ret_2] = prob
        """

        # Ugly but do the job : rewriting it would be great

        ret_req_probs = {}

        if self._fixed_return:
            for num_req_1 in range(MAX_RANGE_REQ_RET):
                ret_req_probs[num_req_1] = {}

                num_ret_1 = LAMBDA_RETURN_1
                ret_req_probs[num_req_1][num_ret_1] = {}

                for num_req_2 in range(MAX_RANGE_REQ_RET):
                    ret_req_probs[num_req_1][num_ret_1][num_req_2] = {}

                    num_ret_2 = LAMBDA_RETURN_2
                    ret_req_probs[num_req_1][num_ret_1][num_req_2][num_ret_2] = \
                        self.poisson_prob(k=num_req_1, lmbda=LAMBDA_REQUEST_1) * \
                        self.poisson_prob(k=num_req_2, lmbda=LAMBDA_REQUEST_2)
        else:
            for num_req_1 in range(MAX_RANGE_REQ_RET):
                ret_req_probs[num_req_1] = {}

                for num_ret_1 in range(MAX_RANGE_REQ_RET):
                    ret_req_probs[num_req_1][num_ret_1] = {}

                    for num_req_2 in range(MAX_RANGE_REQ_RET):
                        ret_req_probs[num_req_1][num_ret_1][num_req_2] = {}

                        for num_ret_2 in range(MAX_RANGE_REQ_RET):
                            ret_req_probs[num_req_1][num_ret_1][num_req_2][num_ret_2] = \
                                self.poisson_prob(k=num_req_1, lmbda=LAMBDA_REQUEST_1) * \
                                self.poisson_prob(k=num_ret_1, lmbda=LAMBDA_RETURN_1) * \
                                self.poisson_prob(k=num_req_2, lmbda=LAMBDA_REQUEST_2) * \
                                self.poisson_prob(k=num_ret_2, lmbda=LAMBDA_RETURN_2)

        return ret_req_probs


    def poisson_prob(self, k, lmbda):
        """ Compute p(X=K) for X ~ Poisson distribution. """
        return (np.exp(-lmbda)*(lmbda**k))/math.factorial(k)


    def get_action_value(self, state, action):
        """ Compute q(s,a), the expected reward by taking action a while being at state s.

        :param state: Initial state
        :param action: Chosen action
        :return: Expected reward.
        """

        q = 0.

        if self._fixed_return:

            # Explore all (s',r)
            for num_req_1 in range(MAX_RANGE_REQ_RET):
                num_ret_1 = LAMBDA_RETURN_1
                for num_req_2 in range(MAX_RANGE_REQ_RET):
                    num_ret_2 = LAMBDA_RETURN_2

                    # Prob to get current (s',r)
                    prob_next_state = self.ret_req_probs[num_req_1][num_ret_1][num_req_2][num_ret_2]

                    n_car_request = np.array([num_req_1, num_req_2])
                    n_car_returned = np.array([num_ret_1, num_ret_2])

                    next_state, reward = self._env.run_step(state, action, n_car_request, n_car_returned)

                    i_next, j_next = next_state

                    q += prob_next_state * (reward + self._gamma * self._values[i_next, j_next])


        else:
            for num_req_1 in range(MAX_RANGE_REQ_RET):
                for num_ret_1 in range(MAX_RANGE_REQ_RET):
                    for num_req_2 in range(MAX_RANGE_REQ_RET):
                        for num_ret_2 in range(MAX_RANGE_REQ_RET):

                            prob_next_state = self.ret_req_probs[num_req_1][num_ret_1][num_req_2][num_ret_2]

                            n_car_request = np.array([num_req_1, num_req_2])
                            n_car_returned = np.array([num_ret_1, num_ret_2])

                            next_state, reward = self._env.run_step(state, action, n_car_request, n_car_returned)

                            i_next, j_next = next_state

                            q += prob_next_state * (reward + self._gamma * self._values[i_next, j_next])

        return q


    def evaluate(self):

        max_step = 0.

        for i in range(self._num_states):
            for j in range(self._num_states):

                state = np.array([i, j])
                action = self._actions[i, j]
                prev_value = self._values[i, j].copy()

                # Values = expected reward from that state => expected rwd from s by taking a (single possble action)
                # summed over all (s',r) => all (req1,ret1,req2,ret2) combinations
                self._values[i, j] = self.get_action_value(state, action)

                max_step = max(max_step, abs(self._values[i, j] - prev_value))

        return max_step


    def get_greedy_policy(self):
        """ Return for each state the action which leads to the max expected reward. """

        greedy_policy = np.zeros((self._num_states, self._num_states))

        for i in range(self._num_states):
            for j in range(self._num_states):

                state = np.array([i, j])
                action_values = []
                associated_actions = []

                for action in self._available_moves:
                    if (0 <= action <= i) or (-j <= action <= 0):

                        action_value = self.get_action_value(state, action)
                        action_values.append(action_value)
                        associated_actions.append(action)

                assert action_values
                imax_val = np.array(action_values).argmax()
                greedy_action = associated_actions[imax_val]

                greedy_policy[i, j] = greedy_action

        return greedy_policy


    def policy_improvement(self):
        """Update current policy with a greedy one corresponding to current values.
        Returns True if policy is optimal, False otherwise."""

        prev_actions = self._actions.copy()
        self._actions = self.get_greedy_policy()

        is_policy_stable = True

        for i in range(self._num_states):
            for j in range(self._num_states):

                if prev_actions[i,j] != self._actions[i,j]:
                    is_policy_stable= False

        return is_policy_stable



    def plot_greedy_policy(self):
        """ Plot the heat map of the current greedy policy.

        Note : Meant to reproduce p81 book's figures."""

        greedy_policy = self.get_greedy_policy()

        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        ax = sns.heatmap(greedy_policy, ax=ax)
        ax.invert_yaxis()
        plt.waitforbuttonpress()





class CarRental:

    def __init__(self, num_max_cars=NUM_MAX_CARS):
        self._num_states = num_max_cars + 1
        self._num_max_cars = num_max_cars


    def run_step(self, state, action, n_car_request, n_car_returned):

        n_car_parked = state.copy()
        reward = 0.

        # --------- Night ---------
        n_car_parked[0] = np.clip( n_car_parked[0]-action, a_min=0, a_max=self._num_max_cars) # move from/to parking 1
        n_car_parked[1] = np.clip( n_car_parked[1]+action, a_min=0, a_max=self._num_max_cars) # move from/to parking 2

        reward -= abs(action) * MOVE_COST

        is_there_one_free_move = action > 0
        if is_there_one_free_move:
            reward += MOVE_COST # subtract one move_cost penalty


        if n_car_parked[0] > FREE_PARKING_LOT_LIMIT:
            reward -= ADDITIONAL_PARKING_LOT_COST
        if n_car_parked[1] > FREE_PARKING_LOT_LIMIT:
            reward -= ADDITIONAL_PARKING_LOT_COST


        # --------- Day ---------

        n_remaining = np.maximum(n_car_parked - n_car_request, 0)
        n_rented = n_car_parked - n_remaining

        next_state = np.minimum(n_remaining + n_car_returned, self._num_max_cars)

        reward += n_rented.sum() * RENT_REWARD

        return next_state, reward


def draw_output_fig(policies, values_opti):
    """Reproduce the book's curves (p 81)."""

    fig = plt.figure(figsize=(18, 12))

    # Initial policy
    plot0_name = 'Init policy'
    ax0 = fig.add_subplot(231)
    ax0 = sns.heatmap(policies[plot0_name], ax=ax0)
    ax0.invert_yaxis()
    ax0.set_title(plot0_name)

    # Policy iterations
    plot1_name = 'Policy n°1'
    ax1 = fig.add_subplot(232)
    ax1 = sns.heatmap(policies[plot1_name], ax=ax1)
    ax1.invert_yaxis()
    ax1.set_title(plot1_name)

    plot2_name = 'Policy n°2'
    ax2 = fig.add_subplot(233)
    ax2 = sns.heatmap(policies[plot2_name], ax=ax2)
    ax2.invert_yaxis()
    ax2.set_title(plot2_name)

    plot3_name = 'Policy n°3'
    ax3 = fig.add_subplot(234)
    ax3 = sns.heatmap(policies[plot3_name], ax=ax3)
    ax3.invert_yaxis()
    ax3.set_title(plot3_name)

    plot4_name = 'Policy n°4'
    ax4 = fig.add_subplot(235)
    ax4 = sns.heatmap(policies[plot4_name], ax=ax4)
    ax4.invert_yaxis()
    ax4.set_title(plot4_name)

    # 3D plot : V*(s)

    plot5_name = 'Optimal value per state'
    i = j = np.arange(0, NUM_MAX_CARS + 1, 1)
    jj, ii = np.meshgrid(i, j)

    ax5 = fig.add_subplot(236, projection='3d')
    ax5.plot_surface(ii, jj, values_opti, cmap=cm.coolwarm)
    ax5.set_xlabel('Num cars 1')
    ax5.set_ylabel('Num cars 2')
    ax5.set_zlabel('V*')
    ax5.set_title(plot5_name)

    plt.waitforbuttonpress()


def run_policy_iteration_jacks_car_rental_with_changes():
    """Exercice 4.7 p.82"""

    car_rental = CarRental()
    policy = Policy(env=car_rental, fix_returned=True)


    policies = {}
    policies['Init policy'] = policy.get_current_policy()

    policy_iteration = True
    policy_it = 0

    while (policy_iteration):

        policy_it += 1
        policy_evaluation = True
        eval_it = 0

        while (policy_evaluation):

            max_step = policy.evaluate()

            eval_it += 1
            print(f"eval_it= {eval_it} | max_step= {max_step}")

            if max_step < STEP_MAX_THRESHOLD:
                policy_evaluation = False

        is_policy_stable = policy.policy_improvement()
        policy_iteration = False if is_policy_stable else True

        if not is_policy_stable:
            policies[f'Policy n°{policy_it}'] = policy.get_current_policy()

        print(f'[policy n°{policy_it}] is_policy_stable = {is_policy_stable}')

    # Plot optimal policy
    values_opti = policy.get_current_values()

    draw_output_fig(policies, values_opti)


if __name__ == '__main__':

    run_policy_iteration_jacks_car_rental_with_changes()

