import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#-------------------------------

WINNING_CAPITAL = 100

NUM_ACTIONS = 99
NUM_STATES = 1 + NUM_ACTIONS + 1 # dummy states : 0 & 100

PROB_HEAD = 0.4

DISCOUNT_FACTOR = 1 #0.9
STEP_MAX_THRESHOLD = 1e-24

FIGURE_SIZE = (8,16)

class Agent:
    def __init__(self):

        self.values = np.zeros((NUM_STATES,))

        self.values_dict = {}


    def get_values(self):
        return self.values[1:-1]


    def get_values_dict(self):
        return self.values_dict


    def compute_value_iteration(self):

        iterate = True
        num_sweep = 0


        while (iterate):

            max_step_value = self.compute_one_sweep()
            num_sweep += 1

            print(f'[sweep {num_sweep}] max_step_value = {max_step_value}')

            iterate = (max_step_value > STEP_MAX_THRESHOLD)

            if (num_sweep in [1,2,3,32]) or (not iterate):
                self.values_dict[num_sweep] = self.get_values().copy()



    def compute_one_sweep(self):

        max_step_value = 0
        all_states = np.arange(WINNING_CAPITAL+1)

        for state in all_states:

            value = self.values[state]
            self.values[state] = self.evaluate_state(state)

            max_step_value = max(max_step_value, abs(self.values[state] - value))

        return max_step_value


    def evaluate_state(self, state):
        """ Compute V(s) """

        all_actions = list(range(1,min(100-state,state)+1,1))

        max_action_value = 0

        for action in all_actions:

            next_states_win = min(state + action, WINNING_CAPITAL)
            next_states_lose = max(state - action, 0)

            rwd_win = 1 if (next_states_win == WINNING_CAPITAL) else 0
            rwd_lose = 0

            action_value = PROB_HEAD * (rwd_win + DISCOUNT_FACTOR*self.values[next_states_win]) + \
                           (1-PROB_HEAD) * (rwd_lose + DISCOUNT_FACTOR*self.values[next_states_lose])

            max_action_value = max(action_value, max_action_value)


        return max_action_value


    def get_deterministic_policy(self):

        policy = np.zeros((NUM_ACTIONS+1,)) # dummy state=0

        all_states = list(range(1, 100, 1))

        for state in all_states:

            all_actions = list(range(1,min(100-state,state)+1,1))

            action_values = []

            for action in all_actions: # discard dummy s=0

                next_states_win = min(state + action, WINNING_CAPITAL)
                next_states_lose = max(state - action, 0)

                rwd_win = 1 if (next_states_win == WINNING_CAPITAL) else 0
                rwd_lose = 0

                action_value = PROB_HEAD * (rwd_win + DISCOUNT_FACTOR*self.values[next_states_win]) + \
                               (1-PROB_HEAD) * (rwd_lose + DISCOUNT_FACTOR*self.values[next_states_lose])

                action_values.append(action_value)

            imax = np.argmax(np.round(action_values,5))

            policy[state] = all_actions[imax]

        return policy[1:] # discard dummy state=0


#-------------------------------

if __name__ == '__main__':

    agent = Agent()

    agent.compute_value_iteration()

    values_dict = agent.get_values_dict()

    final_policy = agent.get_deterministic_policy()


    # Output plots

    fig, ax = plt.subplots(2, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=pd.DataFrame(values_dict), ax=ax[0])
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Value estimates')

    sns.lineplot(x=range(1,len(final_policy)+1),y=final_policy, ax=ax[1])
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final policy (stake)')
    plt.waitforbuttonpress()

