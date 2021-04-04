import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#---------------------------------------------------

K=10 # brandit problem size

FIGURE_SIZE = (16,12)
#---------------------------------------------------


class Bandit:
    def __init__(self, k=K, is_stationary=True):
        self._k = k
        self._q_k = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        self._optimal_action = np.argmax(self._q_k)

        self._is_stationary = is_stationary

    def reward(self, action):
        q_a = self._q_k[action]
        R = np.random.normal(loc=q_a, scale=1.0)

        return R

    def is_optimal_action(self, action):
        return action == self._optimal_action

    def update_target(self):
        if not self._is_stationary:
            # Add random noise
            self._q_k += np.random.normal(loc=0.0, scale=0.01, size=(self._k,))

class Agent:
    def __init__(self, epsilon, alpha=None, k=K):

        # Probability to explore
        self.epsilon = epsilon
        # How much time action A had been selected
        self._n_A = np.zeros(shape=(k,), dtype=np.int32)
        # Estimated value of expected reward by taking action A
        self._Q_k = np.zeros(shape=(k,), dtype=np.float32)
        # Number of possible actions
        self._k = k
        # Learning step rate
        self._alpha = alpha

    def choose_action(self):
        do_explore = np.random.random() < self.epsilon
        if do_explore:
            return np.random.randint(self._k)
        else:
            # Greedy action
            return np.argmax(self._Q_k)


    def update_values(self, action, reward):
        self._n_A[action] += 1

        if self._alpha is None :
            alpha = (1 / self._n_A[action])
        else :
            alpha = self._alpha

        self._Q_k[action] += alpha*(reward-self._Q_k[action])



def run_experiment_1(verbose=False):
    """Demonstrates the difficulties that sample average methods have for non-stationary problems. Compares
        performances with an action-value method that use a constant step-size parameter. (p.33)
        Giving more credit to later observations helps the agent to adapt."""

    print("=" * 50)
    print("Experiment 1")

    records_df = pd.DataFrame()

    eps = 0.1
    num_runs = 2000
    max_steps = 10000

    optimal_action_records = []
    rewards_records = []

    for alpha in [None, 0.1]:

        for i_run in tqdm(range(num_runs), desc=f"Runs : alpha = {alpha if alpha else 'unfixed'}"):

            bandit = Bandit(is_stationary=False)
            agent = Agent(epsilon=eps, alpha=alpha)

            optimal_action_records_i = []
            rewards_records_i = []

            for step in range(max_steps):

                bandit.update_target()

                action = agent.choose_action()
                reward = bandit.reward(action)
                agent.update_values(action, reward)

                if verbose:
                    print('A=', action, '| R=', reward)

                is_optimal_action = bandit.is_optimal_action(action)
                optimal_action_records_i.append(is_optimal_action)

                rewards_records_i.append(reward)

            optimal_action_records.append(optimal_action_records_i)
            rewards_records.append(rewards_records_i)

        # Save results for plotting
        avg_rewards_records = np.array(rewards_records).mean(axis=0)
        optimal_action_ratio_records = np.array(optimal_action_records).mean(axis=0)

        records_df = records_df.append(pd.DataFrame({'reward': avg_rewards_records,
                                                     'optimal_action_ratio': optimal_action_ratio_records,
                                                     'steps': range(max_steps),
                                                     'alpha': np.full((max_steps,),
                                                                      str(alpha) if alpha else 'unfixed')}),
                                       ignore_index=True)

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=records_df, x='steps', y='reward', hue='alpha', ax=ax[0])
    sns.lineplot(data=records_df, x='steps', y='optimal_action_ratio', hue='alpha', ax=ax[1])

    plt.waitforbuttonpress()
    plt.savefig('rwds_step_size_strategies_nonstationnary.png')


def run_experiment_2(verbose=False):
    """ Compare several epsilon-greedy policy on the bandit task. (p.44)
        Exploring helps to find better action-value estimations. A significantly large exploration rate leads
        to a better solution earlier, but quickly plateaus. """

    print("=" * 50)
    print("Experiment 2")

    num_runs = 2000
    max_steps = 10000
    records_df = pd.DataFrame()

    for task in ['stationary', 'nonstationary']:

        print(f'Task : {task}')
        is_stationary = task == 'stationary'

        for eps in [0., 0.1, 0.01]:

            optimal_action_records = []
            rewards_records = []

            for i_run in tqdm(range(num_runs), desc=f'Runs : eps = {eps}'):

                bandit = Bandit(is_stationary=is_stationary)
                agent = Agent(epsilon=eps)

                optimal_action_records_i = []
                rewards_records_i = []

                for step in range(max_steps):

                    bandit.update_target()

                    action = agent.choose_action()
                    reward = bandit.reward(action)
                    agent.update_values(action, reward)

                    if verbose:
                        print('A=', action, '| R=', reward)

                    is_optimal_action = bandit.is_optimal_action(action)
                    optimal_action_records_i.append(is_optimal_action)

                    rewards_records_i.append(reward)

                optimal_action_records.append(optimal_action_records_i)
                rewards_records.append(rewards_records_i)

            # Save results for plotting
            avg_rewards_records = np.array(rewards_records).mean(axis=0)
            optimal_action_ratio_records = np.array(optimal_action_records).mean(axis=0)

            records_df = records_df.append(pd.DataFrame({'reward': avg_rewards_records,
                                                         'optimal_action_ratio': optimal_action_ratio_records,
                                                         'steps': range(max_steps),
                                                         'eps': np.full((max_steps,), eps),
                                                         'task': np.full((max_steps,), task)}),
                                           ignore_index=True)

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=records_df, x='steps', y='reward', hue='eps', style='task', ax=ax[0], palette=sns.color_palette("mako_r", 3))
    sns.lineplot(data=records_df, x='steps', y='optimal_action_ratio', hue='eps', style='task', ax=ax[1], palette=sns.color_palette("mako_r", 3))

    plt.waitforbuttonpress()
    plt.savefig('rwds_epsilon_greedy_both_tasks.png')




if __name__ == '__main__':

    run_experiment_1()

    #run_experiment_2()





