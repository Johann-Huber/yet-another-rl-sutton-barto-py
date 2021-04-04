import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()


N_SERVERS = 10
N_PRIORS = 4
PRIOR_REWARDS = {k: 2 ** k for k in range(4)}
ALL_ACTIONS = {0: 'reject', 1: 'accept'}

ALPHA = 0.01
BETA = 0.01
EPSILON = 0.1

FIGURE_SIZE = (16,12)


# ------------



class Environment:

    def __init__(self, all_actions=ALL_ACTIONS, n_priors=N_PRIORS, prior_rwd=PRIOR_REWARDS,
                 n_server=N_SERVERS):

        self._all_actions = all_actions

        self._n_priors = n_priors
        self._prior_rwd = prior_rwd

        self._n_server_max = n_server

    def get_action_str(self, action):
        return self._all_actions[action]

    def get_random_next_customer(self):
        """Randomly pick a key corresponding to the next customer coming at the head of the queue."""
        return np.random.choice(list(self._prior_rwd.keys()))

    def step(self, state, action):

        n_free_serv, customer_prior = state

        assert (n_free_serv >= 0) and (n_free_serv <= self._n_server_max), "Invalid server number."
        assert (customer_prior >= 0) and (customer_prior < self._n_priors), "Invalid custom prior key."
        assert action in [0, 1], "Invalid action."

        # Apply action
        if (self.get_action_str(action) == 'accept') and (n_free_serv > 0):
            # Accept
            n_free_serv_next = n_free_serv - 1
            reward = self._prior_rwd[customer_prior]
        else :
	    # Refuse
            n_free_serv_next = n_free_serv
            reward = 0

        # New free server
        n_new_available_serv = (np.random.random(self._n_server_max - n_free_serv) < 0.06).astype(int).sum()
        n_free_serv_next += n_new_available_serv

        # Next customer
        customer_prior_next = self.get_random_next_customer()

        next_state = (n_free_serv_next, customer_prior_next)

        return next_state, reward


class Agent:

    def __init__(self, all_actions=ALL_ACTIONS, n_priors=N_PRIORS, prior_rwd=PRIOR_REWARDS,
                 n_server=N_SERVERS, alpha=ALPHA, beta=BETA, epsilon=EPSILON):

        self._q_hat = np.zeros((n_server + 1, n_priors, len(all_actions)))

        self._all_actions = all_actions

        self._n_priors = n_priors
        self._prior_rwd = prior_rwd

        self._n_server_max = n_server

        self._α = alpha
        self._β = beta
        self._ε = epsilon

        self._avg_reward = 0


    @property
    def q_hat(self):
        return self._q_hat

    def get_greedy_policy(self):

        policy = np.zeros((self._n_server_max + 1, self._n_priors))


        for n_free_serv in range(self._n_server_max + 1):
            for n_prior in range(self._n_priors):

                ids_max = np.where(self._q_hat[n_free_serv, n_prior, :] == \
                                   self._q_hat[n_free_serv, n_prior, :].max())[0]

                if len(ids_max) > 1 :
                    print(f'Warning : ({n_free_serv},{n_prior}) did not converged.')

                policy[n_free_serv, n_prior] = self._q_hat[n_free_serv, n_prior, :].argmax()

        return policy


    def get_greedy_policy_values(self):

        q_max = np.zeros((self._n_server_max + 1, self._n_priors))

        for n_free_serv in range(self._n_server_max + 1):
            for n_prior in range(self._n_priors):
                q_max[n_free_serv, n_prior] = self._q_hat[n_free_serv, n_prior, :].max()

        return q_max


    def get_init_state(self):
        n_free_serv = 10
        customer_prior = np.random.choice(list(self._prior_rwd.keys()))
        return n_free_serv, customer_prior


    def policy(self, state, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        n_free_serv, customer_prior = state

        assert (n_free_serv >= 0) and (n_free_serv <= self._n_server_max), "Invalid server number."
        assert (customer_prior >= 0) and (customer_prior < self._n_priors), "Invalid custom prior key."

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = np.random.choice([0, 1])
            return action

        greedy_action_inds = np.where(self._q_hat[n_free_serv, customer_prior, :] == \
                                      self._q_hat[n_free_serv, customer_prior, :].max())[0]
        action = np.random.choice(greedy_action_inds)

        return action


    def run_differential_semi_grad_sarsa(self, env, n_step_max):

        state = self.get_init_state()
        action = self.policy(state)

        for _ in tqdm(range(n_step_max)):

            next_state, reward = env.step(state, action)

            next_action = self.policy(next_state)

            SA = state + (action,)
            SA_next = next_state + (next_action,)

            δ = reward - self._avg_reward + self._q_hat[SA_next] - self._q_hat[SA]

            self._avg_reward += self._β * δ

            self._q_hat[SA] += self._α * δ

            state = next_state
            action = next_action




def get_policy_q_value_plots(n_step_training):
    """Reproduces figure p.252"""

    env = Environment()

    agent = Agent()
    agent.run_differential_semi_grad_sarsa(env, n_step_max=n_step_training)

    policy = agent.get_greedy_policy()
    q_values_max = agent.get_greedy_policy_values()

    # Plot output
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0] = sns.heatmap(policy[1:, :].transpose(), ax=ax[0])
    ax[0].set_xlabel('Number of free servers')
    ax[0].set_ylabel('Priority')
    ax[0].set_xticklabels(range(1, 11))
    ax[0].set_yticklabels([1, 2, 4, 8])
    ax[0].set_title('POLICY | 0 => Reject | 1 => Accept', fontsize=18)

    q_hat = agent.q_hat
    df_vis = pd.DataFrame({f'priority {PRIOR_REWARDS[i]}': q_values_max[:, i] for i in range(N_PRIORS)})
    sns.lineplot(data=df_vis, ax=ax[1])
    ax[1].set_xlabel('Number of free server')
    ax[1].set_ylabel('Differential value of best action')
    ax[1].set_title(f"VALUE FUNCTION", fontsize=18)
    ax[1].axhline(0, color='k', ls=':', lw=0.7)
    plt.savefig('access_control_queuing_task_policy_q_value')

    plt.waitforbuttonpress()


if __name__ == '__main__':

    get_policy_q_value_plots(n_step_training = 30000000)



