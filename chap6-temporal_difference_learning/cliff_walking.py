import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_style("white")

# ---------------

GRID_DIMS = (4, 12)
POS_START = (3, 0)
POS_GOAL = (3, 11)
POS_CLIFF = [(3, j) for j in range (1,11)]

ALL_4_ACTIONS = [(i,j) for i in range(-1,2) for j in range(-1,2) if abs(i) != abs(j)] # (-1,0), (0,-1), (1,0), (0,1)

# TD step size
ALPHA = 0.5
# Discount factor
GAMMA = 1
# Exploration ratio
EPSILON = 0.1



FIGURE_SIZE = (10,4)


# ---------------


class Grid:
    def __init__(self, dims=GRID_DIMS, pos_start=POS_START, pos_goal=POS_GOAL, pos_cliff=POS_CLIFF,
                 fig_size=FIGURE_SIZE):
        self._h, self._w = dims

        self._pos_start, self._pos_goal = pos_start, pos_goal

        self._pos_cliff = pos_cliff

        self._fig_size = fig_size

    @property
    def height(self):
        return self._h

    @property
    def width(self):
        return self._w

    @property
    def pos_start(self):
        return self._pos_start

    @property
    def pos_goal(self):
        return self._pos_goal

    @property
    def pos_cliff(self):
        return self._pos_cliff

    def is_valid_state(self, state):
        i, j = state
        return True if (i >= 0) and (i <= self._h-1) and (j >= 0) and (j <= self._w-1) else False

    def cvt_ij2xy(self, pos_ij):
        return pos_ij[1], self._h - 1 - pos_ij[0]

    def draw(self, sarsa_trajectory=None, q_learn_trajectory=None):
        fig, ax = plt.subplots(1, 1, figsize=self._fig_size)

        # cells
        for i in range(self._h):
            ax.plot([0, self._w], [i, i], color='tab:gray')
        for j in range(self._w):
            ax.plot([j, j], [0, self._h], color='tab:gray')

        # start & goal
        start_xy = self.cvt_ij2xy(self._pos_start)
        goal_xy = self.cvt_ij2xy(self._pos_goal)

        ax.text(start_xy[0] + 0.5, start_xy[1] + 0.5, 'S', fontsize=20, ha='center', va='center')
        ax.text(goal_xy[0] + 0.5, goal_xy[1] + 0.5, 'G', fontsize=20, ha='center', va='center')



        # The cliff

        for id, cliff_ij in enumerate(self._pos_cliff):
            x, y = self.cvt_ij2xy((cliff_ij[0], cliff_ij[1]))
            cliff_rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor='tab:gray')
            ax.add_patch(cliff_rect)
            if id == int(len(self._pos_cliff)/2):
                ax.text(x, y + 0.5, 'The Cliff', fontsize=25, ha='center', va='center', color='w')


        # trajectory
        if sarsa_trajectory is not None:
            n_step = len(sarsa_trajectory)
            for i_step in range(n_step-1):
                from_pt = sarsa_trajectory[i_step]
                to_pt = sarsa_trajectory[i_step+1]

                from_x, from_y = self.cvt_ij2xy(from_pt)
                to_x, to_y = self.cvt_ij2xy(to_pt)

                ax.plot([from_x+0.5, to_x+0.5], [from_y+0.5, to_y+0.5], color='b', linewidth=3)

            legend_x, legend_y = self.cvt_ij2xy((self._h/6, self._w/3))
            ax.text(legend_x, legend_y, 'Sarsa', fontsize=15, ha='center', va='center', color='b')


        if q_learn_trajectory is not None:
            n_step = len(q_learn_trajectory)
            for i_step in range(n_step-1):
                from_pt = q_learn_trajectory[i_step]
                to_pt = q_learn_trajectory[i_step+1]

                from_x, from_y = self.cvt_ij2xy(from_pt)
                to_x, to_y = self.cvt_ij2xy(to_pt)

                ax.plot([from_x+0.5, to_x+0.5], [from_y+0.5, to_y+0.5], color='r', linewidth=3)

            legend_x, legend_y = self.cvt_ij2xy((self._h / 6, self._w * 2 / 3))
            ax.text(legend_x, legend_y, 'Q-learning', fontsize=15, ha='center', va='center', color='r')


        ax.set_xlim(0, self._w)
        ax.set_ylim(0, self._h)

        plt.waitforbuttonpress()


class Environment:
    def __init__(self, grid, all_actions=ALL_4_ACTIONS):
        self._grid = grid
        self._all_actions = all_actions

    def step(self, state, action):

        assert self._grid.is_valid_state(state), 'Invalid state'
        assert action in self._all_actions, 'Invalid action'

        next_state = np.array(state) + np.array(action)

        next_state = (np.clip(next_state[0], a_min=0, a_max=self._grid.height-1),
                      np.clip(next_state[1], a_min=0, a_max=self._grid.width-1))

        if next_state == self._grid.pos_goal:
            return next_state, 0.

        if next_state in self._grid.pos_cliff:
            next_state = self._grid.pos_start
            return next_state, -100.

        return next_state, -1.




class Agent:
    def __init__(self, grid, all_actions=ALL_4_ACTIONS, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):

        self._Q = np.zeros((grid.height, grid.width, len(all_actions))) # states, actions

        self._all_actions = all_actions

        self._grid = grid

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon

        self._reward_hist = []


    @property
    def action_values(self):
        return self._Q

    @property
    def reward_hist(self):
        return self._reward_hist


    def policy(self, state, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        assert self._grid.is_valid_state(state), 'Invalid state'

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = self._all_actions[np.random.choice(range(len(self._all_actions)))]
            return action

        i, j = state

        greedy_action_inds = np.where(self._Q[i,j] == self._Q[i,j].max())[0]
        ind_action = np.random.choice(greedy_action_inds)

        action = self._all_actions[ind_action]
        return action

    def get_start_pos(self):
        return self._grid.pos_start

    def is_terminal_state(self, state):
        return True if state == self._grid.pos_goal else False

    def action2ind(self, action):
        assert action in self._all_actions, 'Invalid action.'
        for ind, a in enumerate(self._all_actions):
            if action == a:
                return ind
        return -1

    def ind2action(self, ind):
        assert (ind >= 0) and ind <= len(self._all_actions)-1
        return self._all_actions[ind]

    def run_sarsa(self, env, n_episodes):
        """Apply SARSA (on-policy TD control) for estimating Q."""
        for e in range(n_episodes):

            state = self.get_start_pos()
            action = self.policy(state)

            n_steps = 0

            sum_rewards = 0

            running = True
            while (running):

                n_steps += 1

                i, j = state
                ind_action = self.action2ind(action)

                next_state, reward = env.step(state, action)

                next_action = self.policy(next_state)

                i_next, j_next = next_state
                ind_next_action = self.action2ind(next_action)

                increment = self._α * reward + self._α * (self._γ * self._Q[i_next, j_next, ind_next_action] - self._Q[i, j, ind_action])


                self._Q[i, j, ind_action] += increment

                sum_rewards += reward

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    state = next_state
                    action = next_action

            self._reward_hist.append(sum_rewards)

    def run_q_learning(self, env, n_episodes):
        """Apply Q-learning (on-policy TD control) for estimating Q."""
        for e in range(n_episodes):

            state = self.get_start_pos()
            action = self.policy(state)

            n_steps = 0

            sum_rewards = 0

            running = True
            while (running):

                n_steps += 1

                i, j = state
                ind_action = self.action2ind(action)

                next_state, reward = env.step(state, action)


                next_action = self.policy(next_state, explore_flg=False) # Always greedy

                i_next, j_next = next_state
                ind_next_action = self.action2ind(next_action)

                increment = self._α * reward + self._α * (self._γ * self._Q[i_next, j_next, ind_next_action] - self._Q[i, j, ind_action])

                self._Q[i, j, ind_action] += increment

                sum_rewards += reward

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    state = next_state
                    action = self.policy(state) # Q-derived policy (= with exploration)

            self._reward_hist.append(sum_rewards)



    def get_greedy_trajectory(self, env):
        """Get the greedy trajectory by following greedy policy (without exploring) until the terminal state
        is reached."""
        state = self.get_start_pos()
        action = self.policy(state)

        trajectory = [state]
        n_steps = 0

        running = True
        while (running):
            n_steps += 1

            next_state, reward = env.step(state, action)
            next_action = self.policy(next_state)

            trajectory.append(next_state)

            if self.is_terminal_state(next_state):
                running = False
            else:
                state = next_state
                action = next_action

        return trajectory



def sarsa_q_learn_reward_plots():
    grid = Grid()
    env = Environment(grid)

    n_runs = 300

    all_sarsa_rwd_hist = []
    all_q_learn_rwd_hist = []

    for _ in tqdm(range(n_runs)):
        # Sarsa
        agent = Agent(grid)
        agent.run_sarsa(env, n_episodes=500)

        all_sarsa_rwd_hist.append(agent.reward_hist)

        # Q-learning
        agent = Agent(grid)
        agent.run_q_learning(env, n_episodes=500)

        all_q_learn_rwd_hist.append(agent.reward_hist)

    all_sarsa_rwd_hist = np.array(all_sarsa_rwd_hist).mean(0)
    all_q_learn_rwd_hist = np.array(all_q_learn_rwd_hist).mean(0)

    # Plot rewards
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=range(len(all_sarsa_rwd_hist)), y=all_sarsa_rwd_hist, ax=ax, label='Sarsa')
    sns.lineplot(x=range(len(all_q_learn_rwd_hist)), y=all_q_learn_rwd_hist, ax=ax, label='Q-learning')
    ax.set_ylim(-100, 0)
    plt.legend()
    plt.waitforbuttonpress()


def sarsa_q_learn_trajectory_plots():
    # Todo : Average over several runs to get the optimal trajectory without decreasing exploration rate.
    #  during training.

    grid = Grid()
    env = Environment(grid)

    # Sarsa
    agent = Agent(grid)
    agent.run_sarsa(env, n_episodes=2000)
    sarsa_trajectory = agent.get_greedy_trajectory(env)

    # Q-learning
    agent = Agent(grid)
    agent.run_q_learning(env, n_episodes=2000)
    q_learn_trajectory = agent.get_greedy_trajectory(env)

    # Plot trajectory
    grid.draw(sarsa_trajectory=sarsa_trajectory,
              q_learn_trajectory=q_learn_trajectory)



if __name__ == '__main__':

    sarsa_q_learn_reward_plots()

    #sarsa_q_learn_trajectory_plots()

