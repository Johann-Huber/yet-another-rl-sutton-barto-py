import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()

# ---------------

GRID_DIMS = (7, 10)

POS_START = (3, 0)
POS_GOAL = (3, 7)

ALL_4_ACTIONS = [(i,j) for i in range(-1,2) for j in range(-1,2) if abs(i) != abs(j)]
ALL_8_ACTIONS = [(i,j) for i in range(-1,2) for j in range(-1,2) if (i != 0) or (j != 0)]
ALL_9_ACTIONS = [(i,j) for i in range(-1,2) for j in range(-1,2)]

# TD step size
ALPHA = 0.5

# Discount factor
GAMMA = 1

# Exploration ratio
EPSILON = 0.1

FIGURE_SIZE = (12,8)

# ---------------



class Grid:
    def __init__(self, dims=GRID_DIMS, pos_start=POS_START, pos_goal=POS_GOAL,fig_size=FIGURE_SIZE):
        self._h, self._w = dims

        self._pos_start, self._pos_goal = pos_start, pos_goal

        self._fig_size = fig_size

        self._wind = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:1, 9:0}

    @property
    def height(self):
        return self._h

    @property
    def width(self):
        return self._w

    @property
    def wind(self):
        return self._wind

    @property
    def pos_start(self):
        return self._pos_start

    @property
    def pos_goal(self):
        return self._pos_goal

    def is_valid_state(self, state):
        i, j = state
        return True if (i >= 0) and (i <= self._h-1) and (j >= 0) and (j <= self._w-1) else False

    def cvt_ij2xy(self, pos_ij):
        return pos_ij[1], self._h - 1 - pos_ij[0]

    def draw(self, trajectory=None):
        fig, ax = plt.subplots(1, 1, figsize=self._fig_size)

        # cells
        for i in range(self._h):
            ax.plot([0, self._w], [i, i], color='k')
        for j in range(self._w):
            ax.plot([j, j], [0, self._h], color='k')

        # start & goal
        start_xy = self.cvt_ij2xy(self._pos_start)
        goal_xy = self.cvt_ij2xy(self._pos_goal)

        ax.text(start_xy[0] + 0.5, start_xy[1] + 0.5, 'S', fontsize=20, ha='center', va='center')
        ax.text(goal_xy[0] + 0.5, goal_xy[1] + 0.5, 'G', fontsize=20, ha='center', va='center')

        # wind arrows
        for i in range(self._h):
            for j in range(self._w):
                if self._wind[j] != 0:
                    x, y = self.cvt_ij2xy((i, j))
                    ax.text(x + 0.5, y, '↑', fontsize=20, ha='center', va='bottom', color='c',
                            alpha=self._wind[j]*0.4)


        # trajectory
        if trajectory is not None:

            n_step = len(trajectory)
            for i_step in range(n_step-1):
                from_pt = trajectory[i_step]
                to_pt = trajectory[i_step+1]

                from_x, from_y = self.cvt_ij2xy(from_pt)
                to_x, to_y = self.cvt_ij2xy(to_pt)

                ax.plot([from_x+0.5, to_x+0.5], [from_y+0.5, to_y+0.5], color='b')

                plt.title(f'Greedy policy trajectory after 170 episodes.', fontsize=18)


        ax.set_xlim(0, self._w)
        ax.set_ylim(0, self._h)

        plt.waitforbuttonpress()


class Environment:
    def __init__(self, grid, all_actions=ALL_8_ACTIONS, stochastic_wind_flg=False):
        self._grid = grid
        self._all_actions = all_actions
        self._stochastic_wind_flg = stochastic_wind_flg

    def step(self, state, action):

        assert self._grid.is_valid_state(state), 'Invalid state'
        assert action in self._all_actions, 'Invalid action'

        next_state = np.array(state) + np.array(action)

        if self._stochastic_wind_flg:
            next_state[0] += np.random.choice([-1, 0, 1])

        next_state[0] -= self._grid.wind[state[1]]


        next_state = (np.clip(next_state[0], a_min=0, a_max=self._grid.height-1),
                      np.clip(next_state[1], a_min=0, a_max=self._grid.width-1))

        if next_state == self._grid.pos_goal:
            return next_state, 0.

        return next_state, -1.


class Agent:
    def __init__(self, grid, all_actions=ALL_4_ACTIONS, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):

        self._Q = np.zeros((grid.height, grid.width, len(all_actions))) # states, actions

        self._all_actions = all_actions

        self._grid = grid

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon

        self._step_episode_list = []


    @property
    def step_episode_list(self):
        return self._step_episode_list

    @property
    def action_values(self):
        return self._Q

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
        for e in tqdm(range(n_episodes)):

            state = self.get_start_pos()
            action = self.policy(state)

            n_steps = 0

            running = True
            while (running):

                self._step_episode_list.append(e)
                n_steps += 1

                i, j = state
                ind_action = self.action2ind(action)

                next_state, reward = env.step(state, action)

                next_action = self.policy(next_state)

                i_next, j_next = next_state
                ind_next_action = self.action2ind(next_action)

                increment = self._α * reward + self._α * (self._γ * self._Q[i_next, j_next, ind_next_action] - self._Q[i, j, ind_action])


                self._Q[i, j, ind_action] += increment

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    state = next_state
                    action = next_action


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
            print('state = ', state, ' n_steps =', n_steps)

            next_state, reward = env.step(state, action)
            next_action = self.policy(next_state)

            trajectory.append(next_state)

            if self.is_terminal_state(next_state):
                running = False
            else:
                state = next_state
                action = next_action

        return trajectory


def get_episodes_cumulated_steps_plot(agent):
    """Reproduce first plot p130 : episodes by cumulated time steps."""
    curr_step_episode = agent.step_episode_list

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(x=range(len(curr_step_episode)), y=curr_step_episode, ax=ax)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Episodes')
    plt.waitforbuttonpress()


def get_greedy_trajectory_plot(grid, env, agent):
    """Reproduce second plot p130 : greedy trajectory."""
    greedy_trajectory = agent.get_greedy_trajectory(env)

    grid.draw(trajectory=greedy_trajectory)


def exercice_6_9():
    """Exercice 6.9 (p131) : Sarsa on windy gridworld. """
    all_actions = ALL_9_ACTIONS

    grid = Grid()
    env = Environment(grid, all_actions=all_actions)
    agent = Agent(grid, all_actions=all_actions)

    # Apply SARSA algorithm
    agent.run_sarsa(env, n_episodes=170)

    # Reproduces p130 first plot
    get_episodes_cumulated_steps_plot(agent)

    # Reproduces p130 second plot
    get_greedy_trajectory_plot(grid, env, agent)


def exercice_6_10():
    """Exercice 6.10 (p131) : Sarsa on stochastic windy gridworld """
    all_actions = ALL_8_ACTIONS

    grid = Grid()
    env = Environment(grid, all_actions=all_actions, stochastic_wind_flg=True)
    agent = Agent(grid, all_actions=all_actions)

    # Apply SARSA algorithm
    agent.run_sarsa(env, n_episodes=170)

    # Reproduces p130 first plot
    get_episodes_cumulated_steps_plot(agent)

    # Reproduces p130 second plot
    get_greedy_trajectory_plot(grid, env, agent)


if __name__ == '__main__':

    exercice_6_9()

    exercice_6_10()

