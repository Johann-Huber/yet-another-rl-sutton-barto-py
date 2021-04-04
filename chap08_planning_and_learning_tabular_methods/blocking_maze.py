import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set_theme()
import matplotlib.gridspec as gridspec

# ---------------

GRID_DIMS = (6, 9)

POS_START = (5, 3)
POS_GOAL = (0, 8)

POS_OBSTACLES = [(3, j) for j in range(8)]

ALL_4_ACTIONS = [(i,j) for i in range(-1,2) for j in range(-1,2) if abs(i) != abs(j)] # up, left, right, down

# TD step size
ALPHA = .1 #0.1

# Discount factor
GAMMA = 0.95

# Exploration ratio
EPSILON = 0.1

# Temporal reward coefficient
TEMPORAL_RWD_COEF = 1e-5

# Number of step from which the map is modified
N_STEP_CHANGE_MAP = 1000

FIGURE_SIZE = (16,12)

RANDOM_SEED = 2

# ---------------

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

class Grid:
    def __init__(self, dims=GRID_DIMS, pos_start=POS_START, pos_goal=POS_GOAL,
                 pos_obstacles=POS_OBSTACLES, fig_size=FIGURE_SIZE, vis_mode='single',
                 modifying_mode='blocking'):

        self._h, self._w = dims
        self._pos_start, self._pos_goal = pos_start, pos_goal
        self._pos_obstacles = pos_obstacles

        assert modifying_mode in ['blocking', 'shortcut']
        self._modifying_mode = modifying_mode
        self._is_map_modified = False

        if self._modifying_mode == 'shortcut':
            # Init : obstacles are shifted to the left to let the right side open
            new_pos_arr = np.array(self._pos_obstacles) + np.array([0, 1])
            self._pos_obstacles = [(i, j) for i, j in new_pos_arr]


        assert vis_mode in ['single', 'parallel']

        if vis_mode == 'single':
            self._fig_size = fig_size
            self.fig, self.ax = plt.subplots(1, 1, figsize=self._fig_size)
        elif vis_mode == 'parallel':
            self.fig = plt.figure(figsize=(22, 16), constrained_layout=False)
            self.gs = self.fig.add_gridspec(2, 2)
            self.ax1 = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[0, 1])
            self.ax3 = self.fig.add_subplot(self.gs[1, :])

        self.n_step_change_map = 1000 if (self._modifying_mode == 'blocking') else 3000


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

    def modify_map(self):
        assert self._modifying_mode in ['blocking', 'shortcut']

        if self._modifying_mode == 'blocking':
            # Shift obstacles to the left
            new_pos_arr = np.array(self._pos_obstacles) + np.array([0, 1])
            self._pos_obstacles = [(i, j) for i, j in new_pos_arr]
        else:
            # Open a shortcut to the right
            self._pos_obstacles = self._pos_obstacles[:-1]

        self._is_map_modified = True

    def is_valid_state(self, state):
        i, j = state
        return True if (i >= 0) and (i <= self._h-1) and (j >= 0) and (j <= self._w-1) else False

    def cvt_ij2xy(self, pos_ij):
        return pos_ij[1], self._h - 1 - pos_ij[0]

    def get_action_char(self, action):
        assert action in ALL_4_ACTIONS

        if action == (-1, 0):
            return '↑'
        if action == (0, -1):
            return '←'
        if action == (1, 0):
            return '↓'
        if action == (0, 1):
            return '→'

        return -1

    def draw(self, agent=None):
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

        # obstacles
        for id, obs_ij in enumerate(self._pos_obstacles):
            x, y = self.cvt_ij2xy((obs_ij[0], obs_ij[1]))
            obs_rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor='tab:gray')
            ax.add_patch(obs_rect)

        # greedy actions
        if agent is not None:
            Q_indices = agent.get_greedy_policy_found()

            for i in range(self._h):
                for j in range(self._w):
                    if Q_indices[i,j] != -1:
                        action = agent.ind2action(Q_indices[i,j])
                        x, y = self.cvt_ij2xy((i, j))
                        arrow_char = self.get_action_char(action)
                        ax.text(x + 0.5, y + 0.5, arrow_char, fontsize=20, ha='center', va='center', color='b')

        ax.set_xlim(0, self._w)
        ax.set_ylim(0, self._h)

        plt.waitforbuttonpress()


    def get_triangles_from_tr_xy(self, x, y):
        """Get the four triangles that compose a grid cell, sharing the center of the square, each triangle having
        a unique side of the square as base.
        Meant to be used to draw q values for each grid cell, w.r.t. action (up, left, down, right)."""

        low_triangle = np.array([[x, y], [x + 0.5, y + 0.5], [x + 1, y]])
        left_triangle = np.array([[x, y], [x, y + 1], [x + 0.5, y + 0.5]])
        right_triangle = np.array([[x + 1, y], [x + 1, y + 1], [x + 0.5, y + 0.5]])
        top_triangle = np.array([[x, y + 1], [x + 1, y + 1], [x + 0.5, y + 0.5]])

        return low_triangle, left_triangle, right_triangle, top_triangle

    def get_q_color(self, val):
        return 'tab:red' if (val < 0) else 'tab:green'

    def draw_q_values(self, agent, plot_time=0.1):

        q_values = agent.action_values
        n_step = agent.n_step
        n_episodes = agent.n_episodes
        curr_state = agent.curr_state
        model_name = agent.model_name

        # cells
        for i in range(self._h):
            self.ax.plot([0, self._w], [i, i], color='k')
        for j in range(self._w):
            self.ax.plot([j, j], [0, self._h], color='k')

        # start & goal
        start_xy = self.cvt_ij2xy(self._pos_start)
        goal_xy = self.cvt_ij2xy(self._pos_goal)

        self.ax.text(start_xy[0] + 0.5, start_xy[1] + 0.5, 'S', fontsize=20, ha='center', va='center')
        self.ax.text(goal_xy[0] + 0.5, goal_xy[1] + 0.5, 'G', fontsize=20, ha='center', va='center')

        # obstacles
        for id, obs_ij in enumerate(self._pos_obstacles):
            x, y = self.cvt_ij2xy((obs_ij[0], obs_ij[1]))
            obs_rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor='tab:gray')
            self.ax.add_patch(obs_rect)


        # Q-values
        for i in range(self._h):
            for j in range(self._w):

                if ((i, j) in self._pos_obstacles) or ((i, j) == self._pos_goal):
                    continue

                # q_values[i,j] <- [q_up, q_left, q_right, q_down]
                q_val = q_values[i, j]

                x, y = self.cvt_ij2xy((i, j))
                # draw relative q values
                low_tri, left_tri, right_tri, top_tri = self.get_triangles_from_tr_xy(x, y)

                alphas = (q_val-q_val.min())/(q_val.max()-q_val.min()+1e-7)
                a_up, a_left, a_right, a_down = alphas

                q_low_tri = plt.Polygon(low_tri, color=self.get_q_color(a_down), alpha=abs(a_down))
                q_left_tri = plt.Polygon(left_tri, color=self.get_q_color(a_left), alpha=abs(a_left))
                q_right_tri = plt.Polygon(right_tri, color=self.get_q_color(a_right), alpha=abs(a_right))
                q_top_tri = plt.Polygon(top_tri, color=self.get_q_color(a_up),  alpha=abs(a_up))

                self.ax.add_patch(q_low_tri)
                self.ax.add_patch(q_left_tri)
                self.ax.add_patch(q_right_tri)
                self.ax.add_patch(q_top_tri)

                # write absolute q values
                q_up, q_left, q_right, q_down = q_val
                self.ax.text(x + 0.5, y + 0.75, f'{q_up:.3f}' , fontsize=13, ha='center', va='center', color='k')
                self.ax.text(x + 0.25, y + 0.5, f'{q_left:.3f}', fontsize=13, ha='center', va='center', color='k')
                self.ax.text(x + 0.75, y + 0.5, f'{q_right:.3f}', fontsize=13, ha='center', va='center', color='k')
                self.ax.text(x + 0.5, y + 0.25, f'{q_down:.3f}', fontsize=13, ha='center', va='center', color='k')


        # Draw agent pos
        x, y = self.cvt_ij2xy(curr_state)
        agent_pos_rect = patches.Circle((x + 0.5, y + 0.5), 0.1, color='tab:orange')
        self.ax.add_patch(agent_pos_rect)



        self.ax.set_xlim(0, self._w)
        self.ax.set_ylim(0, self._h)

        plt.suptitle(f"{model_name} | Step = {n_step} | Episode = {n_episodes} | Modified map = {self._is_map_modified}", fontsize=18)

        plt.pause(plot_time)
        self.ax.clear()


    def draw_q_values_parallel(self, agent, agent_temporal_reward, plot_time=0.1):

        all_cumu_rewards_lists = {}

        for agent2render, ax in zip([agent, agent_temporal_reward], [self.ax1, self.ax2]):
            q_values = agent2render.action_values
            n_step = agent2render.n_step
            n_episodes = agent2render.n_episodes
            curr_state = agent2render.curr_state
            model_name = agent2render.model_name

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

            # obstacles
            for id, obs_ij in enumerate(self._pos_obstacles):
                x, y = self.cvt_ij2xy((obs_ij[0], obs_ij[1]))
                obs_rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor='tab:gray')
                ax.add_patch(obs_rect)


            # Q-values
            for i in range(self._h):
                for j in range(self._w):

                    if ((i, j) in self._pos_obstacles) or ((i, j) == self._pos_goal):
                        continue

                    # q_values[i,j] <- [q_up, q_left, q_right, q_down]

                    q_val = q_values[i, j]

                    x, y = self.cvt_ij2xy((i, j))
                    # draw relative q values
                    low_tri, left_tri, right_tri, top_tri = self.get_triangles_from_tr_xy(x, y)

                    alphas = (q_val-q_val.min())/(q_val.max()-q_val.min()+1e-7)
                    a_up, a_left, a_right, a_down = alphas

                    q_low_tri = plt.Polygon(low_tri, color=self.get_q_color(a_down), alpha=abs(a_down))
                    q_left_tri = plt.Polygon(left_tri, color=self.get_q_color(a_left), alpha=abs(a_left))
                    q_right_tri = plt.Polygon(right_tri, color=self.get_q_color(a_right), alpha=abs(a_right))
                    q_top_tri = plt.Polygon(top_tri, color=self.get_q_color(a_up),  alpha=abs(a_up))

                    ax.add_patch(q_low_tri)
                    ax.add_patch(q_left_tri)
                    ax.add_patch(q_right_tri)
                    ax.add_patch(q_top_tri)

                    # write absolute q values
                    q_up, q_left, q_right, q_down = q_val
                    ax.text(x + 0.5, y + 0.75, f'{q_up:.1f}' , fontsize=8, ha='center', va='center', color='k')
                    ax.text(x + 0.25, y + 0.5, f'{q_left:.1f}', fontsize=8, ha='center', va='center', color='k')
                    ax.text(x + 0.75, y + 0.5, f'{q_right:.1f}', fontsize=8, ha='center', va='center', color='k')
                    ax.text(x + 0.5, y + 0.25, f'{q_down:.1f}', fontsize=8, ha='center', va='center', color='k')


            # Draw agent pos
            x, y = self.cvt_ij2xy(curr_state)
            agent_pos_rect = patches.Circle((x + 0.5, y + 0.5), 0.1, color='tab:orange')
            ax.add_patch(agent_pos_rect)


            ax.set_xlim(0, self._w)
            ax.set_ylim(0, self._h)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax.set_title(f"{model_name} | Episode = {n_episodes}", fontsize=13)

            # get cumulative reward
            model_name = 'Dyna-Q' if (agent2render == agent) else 'Dyna-Q+'
            all_cumu_rewards_lists[f'{model_name}'] = agent2render.cumu_reward_list


        # Plot cumulative reward curves
        df_vis = pd.DataFrame(all_cumu_rewards_lists)
        sns.lineplot(data=df_vis, ax=self.ax3)
        if n_step >= self.n_step_change_map:
            self.ax3.plot([self.n_step_change_map, self.n_step_change_map], [0, df_vis.max().max()], 'k-')
        self.ax3.set_xlabel('Time steps')
        self.ax3.set_ylabel('Cumulative reward')


        plt.suptitle(f"Step = {n_step} | Modified map = {self._is_map_modified}", fontsize=18)

        plt.pause(plot_time)
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()



class Environment:
    def __init__(self, grid, all_actions=ALL_4_ACTIONS):
        self._grid = grid
        self._all_actions = all_actions

    def shift_grid_obstacles(self):
        self._grid.modify_map()

    def step(self, state, action):

        assert self._grid.is_valid_state(state), 'Invalid state'
        assert action in self._all_actions, 'Invalid action'

        next_state = np.array(state) + np.array(action)

        next_state = (np.clip(next_state[0], a_min=0, a_max=self._grid.height-1),
                      np.clip(next_state[1], a_min=0, a_max=self._grid.width-1))

        if next_state in self._grid._pos_obstacles:
            next_state = state

        if next_state == self._grid.pos_goal:
            # goal found
            return next_state, 1.

        return next_state, 0


class Agent:
    def __init__(self, grid, n_planning_steps, all_actions=ALL_4_ACTIONS,
                 alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, seed=RANDOM_SEED,
                 temporal_reward_mode=False, temporal_reward_coef=TEMPORAL_RWD_COEF):

        self._Q = np.zeros((grid.height, grid.width, len(all_actions))) # states, actions

        self.model = {}

        self._all_actions = all_actions

        self._grid = grid

        self._α = alpha
        self._γ = gamma
        self._ε = epsilon

        self._n_planning_steps = n_planning_steps

        self._step_episode_list = []
        self._cumu_reward_list = []

        rand_seed = np.random.randint(100000)
        self._rng = np.random.RandomState(rand_seed)

        self._cumu_reward = 0

        self._n_steps = 0
        self._n_episodes = 0
        self._episode_is_running = False
        self._curr_state = self.get_start_pos()

        # temporal reward
        self._temporal_reward_mode = temporal_reward_mode
        self._last_time_step = np.zeros((grid.height, grid.width, len(all_actions)))
        self._k = temporal_reward_coef


    @property
    def is_episode_running(self):
        return self._episode_is_running

    @property
    def step_episode_list(self):
        return self._step_episode_list

    @property
    def action_values(self):
        return self._Q

    @property
    def n_step(self):
        return self._n_steps

    @property
    def n_episodes(self):
        return self._n_episodes

    @property
    def curr_state(self):
        return self._curr_state

    @property
    def cumu_reward_list(self):
        return self._cumu_reward_list

    @property
    def model_name(self):
        return 'Dyna-Q' if not self._temporal_reward_mode else 'Dyna-Q+'


    def get_greedy_policy_found(self):
        """Returns a (grid.height, grid.width) matrix containing index of the greedy action following the
        found policy. Index is -1 when there is no maximum action value for the state."""

        Q_indices = np.zeros((self._grid.height, self._grid.width), dtype=np.int32) # states, actions

        for i in range(grid.height):
            for j in range(grid.width):
                greedy_action_inds = np.where(self._Q[i, j] == self._Q[i, j].max())[0]

                if len(greedy_action_inds) > 1:
                    Q_indices[i,j] = -1
                else:
                    Q_indices[i, j] = greedy_action_inds[0]

        return Q_indices


    def policy(self, state, explore_flg=True):
        """Apply a ε-greedy policy to choose an action from state."""

        assert self._grid.is_valid_state(state), 'Invalid state'

        if (np.random.random_sample() < self._ε) and explore_flg:
            action = self._all_actions[self._rng.choice(range(len(self._all_actions)))]
            return action

        i, j = state

        greedy_action_inds = np.where(self._Q[i,j] == self._Q[i,j].max())[0]
        ind_action = self._rng.choice(greedy_action_inds)

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


    def q_learning_update(self, env, state, action, planning=False):
        """Apply a q_learning update."""

        if planning:
            next_state, reward = self.model[(state, action)]
        else:
            next_state, reward = env.step(state, action)

        next_action = self.policy(next_state, explore_flg=False)  # pure greedy

        SA = state + (self.action2ind(action),)
        SA_next = next_state + (self.action2ind(next_action),)

        if self._temporal_reward_mode and (planning == True):
            # simulation : bonus reward to encourage exploration in real env
            reward += self._k * (self._n_steps - self._last_time_step[SA])

        self._Q[SA] += self._α * reward + self._α * (self._γ * self._Q[SA_next] - self._Q[SA])

        if self._temporal_reward_mode and (planning == False) :
            # real interaction
            self._last_time_step[SA] = self._n_steps

        return next_state, reward


    def update_model(self, state, action, next_state, reward):
        """Update the model with (S,A) -> (S',R), assuming the environment is deterministic."""

        # update model
        state_cp = copy.deepcopy(state)
        action_cp = copy.deepcopy(action)
        next_state_cp = copy.deepcopy(next_state)
        reward_cp = copy.deepcopy(reward)

        self.model[(state_cp, action_cp)] = (next_state_cp, reward_cp)

        # initialize model such that actions that have not been already taken in the real
        # environment can be picked during planning phase.
        if self._temporal_reward_mode:
            for action2init in self._all_actions:
                if (state_cp, action2init) not in self.model.keys():
                    # Back to the same state, reward of 0
                    action_cp2 = copy.deepcopy(action2init)
                    state_cp2 = copy.deepcopy(state)
                    self.model[(state_cp2, action_cp2)] = (state_cp2, 0.)




    def tabular_dyna_q_iteration(self, env):

        if not self._episode_is_running:
            # Starting new episode
            self._curr_state = self.get_start_pos()
            self._episode_is_running = True

        # Experience
        state = self._curr_state
        action = self.policy(state)

        next_state, reward = self.q_learning_update(env, state, action)
        self._cumu_reward += reward

        self.update_model(state, action, next_state, reward)

        if self.is_terminal_state(next_state):
            self._episode_is_running = False
            self._n_episodes += 1

        self._curr_state = next_state

        # Planning
        for _ in range(self._n_planning_steps):
            state, action = random.choice(list(self.model.keys()))

            self.q_learning_update(env, state, action, planning=True)

        self._n_steps += 1

        self._cumu_reward_list.append(self._cumu_reward)


    def tabular_dyna_q_episode(self, env):

        self._curr_state = self.get_start_pos()
        self._episode_is_running = True

        while (self._episode_is_running):

            # Experience
            state = self._curr_state
            action = self.policy(state)

            next_state, reward = self.q_learning_update(env, state, action)
            self._cumu_reward += reward

            self.model[(state, action)] = (next_state, reward)

            if self.is_terminal_state(next_state):
                self._episode_is_running = False
                self._n_episodes += 1

            self._curr_state = next_state

            # Planning
            for _ in range(self._n_planning_steps):
                state, action = random.choice(list(self.model.keys()))
                self.q_learning_update(env, state, action, planning=True)

            self._n_steps += 1

            self._cumu_reward_list.append(self._cumu_reward)

        self._step_episode_list.append(self._n_steps)


    def run_tabular_dyna_q(self, env, n_episodes):


            curr_state = self.get_start_pos()

            running = True
            while (running):

                # Experience
                state = curr_state
                action = self.policy(state)

                next_state, reward = self.q_learning_update(env, state, action)
                self._cumu_reward += reward

                self.model[(state, action)] = (next_state, reward)

                if self.is_terminal_state(next_state):
                    running = False
                else:
                    curr_state = next_state

                # Planning
                for _ in range(self._n_planning_steps):
                    state, action = random.choice(list(self.model.keys()))
                    self.q_learning_update(env, state, action, planning=True)

                self._n_steps += 1

                self._cumu_reward_list.append(self._cumu_reward)

            self._step_episode_list.append(self._n_steps)


def get_cumu_reward_curves_comparison(modifying_mode):

    assert modifying_mode in ['blocking', 'shortcut']

    n_experiments = 30
    α = 1
    n_planning_steps = 10
    n_steps = 3000 if (modifying_mode == 'blocking') else 6000
    n_step_change_map = 1000 if (modifying_mode == 'blocking') else 3000

    plot_time = 0.5
    draw_update = False # depreciated for this func

    all_cumu_rewards_lists = {}

    for trm in [False, True]:
        cumu_rewards_lists = []

        for _ in tqdm(range(n_experiments)):
            grid = Grid(modifying_mode=modifying_mode)
            env = Environment(grid)
            agent = Agent(grid, alpha=α, n_planning_steps=n_planning_steps, temporal_reward_mode=trm)

            if draw_update:
                grid.draw_q_values(agent, plot_time=plot_time)

            for curr_step in range(n_steps):
                agent.tabular_dyna_q_iteration(env)

                if curr_step == n_step_change_map:
                    grid.modify_map()

                if draw_update and not agent.is_episode_running:
                    grid.draw_q_values(agent, plot_time=plot_time)

            cumu_rewards_lists.append(agent.cumu_reward_list)

        cumu_rewards_lists = np.array(cumu_rewards_lists).mean(0)

        model_name = 'Dyna-Q+' if trm else 'Dyna-Q'
        all_cumu_rewards_lists[f'{model_name}_nps{n_planning_steps}_alpha{α}'] = cumu_rewards_lists

    # Plot output
    df_vis = pd.DataFrame(all_cumu_rewards_lists)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(data=df_vis, ax=ax)
    ax.plot([n_step_change_map, n_step_change_map], [0, df_vis.max().max()], 'k-')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Cumulative reward')
    title_desc = "shortcut opening" if (modifying_mode == 'shortcut') else "map modification"
    plt.suptitle(f"Dyna-Q and Dyna-Q+ adaptation to {title_desc}", fontsize=18)
    plt.waitforbuttonpress()



def run_real_time_dislay_modified_maze(modifying_mode):

    assert modifying_mode in ['blocking', 'shortcut']

    n_steps = 3000 if (modifying_mode == 'blocking') else 6000
    n_step_change_map = 1000 if (modifying_mode == 'blocking') else 3000
    α = 1
    n_planning_steps = 10
    plot_time = 0.5

    draw_update = True

    grid = Grid(vis_mode='parallel', modifying_mode=modifying_mode)
    env = Environment(grid)
    agent = Agent(grid, alpha=α, n_planning_steps=n_planning_steps, temporal_reward_mode=False)
    agent_temporal_reward = Agent(grid, alpha=α, n_planning_steps=n_planning_steps, temporal_reward_mode=True)

    # init setup
    if draw_update:
        grid.draw_q_values_parallel(agent, agent_temporal_reward, plot_time=plot_time)

    # running
    for curr_step in range(n_steps):

        agent.tabular_dyna_q_iteration(env)
        agent_temporal_reward.tabular_dyna_q_iteration(env)

        if draw_update:
            grid.draw_q_values_parallel(agent, agent_temporal_reward, plot_time=plot_time)

        if curr_step == n_step_change_map:
            grid.modify_map()

    # ending setup
    if draw_update:
        grid.draw_q_values_parallel(agent, agent_temporal_reward, plot_time=plot_time)


if __name__ == '__main__':

    ## 1) Shifting obstacles

    # Reproduces book's figure 8.4
    get_cumu_reward_curves_comparison(modifying_mode='blocking')

    # Video comparison to visualize Dyna-Q & DynaQ+ on the blocking maze task
    #run_real_time_dislay_modified_maze(modifying_mode='blocking')

    ## 2) Opening shortcut

    # Reproduces book's figure 8.5
    get_cumu_reward_curves_comparison(modifying_mode='shortcut')

    # Video comparison to visualize Dyna-Q & DynaQ+ on the shortcut opening maze task
    #run_real_time_dislay_modified_maze(modifying_mode='shortcut')


