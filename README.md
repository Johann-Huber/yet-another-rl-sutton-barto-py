# Yet another "Reinforcement-learning : an introduction" repository

Replication of examples in python : solutions repo for the Sutton & Barto's book : "Reinforcement-learning : an introduction"yet-another-rl-sutton-barto-py

## Acknowledgment
The great [reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) repo inspired me to get the book and reproduce examples myself.
I encourage the interested reader to cross compare solutions as needed.

## Figures & examples

### Chapter 1 : Introduction
* [Tic-tac-toe : Win rate vs random policy agent](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap01-Tictactoe_greedy_temporal_learning/figures/play_against_random_opponent_win_rate.png)
* [Tic-tac-toe : Impact of exploration rate on score](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap01-Tictactoe_greedy_temporal_learning/figures/score_wrt_exploration_rate.png?token=AKN4L7P2HSEKESZO4SP4EA3ANHTRK)
* [Tic-tac-toe : Self-play win rate](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap01-Tictactoe_greedy_temporal_learning/figures/self_play_win_rate.png?token=AKN4L7LMHLGPPF5YHCREHH3ANHTRM)



### Chapter 2 : Multi-armed Bandits
* [Epsilon-greedy policy on a stationary task](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap02-bandit_problem/figures/rwds_epsilon_greedy_stationary.png?token=AKN4L7MDSTL7BDZ55Z25ZKLANHTWY)
* [Epsilon-greedy policy on a nonstationary task](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap02-bandit_problem/figures/rwds_epsilon_greedy_nonstationary.png?token=AKN4L7LPIR6KO6TGJYCIKU3ANHTWW)
* [Impact of the step-size strategy on a nonstationary task](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap02-bandit_problem/figures/rwds_step_size_strategies_nonstationnary.png?token=AKN4L7JSTQKZYPRUXBTWC3TANHTW2)


### Chapter 3 : Finite Markov Decision Processes
* [GridWorld : Exact value function](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap03-finite_MDP/figures/gridworld_exact_value_function.png?token=AKN4L7OHZGQLYCHSBRGT53TANHTZK)


### Chapter 4 : Dynamic Programming
* [relative path](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap04-dynamic_programming/figures/policy_opti_book_case_subplots.png)


### Chapter 5 : Monte Carlo Methods
* []()

### Chapter 6 : Temporal-Difference Learning
* []()

### Chapter 7 : n-step Bootstrapping
* []()

### Chapter 8 : Planning and Learning with Tabular Methods
* []()

### Chapter 9 : On-policy Prediction with Approximation
* []()

### Chapter 10 : On-policy Control with Approximation
* []()

### Chapter 11 : Off-policy Methods with Approximation
* []()

### Chapter 12 : Eligibility Traces
* []()

### Chapter 13 : Policy Gradient Methods
* []()

## Additional content

TODO: Add video


## Requirements
* numpy
* pandas
* tqdm
* matplotlib, seaborn

## Credits
I used external pieces of code for a couple of examples :
* The Cart and Pole environment's code has been taken from [openai gym source code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L14).
* The tile coding software comes from [Sutton's website](http://www.incompleteideas.net/tiles/tiles3.html).
