# Yet another "Reinforcement-learning : an introduction" repository

Replication of examples in python : solutions repo for the **Sutton & Barto's book : "Reinforcement-learning : an introduction"**.

See the the [book's official webpage](http://incompleteideas.net/book/the-book-2nd.html).


## Acknowledgment
The great ["reinforcement-learning-an-introduction" repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) inspired me to get the book and reproduce examples myself. I encourage the interested reader to cross compare solutions as needed.


## Figures & examples

### Chapter 1 : Introduction
* \[Tic-tac-toe\] [Win rate vs random policy agent](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap01-Tictactoe_greedy_temporal_learning/figures/play_against_random_opponent_win_rate.png)
* \[Tic-tac-toe\] [Impact of exploration rate on score](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap01-Tictactoe_greedy_temporal_learning/figures/score_wrt_exploration_rate.png)
* \[Tic-tac-toe\] [Self-play win rate](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap01-Tictactoe_greedy_temporal_learning/figures/self_play_win_rate.png)



### Chapter 2 : Multi-armed Bandits
* \[10-armed testbed\] [Epsilon-greedy policy on a stationary task](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap02-bandit_problem/figures/rwds_epsilon_greedy_stationary.png)
* \[10-armed testbed\] [Epsilon-greedy policy on a nonstationary task](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap02-bandit_problem/figures/rwds_epsilon_greedy_nonstationary.png)
* \[10-armed testbed\] [Impact of the step-size strategy on a nonstationary task](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap02-bandit_problem/figures/rwds_step_size_strategies_nonstationnary.png)


### Chapter 3 : Finite Markov Decision Processes
* \[GridWorld\] [Exact value function](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap03-finite_MDP/figures/gridworld_exact_value_function.png)


### Chapter 4 : Dynamic Programming
* \[GridWorld\] Iterative policy evaluation : [it=10](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap04-dynamic_programming/figures/iterative_policy_evaluation_k10.png), [it=131 (optimal)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap04-dynamic_programming/figures/iterative_policy_evaluation_optimal_policy.png)
* \[Jack's Car Rental\] Found policy : [basic setup](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap04-dynamic_programming/figures/policy_opti_book_case_subplots.png), [modified setup (ex 4.7)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap04-dynamic_programming/figures/policy_opti_changed_case_subplots.png)
* \[Gambler's problem\] [Solution](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap04-dynamic_programming/figures/gamblers_problem_curves.png)


### Chapter 5 : Monte Carlo Methods
* \[Blackjack\] [Approximate state-value function, policy that sticks on 20 or 21](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap05-monte_carlo_methods/figures/first_visit_mc_prediction_blackjack.png)
* \[Blackjack\] [Optimal policy and state-value function, found by Monte Carlo ES](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap05-monte_carlo_methods/figures/mc_exploring_starts.png)
* \[Blackjack\] [Ordinary vs Weighted importance sampling](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap05-monte_carlo_methods/figures/black_jack_offpolicy_ordinary_weighted.png)
* \[One-state MDP\] [Ordinary importance sampling produces surprinsingly unstable estimates](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap05-monte_carlo_methods/figures/off_policy_ordinary_infinite_variance.png)
* \[Racetrack\] [Large map](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap05-monte_carlo_methods/figures/racetrack_large.png)


### Chapter 6 : Temporal-Difference Learning
* \[Random Walk\] [TD(0) estimated values](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/random_walk_td_values.png), [learning curves for various values of alpha](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/random_walk_rms_errors.png)
* \[Random Walk\] [TD(0) vs constant-alpha MC under batch training](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/random_walk_batch_training.png)
* \[Windy GridWorld\] [Cumulated number of episodes (4 actions)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/windy_gridworld_4actions_cumstep_episodes.png), [(8 actions)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/windy_gridworld_8actions_cumstep_episodes.png), [(9 actions)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/windy_gridworld_9actions_cumstep_episodes.png), [optimal trajectory (ex 6.10)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/windy_gridworld_optimal_trajectory.png)
* \[Cliff Walking\] Q-learning vs Sarsa : [found trajectories](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/cliff_walking_trajectories.png), [cumulated rewards w.r.t. episodes](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/cliff_walking_rewards.png)
* \[Simple episodic MDP\] [Q-learning vs Double Q-learning](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap06-temporal_difference_learning/figures/maximization_bias_double_learning.png)


### Chapter 7 : n-step Bootstrapping
* \[19-state Random Walk\] [n-step TD methods performances](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap07-n_step_bootstrapping/figures/n_step_td_random_walk.png)
* \[5-state Random Walk\] [n-step TD methods performances (ex 7.3)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap07-n_step_bootstrapping/figures/n_step_td_random_walk_5_states.png)


### Chapter 8 : Planning and Learning with Tabular Methods
* \[Simple Maze\] [Learning curves comparison for Dyna-Q agents varying in their number of planning steps](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/steps_per_episodes_wrt_planning.png)
* Simple Maze : Policy found - [nonplanning Dyna-Q agent](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/without_planning_n0_episode2.png?token=AKN4L7K7KHHRIGD6H5VEKCLANSFHK), [planning Dyna-Q agent](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/without_planning_n0_episode2.png?token=AKN4L7K7KHHRIGD6H5VEKCLANSFHK)
* \[Blocking Maze\] [Average performance : Dyna-Q vs Dyna-Q+](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/dyna_q_obstacle_shift_adaptation.png),
	* **Video** : [Solving Blocking Maze](https://www.youtube.com/watch?v=99SmY9es3ow)
* \[Shortcut Maze\] [Average performance : Dyna-Q vs Dyna-Q+](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/dyna_q_shortcut_opening_adaptation.png)
* \[Large-sized Gridworlds\] [Updates until optimal solution : Dyna-Q vs Prioritized sweeping](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/prioritized_sweeping_on_mazes.png)
* \[1000-states, various branching factors\] [Comparison of relative efficiency of updates : simulated on-policy trajectories vs uniformly distributed](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/trajectory_sampling_8_8_up.png)
* \[10000-states, various branching factors\] [Comparison of relative efficiency of updates : simulated on-policy trajectories vs uniformly distributed](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap08_planning_and_learning_tabular_methods/figures/trajectory_sampling_8_8_down.png)


### Chapter 9 : On-policy Prediction with Approximation
* \[1000-states RandomWalk\] [Function approximation by state aggregation, gradient Monte Carlo algorithm](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap09-on_policy_prediction_with_approximation/figures/gradient_mc_state_aggreg_random_walk.png)
* \[1000-states RandomWalk\] [Function approximation by state aggregation, semi-gradient TD algorithm](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap09-on_policy_prediction_with_approximation/figures/semi_grad_td0_state_aggreg_random_walk.png)


### Chapter 10 : On-policy Control with Approximation
* \[Mountain Car\] [Cost-to-go function learned](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap10-on_policy_control_with_approximation/figures/mountain_car_semi_gradient_sarsa_3d_plots_(ticks2correct).png)
* \[Mountain Car\] [Learning curves for the semi-gradient Sarsa method with tile coding](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap10-on_policy_control_with_approximation/figures/mountain_car_semi_gradient_sarsa.png)
* \[Mountain Car\] [Performance of one-step vs 8-step semi-gradient Sarsa](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap10-on_policy_control_with_approximation/figures/mountain_car_semi_gradient_sarsa_n_steps.png)
* \[Access-Control Queuing Task\] [Policy and value function found by differential semi-gradient one-step Sarsa](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap10-on_policy_control_with_approximation/figures/access_control_queuing_task_policy_q_value.png)


### Chapter 11 : Off-policy Methods with Approximation
* \[Baird's counterexample\] Demonstration of instability : [Semi-gradient Off-policy TD](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap11-off_policy_methods_with_approximation/figures/baird_counterexemple_divergent_weights_TD_method.png), [Semi-gradient DP](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap11-off_policy_methods_with_approximation/figures/baird_counterexemple_divergent_weights_DP_method.png)


### Chapter 12 : Eligibility Traces
* \[19-state Random Walk\] [Performance of the Off-line lambda-return algorithm](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap12-eligibility_traces/figures/offline_lambda_return_random_walk.png)
* \[19-state Random Walk\] [Performance of TD(lambda)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap12-eligibility_traces/figures/td_lambda_random_walk.png)
* \[19-state Random Walk\] [Performance of True Online TD(lambda)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap12-eligibility_traces/figures/true_online_td_lambda_random_walk_alpha_range.png)
* \[Mountain Car\] [Early performance of Sarsa(lambda) with replacing trace](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap12-eligibility_traces/figures/mountain_car_sarsa_lambda_with_replacing_traces.png)
* \[Mountain Car\] [Summary comparison of Sarsa(lambda)](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap12-eligibility_traces/figures/mountain_car_sarsa_lambda_algo_compare.png)
* \[Mountain Car\] [Random Walk, Puddle World, Cart and Pole : The effect of lambda on reinforcement learning performance](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap12-eligibility_traces/figures/lambda_effect_on_rl.png)


### Chapter 13 : Policy Gradient Methods
* \[Short-corridor GridWorld\] [State-value distribution](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap13-policy_gradient_methods/figures/short_corridor_switched_actions_state_values_distrib.png)
* \[Short-corridor GridWorld\] [REINFORCE with different step sizes](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap13-policy_gradient_methods/figures/reinforce_mc_reward_curves.png)
* \[Short-corridor GridWorld\] [Effect of baseline on REINFORCE learning curve](https://raw.githubusercontent.com/Johann-Huber/yet-another-rl-sutton-barto-py/master/chap13-policy_gradient_methods/figures/reinforce_mc_baseline_reward_curves.png)



## Requirements
* numpy
* pandas
* tqdm
* matplotlib, seaborn


## Credits
I used external pieces of code for a couple of examples :
* The Cart and Pole environment's code has been taken from [openai gym source code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L14).
* The tile coding software comes from [Sutton's website](http://www.incompleteideas.net/tiles/tiles3.html).
