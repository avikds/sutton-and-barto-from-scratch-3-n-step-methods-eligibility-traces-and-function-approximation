# Sutton and Barto from Scratch 3: n-Step Methods, Eligibility Traces and Function Approximation

Implement n-step TD, lambda-returns, eligibility traces and linear function approximation from Sutton and Barto. Reproduce the n-step sweep on the 19-state random walk, compare offline lambda-returns with TD(lambda) traces, fit the 1000-state random walk with state aggregation by gradient Monte Carlo and semi-gradient TD(0), then solve Mountain Car with tile coding and Sarsa(lambda).

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** random_walk_true_values
- [x] **2.** random_walk_step
- [x] **3.** generate_random_walk_episode
- [x] **4.** n_step_return
- [x] **5.** n_step_td_episode
- [x] **6.** n_step_td_prediction
- [x] **7.** n_step_parameter_sweep
- [x] **8.** lambda_return
- [x] **9.** offline_lambda_return_episode
- [x] **10.** update_traces
- [x] **11.** td_lambda_episode
- [x] **12.** lambda_prediction
- [x] **13.** large_walk_true_values
- [x] **14.** state_aggregation_features
- [x] **15.** gradient_mc_episode
- [x] **16.** semi_gradient_td0_episode
- [x] **17.** compare_value_approximations
- [x] **18.** mountain_car_step
- [x] **19.** tile_code
- [x] **20.** tile_epsilon_greedy
- [x] **21.** sarsa_lambda_episode
- [x] **22.** mountain_car_training

## Results

```
19-state true values (first 5): [-0.9, -0.8, -0.7, -0.6, -0.5]
Sample episode length: 96 exit reward: 1.0
n-step returns at t=0 for n=1,4,16: [0.1, -0.2, 0.4]
n-step sweep: mean RMS over 10 episodes x 20 runs (rows n, columns alpha)
  n= 1: [0.506, 0.471, 0.413, 0.368, 0.345, 0.388]  best alpha=0.8 RMS=0.3454
  n= 2: [0.47, 0.407, 0.313, 0.277, 0.316, 0.444]  best alpha=0.6 RMS=0.2772
  n= 4: [0.413, 0.325, 0.266, 0.309, 0.391, 0.527]  best alpha=0.4 RMS=0.2658
  n= 8: [0.35, 0.283, 0.321, 0.397, 0.477, 0.578]  best alpha=0.2 RMS=0.2830
  n=16: [0.323, 0.328, 0.421, 0.502, 0.573, 0.653]  best alpha=0.1 RMS=0.3232
Best n overall (intermediate n wins, as in figure 7.2): 4
lambda-return at t=0 for lambda=0,0.5,1 on a toy episode: [0.4, 0.4, 1.0]
TD(lambda=0) == TD(0): True | offline lambda=1 == constant-alpha MC: True
lambda study: mean RMS over 10 episodes x 20 runs, alpha=0.3 (columns lambda = [0.0, 0.4, 0.8, 0.9, 1.0] )
       offline: [0.446, 0.389, 0.29, 0.317, 0.73]
  accumulating: [0.441, 0.38, 0.281, 0.351, 'diverged']
     replacing: [0.441, 0.387, 0.301, 0.288, 0.412]
1000-state true values at 0, 250, 500, 750, 999: [-0.922, -0.471, 0.001, 0.473, 0.922]
Feature of state 250 (10 groups): [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
After 500 episodes: gradient MC RMS=0.2476  semi-gradient TD(0) RMS=0.1894
After 3000 episodes: gradient MC RMS=0.0821  semi-gradient TD(0) RMS=0.1293
(TD(0) learns faster early; gradient MC ends closer to the true curve, as in figures 9.1/9.2)
Group centres   : [-0.85, -0.66, -0.47, -0.28, -0.09, 0.1, 0.28, 0.47, 0.66, 0.85]
MC staircase    : [-0.83, -0.65, -0.41, -0.16, 0.02, 0.12, 0.28, 0.45, 0.64, 0.78]
TD(0) staircase : [-0.72, -0.52, -0.37, -0.26, -0.12, 0.0, 0.18, 0.32, 0.49, 0.7]
One Mountain Car step from (-0.5, 0) with full throttle: -0.4992 0.0008 -1.0 False
Active tiles at (-0.5, 0): [31, 112, 193, 274, 355, 436, 526, 607]
Steps per episode (first 10): [1848, 1087, 657, 423, 403, 228, 231, 325, 260, 156]
Steps per episode (last 10) : [110, 106, 170, 150, 107, 107, 107, 107, 107, 107]
Mean steps, episodes 1-10 vs 41-50: 561.8 -> 117.8
Learned q(start, a) for a=0,1,2: [-135.95, -134.4, -115.68]
```
