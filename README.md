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
- [ ] **19.** tile_code
- [ ] **20.** tile_epsilon_greedy
- [ ] **21.** sarsa_lambda_episode
- [ ] **22.** mountain_car_training

---

Built on Deep-ML.
