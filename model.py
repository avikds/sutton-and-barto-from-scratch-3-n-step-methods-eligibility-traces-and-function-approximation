"""
Sutton and Barto from Scratch 3: n-Step Methods, Eligibility Traces and Function Approximation

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - random_walk_true_values
import numpy as np

def random_walk_true_values(n_states):
    # State indices range from 0 to n_states - 1.
    states = np.arange(n_states, dtype=float)

    # Exact true value of each state in the random walk.
    true_values = -1 + 2 * (states + 1) / (n_states + 1)

    return true_values

# Step 2 - random_walk_step (not yet solved)
# TODO: implement

# Step 3 - generate_random_walk_episode (not yet solved)
# TODO: implement

# Step 4 - n_step_return (not yet solved)
# TODO: implement

# Step 5 - n_step_td_episode (not yet solved)
# TODO: implement

# Step 6 - n_step_td_prediction (not yet solved)
# TODO: implement

# Step 7 - n_step_parameter_sweep (not yet solved)
# TODO: implement

# Step 8 - lambda_return (not yet solved)
# TODO: implement

# Step 9 - offline_lambda_return_episode (not yet solved)
# TODO: implement

# Step 10 - update_traces (not yet solved)
# TODO: implement

# Step 11 - td_lambda_episode (not yet solved)
# TODO: implement

# Step 12 - lambda_prediction (not yet solved)
# TODO: implement

# Step 13 - large_walk_true_values (not yet solved)
# TODO: implement

# Step 14 - state_aggregation_features (not yet solved)
# TODO: implement

# Step 15 - gradient_mc_episode (not yet solved)
# TODO: implement

# Step 16 - semi_gradient_td0_episode (not yet solved)
# TODO: implement

# Step 17 - compare_value_approximations (not yet solved)
# TODO: implement

# Step 18 - mountain_car_step (not yet solved)
# TODO: implement

# Step 19 - tile_code (not yet solved)
# TODO: implement

# Step 20 - tile_epsilon_greedy (not yet solved)
# TODO: implement

# Step 21 - sarsa_lambda_episode (not yet solved)
# TODO: implement

# Step 22 - mountain_car_training (not yet solved)
# TODO: implement

