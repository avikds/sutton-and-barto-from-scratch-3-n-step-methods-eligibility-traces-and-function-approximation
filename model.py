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

# Step 2 - random_walk_step
def random_walk_step(state, n_states, rng, max_jump=1):
    # Decide the direction first: left with probability 0.5, otherwise right.
    go_left = rng.random() < 0.5

    # Draw the jump size only when max_jump is greater than 1.
    if max_jump == 1:
        jump = 1
    else:
        jump = int(rng.integers(1, max_jump + 1))

    # Calculate the next state based on the chosen direction.
    if go_left:
        next_state = state - jump
    else:
        next_state = state + jump

    # Check whether the move reaches or overshoots a terminal state.
    if next_state < 0:
        return (-1, -1.0, True)
    elif next_state >= n_states:
        return (n_states, 1.0, True)

    # Non-terminal transitions have zero reward.
    return (int(next_state), 0.0, False)

# Step 3 - generate_random_walk_episode
def generate_random_walk_episode(n_states, rng, max_jump=1):
    # Start at the centre non-terminal state.
    state = n_states // 2

    # Store the visited non-terminal states and rewards.
    states = []
    rewards = []

    # Continue until the walk reaches a terminal state.
    while True:
        # Record the current state; never store the terminal state.
        states.append(int(state))

        # Take one transition using the previously implemented function.
        next_state, reward, done = random_walk_step(
            state, n_states, rng, max_jump
        )

        # Store the reward received when leaving the current state.
        rewards.append(float(reward))

        # Stop after recording the terminal transition's reward.
        if done:
            break

        # Move to the next non-terminal state.
        state = next_state

    return states, rewards

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

