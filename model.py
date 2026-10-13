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

# Step 4 - n_step_return
def n_step_return(rewards, states, t, n, values, gamma=1.0):
    T = len(rewards)
    G = 0.0

    # Sum up to n discounted rewards, stopping at the episode's end.
    for k in range(t, min(t + n, T)):
        G += (gamma ** (k - t)) * rewards[k]

    # Bootstrap only if the full n-step horizon remains within the episode.
    if t + n < T:
        G += (gamma ** n) * values[states[t + n]]

    return float(G)

# Step 5 - n_step_td_episode
def n_step_td_episode(values, states, rewards, n, alpha, gamma=1.0):
    # Copy the input so the original value array remains unchanged.
    updated_values = values.copy()

    T = len(rewards)

    # Process each visited state in chronological order.
    for t in range(T):
        # Compute the n-step return using the current values,
        # including updates made at earlier time steps.
        G = n_step_return(
            rewards, states, t, n, updated_values, gamma
        )

        # Apply the n-step TD prediction update.
        state = states[t]
        updated_values[state] += alpha * (
            G - updated_values[state]
        )

    return updated_values

# Step 6 - n_step_td_prediction
def n_step_td_prediction(n_states, n, alpha, n_episodes, rng, gamma=1.0):
    # Initialize state-value estimates to zero.
    values = np.zeros(n_states, dtype=float)

    # Calculate the exact state values for RMS error evaluation.
    true_values = random_walk_true_values(n_states)

    # Store the RMS error after each episode.
    rms_errors = np.zeros(n_episodes, dtype=float)

    for episode in range(n_episodes):
        # Generate one complete random-walk episode.
        states, rewards = generate_random_walk_episode(n_states, rng)

        # Update the estimates using n-step TD prediction.
        values = n_step_td_episode(
            values, states, rewards, n, alpha, gamma
        )

        # Measure prediction accuracy after this episode.
        rms_errors[episode] = np.sqrt(
            np.mean((values - true_values) ** 2)
        )

    return values, rms_errors

# Step 7 - n_step_parameter_sweep
def n_step_parameter_sweep(n_states, ns, alphas, n_episodes, n_runs, seed):
    # Store the average learning error for every (n, alpha) combination.
    errors = np.zeros((len(ns), len(alphas)), dtype=float)

    for i, n in enumerate(ns):
        for j, alpha in enumerate(alphas):
            run_errors = []

            for run in range(n_runs):
                # Use the same seed for each parameter combination
                # within a given run, ensuring comparable episodes.
                rng = np.random.default_rng(seed + run)

                # Run n-step TD prediction for the specified parameters.
                _, rms_errors = n_step_td_prediction(
                    n_states, n, alpha, n_episodes, rng
                )

                # Average the RMS errors across episodes for this run.
                run_errors.append(np.mean(rms_errors))

            # Average the episode-averaged errors across independent runs.
            errors[i, j] = np.mean(run_errors)

    return errors

# Step 8 - lambda_return
def lambda_return(rewards, states, t, lam, values, gamma=1.0):
    T = len(rewards)
    G_lambda = 0.0

    # Combine the n-step returns for n = 1, ..., T-t-1.
    for n in range(1, T - t):
        G_n = n_step_return(rewards, states, t, n, values, gamma)
        G_lambda += (1.0 - lam) * (lam ** (n - 1)) * G_n

    # Add the full Monte Carlo return, weighted by lam ** (T-t-1).
    G_full = n_step_return(rewards, states, t, T - t, values, gamma)
    G_lambda += (lam ** (T - t - 1)) * G_full

    return float(G_lambda)

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

