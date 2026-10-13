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

    # The full return is the Monte Carlo return, with no bootstrap term.
    G_full = n_step_return(rewards, states, t, T - t, values, gamma)

    G_lambda = (lam ** (T - t - 1)) * G_full

    # Add the weighted n-step returns for shorter horizons.
    for n in range(1, T - t):
        G_n = n_step_return(rewards, states, t, n, values, gamma)
        G_lambda += (1.0 - lam) * (lam ** (n - 1)) * G_n

    return float(G_lambda)

# Step 9 - offline_lambda_return_episode
def offline_lambda_return_episode(values, states, rewards, lam, alpha, gamma=1.0):
    # Preserve the original values and use them to compute all targets.
    updated_values = values.copy()
    T = len(rewards)
    targets = np.zeros(T, dtype=float)

    # Compute lambda-return targets backward using the values at the
    # start of the episode, before applying any updates.
    G = 0.0

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            # The episode has terminated, so the terminal value is zero.
            G = float(rewards[t])
        else:
            next_state_value = values[states[t + 1]]
            G = rewards[t] + gamma * (
                (1.0 - lam) * next_state_value + lam * G
            )

        targets[t] = G

    # Apply updates in chronological order using the precomputed targets.
    for t in range(T):
        state = states[t]
        updated_values[state] += alpha * (
            targets[t] - updated_values[state]
        )

    return updated_values

# Step 10 - update_traces
def update_traces(traces, state, lam, gamma, mode):
    # Validate the trace mode before modifying the copied array.
    if mode not in ("accumulating", "replacing"):
        raise ValueError(
            "mode must be 'accumulating' or 'replacing'"
        )

    # Copy the input to ensure it remains unchanged.
    updated_traces = traces.copy()

    # Decay all eligibility traces.
    updated_traces *= gamma * lam

    # Update the eligibility trace for the current state.
    if mode == "accumulating":
        updated_traces[state] += 1.0
    else:  # mode == "replacing"
        updated_traces[state] = 1.0

    return updated_traces

# Step 11 - td_lambda_episode
def td_lambda_episode(values, states, rewards, lam, alpha, gamma, mode):
    # Copy the input values so the original array remains unchanged.
    updated_values = values.copy()

    # Initialize all eligibility traces to zero.
    traces = np.zeros_like(updated_values, dtype=float)

    T = len(rewards)

    # Process each transition in chronological order.
    for t in range(T):
        state = states[t]

        # The terminal state's value is zero.
        if t + 1 < T:
            next_state_value = updated_values[states[t + 1]]
        else:
            next_state_value = 0.0

        # Compute the TD error using the current value estimates.
        delta = (
            rewards[t]
            + gamma * next_state_value
            - updated_values[state]
        )

        # Decay traces and mark the current state.
        traces = update_traces(
            traces, state, lam, gamma, mode
        )

        # Update every state's value in proportion to its eligibility trace.
        updated_values += alpha * delta * traces

    return updated_values

# Step 12 - lambda_prediction
def lambda_prediction(n_states, lam, alpha, n_episodes, method, rng, gamma=1.0):
    # Validate the selected learning method.
    if method not in ("offline", "accumulating", "replacing"):
        raise ValueError(
            "method must be 'offline', 'accumulating', or 'replacing'"
        )

    # Initialize state values and calculate the exact true values.
    values = np.zeros(n_states, dtype=float)
    true_values = random_walk_true_values(n_states)

    # Store RMS error after each episode.
    rms_errors = np.zeros(n_episodes, dtype=float)

    for episode in range(n_episodes):
        # Generate one complete random-walk trajectory.
        states, rewards = generate_random_walk_episode(n_states, rng)

        # Apply the selected lambda-return or eligibility-trace method.
        if method == "offline":
            values = offline_lambda_return_episode(
                values, states, rewards, lam, alpha, gamma
            )
        else:
            values = td_lambda_episode(
                values, states, rewards, lam, alpha, gamma, method
            )

        # Evaluate the updated estimates against the true state values.
        rms_errors[episode] = np.sqrt(
            np.mean((values - true_values) ** 2)
        )

    return values, rms_errors

# Step 13 - large_walk_true_values
def large_walk_true_values(n_states, max_jump):
    # Transition matrix between non-terminal states.
    P = np.zeros((n_states, n_states), dtype=float)

    # Expected immediate reward for each non-terminal state.
    r = np.zeros(n_states, dtype=float)

    # Every direction-jump combination has equal probability.
    probability = 0.5 / max_jump

    for state in range(n_states):
        for jump in range(1, max_jump + 1):
            # Consider moving left.
            next_state = state - jump

            if next_state < 0:
                # Left terminal: reward -1.
                r[state] -= probability
            else:
                P[state, next_state] += probability

            # Consider moving right.
            next_state = state + jump

            if next_state >= n_states:
                # Right terminal: reward +1.
                r[state] += probability
            else:
                P[state, next_state] += probability

    # Solve the Bellman equation: (I - P) v = r.
    identity = np.eye(n_states, dtype=float)
    true_values = np.linalg.solve(identity - P, r)

    return true_values.astype(float)

# Step 14 - state_aggregation_features
def state_aggregation_features(state, n_states, n_groups):
    # Map the state to its corresponding aggregation group.
    group = state * n_groups // n_states

    # Create a one-hot feature vector.
    features = np.zeros(n_groups, dtype=float)
    features[group] = 1.0

    return features

# Step 15 - gradient_mc_episode
def gradient_mc_episode(weights, states, rewards, n_states, n_groups, alpha, gamma=1.0):
    # Copy the weights so the input array is not modified.
    updated_weights = weights.copy()

    T = len(rewards)

    # Compute the complete discounted return for every time step.
    returns = np.zeros(T, dtype=float)
    G = 0.0

    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Apply gradient Monte Carlo updates in chronological order.
    for t in range(T):
        # Get the one-hot state-aggregation feature vector.
        x = state_aggregation_features(states[t], n_states, n_groups)

        # Estimate the value using the current weights.
        estimated_value = updated_weights @ x

        # Update the weights using the complete return as the target.
        updated_weights += alpha * (returns[t] - estimated_value) * x

    return updated_weights

# Step 16 - semi_gradient_td0_episode
def semi_gradient_td0_episode(weights, states, rewards, n_states, n_groups, alpha, gamma=1.0):
    # Copy the weights so the input array remains unchanged.
    updated_weights = weights.copy()

    T = len(rewards)

    # Process each transition in chronological order.
    for t in range(T):
        # Feature vector for the current state.
        x = state_aggregation_features(
            states[t], n_states, n_groups
        )

        # Estimate the next state's value, or zero at termination.
        if t + 1 < T:
            next_features = state_aggregation_features(
                states[t + 1], n_states, n_groups
            )
            next_value = updated_weights @ next_features
        else:
            next_value = 0.0

        # Estimate the current state's value using the latest weights.
        current_value = updated_weights @ x

        # Compute the one-step TD error.
        delta = (
            rewards[t]
            + gamma * next_value
            - current_value
        )

        # Apply the semi-gradient TD(0) weight update.
        updated_weights += alpha * delta * x

    return updated_weights

# Step 17 - compare_value_approximations
def compare_value_approximations(
    n_states, n_groups, max_jump,
    alpha_mc, alpha_td, n_episodes, seed
):
    # Initialize the random number generator and compute exact values.
    rng = np.random.default_rng(seed)
    true_values = large_walk_true_values(n_states, max_jump)

    # Initialize the state-aggregation weights for both methods.
    mc_weights = np.zeros(n_groups, dtype=float)
    td_weights = np.zeros(n_groups, dtype=float)

    # Train both methods using exactly the same episodes.
    for episode in range(n_episodes):
        states, rewards = generate_random_walk_episode(
            n_states, rng, max_jump
        )

        # Gradient Monte Carlo update.
        mc_weights = gradient_mc_episode(
            mc_weights,
            states,
            rewards,
            n_states,
            n_groups,
            alpha_mc
        )

        # Semi-gradient TD(0) update using the same trajectory.
        td_weights = semi_gradient_td0_episode(
            td_weights,
            states,
            rewards,
            n_states,
            n_groups,
            alpha_td
        )

    # Map each state's index to its state-aggregation group.
    state_indices = np.arange(n_states)
    group_indices = state_indices * n_groups // n_states

    # Obtain the estimated value of every non-terminal state.
    mc_values = mc_weights[group_indices].astype(float)
    td_values = td_weights[group_indices].astype(float)

    # Calculate RMS errors against the exact state values.
    mc_rms = float(np.sqrt(np.mean((mc_values - true_values) ** 2)))
    td_rms = float(np.sqrt(np.mean((td_values - true_values) ** 2)))

    return {
        "mc_values": mc_values,
        "td_values": td_values,
        "mc_rms": mc_rms,
        "td_rms": td_rms
    }

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

