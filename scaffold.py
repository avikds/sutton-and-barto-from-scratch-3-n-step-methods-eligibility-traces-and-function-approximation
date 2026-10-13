"""
Sutton and Barto from Scratch 3: n-Step Methods, Eligibility Traces and Function Approximation scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sutton & Barto from scratch 3: n-step TD, eligibility traces and function approximation demo."""
import numpy as np


def main():
    # --- Part 1: n-step TD on the 19-state random walk (book figure 7.2 at small scale) ---
    n_states = 19
    true_values = random_walk_true_values(n_states)
    print("19-state true values (first 5):", np.round(true_values[:5], 2).tolist())

    rng = np.random.default_rng(0)
    states, rewards = generate_random_walk_episode(n_states, rng)
    print("Sample episode length:", len(states), "exit reward:", rewards[-1])
    print("n-step returns at t=0 for n=1,4,16:",
          [round(n_step_return(rewards, states, 0, n, true_values), 4) for n in (1, 4, 16)])

    ns = [1, 2, 4, 8, 16]
    alphas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    errors = n_step_parameter_sweep(n_states, ns, alphas, n_episodes=10, n_runs=20, seed=0)
    print("n-step sweep: mean RMS over 10 episodes x 20 runs (rows n, columns alpha)")
    for i, n in enumerate(ns):
        best = int(np.argmin(errors[i]))
        print(f"  n={n:>2}: {np.round(errors[i], 3).tolist()}  best alpha={alphas[best]} RMS={errors[i, best]:.4f}")
    best_n = ns[int(np.argmin(errors.min(axis=1)))]
    print("Best n overall (intermediate n wins, as in figure 7.2):", best_n)

    # --- Part 2: lambda-returns and eligibility traces (book figures 12.3 and 12.6) ---
    states, rewards = [2, 3, 2, 3, 4], [0.0, 0.0, 0.0, 0.0, 1.0]
    print("lambda-return at t=0 for lambda=0,0.5,1 on a toy episode:",
          [round(lambda_return(rewards, states, 0, lam, np.array([0.0, 0.0, 0.2, 0.4, 0.6])), 4) for lam in (0.0, 0.5, 1.0)])

    rng = np.random.default_rng(3)
    states, rewards = generate_random_walk_episode(n_states, rng)
    td0 = n_step_td_episode(np.zeros(n_states), states, rewards, 1, 0.3)
    tdl0 = td_lambda_episode(np.zeros(n_states), states, rewards, 0.0, 0.3, 1.0, 'accumulating')
    mc = n_step_td_episode(np.zeros(n_states), states, rewards, 10 ** 6, 0.3)
    off1 = offline_lambda_return_episode(np.zeros(n_states), states, rewards, 1.0, 0.3)
    print("TD(lambda=0) == TD(0):", bool(np.allclose(td0, tdl0)),
          "| offline lambda=1 == constant-alpha MC:", bool(np.allclose(mc, off1)))

    lams = [0.0, 0.4, 0.8, 0.9, 1.0]
    print("lambda study: mean RMS over 10 episodes x 20 runs, alpha=0.3 (columns lambda =", lams, ")")
    for method in ('offline', 'accumulating', 'replacing'):
        row = []
        for lam in lams:
            total = 0.0
            for run in range(20):
                _, rms = lambda_prediction(n_states, lam, 0.3, 10, method, np.random.default_rng(100 + run))
                total += rms.mean()
            mean_rms = float(total / 20)
            row.append(round(mean_rms, 3) if mean_rms < 10 else 'diverged')
        print(f"  {method:>12}: {row}")

    # --- Part 3: 1000-state random walk with state aggregation (book figures 9.1 / 9.2) ---
    big = large_walk_true_values(1000, 100)
    print("1000-state true values at 0, 250, 500, 750, 999:", np.round(big[[0, 250, 500, 750, 999]], 3).tolist())
    print("Feature of state 250 (10 groups):", state_aggregation_features(250, 1000, 10).astype(int).tolist())
    for n_episodes in (500, 3000):
        result = compare_value_approximations(1000, 10, 100, alpha_mc=0.0005, alpha_td=0.01,
                                              n_episodes=n_episodes, seed=0)
        print(f"After {n_episodes} episodes: gradient MC RMS={result['mc_rms']:.4f}  semi-gradient TD(0) RMS={result['td_rms']:.4f}")
    print("(TD(0) learns faster early; gradient MC ends closer to the true curve, as in figures 9.1/9.2)")
    print("Group centres   :", np.round(big[50::100], 2).tolist())
    print("MC staircase    :", np.round(result['mc_values'][50::100], 2).tolist())
    print("TD(0) staircase :", np.round(result['td_values'][50::100], 2).tolist())

    # --- Part 4: Mountain Car with tile coding and Sarsa(lambda) (book figure 10.1) ---
    position, velocity, reward, done = mountain_car_step(-0.5, 0.0, 2)
    print("One Mountain Car step from (-0.5, 0) with full throttle:", round(position, 4), round(velocity, 4), reward, done)
    print("Active tiles at (-0.5, 0):", tile_code(-0.5, 0.0, 8, 8).tolist())
    weights, steps = mountain_car_training(n_episodes=50, n_tilings=8, tiles_per_dim=8, lam=0.9,
                                           alpha=0.5 / 8, gamma=1.0, epsilon=0.0, seed=0, max_steps=5000)
    print("Steps per episode (first 10):", steps[:10].tolist())
    print("Steps per episode (last 10) :", steps[-10:].tolist())
    print("Mean steps, episodes 1-10 vs 41-50:", round(float(steps[:10].mean()), 1), "->", round(float(steps[-10:].mean()), 1))
    start_tiles = tile_code(-0.5, 0.0, 8, 8)
    print("Learned q(start, a) for a=0,1,2:", np.round(weights[:, start_tiles].sum(axis=1), 2).tolist())


if __name__ == "__main__":
    main()

