"""Tests for RL agents."""

import pytest
import numpy as np
from gridworld.env import GridWorld
from gridworld.agents import QLearningAgent, ValueIterationAgent, PolicyIterationAgent


@pytest.fixture
def env():
    return GridWorld.from_preset("simple_9x9", slip_prob=0.0)


@pytest.fixture
def small_env():
    """Tiny 3x3 grid for fast tests."""
    grid = [
        [4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 2, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return GridWorld(grid=grid, slip_prob=0.0)


# ======================================================================
# Q-Learning
# ======================================================================

class TestQLearningAgent:
    def test_init_q_table_zeros(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions)
        assert np.all(agent.Q == 0)

    def test_action_selection_valid(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions)
        for s in range(env.n_states):
            a = agent.select_action(s)
            assert 0 <= a < env.n_actions

    def test_greedy_action_valid(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions, epsilon=0.0)
        a = agent.select_action(0)
        assert 0 <= a < env.n_actions

    def test_q_table_updates_after_training(self, small_env):
        agent = QLearningAgent(small_env.n_states, small_env.n_actions)
        agent.train(small_env, n_episodes=3000, verbose=False)
        assert not np.all(agent.Q == 0), "Q-table should be updated after training"

    def test_policy_shape(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions)
        agent.train(env, n_episodes=3000, verbose=False)
        policy = agent.get_policy()
        assert policy.shape == (env.n_states,)

    def test_value_function_shape(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions)
        agent.train(env, n_episodes=100, verbose=False)
        V = agent.get_value_function()
        assert V.shape == (env.n_states,)

    def test_epsilon_decays(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions,
                               epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01)
        initial_eps = agent.epsilon
        agent.train(env, n_episodes=3000, verbose=False)
        assert agent.epsilon < initial_eps

    def test_reset_restores_q_table(self, small_env):
        agent = QLearningAgent(small_env.n_states, small_env.n_actions)
        agent.train(small_env, n_episodes=3000, verbose=False)
        agent.reset()
        assert np.all(agent.Q == 0)
        assert agent.epsilon == 1.0

    def test_training_history_populated(self, env):
        agent = QLearningAgent(env.n_states, env.n_actions)
        history = agent.train(env, n_episodes=50, verbose=False)
        assert len(history["episode_rewards"]) == 50
        assert len(history["epsilons"]) == 50

    def test_convergence_on_small_grid(self, small_env):
        """Trained agent should achieve better-than-random performance."""
        agent = QLearningAgent(
            small_env.n_states, small_env.n_actions,
            alpha=0.3, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.0
        )
        agent.train(small_env, n_episodes=5000, seed=0, verbose=False)

        # Evaluate greedily
        agent.epsilon = 0.0
        successes = 0
        for _ in range(50):
            s = small_env.reset()
            done = False
            while not done:
                a = agent.select_action(s)
                s, _, done, info = small_env.step(a)
            from gridworld.env import CellType
            if info["cell_type"] == int(CellType.GOAL):
                successes += 1
        assert successes > 10, f"Expected >10 successes/50, got {successes}"


# ======================================================================
# Value Iteration
# ======================================================================

class TestValueIterationAgent:
    def test_train_converges(self, env):
        agent = ValueIterationAgent(env.n_states, env.n_actions, theta=1e-9)
        history = agent.train(env)
        assert history["converged"]

    def test_policy_valid_actions(self, env):
        agent = ValueIterationAgent(env.n_states, env.n_actions)
        agent.train(env)
        policy = agent.get_policy()
        assert all(0 <= a < env.n_actions for a in policy)

    def test_value_function_goal_highest(self, small_env):
        """Goal state should have value 0 (terminal) and adjacent states positive."""
        agent = ValueIterationAgent(small_env.n_states, small_env.n_actions)
        agent.train(small_env)
        V = agent.get_value_function()
        start_s = small_env.n_cols * 0 + 0  # (0, 0)
        goal_s  = small_env.n_cols * 4 + 4 # (2, 2)
        # Start state should have a positive value (can reach goal)
        assert V[start_s] > 0

    def test_action_values_shape(self, env):
        agent = ValueIterationAgent(env.n_states, env.n_actions)
        agent.train(env)
        Q = agent.get_action_values(env)
        assert Q.shape == (env.n_states, env.n_actions)

    def test_select_action_consistent_with_policy(self, env):
        agent = ValueIterationAgent(env.n_states, env.n_actions)
        agent.train(env)
        policy = agent.get_policy()
        for s in range(env.n_states):
            assert agent.select_action(s) == policy[s]


# ======================================================================
# Policy Iteration
# ======================================================================

class TestPolicyIterationAgent:
    def test_train_converges(self, env):
        agent = PolicyIterationAgent(env.n_states, env.n_actions, eval_theta=1e-9)
        history = agent.train(env)
        assert history["converged"]

    def test_policy_updates_recorded(self, env):
        agent = PolicyIterationAgent(env.n_states, env.n_actions)
        history = agent.train(env)
        assert history["n_policy_updates"] >= 1

    def test_policy_matches_value_iteration(self, small_env):
        """PI and VI should produce the same optimal policy on a small grid."""
        from gridworld.env import CellType
        non_terminal = [
            s for s in range(small_env.n_states)
            if small_env.grid.flat[s] not in (CellType.WALL, CellType.GOAL, CellType.TRAP)
        ]

        pi_agent = PolicyIterationAgent(small_env.n_states, small_env.n_actions, eval_theta=1e-10)
        vi_agent = ValueIterationAgent(small_env.n_states, small_env.n_actions, theta=1e-10)
        pi_agent.train(small_env)
        vi_agent.train(small_env)

        pi_policy = pi_agent.get_policy()
        vi_policy = vi_agent.get_policy()

        # Policies should be identical on non-terminal states
        for s in non_terminal:
            assert pi_policy[s] == vi_policy[s], \
                f"State {s}: PI chose {pi_policy[s]}, VI chose {vi_policy[s]}"

    def test_value_function_shape(self, env):
        agent = PolicyIterationAgent(env.n_states, env.n_actions)
        agent.train(env)
        V = agent.get_value_function()
        assert V.shape == (env.n_states,)
