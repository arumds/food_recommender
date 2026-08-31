"""
LinUCB contextual bandit for real-time re-ranking.

The base ranker (LambdaMART) is trained offline on historical data and doesn't
see real-time context (weather, time of day, active discounts) -- and can't
adapt online as it learns which restaurants actually perform well under which
conditions. A contextual bandit sits as a final re-ranking layer that:

  1. Takes the base ranker's top-N candidates
  2. Scores each candidate (an "arm") using a per-arm linear model over the
     CURRENT context vector, with an upper-confidence-bound exploration term
  3. Blends explore/exploit: arms with little data get an exploration boost,
     arms with a well-estimated high reward get pushed up on merit
  4. After the (simulated) outcome is observed, updates that arm's linear
     model -- so the bandit *improves online*, unlike the static ranker.

This is the LinUCB algorithm (Li et al., 2010 -- the standard contextual
bandit formulation for exactly this kind of "which arm performs best under
this context" problem, popularized by Yahoo's news article recommendation).
"""
import numpy as np


class LinUCB:
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        """
        n_arms       : number of restaurants (arms) the bandit can choose among
        context_dim  : dimensionality of the real-time context vector
        alpha        : exploration strength (higher = more exploration)
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        # per-arm ridge regression sufficient statistics
        self.A = [np.eye(context_dim) for _ in range(n_arms)]        # (d, d)
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]      # (d,)

    def score(self, arm_ids, context_vec):
        """Return UCB scores for the given arms under the current context."""
        scores = np.zeros(len(arm_ids))
        for i, arm in enumerate(arm_ids):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean = theta @ context_vec
            uncertainty = self.alpha * np.sqrt(context_vec @ A_inv @ context_vec)
            scores[i] = mean + uncertainty
        return scores

    def rank(self, arm_ids, context_vec):
        """Return arm_ids sorted by UCB score, descending."""
        scores = self.score(arm_ids, context_vec)
        order = np.argsort(-scores)
        return [arm_ids[i] for i in order]

    def update(self, arm, context_vec, reward):
        """Update the arm's linear model after observing `reward` (0/1 or continuous) for this context."""
        self.A[arm] += np.outer(context_vec, context_vec)
        self.b[arm] += reward * context_vec


def simulate_bandit_vs_static(
    n_rounds, restaurants, context_dim, rng, alpha=0.5, top_k_candidates=15,
):
    """
    Simulates `n_rounds` of serving requests: each round samples a context,
    a candidate pool (proxying the base ranker's top-K output), and compares
    the reward the LinUCB bandit's #1 pick earns vs. a static baseline that
    always picks the highest-popularity candidate regardless of context.

    Returns cumulative reward curves for both policies so the online-learning
    lift is visible over time (the bandit starts no better than static, and
    should pull ahead as it accumulates context-outcome data).
    """
    from context import sample_context, context_reward_boost, context_to_vector

    n_restaurants = len(restaurants)
    bandit = LinUCB(n_arms=n_restaurants, context_dim=context_dim, alpha=alpha)

    bandit_rewards, static_rewards = [], []

    for t in range(n_rounds):
        context = sample_context(rng)
        has_discount = rng.random() < 0.15
        is_repeat = rng.random() < 0.3
        ctx_vec = context_to_vector(context, has_discount, is_repeat)

        # candidate pool: a random-ish top-K slice standing in for the base ranker's output
        candidates = rng.choice(n_restaurants, size=top_k_candidates, replace=False)

        # --- static baseline: always picks highest popularity, ignores context ---
        cand_pop = restaurants.iloc[candidates]["popularity"].values
        static_pick = candidates[np.argmax(cand_pop)]

        # --- bandit: ranks by contextual UCB score ---
        bandit_order = bandit.rank(candidates.tolist(), ctx_vec)
        bandit_pick = bandit_order[0]

        # --- true reward model: base conversion rate + context-cuisine boost ---
        def true_reward(rest_idx):
            cuisine = restaurants.iloc[rest_idx]["cuisine"]
            base = 0.15 + 0.1 * restaurants.iloc[rest_idx]["popularity"]
            boost = context_reward_boost(context, cuisine)
            discount_boost = 0.05 if has_discount else 0.0
            p = min(base + boost + discount_boost, 0.95)
            return float(rng.random() < p)

        bandit_reward = true_reward(bandit_pick)
        static_reward = true_reward(static_pick)

        bandit.update(bandit_pick, ctx_vec, bandit_reward)

        bandit_rewards.append(bandit_reward)
        static_rewards.append(static_reward)

    return {
        "bandit_cumulative_reward": np.cumsum(bandit_rewards),
        "static_cumulative_reward": np.cumsum(static_rewards),
        "bandit_mean_reward_last_20pct": float(np.mean(bandit_rewards[int(n_rounds * 0.8):])),
        "static_mean_reward_last_20pct": float(np.mean(static_rewards[int(n_rounds * 0.8):])),
    }