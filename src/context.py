import numpy as np

CUISINES = [
    "pizza", "sushi", "burgers", "vegan", "indian",
    "thai", "italian", "chinese", "kebab", "salads",
]

WEATHER_OPTIONS = ["sunny", "rainy", "cold", "hot"]
TIME_OF_DAY_OPTIONS = ["breakfast", "lunch", "afternoon", "dinner", "late_night"]

# Cuisine affinity boost by weather -- e.g. rainy/cold days favor warm comfort food
WEATHER_CUISINE_BOOST = {
    "rainy":  {"indian": 0.35, "thai": 0.30, "chinese": 0.25, "italian": 0.15},
    "cold":   {"indian": 0.30, "pizza": 0.20, "burgers": 0.15, "italian": 0.20},
    "hot":    {"salads": 0.35, "sushi": 0.25, "vegan": 0.20},
    "sunny":  {"salads": 0.15, "burgers": 0.10, "kebab": 0.10},
}

TIME_CUISINE_BOOST = {
    "late_night": {"burgers": 0.30, "kebab": 0.30, "pizza": 0.20},
    "lunch":      {"salads": 0.15, "sushi": 0.15},
    "breakfast":  {"salads": 0.05},
    "dinner":     {"italian": 0.10, "indian": 0.10, "thai": 0.10},
    "afternoon":  {},
}


def sample_context(rng: np.random.Generator):
    """Sample one realistic real-time request context."""
    weather = rng.choice(WEATHER_OPTIONS, p=[0.4, 0.25, 0.2, 0.15])
    time_of_day = rng.choice(TIME_OF_DAY_OPTIONS, p=[0.1, 0.25, 0.15, 0.35, 0.15])
    has_discount_map = {}  # filled in per-restaurant by caller if needed
    return {"weather": weather, "time_of_day": time_of_day}


def context_reward_boost(context: dict, cuisine: str) -> float:
    """Additive boost to true relevance/conversion probability from context, for a given cuisine."""
    boost = 0.0
    boost += WEATHER_CUISINE_BOOST.get(context["weather"], {}).get(cuisine, 0.0)
    boost += TIME_CUISINE_BOOST.get(context["time_of_day"], {}).get(cuisine, 0.0)
    return boost


def context_to_vector(context: dict, has_discount: bool, is_repeat_customer: bool) -> np.ndarray:
    """One-hot encode context into a fixed-length feature vector for the bandit/MMoE models."""
    weather_oh = [1.0 if context["weather"] == w else 0.0 for w in WEATHER_OPTIONS]
    time_oh = [1.0 if context["time_of_day"] == t else 0.0 for t in TIME_OF_DAY_OPTIONS]
    extra = [1.0 if has_discount else 0.0, 1.0 if is_repeat_customer else 0.0]
    return np.array(weather_oh + time_oh + extra, dtype=np.float32)


CONTEXT_VECTOR_DIM = len(WEATHER_OPTIONS) + len(TIME_OF_DAY_OPTIONS) + 2