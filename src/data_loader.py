"""
Loads previously-generated synthetic data from disk.

If the expected CSVs aren't present, load_all() raises a clear error telling
you to run the generation script first, rather than silently generating data
as a fallback.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

REQUIRED_FILES = ["users.csv", "restaurants.csv", "interactions.csv"]


def _check_data_exists(data_dir: str):
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Missing generated data file(s): " + ", ".join(missing) + "\n\n"
            "Data generation is a separate step from the recommender pipeline.\n"
            "Run this first, then re-run your pipeline script:\n\n"
            "    python3 src/data_gen.py\n\n"
            f"Expected data directory: {data_dir}"
        )


def load_users(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "users.csv"))


def load_restaurants(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "restaurants.csv"))


def load_interactions(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "interactions.csv"))


def load_all(data_dir: str = DATA_DIR):
    """
    Load users, restaurants, and interactions from disk.
    Fails loudly with clear instructions if `data_gen.py` hasn't been run yet.
    """
    _check_data_exists(data_dir)
    users = load_users(data_dir)
    restaurants = load_restaurants(data_dir)
    interactions = load_interactions(data_dir)
    return users, restaurants, interactions