"""
Group project by Khai Kieu
Phase 2: data preparation, baselines and svd matrix factorization on movielens 1m."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed since we only save figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).parent / "ml-1m" / "ml-1m"
RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
GROUP_BINS = [0, 19, 100, np.inf]  # <20, 20-100, >100 training ratings
GROUP_LABELS = ["<20", "20-100", ">100"]
FACTOR_GRID = [10, 20, 50, 100]
N_NEIGHBORS = 30


def load_and_clean(data_dir: Path) -> pd.DataFrame:
    """load ratings, run quality checks, and add contiguous user/item indices."""
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        data_dir / "ratings.dat", sep="::", names=columns, engine="python"
    )
    n_raw = len(ratings)
    ratings = ratings.dropna()
    ratings = ratings[ratings["rating"].between(1, 5)]
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"])
    print(f"rows: {n_raw} raw -> {len(ratings)} after cleaning")

    # matrices need 0..n-1 indices; ids alone are not contiguous
    ratings = ratings.assign(
        u=pd.factorize(ratings["user_id"])[0],
        i=pd.factorize(ratings["movie_id"])[0],
    )
    return ratings.reset_index(drop=True)


def describe_data(ratings: pd.DataFrame) -> None:
    """print summary statistics and save the eda figure."""
    n_users, n_items = ratings["u"].nunique(), ratings["i"].nunique()
    sparsity = 1 - len(ratings) / (n_users * n_items)
    per_user = ratings.groupby("u").size()
    print(f"users: {n_users}, movies: {n_items}, sparsity: {sparsity:.2%}")
    print(f"ratings per user (min/median/max): "
          f"{per_user.min()}/{per_user.median():.0f}/{per_user.max()}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    ratings["rating"].value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set(title="rating distribution", xlabel="stars", ylabel="count")
    axes[1].hist(per_user, bins=50, log=True)
    axes[1].set(title="ratings per user", xlabel="ratings", ylabel="users (log)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda.png", dpi=200)


def split_per_user(ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """split each user's ratings 80/10/10 with a fixed seed."""
    shuffled = ratings.sample(frac=1.0, random_state=SEED)
    by_user = shuffled.groupby("u")
    # relative position of each rating inside its user's shuffled list
    position = by_user.cumcount() / by_user["rating"].transform("size")
    labels = np.select(
        [position < TRAIN_FRAC, position < TRAIN_FRAC + VAL_FRAC],
        ["train", "val"],
        default="test",
    )
    return {name: shuffled[labels == name] for name in ("train", "val", "test")}


def mean_predictions(train: pd.DataFrame, test: pd.DataFrame, key: str) -> np.ndarray:
    """predict the mean rating per user ('u') or item ('i'); global mean if unseen."""
    means = train.groupby(key)["rating"].mean()
    return test[key].map(means).fillna(train["rating"].mean()).to_numpy()


def svd_predictions(
    train: pd.DataFrame, test: pd.DataFrame, shape: tuple[int, int], n_factors: int
) -> np.ndarray:
    """svd-style mf: truncated svd of the user-mean-centered rating matrix."""
    user_mean = train.groupby("u")["rating"].mean()
    centered = train["rating"] - train["u"].map(user_mean)
    matrix = csr_matrix((centered, (train["u"], train["i"])), shape=shape)

    # missing entries count as 0 (= the user's mean) after centering
    user_f, sigma, item_f_t = svds(matrix, k=n_factors)
    scores = np.einsum(
        "ij,ij->i", user_f[test["u"]] * sigma, item_f_t.T[test["i"]]
    )
    base = test["u"].map(user_mean).fillna(train["rating"].mean()).to_numpy()
    return np.clip(base + scores, 1, 5)


def knn_predictions(
    train: pd.DataFrame, test: pd.DataFrame, shape: tuple[int, int], k: int
) -> np.ndarray:
    """item-based knn: item mean plus similarity-weighted neighbor residuals."""
    global_mean = train["rating"].mean()
    item_mean = (
        train.groupby("i")["rating"].mean().reindex(range(shape[1])).fillna(global_mean)
    ).to_numpy()
    residual = train["rating"].to_numpy() - item_mean[train["i"]]
    by_user = csr_matrix((residual, (train["u"], train["i"])), shape=shape)

    # cosine similarity between item columns of the residual matrix
    item_vectors = normalize(by_user.T.tocsr())
    sim = (item_vectors @ item_vectors.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)

    preds = np.empty(len(test))
    for n, (u, i) in enumerate(zip(test["u"], test["i"])):
        start, end = by_user.indptr[u], by_user.indptr[u + 1]
        rated, resid = by_user.indices[start:end], by_user.data[start:end]
        sims = sim[i, rated]
        if len(sims) > k:  # keep only the k most similar rated items
            top = np.argpartition(-sims, k)[:k]
            sims, resid = sims[top], resid[top]
        positive = sims > 0
        denom = sims[positive].sum()
        adjust = (sims[positive] * resid[positive]).sum() / denom if denom > 0 else 0.0
        preds[n] = item_mean[i] + adjust
    return np.clip(preds, 1, 5)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate(
    name: str, test: pd.DataFrame, preds: np.ndarray, groups: pd.Series
) -> dict:
    """overall rmse/mae plus rmse per user-activity group."""
    y = test["rating"].to_numpy()
    row = {"model": name, "rmse": rmse(y, preds), "mae": mae(y, preds)}
    for label in GROUP_LABELS:
        mask = (groups == label).to_numpy()
        row[f"rmse {label}"] = rmse(y[mask], preds[mask])
    return row


def main() -> None:
    ratings = load_and_clean(DATA_DIR)
    describe_data(ratings)
    shape = (ratings["u"].nunique(), ratings["i"].nunique())
    splits = split_per_user(ratings)
    train, val, test = splits["train"], splits["val"], splits["test"]
    print({name: len(df) for name, df in splits.items()})

    # tune the number of factors on validation only, never on test
    val_rmse = {
        f: rmse(val["rating"].to_numpy(), svd_predictions(train, val, shape, f))
        for f in FACTOR_GRID
    }
    best_f = min(val_rmse, key=val_rmse.get)
    print(f"validation rmse by n_factors: {val_rmse} -> best {best_f}")

    # activity group is defined by the number of training ratings
    train_counts = train.groupby("u").size()
    groups = pd.cut(
        test["u"].map(train_counts), bins=GROUP_BINS, labels=GROUP_LABELS
    )

    global_pred = np.full(len(test), train["rating"].mean())
    models = {
        "global mean": global_pred,
        "user mean": mean_predictions(train, test, "u"),
        "item mean": mean_predictions(train, test, "i"),
        f"item knn (k={N_NEIGHBORS})": knn_predictions(train, test, shape, N_NEIGHBORS),
        f"svd mf (f={best_f})": svd_predictions(train, test, shape, best_f),
    }
    table = pd.DataFrame(
        [evaluate(name, test, preds, groups) for name, preds in models.items()]
    )
    print(table.round(4).to_string(index=False))
    table.round(4).to_csv(RESULTS_DIR / "phase2_results.csv", index=False)
    print(groups.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()