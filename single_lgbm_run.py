import gc
import os
import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass
class RunConfig:
    emb_cols: list[str] = field(default_factory=list)
    chunk_size: int = 100_000
    data_file: str = "combined_no_ab.parquet"
    pca_components: int | None = None
    oversampling_factor: float | None = None
    relative_price: bool = False
    exclude_amenities: bool = True
    lgbm_params: dict[str, any] = field(default_factory=dict)

    def model_fname(self) -> str:
        # Define the exact keys you want to exclude from the filename
        skip_params = {"eval_at", "seed", "num_threads", "objective"}

        param_parts = []
        for k, v in self.lgbm_params.items():
            if k in skip_params:
                continue

            # If the value is a list or tuple, convert it to a flat string separated by hyphens
            if isinstance(v, (list, tuple)):
                v_str = "-".join(map(str, v))
            else:
                v_str = str(v)

            param_parts.append(f"{k}_{v_str}")

        param_str = "_".join(param_parts)
        emb_str = "-".join(self.emb_cols) if self.emb_cols else "none"

        return (
            f"emb_{emb_str}_pca_{self.pca_components}"
            + f"_osf_{self.oversampling_factor if self.oversampling_factor else 0}"
            + f"_rp_{self.relative_price}_excamen_{self.exclude_amenities}"
            + (f"_{param_str}" if param_str else "")
        )


class EmbeddingLookup:
    def __init__(
        self,
        pca_components: int | None,
        embeddings_path: str = "embeddings.npz",
        ids_path: str = "listing_ids.npy",
    ):
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.emb_views = {
            col: self.embeddings[col] for col in self.embeddings.files
        }

        listing_ids = np.load(ids_path)

        self.id_series = pd.Series(
            np.arange(len(listing_ids), dtype=np.int64),
            index=listing_ids.astype(np.int64),
        )

        self.pcas: dict[str, PCA] = {}
        if pca_components is not None:
            for col, emb in self.emb_views.items():
                pca = PCA(n_components=pca_components)
                pca.fit(emb.astype(np.float32))
                self.pcas[col] = pca

    def fetch(self, listing_ids: np.ndarray, col: str) -> np.ndarray:
        indices = self.id_series[listing_ids.astype(np.int64)]
        vectors = self.emb_views[col][indices].astype(np.float32)
        if col in self.pcas:
            vectors = self.pcas[col].transform(vectors)
        return vectors


class FeatureSequence(lgb.Sequence):
    def __init__(
        self,
        data: pd.DataFrame,
        lookup: EmbeddingLookup,
        tabular_features: list[str],
        emb_cols: list,
        batch_size: int = 100_000,
    ):
        self.length = len(data)
        self.listing_ids = data["listing_id"].to_numpy()

        self.X_tab = data[tabular_features].to_numpy(dtype=np.float32)

        self.lookup = lookup
        self.emb_cols = emb_cols
        self.batch_size = batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        is_single = isinstance(idx, (int, np.integer))
        if is_single:
            ids = self.listing_ids[[idx]]
            X = self.X_tab[[idx]]
        else:
            ids = self.listing_ids[idx]
            X = self.X_tab[idx]

        emb_parts = [self.lookup.fetch(ids, col) for col in self.emb_cols]
        combined = np.hstack([X] + emb_parts)

        combined_double = combined.astype(np.float64)
        return combined_double[0] if is_single else combined_double


def load_and_split_data(
    lookup: EmbeddingLookup,
    data_file: str,
    emb_cols: list[str],
    chunk_size: int,
    oversampling_factor: float | None,
    relative_price: bool,
    exclude_amenities: bool,
    rows_limit: int | None = None,
    tail: bool = False,
):
    print(f"Reading {data_file}...")

    if rows_limit:
        # polars is reading parquet faster with less ram, but for compatiblity with rest of the code we are converting to pandas
        import polars as pl

        if tail:
            data = (
                pl.scan_parquet(data_file)
                .tail(rows_limit)
                .collect()
                .to_pandas()
            )
        else:
            data = pl.read_parquet(data_file, n_rows=rows_limit).to_pandas()
    else:
        data = pd.read_parquet(data_file)

    data = data.dropna(subset=["booked", "session_id", "listing_id"])

    print("Combined data additional processing...")
    session_booked_map = data[data["booked"] == 1].drop_duplicates(
        "session_id"
    )[["session_id", "listing_id"]]
    session_booked_map = session_booked_map.rename(
        columns={"listing_id": "booked_listing_id"}
    )
    data = data.merge(session_booked_map, on="session_id", how="inner")

    data["host_since"] = pd.to_datetime(data["host_since"], errors="coerce")
    reference_date = data["host_since"].max()
    data["host_days_active"] = (reference_date - data["host_since"]).dt.days

    meta_cols = [
        "session_id",
        "host_days_active",
        "listing_id",
        "booked_listing_id",
    ]
    meta_data = data[meta_cols]

    data = data.select_dtypes(include=["number", "bool"])

    for col in meta_cols:
        if col not in data.columns:
            data[col] = meta_data[col]

    naive_cols = [
        "review_scores_value",
        "review_scores_cleanliness",
        "review_scores_location",
        "review_scores_accuracy",
        "review_scores_rating",
        "review_scores_checkin",
        "review_scores_communication",
    ]

    if relative_price:
        print("Calculative realtive price features for sessions...")

        price_col = "price"

        session_price_stats = (
            data.groupby("session_id")[price_col]
            .agg(["mean", "median", "std", "min"])
            .reset_index()
        )
        session_price_stats.columns = [
            "session_id",
            "session_price_mean",
            "session_price_median",
            "session_price_std",
            "session_price_min",
        ]
        data = data.merge(session_price_stats, on="session_id", how="left")

        epsilon = 1e-6

        data["price_diff_from_mean"] = (
            data[price_col] - data["session_price_mean"]
        )
        data["price_ratio_to_mean"] = data[price_col] / (
            data["session_price_mean"] + epsilon
        )
        data["price_ratio_to_median"] = data[price_col] / (
            data["session_price_median"] + epsilon
        )
        data["price_diff_from_min"] = (
            data[price_col] - data["session_price_min"]
        )
        data["price_z_score_session"] = (
            data[price_col] - data["session_price_mean"]
        ) / (data["session_price_std"] + epsilon)

        data = data.drop(
            columns=[
                "session_price_mean",
                "session_price_median",
                "session_price_std",
                "session_price_min",
                price_col,
            ]
        )

    print("Splitting data...")

    unique_booked_listings = data["booked_listing_id"].unique()

    train_val_booked, test_booked = train_test_split(
        unique_booked_listings, test_size=0.2, random_state=42
    )
    train_booked, eval_booked = train_test_split(
        train_val_booked, test_size=0.15, random_state=42
    )

    train_data = data[
        data["booked_listing_id"].isin(train_booked)
    ].sort_values("session_id")
    eval_data = data[data["booked_listing_id"].isin(eval_booked)].sort_values(
        "session_id"
    )
    test_data = data[data["booked_listing_id"].isin(test_booked)].sort_values(
        "session_id"
    )

    del data, meta_data
    gc.collect()

    if oversampling_factor:
        print("Oversampling top 20%...")

        threshold_20 = train_data["host_days_active"].quantile(0.2)
        new_host_sessions = train_data.loc[
            train_data["host_days_active"] <= threshold_20, "session_id"
        ].unique()
        n_sessions_to_sample = int(
            len(new_host_sessions) * oversampling_factor
        )
        sampled_sessions = np.random.default_rng(42).choice(
            new_host_sessions, size=n_sessions_to_sample, replace=True
        )

        new_ids = [f"os_{i}_{sid}" for i, sid in enumerate(sampled_sessions)]
        samples_df = pd.DataFrame(
            {"session_id": sampled_sessions, "new_session_id": new_ids}
        )

        oversampled = samples_df.merge(
            train_data, on="session_id", how="inner"
        )
        oversampled["session_id"] = oversampled["new_session_id"]
        oversampled = oversampled.drop(columns=["new_session_id"])

        n_original = len(train_data)
        train_data = pd.concat(
            [train_data, oversampled], ignore_index=True
        ).sort_values("session_id")
        n_oversampled = len(oversampled)
        print(
            f"Original rows: {n_original}, Oversampled rows: {n_oversampled}, Ratio: {n_oversampled / n_original:.2f}"
        )

    train_groups = train_data.groupby("session_id", sort=False).size().values
    eval_groups = eval_data.groupby("session_id", sort=False).size().values

    cols_to_drop = [
        "booked",
        "session_id",
        "host_days_active",
        "listing_id",
        "booked_listing_id",
    ] + [c for c in naive_cols if c in train_data.columns]

    if exclude_amenities:
        cols_to_drop += [
            c for c in train_data.columns if c.startswith("amenity_")
        ]

    tabular_features = [c for c in train_data.columns if c not in cols_to_drop]
    print("\n" + "=" * 50)
    print(f"LightGBM Tabular Features ({len(tabular_features)}):")
    print(tabular_features)
    print(f"\nLightGBM Embedding Features ({len(emb_cols)}):")
    print(emb_cols)
    print("=" * 50 + "\n")

    print("Building LightGBM Sequence datasets...")
    train_seq = FeatureSequence(
        data=train_data,
        lookup=lookup,
        tabular_features=tabular_features,
        emb_cols=emb_cols,
        batch_size=chunk_size,
    )
    eval_seq = FeatureSequence(
        data=eval_data,
        lookup=lookup,
        tabular_features=tabular_features,
        emb_cols=emb_cols,
        batch_size=chunk_size,
    )

    y_train = train_data["booked"].to_numpy()
    y_eval = eval_data["booked"].to_numpy()

    del train_data, eval_data
    gc.collect()

    feature_names = list(tabular_features)
    for col in emb_cols:
        if col in lookup.pcas:
            dim = lookup.pcas[col].n_components
            feature_names.extend([f"{col}_pca_{i}" for i in range(dim)])
        else:
            dim = lookup.emb_views[col].shape[1]
            feature_names.extend([f"{col}_emb_{i}" for i in range(dim)])

    clean_feature_names = [f.replace(":", "_") for f in feature_names]

    train_ds = lgb.Dataset(
        train_seq,
        label=y_train,
        group=train_groups,
        free_raw_data=True,
        feature_name=clean_feature_names,
    )

    eval_ds = lgb.Dataset(
        eval_seq,
        label=y_eval,
        group=eval_groups,
        feature_name=clean_feature_names,
        reference=train_ds,
        free_raw_data=True,
    )

    print("Constructing train_ds for binary save...")
    train_ds.construct()
    if os.path.exists("train_dataset.bin"):
        os.remove("train_dataset.bin")
    train_ds.save_binary("train_dataset.bin")
    print("train_dataset.bin created")

    print("Constructing eval_ds for binary save...")
    eval_ds.construct()
    if os.path.exists("eval_dataset.bin"):
        os.remove("eval_dataset.bin")
    eval_ds.save_binary("eval_dataset.bin")
    print("eval_dataset.bin created")

    print("Saving test data and column metadata...")
    test_data.to_parquet("test_data_cache.parquet")
    with open("metadata_cache.pkl", "wb") as f:
        pickle.dump(
            {"naive_cols": naive_cols, "cols_to_drop": cols_to_drop}, f
        )

    return (
        train_ds,
        eval_ds,
        test_data,
        naive_cols,
        cols_to_drop,
    )


def get_cached_data_or_build(
    lookup: EmbeddingLookup,
    run_config: RunConfig,
    force_build: bool,
    rows_limit: int | None = None,
    tail: bool = False,
):
    cache_files = [
        "train_dataset.bin",
        "eval_dataset.bin",
        "test_data_cache.parquet",
        "metadata_cache.pkl",
    ]

    if all(os.path.exists(f) for f in cache_files) and not force_build:
        print("Found cached datasets, loading...")

        train_ds = lgb.Dataset("train_dataset.bin")
        eval_ds = lgb.Dataset("eval_dataset.bin", reference=train_ds)

        test_data = pd.read_parquet("test_data_cache.parquet")

        with open("metadata_cache.pkl", "rb") as f:
            meta = pickle.load(f)
            naive_cols = meta["naive_cols"]
            cols_to_drop = meta["cols_to_drop"]

        print("Cache loaded successfully.")
        return train_ds, eval_ds, test_data, naive_cols, cols_to_drop
    else:
        print("Cache missing or incomplete. Processing data from scratch...")
        return load_and_split_data(
            lookup=lookup,
            rows_limit=rows_limit,
            tail=tail,
            data_file=run_config.data_file,
            emb_cols=run_config.emb_cols,
            chunk_size=run_config.chunk_size,
            oversampling_factor=run_config.oversampling_factor,
            exclude_amenities=run_config.exclude_amenities,
            relative_price=run_config.relative_price,
        )


def evaluate_ranking_performance(df, score_column):
    sorted_df = df.sort_values(
        ["session_id", score_column], ascending=[True, False]
    )
    sorted_df["rank"] = sorted_df.groupby("session_id").cumcount() + 1

    booked_items = sorted_df[sorted_df["booked"] == 1]
    if len(booked_items) == 0:
        return None

    mrr = (1 / booked_items["rank"]).mean()
    hit_rate_1 = (booked_items["rank"] == 1).mean()
    hit_rate_5 = (booked_items["rank"] <= 5).mean()
    hit_rate_6 = (booked_items["rank"] <= 6).mean()

    def calculate_ndcg_at_k(df, k):
        top_k = sorted_df[sorted_df["rank"] <= k].copy()
        top_k["dcg"] = top_k["booked"] / np.log2(top_k["rank"] + 1)
        dcg_per_session = top_k.groupby("session_id")["dcg"].sum()

        ideal_df = df.sort_values(
            ["session_id", "booked"], ascending=[True, False]
        )
        ideal_df["ideal_rank"] = ideal_df.groupby("session_id").cumcount() + 1
        ideal_top_k = ideal_df[ideal_df["ideal_rank"] <= k].copy()

        ideal_top_k["idcg"] = ideal_top_k["booked"] / np.log2(
            ideal_top_k["ideal_rank"] + 1
        )
        idcg_per_session = ideal_top_k.groupby("session_id")["idcg"].sum()

        ndcg_per_session = dcg_per_session / idcg_per_session
        return ndcg_per_session.fillna(0).mean()

    return {
        "NDCG@5": round(calculate_ndcg_at_k(df, 5), 4),
        "NDCG@6": round(calculate_ndcg_at_k(df, 6), 4),
        "NDCG@10": round(calculate_ndcg_at_k(df, 10), 4),
        "MRR": round(mrr, 4),
        "HitRate@1": f"{hit_rate_1:.2%}",
        "HitRate@5": f"{hit_rate_5:.2%}",
        "HitRate@6": f"{hit_rate_6:.2%}",
    }


def train_and_eval_lgbm(
    run_config: RunConfig, force_dataset_build: bool = False
):
    lookup = EmbeddingLookup(pca_components=run_config.pca_components)
    (
        train_ds,
        eval_ds,
        test_data,
        naive_cols,
        cols_to_drop,
    ) = get_cached_data_or_build(
        lookup, run_config, force_build=force_dataset_build
    )

    print("Training LightGBM Lambdarank...")

    lgb_ranker = lgb.train(
        params=run_config.lgbm_params,
        train_set=train_ds,
        num_boost_round=1000,
        valid_sets=[train_ds, eval_ds],
        valid_names=["train", "eval"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)],
    )

    print("Saving model...")
    lgb_ranker.save_model(
        f"lgbm_model_{run_config.model_fname()}.txt",
        num_iteration=lgb_ranker.best_iteration,
    )

    print("Running Inference...")
    predictions = []

    tabular_features = [c for c in test_data.columns if c not in cols_to_drop]

    for start in range(0, len(test_data), run_config.chunk_size):
        chunk = test_data.iloc[start : start + run_config.chunk_size]
        listing_ids = chunk["listing_id"].to_numpy()

        X_chunk_array = chunk[tabular_features].to_numpy(dtype=np.float32)

        emb_parts = []
        for col in run_config.emb_cols:
            raw = lookup.fetch(listing_ids, col)
            emb_parts.append(raw)

        X_test_chunk = np.hstack([X_chunk_array] + emb_parts)

        chunk_preds = lgb_ranker.predict(
            X_test_chunk, num_iteration=lgb_ranker.best_iteration
        )
        predictions.append(chunk_preds)

    eval_cols = ["session_id", "booked", "host_days_active"] + [
        c for c in naive_cols if c in test_data.columns
    ]
    eval_df = test_data[eval_cols].copy().reset_index(drop=True)
    eval_df["lgb_score"] = np.concatenate(predictions)

    existing_rating_cols = [c for c in naive_cols if c in eval_df.columns]
    eval_df["naive_score"] = (
        eval_df[existing_rating_cols].mean(axis=1).fillna(0)
    )

    booked_items = (
        eval_df[eval_df["booked"] == 1]
        .copy()
        .dropna(subset=["host_days_active"])
    )
    naive_res, lgb_res = [], []
    percentiles = np.arange(0.2, 1.01, 0.2)

    for p in percentiles:
        threshold = booked_items["host_days_active"].quantile(p)
        cumulative_sessions = booked_items[
            booked_items["host_days_active"] <= threshold
        ]["session_id"]
        eval_subset = eval_df[eval_df["session_id"].isin(cumulative_sessions)]

        for name, col, res_list in [
            ("Naive", "naive_score", naive_res),
            ("LightGBM", "lgb_score", lgb_res),
        ]:
            metrics = evaluate_ranking_performance(eval_subset, col)
            if metrics:
                metrics["group"] = f"Top {int(round(p * 100))}%"
                metrics["num sessions"] = len(cumulative_sessions)
                res_list.append(metrics)

    cols = [
        "group",
        "num sessions",
        "NDCG@5",
        "NDCG@6",
        "NDCG@10",
        "MRR",
        "HitRate@1",
        "HitRate@5",
        "HitRate@6",
    ]
    print("\n" + "=" * 60 + "\nNAIVE EVALUATION\n" + "=" * 60)
    print(pd.DataFrame(naive_res)[cols].to_string(index=False))
    print("\n" + "=" * 60 + "\nLIGHTGBM EVALUATION\n" + "=" * 60)
    print(pd.DataFrame(lgb_res)[cols].to_string(index=False))


if __name__ == "__main__":
    # winning config
    params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 10,
        "metric": "ndcg",
        "learning_rate": 0.01,
        "num_leaves": 36,
        "feature_fraction": 0.6,
        "min_data_in_leaf": 200,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "num_threads": -1,
        "eval_at": [5, 6, 10],
    }
    run_config = RunConfig(
        lgbm_params=params, exclude_amenities=False, relative_price=True
    )
    train_and_eval_lgbm(run_config, force_dataset_build=True)
