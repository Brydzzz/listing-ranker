import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CRITICAL GPU ENGINE HARDWARE INITIALIZATION
# ============================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🌲 Hardware engine configured. Active GPU device: {gpus}")
    except RuntimeError as e:
        print(f"Hardware engine assignment failed: {e}")
else:
    print(
        "⚠️ WARNING: No GPU detected. Defaulting to fallback CPU execution pipeline."
    )


class EmbeddingLookup:
    def __init__(
        self,
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

    def fetch(self, listing_ids: np.ndarray, col: str) -> np.ndarray:
        indices = self.id_series[listing_ids.astype(np.int64)]
        return self.emb_views[col][indices].astype(np.float32)


EMB_COLS = ["name", "description"]
CHUNK_SIZE = 500_000


def build_keras_ranker(max_list_size, num_features):
    inputs = tf.keras.Input(
        shape=(max_list_size, num_features),
        dtype=tf.float32,
        name="features_input",
    )
    mask = tf.keras.Input(
        shape=(max_list_size,), dtype=tf.bool, name="mask_input"
    )

    # --- STAGE 1: DEEP RESIDUAL INDEPENDENT DOCUMENT ENCODING ---
    x = tf.keras.layers.Dense(256, activation="swish")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)

    x_skip = x
    x = tf.keras.layers.Dense(256, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.LayerNormalization()(x)

    # --- STAGE 2: MASKED GLOBAL SLATE CONTEXT AGGREGATION ---
    def compute_slate_context(args):
        doc_feats, m = args
        mask_expanded = tf.cast(tf.expand_dims(m, -1), tf.float32)
        valid_counts = (
            tf.reduce_sum(mask_expanded, axis=1, keepdims=True) + 1e-9
        )
        slate_sum = tf.reduce_sum(
            doc_feats * mask_expanded, axis=1, keepdims=True
        )
        slate_mean = slate_sum / valid_counts
        ones_broadcast = tf.ones_like(doc_feats[:, :, :1])
        return slate_mean * ones_broadcast

    slate_context = tf.keras.layers.Lambda(
        compute_slate_context, name="slate_context_pooling"
    )([x, mask])

    combined_features = tf.keras.layers.Concatenate()([x, slate_context])
    combined_features = tf.keras.layers.Dense(128, activation="swish")(
        combined_features
    )
    combined_features = tf.keras.layers.LayerNormalization()(combined_features)

    # --- STAGE 3: MULTI-HEAD CROSS-DOCUMENT ATTENTION ---
    attended = tfr.keras.layers.DocumentInteractionAttention(
        num_heads=4,
        head_size=32,
        num_layers=2,
        dropout=0.2,
    )((combined_features, mask))

    # --- STAGE 4: FINAL SCORING LAYER ---
    scores = tf.keras.layers.Dense(64, activation="swish")(attended)
    scores = tf.keras.layers.Dropout(0.3)(scores)
    scores = tf.keras.layers.Dense(1)(scores)
    scores = tf.keras.layers.Reshape((max_list_size,))(scores)

    return tf.keras.Model(inputs=[inputs, mask], outputs=scores)


def custom_softmax_ranking_loss(y_true, y_pred):
    is_valid = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    y_true_clean = tf.where(
        tf.equal(y_true, -1.0), tf.zeros_like(y_true), y_true
    )

    mask_large_negative = (1.0 - is_valid) * -1e9
    y_pred_masked = y_pred + mask_large_negative

    target_probs = y_true_clean / (
        tf.reduce_sum(y_true_clean, axis=-1, keepdims=True) + 1e-9
    )

    log_probs = tf.nn.log_softmax(y_pred_masked, axis=-1)
    loss_per_session = -tf.reduce_sum(target_probs * log_probs, axis=-1)

    has_target = tf.cast(tf.reduce_sum(y_true_clean, axis=-1) > 0, tf.float32)
    return tf.reduce_sum(loss_per_session * has_target) / (
        tf.reduce_sum(has_target) + 1e-9
    )


def load_and_split_data(rows_limit: int | None = None):
    print("Reading combined.parquet...")

    if rows_limit:
        from fastparquet import ParquetFile

        pf = ParquetFile("combined.parquet")
        data = pf.head(rows_limit)
    else:
        data = pd.read_parquet("combined.parquet")

    data = data.dropna(subset=["booked", "session_id", "listing_id"])

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

    data = data.select_dtypes(
        include=[
            "int",
            "float",
            "bool",
            "int32",
            "int64",
            "float32",
            "float64",
        ]
    )

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

    removed_cols = [
        "user_id",
        "id",
        "host_id",
        "estimated_revenue_l365d",
        "scrape_id",
        "number_of_reviews",
        "reviews_per_month",
        "latitude",
        "longitude",
        "number_of_reviews_ltm",
        "number_of_reviews_ly",
        "availability_365",
        "availability_30",
        "availability_eoy",
        "availability_60",
        "availability_90",
        "estimated_occupancy_l365d",
        "number_of_reviews_l30d",
        "host_total_listings_count",
        "maximum_nights",
        "minimum_maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_maximum_nights",
        "maximum_minimum_nights",
        "maximum_nights_avg_ntm",
        "host_listings_count",
        "minimum_nights",
        "minimum_minimum_nights",
    ]
    data = data.drop(columns=[c for c in removed_cols if c in data.columns])

    unique_booked_listings = data["booked_listing_id"].unique()
    train_val_booked, test_booked = train_test_split(
        unique_booked_listings, test_size=0.2, random_state=42
    )
    train_booked, val_booked = train_test_split(
        train_val_booked, test_size=0.15, random_state=42
    )

    train_data = data[
        data["booked_listing_id"].isin(train_booked)
    ].sort_values("session_id")
    val_data = data[data["booked_listing_id"].isin(val_booked)].sort_values(
        "session_id"
    )
    test_data = data[data["booked_listing_id"].isin(test_booked)].sort_values(
        "session_id"
    )

    train_groups = train_data.groupby("session_id", sort=False).size().values
    val_groups = val_data.groupby("session_id", sort=False).size().values
    test_groups = test_data.groupby("session_id", sort=False).size().values

    cols_to_drop = [
        "booked",
        "session_id",
        "host_days_active",
        "listing_id",
        "booked_listing_id",
    ] + [c for c in naive_cols if c in train_data.columns]

    tabular_features = [c for c in train_data.columns if c not in cols_to_drop]
    print("\n" + "=" * 50)
    print(f"TF NN Tabular Features ({len(tabular_features)}):")
    print(tabular_features)
    print(f"\nTF NN Embedding Features ({len(EMB_COLS)}):")
    print(EMB_COLS)
    print("=" * 50 + "\n")

    X_train_raw = (
        train_data.drop(columns=cols_to_drop)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    X_val_raw = (
        val_data.drop(columns=cols_to_drop)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    X_test_raw = (
        test_data.drop(columns=cols_to_drop)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    non_constant_cols = [
        c for c in X_train_raw.columns if X_train_raw[c].nunique() > 1
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train_raw[non_constant_cols]
    ).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw[non_constant_cols]).astype(
        np.float32
    )
    X_test_scaled = scaler.transform(X_test_raw[non_constant_cols]).astype(
        np.float32
    )

    y_train = train_data["booked"].to_numpy(dtype=np.float32)
    y_val = val_data["booked"].to_numpy(dtype=np.float32)
    y_test = test_data["booked"].to_numpy(dtype=np.float32)

    train_ids = train_data["listing_id"].to_numpy()
    val_ids = val_data["listing_id"].to_numpy()
    test_ids = test_data["listing_id"].to_numpy()

    return (
        X_train_scaled,
        y_train,
        train_ids,
        train_groups,
        X_val_scaled,
        y_val,
        val_ids,
        val_groups,
        X_test_scaled,
        y_test,
        test_ids,
        test_groups,
        test_data,
        naive_cols,
        scaler,
    )


def build_tf_dataset(
    X_tab: np.ndarray,
    y: np.ndarray,
    listing_ids: np.ndarray,
    groups: np.ndarray,
    lookup: EmbeddingLookup,
    max_list_size: int,
    batch_size: int,
    is_training: bool = True,
):
    dummy_emb = [lookup.fetch(listing_ids[0], col) for col in EMB_COLS]
    emb_dim = sum(e.shape[0] for e in dummy_emb)
    num_features = X_tab.shape[1] + emb_dim

    def session_generator():
        idx = 0
        for g_size in groups:
            size = min(g_size, max_list_size)

            x_slice = X_tab[idx : idx + size]
            y_slice = y[idx : idx + size]
            ids_slice = listing_ids[idx : idx + size]
            idx += g_size

            emb_parts = [lookup.fetch(ids_slice, col) for col in EMB_COLS]

            if len(x_slice) > 0:
                x_combined = np.hstack([x_slice] + emb_parts)
            else:
                x_combined = np.empty((0, num_features), dtype=np.float32)

            pad_len = max_list_size - size
            if pad_len > 0:
                x_padded = np.pad(
                    x_combined, ((0, pad_len), (0, 0)), mode="constant"
                )
                y_padded = np.pad(
                    y_slice,
                    (0, pad_len),
                    mode="constant",
                    constant_values=-1.0,
                )
                mask_padded = np.pad(
                    np.ones(size, dtype=np.bool_),
                    (0, pad_len),
                    mode="constant",
                )
            else:
                x_padded = x_combined
                y_padded = y_slice
                mask_padded = np.ones(size, dtype=np.bool_)

            yield (
                {"features_input": x_padded, "mask_input": mask_padded},
                y_padded,
            )

    dataset = tf.data.Dataset.from_generator(
        session_generator,
        output_signature=(
            {
                "features_input": tf.TensorSpec(
                    shape=(max_list_size, num_features), dtype=tf.float32
                ),
                "mask_input": tf.TensorSpec(
                    shape=(max_list_size,), dtype=tf.bool
                ),
            },
            tf.TensorSpec(shape=(max_list_size,), dtype=tf.float32),
        ),
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, num_features


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
        "NDCG@10": round(calculate_ndcg_at_k(df, 10), 4),
        "MRR": round(mrr, 4),
        "HitRate@1": f"{hit_rate_1:.2%}",
        "HitRate@5": f"{hit_rate_5:.2%}",
    }


def main():
    lookup = EmbeddingLookup()

    (
        X_train_tab,
        y_train,
        train_ids,
        train_groups,
        X_val_tab,
        y_val,
        val_ids,
        val_groups,
        X_test_tab,
        y_test,
        test_ids,
        test_groups,
        test_data,
        naive_cols,
        scaler,
    ) = load_and_split_data(rows_limit=4_200_000)


    print("\n" + "=" * 40)
    print("DATASET SPLIT SIZES")
    print("=" * 40)
    print(f"Training:   {len(X_train_tab):>9,} single sessions | {len(train_groups):>7,} lists")
    print(f"Validation: {len(X_val_tab):>9,} single sessions | {len(val_groups):>7,} lists")
    print(f"Testing:    {len(X_test_tab):>9,} single sessions | {len(test_groups):>7,} lists")
    print("=" * 40 + "\n")

    max_list_size = int(max(max(train_groups), max(test_groups)))
    eval_df = test_data.copy().reset_index(drop=True)

    print("\nPreparing Lazy-Loading TensorFlow Datasets...")
    train_ds, num_features = build_tf_dataset(
        X_train_tab,
        y_train,
        train_ids,
        train_groups,
        lookup,
        max_list_size,
        batch_size=64,
        is_training=True,
    )

    val_ds, _ = build_tf_dataset(
        X_val_tab,
        y_val,
        val_ids,
        val_groups,
        lookup,
        max_list_size,
        batch_size=64,
        is_training=False,
    )

    test_ds, _ = build_tf_dataset(
        X_test_tab,
        y_test,
        test_ids,
        test_groups,
        lookup,
        max_list_size,
        batch_size=64,
        is_training=False,
    )
    print(
        f"Dataset Pipeline Prepared: max_list_size={max_list_size}, num_features={num_features}"
    )

    model = build_keras_ranker(max_list_size, num_features)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_softmax_ranking_loss,
    )


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    print("Training TF-Ranking Neural Model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        callbacks=[early_stopping],
        verbose=1,
    )

    print("Generating Neural Predictions...")
    preds = model.predict(test_ds)

    flat_preds = []
    for i, g_size in enumerate(test_groups):
        flat_preds.extend(preds[i, :g_size])
    eval_df["neural_score"] = flat_preds

    existing_rating_cols = [c for c in naive_cols if c in eval_df.columns]
    eval_df["naive_score"] = (
        eval_df[existing_rating_cols].mean(axis=1).fillna(0)
    )

    booked_items = (
        eval_df[eval_df["booked"] == 1]
        .copy()
        .dropna(subset=["host_days_active"])
    )
    naive_res, neural_res = [], []
    percentiles = np.arange(0.2, 1.01, 0.2)

    for p in percentiles:
        threshold = booked_items["host_days_active"].quantile(p)
        cumulative_sessions = booked_items[
            booked_items["host_days_active"] <= threshold
        ]["session_id"]
        eval_subset = eval_df[eval_df["session_id"].isin(cumulative_sessions)]

        for name, col, res_list in [
            ("Naive", "naive_score", naive_res),
            ("NeuralNet", "neural_score", neural_res),
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
        "NDCG@10",
        "MRR",
        "HitRate@1",
        "HitRate@5",
    ]
    print("\n" + "=" * 60 + "\nNAIVE EVALUATION\n" + "=" * 60)
    print(pd.DataFrame(naive_res)[cols].to_string(index=False))
    print("\n" + "=" * 60 + "\nTF-RANKING NET EVALUATION\n" + "=" * 60)
    print(pd.DataFrame(neural_res)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
