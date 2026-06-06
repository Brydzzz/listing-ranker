from single_lgbm_run import RunConfig, train_and_eval_lgbm


def main():
    lgbm_no_emb_params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 10,
        "metric": "ndcg",
        "learning_rate": 0.01,
        "num_leaves": 36,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "num_threads": -1,
        "eval_at": [5, 6, 10],
    }
    lgbm_emb_params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 10,
        "metric": "ndcg",
        "learning_rate": 0.01,
        "num_leaves": 62,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "num_threads": -1,
        "eval_at": [5, 6, 10],
    }
    lgbm_emb_pca_params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 10,
        "metric": "ndcg",
        "learning_rate": 0.01,
        "num_leaves": 43,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "num_threads": -1,
        "eval_at": [5, 6, 10],
    }

    configs = [
        # baseline - host response rate, host acceptance rate, host is superhost, host has profile pifc, host identity verified
        # bathrooms, bedrooms, beds, price, license, host response time, property type, room type
        RunConfig(lgbm_params=lgbm_no_emb_params),
        # no embeddings configs
        RunConfig(lgbm_params=lgbm_no_emb_params, exclude_amenities=False),
        RunConfig(lgbm_params=lgbm_no_emb_params, relative_price=True),
        RunConfig(lgbm_params=lgbm_no_emb_params, oversampling_factor=0.8),
        RunConfig(lgbm_params=lgbm_no_emb_params, oversampling_factor=0.4),
        # embeddings configs
        RunConfig(
            lgbm_params=lgbm_emb_params, emb_cols=["name", "description"]
        ),
        RunConfig(lgbm_params=lgbm_emb_params, emb_cols=["name"]),
        RunConfig(lgbm_params=lgbm_emb_params, emb_cols=["description"]),
        # embedding but with pca
        RunConfig(
            lgbm_params=lgbm_emb_pca_params,
            emb_cols=["name", "description"],
            pca_components=32,
        ),
        RunConfig(
            lgbm_params=lgbm_emb_pca_params,
            emb_cols=["name"],
            pca_components=32,
        ),
        RunConfig(
            lgbm_params=lgbm_emb_pca_params,
            emb_cols=["description"],
            pca_components=32,
        ),
    ]

    for i, c in enumerate(configs):
        print(f"CONFIG {i+1}/{len(configs)}")
        train_and_eval_lgbm(run_config=c, force_dataset_build=True)


if __name__ == "__main__":
    main()
