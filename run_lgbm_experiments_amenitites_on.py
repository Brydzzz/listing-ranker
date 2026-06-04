from simple_model_only_lgbm import RunConfig, train_and_eval_lgbm


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

    configs = [
        # baseline - host response rate, host acceptance rate, host is superhost, host has profile pifc, host identity verified
        # bathrooms, bedrooms, beds, price, license, host response time, property type, room type
        RunConfig(lgbm_params=lgbm_no_emb_params),
        # with amenitites
        RunConfig(lgbm_params=lgbm_no_emb_params, exclude_amenities=False, relative_price=True),
        RunConfig(lgbm_params=lgbm_no_emb_params, exclude_amenities=False, relative_price=True, oversampling_factor=0.8),
    ]

    for i, c in enumerate(configs):
        print(f"CONFIG {i+1}/{len(configs)}")
        train_and_eval_lgbm(run_config=c, force_dataset_build=True)


if __name__ == "__main__":
    main()
