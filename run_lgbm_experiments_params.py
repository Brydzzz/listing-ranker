from sklearn.model_selection import ParameterGrid

from single_lgbm_run import RunConfig, train_and_eval_lgbm


def main():
    param_grid = {
        "objective": ["lambdarank"],
        "lambdarank_truncation_level": [10],
        "metric": ["ndcg"],
        "learning_rate": [0.01, 0.05],
        "num_leaves": [36, 45, 63, 127],
        "min_data_in_leaf": [200, 500, 1000],
        "feature_fraction": [0.6, 0.8],
        "lambda_l1": [0.1],
        "lambda_l2": [0.1],
        "seed": [42],
        "num_threads": [-1],
        "eval_at": [[5, 6, 10]],
    }

    grid = list(ParameterGrid(param_grid))

    print(f"Starting search with {len(grid)} configurations...")

    configs = []
    for params in grid:
        c = RunConfig(
            lgbm_params=params,
            exclude_amenities=False,
            relative_price=True,
            emb_cols=[],
        )
        configs.append(c)

    for i, c in enumerate(configs):
        print("\n" + "=" * 50)
        print(f"CONFIG {i + 1}/{len(configs)}")
        print(
            f"Leaves: {c.lgbm_params['num_leaves']} | LR: {c.lgbm_params['learning_rate']} | "
            f"Min Data: {c.lgbm_params['min_data_in_leaf']} | FF: {c.lgbm_params['feature_fraction']}"
        )
        print("=" * 50)

        if i == 0:
            train_and_eval_lgbm(run_config=c, force_dataset_build=True)
        else:
            train_and_eval_lgbm(run_config=c, force_dataset_build=False)


if __name__ == "__main__":
    main()
