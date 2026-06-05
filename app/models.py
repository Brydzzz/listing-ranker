import lightgbm as lgb
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from app.schemas import Listing


class RankingModel(ABC):
    @abstractmethod
    def rank(self, listings: list[Listing]) -> list[Listing]:
        ...


class Model1(RankingModel):
    def rank(self, listings: list[Listing]) -> list[Listing]:
        return sorted(
            listings,
            key=lambda l: l.review_scores_rating / (l.price + 1),
            reverse=True,
        )


class Model2(RankingModel):
    def __init__(self, model_path: str = "lgbm_model_with_feature_names.txt"):
        self.bst = lgb.Booster(model_file=model_path)
        self.feature_names = self.bst.feature_name()

    def rank(self, listings: list[Listing]) -> list[Listing]:
        if not listings:
            return []

        df = pd.DataFrame([l.model_dump() for l in listings])

        prices = df["price"]
        price_mean = prices.mean()
        price_median = prices.median()
        price_min = prices.min()
        price_std = prices.std()

        df["price_diff_from_mean"] = prices - price_mean
        df["price_ratio_to_mean"] = prices / price_mean if price_mean else 0
        df["price_ratio_to_median"] = prices / price_median if price_median else 0
        df["price_diff_from_min"] = prices - price_min
        df["price_z_score_session"] = (
            (prices - price_mean) / price_std if pd.notnull(price_std) and price_std > 0 else 0
        )

        X = pd.DataFrame(0.0, index=df.index, columns=self.feature_names)

        for col in self.feature_names:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                    X[col] = df[col]
                else:
                    X[col] = pd.to_numeric(df[col], errors='coerce')

        for idx, row in df.iterrows():
            pt = f"property_type_{str(row.get('property_type', '')).replace(' ', '_')}"
            if pt in self.feature_names:
                X.at[idx, pt] = 1.0

            rt = f"room_type_{str(row.get('room_type', '')).replace(' ', '_')}"
            if rt in self.feature_names:
                X.at[idx, rt] = 1.0

            hrt = f"host_response_time_{str(row.get('host_response_time', '')).replace(' ', '_')}"
            if hrt in self.feature_names:
                X.at[idx, hrt] = 1.0

            for am in row.get("amenities", []):
                am_cleaned = str(am).replace(" ", "_").replace(":", "_")
                am_col = f"amenity_{am_cleaned}"
                if am_col in self.feature_names:
                    X.at[idx, am_col] = 1.0

        preds = self.bst.predict(X)
        ranked_indices = np.argsort(preds)[::-1]
        
        return [listings[i] for i in ranked_indices]
