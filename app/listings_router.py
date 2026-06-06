import hashlib
import random
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from app.schemas import Listing

LISTINGS_PARQUET = Path("combined_ab.parquet")

router = APIRouter()

HOST_RESPONSE_TIME_COLS = [
    "host_response_time_a few days or more",
    "host_response_time_within a day",
    "host_response_time_within a few hours",
    "host_response_time_within an hour",
]

PROPERTY_TYPE_PREFIX = "property_type_"
ROOM_TYPE_PREFIX = "room_type_"
AMENITY_PREFIX = "amenity_"


def _decode_onehot(row: pd.Series, prefix: str, columns: list[str]) -> str:
    for col in columns:
        if row.get(col, 0) == 1:
            return col[len(prefix):]
    return "Unknown"


def _decode_amenities(row: pd.Series, columns: list[str]) -> list[str]:
    return [col[len(AMENITY_PREFIX):] for col in columns if row.get(col, 0) == 1]


def _row_to_listing(row: pd.Series, property_type_cols: list[str], room_type_cols: list[str], amenity_cols: list[str]) -> Listing:
    return Listing(
        listing_id=int(row["listing_id"]),
        booked=int(row["booked"]),
        session_id=hashlib.sha256(str(row["session_id"]).encode()).hexdigest(),
        host_since=str(row["host_since"]),
        host_response_rate=float(row["host_response_rate"]),
        host_acceptance_rate=float(row["host_acceptance_rate"]),
        host_is_superhost=bool(row["host_is_superhost"]),
        host_has_profile_pic=bool(row["host_has_profile_pic"]),
        host_identity_verified=bool(row["host_identity_verified"]),
        bathrooms=int(row["bathrooms"]),
        bedrooms=int(row["bedrooms"]),
        beds=int(row["beds"]),
        price=float(row["price"]),
        review_scores_rating=float(row["review_scores_rating"]),
        review_scores_accuracy=float(row["review_scores_accuracy"]),
        review_scores_cleanliness=float(row["review_scores_cleanliness"]),
        review_scores_checkin=float(row["review_scores_checkin"]),
        review_scores_communication=float(row["review_scores_communication"]),
        review_scores_location=float(row["review_scores_location"]),
        review_scores_value=float(row["review_scores_value"]),
        license=str(row["license"]),
        instant_bookable=bool(row["instant_bookable"]),
        host_response_time=_decode_onehot(row, "host_response_time_", HOST_RESPONSE_TIME_COLS),
        property_type=_decode_onehot(row, PROPERTY_TYPE_PREFIX, property_type_cols),
        room_type=_decode_onehot(row, ROOM_TYPE_PREFIX, room_type_cols),
        amenities=_decode_amenities(row, amenity_cols),
    )


@router.get("/listings", response_model=list[Listing])
def get_listings():
    df = pd.read_parquet(LISTINGS_PARQUET)

    numeric_cols = [
        "host_response_rate", "host_acceptance_rate", "bathrooms", "bedrooms",
        "beds", "price", "review_scores_rating", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location",
        "review_scores_value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    bool_cols = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    str_cols = ["host_since", "license", "session_id"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    property_type_cols = [c for c in df.columns if c.startswith(PROPERTY_TYPE_PREFIX)]
    room_type_cols = [c for c in df.columns if c.startswith(ROOM_TYPE_PREFIX)]
    amenity_cols = [c for c in df.columns if c.startswith(AMENITY_PREFIX)]

    session_ids = df["session_id"].unique().tolist()
    chosen_session_id = random.choice(session_ids)

    session_df = df[df["session_id"] == chosen_session_id]
    return [
        _row_to_listing(row, property_type_cols, room_type_cols, amenity_cols)
        for _, row in session_df.iterrows()
    ]
