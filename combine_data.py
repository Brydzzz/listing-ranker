import ast
from collections import Counter

import pandas as pd

COLS_TO_IGNORE = [
    "name",
    "description",
    "listing_url",
    "scrape_id",
    "last_scraped",
    "source",
    "picture_url",
    "host_id",
    "host_url",
    "host_name",
    "host_location",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms_text",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "calendar_last_scraped",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "availability_eoy",
    "number_of_reviews_ly",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "first_review",
    "last_review",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "reviews_per_month",
]


def encode_amenities(listings_read):
    if "amenities" not in listings_read.columns:
        return listings_read

    def parse_amenities(val):
        if pd.isnull(val):
            return []
        try:
            parsed = ast.literal_eval(val)
            return (
                [str(a).strip() for a in parsed]
                if isinstance(parsed, list)
                else []
            )
        except (ValueError, SyntaxError):
            return []

    parsed = listings_read["amenities"].apply(parse_amenities)
    listings_read = listings_read.drop(columns=["amenities"])

    all_amenities = [a for sublist in parsed for a in sublist]
    top_100 = [item for item, _ in Counter(all_amenities).most_common(100)]

    amenity_cols = {}
    for amenity in top_100:
        amenity_cols[f"amenity_{amenity}"] = parsed.apply(
            lambda x: 1 if amenity in x else 0
        ).astype("int8")

    amenities_df = pd.DataFrame(amenity_cols, index=listings_read.index)
    listings_read = pd.concat([listings_read, amenities_df], axis=1)
    print(f"Amenity vector length: {len(top_100)} unique amenities")
    return listings_read


def prepare_listings(listings_read):
    """
    Applies transformations to the listings data:
    1. One-hot encodes host_response_time.
    2. Converts host_response_rate and host_acceptance_rate from percentages to floats.
    3. Converts host_is_superhost from 't'/'f' to 1/0 binary values.
    """
    # 1. Convert host_response_rate and host_acceptance_rate from strings (e.g., '95%') to float decimals (e.g., 0.95)
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in listings_read.columns:
            listings_read[col] = (
                listings_read[col].str.rstrip("%").astype(float) / 100.0
            )

    # 2. Convert host_is_superhost from 't'/'f' to 1/0 binary integers
    if "host_is_superhost" in listings_read.columns:
        listings_read["host_is_superhost"] = listings_read[
            "host_is_superhost"
        ].map({"t": 1, "f": 0})

    listings_read["host_has_profile_pic"] = listings_read[
        "host_has_profile_pic"
    ].map({"t": 1, "f": 0})
    listings_read["host_identity_verified"] = listings_read[
        "host_identity_verified"
    ].map({"t": 1, "f": 0})

    # 3. One-hot encode host_response_time into binary columns (0 or 1)
    categorical_cols = ["host_response_time", "property_type", "room_type"]
    for col in categorical_cols:
        if col in listings_read.columns:
            listings_read = pd.get_dummies(
                listings_read, columns=[col], prefix=col, dtype="int8"
            )

    # skipping for now if not sparse out of memory if sparse cpu can't handle merge later
    listings_read = encode_amenities(listings_read)
    return listings_read


def prepare_sessions(sessions_read, listings_read):
    valid_listing_ids = set(listings_read["id"])
    all_session_ids = set(sessions_read["listing_id"].dropna())
    missing = all_session_ids - valid_listing_ids

    print(
        f"{len(missing):>6} / {len(all_session_ids):>6} listing IDs not in listings.csv ({100 * len(missing) / len(all_session_ids):.1f}%)"
    )

    viewed_sessions = sessions_read[sessions_read["action"] == "view_listing"][
        ["user_id", "listing_id", "timestamp"]
    ]
    booked_sessions = sessions_read[sessions_read["action"] == "book_listing"][
        ["user_id", "listing_id", "booking_id"]
    ]
    combined_sessions = viewed_sessions.merge(
        booked_sessions, on=["user_id", "listing_id"], how="left"
    )
    combined_sessions.rename(columns={"booking_id": "booked"}, inplace=True)
    combined_sessions["booked"] = combined_sessions["booked"].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    combined_sessions["timestamp"] = pd.to_datetime(
        combined_sessions["timestamp"]
    )
    combined_sessions["session_id"] = (
        combined_sessions["user_id"].astype(str)
        + "_"
        + combined_sessions["timestamp"].dt.date.astype(str)
    )

    return combined_sessions


def combine_sessions_listings(sessions, listings_read):
    valid_listing_ids = set(listings_read["id"])

    booked_mask = sessions["booked"] == 1
    unlisted_mask = ~sessions["listing_id"].isin(valid_listing_ids)
    users_who_booked_unlisted = set(
        sessions[booked_mask & unlisted_mask]["user_id"]
    )

    sessions_filtered = sessions[
        ~sessions["user_id"].isin(users_who_booked_unlisted)
    ]

    combined = sessions_filtered.merge(
        listings_read, left_on="listing_id", right_on="id", how="inner"
    )

    print(f"combined after merge: {combined.shape}")

    cols_to_drop = [
        # listings columns
        "id",
        # sessions columns
        "timestamp",
        "user_id",
    ]

    combined = combined.drop(columns=cols_to_drop)
    print(f"combined after drop shape: {combined.shape}")

    combined.to_parquet("combined.parquet", index=False, engine="fastparquet")
    print("Saved combined.parquet")


if __name__ == "__main__":
    print("Reading sessions...")
    sessions_read = pd.read_csv("sessions.csv", dtype={"listing_id": "Int64"})

    valid_sampled_listing_ids = sessions_read["listing_id"].dropna().unique()

    print("Reading listings...")
    all_listings_cols = pd.read_csv("listings.csv", nrows=0).columns
    cols_to_keep = [
        col for col in all_listings_cols if col not in COLS_TO_IGNORE
    ]
    listings_read = pd.read_csv("listings.csv", usecols=cols_to_keep)
    listings_read = listings_read[
        listings_read["id"].isin(valid_sampled_listing_ids)
    ]

    print("Removing duplicates...")
    listings_read = listings_read.drop_duplicates(subset="id")
    sessions_read = sessions_read.drop_duplicates()

    print("Preprocessing listings...")
    listings_read = prepare_listings(listings_read)

    print("Preparing sessions...")
    sessions = prepare_sessions(sessions_read, listings_read)

    del sessions_read

    print("Combining sessions with listings...")
    combine_sessions_listings(sessions, listings_read)

    print("Done :)")
