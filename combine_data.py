import pandas as pd
import numpy as np


def align_embeddings_to_sessions(combined, chunk_size=1_000_000):
    print("Loading listings in original order...")

    listings = pd.read_csv("listings.csv")
    embeddings_data = np.load("embeddings.npz")

    listing_to_idx = pd.Series(
        np.arange(len(listings), dtype=np.int32),
        index=listings["id"]
    )

    print("Mapping listing ids to embedding indices...")
    embedding_indices = combined["listing_id"].map(
        listing_to_idx
    ).to_numpy(dtype=np.int32)

    n_sessions = len(combined)

    output_files = []

    for key in embeddings_data.files:
        print(f"Aligning {key} embeddings...")

        source_embeddings = embeddings_data[key]
        emb_dim = source_embeddings.shape[1]
        dtype = source_embeddings.dtype

        out_path = f"{key}_combined.npy"

        output = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=dtype,
            shape=(n_sessions, emb_dim),
        )

        for start in range(0, n_sessions, chunk_size):
            end = min(start + chunk_size, n_sessions)

            idx_chunk = embedding_indices[start:end]

            output[start:end] = source_embeddings[idx_chunk]

            print(
                f"{key}: {end:,}/{n_sessions:,}"
            )

        del output
        output_files.append(out_path)

    print("Aligning done.")


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
    users_who_booked_unlisted = set(
        sessions[
            booked_mask & ~sessions["listing_id"].isin(valid_listing_ids)
        ]["user_id"]
    )

    sessions_filtered = sessions[
        ~sessions["user_id"].isin(users_who_booked_unlisted)
    ]

    combined = sessions_filtered.merge(
        listings_read, left_on="listing_id", right_on="id", how="left"
    )
    combined = combined[combined["id"].notna()]

    print(f"combined after merge: {combined.shape}")

    print("Aligning listings embeddings...")
    align_embeddings_to_sessions(combined)

    cols_to_drop = [
        # listings columns
        "id",
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
        "host_has_profile_pic",
        "host_identity_verified",
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
        # sessions columns
        "timestamp",
        "user_id",
        "listing_id",
    ]

    combined = combined.drop(columns=cols_to_drop)
    print(f"combined after drop shape: {combined.shape}")

    combined.to_csv("combined.csv", index=False)


if __name__ == "__main__":
    print("Reading sessions...")
    sessions_read = pd.read_csv("sessions.csv", dtype={"listing_id": "Int64"})

    valid_sampled_listing_ids = sessions_read["listing_id"].dropna().unique()

    print("Reading listings...")
    listings_read = pd.read_csv("listings.csv")
    listings_read = listings_read[
        listings_read["id"].isin(valid_sampled_listing_ids)
    ]
    print(listings_read["id"].duplicated().sum())

    print("Removing duplicates...")

    # Remove duplicate listing IDs
    n_dup_listings = listings_read["id"].duplicated().sum()
    print(f"{n_dup_listings} duplicate listing IDs — dropping")
    listings_read = listings_read.drop_duplicates(subset="id")

    # Remove fully identical session rows, then targeted key duplicates
    n_dup_sessions_full = sessions_read.duplicated().sum()
    print(f"{n_dup_sessions_full} fully duplicate session rows — dropping")
    sessions_read = sessions_read.drop_duplicates()

    print("Preparing sessions...")
    sessions = prepare_sessions(sessions_read, listings_read)

    print("Combining sessions with listings...")
    combine_sessions_listings(sessions, listings_read)

    print("Done :)")
