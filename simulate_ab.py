import polars as pl
import numpy as np
import uuid
import csv

INPUT_FILE = "combined_ab.parquet"
OUTPUT_FILE = "ab_results.csv"

SEED = 42
DESIRED_M1_MEAN = 8.5
DESIRED_M2_MEAN = 6.2

rng = np.random.default_rng(SEED)

df = pl.read_parquet(INPUT_FILE).drop("booked")
listing_columns = [c for c in df.columns if c != "session_id"]

session_size_df = df.group_by("session_id").len().sort("len")
session_list = session_size_df["session_id"].to_list()
sizes_list = session_size_df["len"].to_list()
session_sizes = dict(zip(session_list, sizes_list))

fifth = len(session_list) // 5
bottom = session_list[:fifth]
middle = session_list[fifth:4 * fifth]
top = session_list[4 * fifth:]

middle_shuffled = list(middle)
rng.shuffle(middle_shuffled)
mid_half = len(middle_shuffled) // 2

model2_sessions = set(bottom + middle_shuffled[:mid_half])
model1_sessions = set(top + middle_shuffled[mid_half:])

m1_sizes = [session_sizes[s] for s in model1_sessions]
m2_sizes = [session_sizes[s] for s in model2_sessions]
m1_avg = np.mean(m1_sizes)
m2_avg = np.mean(m2_sizes)
m1_ratio = DESIRED_M1_MEAN / m1_avg
m2_ratio = DESIRED_M2_MEAN / m2_avg

print(f"model1 — raw avg: {m1_avg:.2f}, scale ratio: {m1_ratio:.3f}")
print(f"model2 — raw avg: {m2_avg:.2f}, scale ratio: {m2_ratio:.3f}")

targets = {}
for sid in model1_sessions:
    n = session_sizes[sid]
    targets[sid] = max(1, min(int(round(n * m1_ratio)), n))
for sid in model2_sessions:
    n = session_sizes[sid]
    targets[sid] = max(1, min(int(round(n * m2_ratio)), n))

fieldnames = ["user_uid", "session_type", "ranking_model"] + listing_columns + ["session_id"]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for (session_id,), group in df.group_by("session_id"):
        model = "model1" if session_id in model1_sessions else "model2"
        n = group.height
        target = targets[session_id]

        all_idx = np.arange(n)
        shown_idx = set(rng.choice(all_idx, size=target, replace=False).tolist())
        booked_idx = rng.choice(list(shown_idx)).item()

        user_uid = str(uuid.uuid4())

        session_types = np.full(n, "not_viewed", dtype=object)
        for idx in shown_idx:
            session_types[idx] = "view_listing"
        session_types[booked_idx] = "book_listing"

        chunk = group.with_columns(
            pl.lit(user_uid).alias("user_uid"),
            pl.Series("session_type", session_types),
            pl.lit(model).alias("ranking_model"),
        ).select(fieldnames)

        writer.writerows(chunk.to_dicts())
