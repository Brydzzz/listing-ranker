import polars as pl
import pandas as pd
import gc

def check_sessions_age_and_size(parquet_path="combined.parquet", chunk_size=2_000_000):
    print("Scanning Parquet Schema...")
    lazy_file = pl.scan_parquet(parquet_path)
    total_rows = lazy_file.select(pl.len()).collect().item()

    print(f"Detected Total Rows: {total_rows:,}\n")

    meta_cols = ["booked", "session_id", "listing_id", "host_since"]

    ref_date = pd.Timestamp.now()

    results = []
    for offset in range(0, total_rows, chunk_size):
        end_idx = min(offset + chunk_size, total_rows)
        chunk_label = f"{offset / 1_000_000:.1f}M - {end_idx / 1_000_000:.1f}M"

        print(f"Processing Chunk: {chunk_label}...")

        lazy_df = (
            pl.scan_parquet(parquet_path)
            .slice(offset, chunk_size)
            .select(meta_cols)
        )

        df = lazy_df.collect().to_pandas()
        df = df.dropna(subset=["booked", "session_id", "listing_id"])

        if len(df) == 0:
            continue

        session_sizes = df.groupby("session_id").size()
        med_size = session_sizes.median()
        p90_size = session_sizes.quantile(0.90)
        p99_size = session_sizes.quantile(0.99)
        max_size = session_sizes.max()

        df["host_since"] = pd.to_datetime(df["host_since"], errors="raise")
        df["host_days_active"] = (ref_date - df["host_since"]).dt.days
        age_p10 = df["host_days_active"].quantile(0.10)
        age_med = df["host_days_active"].median()
        age_p90 = df["host_days_active"].quantile(0.90)
        younger_than_2_years = (df["host_days_active"].fillna(0) <= 365 * 2).mean()


        results.append({
            "Row Range": chunk_label,
            "Age P10": f"{age_p10:.0f}d",
            "Age Med": f"{age_med:.0f}d",
            "Age P90": f"{age_p90:.0f}d",
            "New%": f"{younger_than_2_years * 100:.2f}%",
            "Size Med": f"{med_size:.0f}",
            "Size P90": f"{p90_size:.0f}",
            "Size P99": f"{p99_size:.0f}",
            "Size Max": f"{max_size:.0f}"
        })

        del df, session_sizes
        gc.collect()

    print("\n" + "="*125)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*125)

if __name__ == "__main__":
    check_sessions_age_and_size()