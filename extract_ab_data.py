import polars as pl

INPUT_FILE = "combined.parquet"
OUTPUT_AB = "combined_ab.parquet"
OUTPUT_NO_AB = "combined_no_ab.parquet"
N_ROWS = 1_000_000

total_rows = pl.scan_parquet(INPUT_FILE).select(pl.len()).collect().item()
offset = max(total_rows - N_ROWS, 0)

print(f"Total rows: {total_rows}, splitting at offset {offset}")

df_no_ab = pl.scan_parquet(INPUT_FILE).slice(0, offset).collect()
df_no_ab.write_parquet(OUTPUT_NO_AB)
print(f"Wrote {df_no_ab.height} rows to {OUTPUT_NO_AB}")

df_ab = pl.scan_parquet(INPUT_FILE).slice(offset, N_ROWS).collect()
df_ab.write_parquet(OUTPUT_AB)
print(f"Wrote {df_ab.height} rows to {OUTPUT_AB}")
