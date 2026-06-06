import gc
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

INPUT_FILE = "combined.parquet"
OUTPUT_AB = "combined_ab.parquet"
OUTPUT_NO_AB = "combined_no_ab.parquet"
AB_FRACTION = 1 / 13
SEED = 42
BATCH_SIZE = 10_000

rng = np.random.default_rng(SEED)

session_counts = Counter()
pf = pq.ParquetFile(INPUT_FILE, memory_map=True)
for batch in pf.iter_batches(
    batch_size=BATCH_SIZE, columns=["session_id"], use_threads=False
):
    session_counts.update(batch.column("session_id").to_pylist())

total_rows = sum(session_counts.values())

session_ids = list(session_counts.keys())
rng.shuffle(session_ids)

target_rows = int(total_rows * AB_FRACTION)
ab_sessions = []
ab_row_count = 0
for sid in session_ids:
    if ab_row_count >= target_rows:
        break
    ab_sessions.append(sid)
    ab_row_count += session_counts[sid]

ab_set = set(ab_sessions)
ab_array = pa.array(list(ab_set))
del session_ids, session_counts, ab_sessions
gc.collect()


pf = pq.ParquetFile(INPUT_FILE, memory_map=True)
writer = None
rows = 0
for batch in pf.iter_batches(batch_size=BATCH_SIZE, use_threads=False):
    table = pa.Table.from_batches([batch])
    filtered = table.filter(pc.is_in(table.column("session_id"), ab_array))
    if filtered.num_rows == 0:
        continue
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_AB, filtered.schema)
    writer.write_table(filtered)
    rows += filtered.num_rows
if writer is not None:
    writer.close()

gc.collect()


pf = pq.ParquetFile(INPUT_FILE, memory_map=True)
writer = None
rows = 0
for batch in pf.iter_batches(batch_size=BATCH_SIZE, use_threads=False):
    table = pa.Table.from_batches([batch])
    filtered = table.filter(
        pc.invert(pc.is_in(table.column("session_id"), ab_array))
    ).drop("booked")
    if filtered.num_rows == 0:
        continue
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_NO_AB, filtered.schema)
    writer.write_table(filtered)
    rows += filtered.num_rows
if writer is not None:
    writer.close()
