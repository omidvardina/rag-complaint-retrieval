import json
import time
from pathlib import Path

import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

drive = Path.home() / "Library/CloudStorage/GoogleDrive-dinaomidvar1377@gmail.com/My Drive"
embedding_dir = drive / "consumer_complaints_bge_embeddings_v1_parts"
checkpoint_path = Path("fixed_pgvector_load_checkpoint.json")

embedding_files = sorted(
    embedding_dir.glob("bge_chunks_part_*_embeddings.parquet")
)

print("Embedding files found:", len(embedding_files))

start_file = 0

if checkpoint_path.exists():
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
    start_file = checkpoint["last_file"]
    print("Resuming from file:", start_file)
else:
    print("Starting fresh")

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

register_vector(conn)
cur = conn.cursor()

insert_sql = """
INSERT INTO fixed_size_chunks (
    chunk_id,
    representative_complaint_id,
    chunk_index,
    chunk_count_for_complaint,
    chunking_method,
    chunk_text,
    product,
    sub_product,
    issue,
    sub_issue,
    company,
    state,
    final_chunk_token_count,
    embedding
)
VALUES %s
ON CONFLICT (chunk_id) DO NOTHING;
"""

start_time = time.time()

for file_idx in range(start_file, len(embedding_files)):

    file_path = embedding_files[file_idx]
    df = pd.read_parquet(file_path)

    rows = []

    for _, r in df.iterrows():
        rows.append((
            str(r["chunk_id"]),
            str(r["representative_complaint_id"]),
            int(r["chunk_index"]),
            int(r["chunk_count_for_complaint"]),
            "fixed_size_token_chunking",
            str(r["chunk_text"]),
            str(r["Product"]) if pd.notna(r["Product"]) else None,
            str(r["Sub-product"]) if pd.notna(r["Sub-product"]) else None,
            str(r["Issue"]) if pd.notna(r["Issue"]) else None,
            str(r["Sub-issue"]) if pd.notna(r["Sub-issue"]) else None,
            str(r["Company"]) if pd.notna(r["Company"]) else None,
            str(r["State"]) if pd.notna(r["State"]) else None,
            int(r["final_chunk_token_count"]),
            r["embedding"]
        ))

    execute_values(
        cur,
        insert_sql,
        rows,
        page_size=500
    )

    conn.commit()

    with open(checkpoint_path, "w") as f:
        json.dump({"last_file": file_idx + 1}, f)

    if (file_idx + 1) % 50 == 0 or file_idx == 0:
        cur.execute("SELECT COUNT(*) FROM fixed_size_chunks;")
        count = cur.fetchone()[0]

        elapsed = time.time() - start_time
        files_done_now = (file_idx + 1) - start_file
        files_left = len(embedding_files) - (file_idx + 1)
        avg_time_per_file = elapsed / max(files_done_now, 1)
        remaining_seconds = files_left * avg_time_per_file

        print(
            f"Loaded {file_idx + 1}/{len(embedding_files)} files | "
            f"table rows: {count:,} | "
            f"elapsed: {elapsed/60:.1f} min | "
            f"estimated remaining: {remaining_seconds/60:.1f} min"
        )

cur.close()
conn.close()

print("\nDONE loading fixed-size chunks.")