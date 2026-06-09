import os
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

questions_df = pd.read_csv("rag_generated_questions_clean_v1.csv")

sample_df = questions_df.sample(
    n=100,
    random_state=42
).reset_index(drop=True)

output_path = "cross_encoder_eval_recursive_100_results.csv"

model_embed = SentenceTransformer("BAAI/bge-small-en-v1.5")
model_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

question_embeddings = model_embed.encode(
    sample_df["question"].astype(str).tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

register_vector(conn)
cur = conn.cursor()

table_name = "recursive_chunks"

def get_full_complaint_text(complaint_id):
    cur.execute(
        f"""
        SELECT chunk_text
        FROM {table_name}
        WHERE representative_complaint_id = %s
        ORDER BY chunk_index;
        """,
        (str(complaint_id),)
    )

    chunks = [r[0] for r in cur.fetchall()]
    text = "\n".join(chunks)

    return text[:3000]

results = []

for i, row in sample_df.iterrows():

    question = row["question"]
    embedding = question_embeddings[i]

    cur.execute(
        f"""
        SELECT
            representative_complaint_id,
            product,
            issue,
            company,
            embedding <=> %s AS distance
        FROM {table_name}
        ORDER BY embedding <=> %s
        LIMIT 3;
        """,
        (embedding, embedding)
    )

    retrieved = cur.fetchall()

    pairs = []
    retrieved_info = []

    for rank, r in enumerate(retrieved, start=1):
        complaint_id = str(r[0])
        full_text = get_full_complaint_text(complaint_id)

        pairs.append((question, full_text))

        retrieved_info.append({
            "rank": rank,
            "complaint_id": complaint_id,
            "product": r[1],
            "issue": r[2],
            "company": r[3],
            "distance": r[4]
        })

    scores = model_cross.predict(pairs)

    max_score = float(max(scores))
    best_index = int(scores.argmax())
    best_result = retrieved_info[best_index]

    results.append({
        "method": "fixed_size",
        "question_index": i + 1,
        "question": question,
        "question_type": row["question_type"],
        "source_complaint_id": row["source_complaint_id"],
        "best_cross_encoder_score": max_score,
        "best_rank": best_result["rank"],
        "best_complaint_id": best_result["complaint_id"],
        "best_product": best_result["product"],
        "best_issue": best_result["issue"],
        "best_company": best_result["company"]
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/100")

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

cur.close()
conn.close()

print("\nSaved to:", output_path)

print("\nScore summary:")
print(results_df["best_cross_encoder_score"].describe())

print("\nAverage score:")
print(round(results_df["best_cross_encoder_score"].mean(), 4))