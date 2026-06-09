import os
import json
import time
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

questions_df = pd.read_csv("rag_generated_questions_clean_v1.csv")

sample_df = questions_df.sample(
    n=100,
    random_state=42
).reset_index(drop=True)

output_path = "llm_primary_eval_fixed_100_results.csv"

if os.path.exists(output_path):
    results_df = pd.read_csv(output_path)
    done_questions = set(results_df["question"].astype(str))
    results = results_df.to_dict("records")
    print("Existing results loaded:", len(results))
else:
    done_questions = set()
    results = []
    print("Starting fresh")

remaining_df = sample_df[
    ~sample_df["question"].astype(str).isin(done_questions)
].reset_index(drop=True)

print("Total questions:", len(sample_df))
print("Remaining:", len(remaining_df))

if len(remaining_df) == 0:
    print("All questions already evaluated.")
    exit()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

question_embeddings = model.encode(
    remaining_df["question"].astype(str).tolist(),
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

table_name = "fixed_size_chunks"

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


def judge_with_llm(question, retrieved_text):
    prompt = f"""
You are judging retrieval relevance for a RAG system.

User question:
{question}

Retrieved complaints:
{retrieved_text}

Decide whether the retrieved complaints are useful for answering or matching the user question.

Use these labels:
y = at least one retrieved complaint is strongly relevant to the question
partial = the retrieved complaints are related, but miss important details
n = the retrieved complaints are not meaningfully relevant

Return only valid JSON in this format:
{{"label": "y", "reason": "short reason"}}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0
    )

    text = response.output_text.strip()

    try:
        data = json.loads(text)
        label = data.get("label", "").strip().lower()
        reason = data.get("reason", "").strip()
    except:
        label = "parse_error"
        reason = text

    if label not in ["y", "partial", "n"]:
        label = "parse_error"

    return label, reason


for i, row in remaining_df.iterrows():

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

    retrieved_text = ""

    for rank, r in enumerate(retrieved, start=1):
        complaint_id = str(r[0])
        full_text = get_full_complaint_text(complaint_id)

        retrieved_text += f"""
Retrieved complaint {rank}
Complaint ID: {complaint_id}
Product: {r[1]}
Issue: {r[2]}
Company: {r[3]}
Distance: {round(r[4], 4)}
Complaint text:
{full_text}

"""

    label, reason = judge_with_llm(question, retrieved_text)

    results.append({
        "method": "fixed_size",
        "question_index": len(results) + 1,
        "question": question,
        "question_type": row["question_type"],
        "source_complaint_id": row["source_complaint_id"],
        "llm_primary_label": label,
        "llm_reason": reason
    })

    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Done {len(results)}/100 | Label: {label} | Reason: {reason}")

    time.sleep(0.5)

cur.close()
conn.close()

final_df = pd.DataFrame(results)
print("\nFinal results:")
print(final_df["llm_primary_label"].value_counts())

score = (
    (final_df["llm_primary_label"] == "y").sum()
    + 0.5 * (final_df["llm_primary_label"] == "partial").sum()
) / len(final_df)

print("Weighted relevance score:", round(score, 4))