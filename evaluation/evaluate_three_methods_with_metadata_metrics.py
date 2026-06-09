import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

questions_df = pd.read_csv("rag_generated_questions_clean_v1.csv")

eval_sample_df = questions_df.sample(
    n=100,
    random_state=42
).reset_index(drop=True)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

question_embeddings = model.encode(
    eval_sample_df["question"].astype(str).tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

tables = {
    "fixed_size": "fixed_size_chunks",
    "recursive": "recursive_chunks",
    "token_aware_semantic": "token_aware_semantic_chunks"
}

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

register_vector(conn)
cur = conn.cursor()

summary = []

for method_name, table_name in tables.items():

    exact_hits = 0
    product_hits = 0
    issue_hits = 0
    company_hits = 0
    product_issue_hits = 0

    print("\n" + "="*80)
    print("Evaluating:", method_name)

    for i, row in eval_sample_df.iterrows():

        true_id = str(row["source_complaint_id"])
        true_product = str(row["Product"])
        true_issue = str(row["Issue"])
        true_company = str(row["Company"])

        embedding = question_embeddings[i]

        cur.execute(
            f"""
            SELECT
                representative_complaint_id,
                product,
                issue,
                company
            FROM {table_name}
            ORDER BY embedding <=> %s
            LIMIT 5;
            """,
            (embedding,)
        )

        retrieved = cur.fetchall()

        retrieved_ids = [str(r[0]) for r in retrieved]
        retrieved_products = [str(r[1]) for r in retrieved]
        retrieved_issues = [str(r[2]) for r in retrieved]
        retrieved_companies = [str(r[3]) for r in retrieved]

        if true_id in retrieved_ids:
            exact_hits += 1

        if true_product in retrieved_products:
            product_hits += 1

        if true_issue in retrieved_issues:
            issue_hits += 1

        if true_company in retrieved_companies:
            company_hits += 1

        product_issue_match = False

        for r in retrieved:
            retrieved_product = str(r[1])
            retrieved_issue = str(r[2])

            if retrieved_product == true_product and retrieved_issue == true_issue:
                product_issue_match = True
                break

        if product_issue_match:
            product_issue_hits += 1

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/100")

    total = len(eval_sample_df)

    summary.append({
        "method": method_name,
        "exact_id_recall_at_5": exact_hits / total,
        "product_match_at_5": product_hits / total,
        "issue_match_at_5": issue_hits / total,
        "company_match_at_5": company_hits / total,
        "product_issue_match_at_5": product_issue_hits / total,
        "exact_hits": exact_hits,
        "product_hits": product_hits,
        "issue_hits": issue_hits,
        "company_hits": company_hits,
        "product_issue_hits": product_issue_hits,
        "total_questions": total
    })

results_df = pd.DataFrame(summary)

print("\nFinal automatic metric comparison:")
print(results_df)

results_df.to_csv(
    "small_eval_metadata_metrics_100_questions.csv",
    index=False
)

cur.close()
conn.close()