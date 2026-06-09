# ---------------------------------------
# 1. Import required libraries
# ---------------------------------------

import os
import pandas as pd

# psycopg2 is used to connect Python to PostgreSQL
import psycopg2

# register_vector allows psycopg2 to work with pgvector embeddings
from pgvector.psycopg2 import register_vector

# SentenceTransformer is used for embedding-based retrieval
# CrossEncoder is used for relevance scoring after retrieval
from sentence_transformers import SentenceTransformer, CrossEncoder


# ---------------------------------------
# 2. Load the generated questions dataset
# ---------------------------------------

# This CSV contains the generated questions used for evaluation
questions_df = pd.read_csv("rag_generated_questions_clean_v1.csv")


# ---------------------------------------
# 3. Select 100 questions for evaluation
# ---------------------------------------

# Randomly sample 100 questions from the full question dataset.
# random_state=42 makes the sample reproducible,
# so the same 100 questions are selected every time.
sample_df = questions_df.sample(
    n=100,
    random_state=42
).reset_index(drop=True)


# ---------------------------------------
# 4. Define the output file
# ---------------------------------------

# This file will store the cross-encoder evaluation results
# for the fixed-size chunking method.
output_path = "cross_encoder_eval_fixed_100_results.csv"


# ---------------------------------------
# 5. Load the embedding model and cross-encoder model
# ---------------------------------------

# This model converts questions into embeddings.
# These embeddings are used for the initial vector search in PostgreSQL.
model_embed = SentenceTransformer("BAAI/bge-small-en-v1.5")

# This model scores the relevance between a question and a retrieved complaint.
# It reads the question and complaint together as a pair.
model_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------
# 6. Create embeddings for all sampled questions
# ---------------------------------------

# Each question is converted into a numerical vector.
# normalize_embeddings=True makes the vectors normalized,
# which is useful for vector similarity search.
# convert_to_numpy=True returns the embeddings as NumPy arrays.
question_embeddings = model_embed.encode(
    sample_df["question"].astype(str).tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)


# ---------------------------------------
# 7. Connect to PostgreSQL database
# ---------------------------------------

# This connects Python to the local PostgreSQL database
# where the chunk texts and embeddings are stored.
conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

# Register pgvector support for this database connection.
# This allows Python to send vector embeddings to PostgreSQL correctly.
register_vector(conn)

# Create a cursor to execute SQL queries.
cur = conn.cursor()


# ---------------------------------------
# 8. Choose the chunking table to evaluate
# ---------------------------------------

# This code evaluates the fixed-size chunking method,
# so it uses the fixed_size_chunks table.
table_name = "fixed_size_chunks"


# ---------------------------------------
# 9. Function to reconstruct full complaint text
# ---------------------------------------

def get_full_complaint_text(complaint_id):
    """
    This function receives one complaint ID.
    It retrieves all chunks that belong to that complaint,
    orders them correctly by chunk_index,
    joins them together,
    and returns the first 3000 characters.
    """

    # Get all chunks for this complaint from the fixed_size_chunks table
    cur.execute(
        f"""
        SELECT chunk_text
        FROM {table_name}
        WHERE representative_complaint_id = %s
        ORDER BY chunk_index;
        """,
        (str(complaint_id),)
    )

    # Extract the chunk_text values from the database result
    chunks = [r[0] for r in cur.fetchall()]

    # Join all chunks together to rebuild the complaint text
    text = "\n".join(chunks)

    # Return only the first 3000 characters.
    # This keeps the text short enough for the cross-encoder model.
    return text[:3000]


# ---------------------------------------
# 10. Create an empty list to store results
# ---------------------------------------

# Each evaluated question will add one dictionary to this list.
results = []


# ---------------------------------------
# 11. Loop through each sampled question
# ---------------------------------------

for i, row in sample_df.iterrows():

    # Get the current question text
    question = row["question"]

    # Get the embedding of the current question
    embedding = question_embeddings[i]


    # ---------------------------------------
    # 12. Retrieve top 3 similar results using vector search
    # ---------------------------------------

    # This SQL query compares the current question embedding
    # with the stored chunk embeddings in the fixed_size_chunks table.
    #
    # embedding <=> %s calculates vector distance.
    # Smaller distance means the chunk is more similar to the question.
    #
    # The query orders all chunks by similarity and returns the top 3.
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

    # Fetch the top 3 retrieved rows from the database
    retrieved = cur.fetchall()


    # ---------------------------------------
    # 13. Prepare pairs for cross-encoder scoring
    # ---------------------------------------

    # This list will contain pairs in this format:
    # (question, complaint_text)
    pairs = []

    # This list stores metadata about each retrieved complaint.
    # We keep this so later we can identify which complaint got the best score.
    retrieved_info = []


    # Go through the top 3 retrieved results
    for rank, r in enumerate(retrieved, start=1):

        # Get the representative complaint ID
        complaint_id = str(r[0])

        # Rebuild the full complaint text from all its chunks
        full_text = get_full_complaint_text(complaint_id)

        # Add the question and complaint text as a pair for the cross-encoder
        pairs.append((question, full_text))

        # Save metadata about this retrieved complaint
        retrieved_info.append({
            "rank": rank,             # The original vector-search rank: 1, 2, or 3
            "complaint_id": complaint_id,
            "product": r[1],
            "issue": r[2],
            "company": r[3],
            "distance": r[4]          # Vector distance from pgvector
        })


    # ---------------------------------------
    # 14. Score each question-complaint pair with the cross-encoder
    # ---------------------------------------

    # The cross-encoder reads each pair together:
    # question + complaint text
    #
    # It returns one relevance score for each retrieved complaint.
    # Higher score means the complaint is more relevant to the question.
    scores = model_cross.predict(pairs)


    # ---------------------------------------
    # 15. Find the best retrieved complaint
    # ---------------------------------------

    # Get the highest cross-encoder score among the top 3 retrieved complaints
    max_score = float(max(scores))

    # Get the index of the complaint with the highest score
    best_index = int(scores.argmax())

    # Use that index to get the metadata of the best complaint
    best_result = retrieved_info[best_index]


    # ---------------------------------------
    # 16. Save the evaluation result for this question
    # ---------------------------------------

    # For each question, we save only the best cross-encoder result
    # among the top 3 retrieved complaints.
    results.append({
        "method": "fixed_size",                       # Chunking method evaluated
        "question_index": i + 1,                      # Question number from 1 to 100
        "question": question,                         # The generated question
        "question_type": row["question_type"],        # Type/category of question
        "source_complaint_id": row["source_complaint_id"],  # Complaint used to generate the question

        "best_cross_encoder_score": max_score,        # Highest cross-encoder score among top 3
        "best_rank": best_result["rank"],             # Original vector-search rank of the best result
        "best_complaint_id": best_result["complaint_id"],
        "best_product": best_result["product"],
        "best_issue": best_result["issue"],
        "best_company": best_result["company"]
    })


    # ---------------------------------------
    # 17. Print progress every 10 questions
    # ---------------------------------------

    # This helps us know that the code is still running.
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/100")


# ---------------------------------------
# 18. Convert results to DataFrame and save to CSV
# ---------------------------------------

# Convert the list of dictionaries into a pandas DataFrame
results_df = pd.DataFrame(results)

# Save the evaluation results to a CSV file
results_df.to_csv(output_path, index=False)


# ---------------------------------------
# 19. Close database connection
# ---------------------------------------

# Close the cursor
cur.close()

# Close the PostgreSQL connection
conn.close()


# ---------------------------------------
# 20. Print final output information
# ---------------------------------------

# Show where the result file was saved
print("\nSaved to:", output_path)


# ---------------------------------------
# 21. Print score summary
# ---------------------------------------

# describe() gives summary statistics:
# count, mean, std, min, 25%, 50%, 75%, max
print("\nScore summary:")
print(results_df["best_cross_encoder_score"].describe())


# ---------------------------------------
# 22. Print average cross-encoder score
# ---------------------------------------

# This gives one overall average score for the fixed-size chunking method.
print("\nAverage score:")
print(round(results_df["best_cross_encoder_score"].mean(), 4))