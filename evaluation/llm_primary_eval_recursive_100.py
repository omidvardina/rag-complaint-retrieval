# -----------------------------
# 1. Import required libraries
# -----------------------------

import os                  # Used to access environment variables, like the OpenAI API key
import json                # Used to parse the JSON response returned by the LLM
import time                # Used to pause between OpenAI API calls
import pandas as pd        # Used for reading and saving CSV files

import psycopg2            # Used to connect Python to PostgreSQL
from pgvector.psycopg2 import register_vector  # Allows PostgreSQL pgvector embeddings to work with psycopg2

from sentence_transformers import SentenceTransformer  # Used to create embeddings for questions
from openai import OpenAI  # Used to call the OpenAI API


# ---------------------------------------
# 2. Create the OpenAI client
# ---------------------------------------

# The API key is read from the environment variable OPENAI_API_KEY.
# This is better than writing the API key directly inside the code.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------
# 3. Load the generated questions dataset
# ---------------------------------------

# This CSV file contains the generated questions that will be used for evaluation.
questions_df = pd.read_csv("rag_generated_questions_clean_v1.csv")


# --------------------------------------------------
# 4. Randomly select 100 questions for evaluation
# --------------------------------------------------

# We sample 100 questions from the dataset.
# random_state=42 makes sure we get the same 100 questions every time we run the code.
sample_df = questions_df.sample(
    n=100,
    random_state=42
).reset_index(drop=True)


# ---------------------------------------------------
# 5. Define the output file for saving results
# ---------------------------------------------------

# This file will store the LLM evaluation results for the recursive chunking method.
output_path = "llm_primary_eval_recursive_100_results.csv"


# --------------------------------------------------------
# 6. Check if previous evaluation results already exist
# --------------------------------------------------------

# This part allows the code to continue from where it stopped.
# If the output file already exists, we load the previous results.
if os.path.exists(output_path):
    
    # Read the existing results file
    results_df = pd.read_csv(output_path)
    
    # Store the questions that were already evaluated
    done_questions = set(results_df["question"].astype(str))
    
    # Convert previous results into a list of dictionaries
    results = results_df.to_dict("records")
    
    print("Existing results loaded:", len(results))

else:
    # If no output file exists, start with an empty result list
    done_questions = set()
    results = []
    
    print("Starting fresh")


# -------------------------------------------------------
# 7. Keep only the questions that are not evaluated yet
# -------------------------------------------------------

# This removes questions that were already saved in the output file.
# So if the code stopped before, it will not repeat the same questions.
remaining_df = sample_df[
    ~sample_df["question"].astype(str).isin(done_questions)
].reset_index(drop=True)


# Print how many questions are in the sample and how many still need evaluation
print("Total questions:", len(sample_df))
print("Remaining:", len(remaining_df))


# If all 100 questions were already evaluated, stop the code.
if len(remaining_df) == 0:
    print("All questions already evaluated.")
    exit()


# ------------------------------------------------
# 8. Load the embedding model
# ------------------------------------------------

# This is the same model used to create embeddings.
# It converts text into numerical vectors.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# --------------------------------------------------------
# 9. Create embeddings for the remaining questions
# --------------------------------------------------------

# Each question is converted into an embedding vector.
# normalize_embeddings=True is useful for similarity search.
# convert_to_numpy=True returns the embeddings as NumPy arrays.
question_embeddings = model.encode(
    remaining_df["question"].astype(str).tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)


# ----------------------------------------------
# 10. Connect to the PostgreSQL database
# ----------------------------------------------

# This connects Python to the local PostgreSQL database where the chunks are stored.
conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)


# Register pgvector so Python can send vector embeddings to PostgreSQL correctly
register_vector(conn)


# Create a cursor.
# The cursor is used to run SQL queries.
cur = conn.cursor()


# ------------------------------------------------
# 11. Choose which chunking table to evaluate
# ------------------------------------------------

# This code is evaluating the recursive chunking method,
# so it uses the recursive_chunks table.
table_name = "recursive_chunks"


# ---------------------------------------------------------
# 12. Function to reconstruct the full complaint text
# ---------------------------------------------------------

def get_full_complaint_text(complaint_id):
    """
    This function receives a complaint ID.
    Then it retrieves all chunks belonging to that complaint.
    Finally, it joins the chunks together to rebuild the complaint text.
    """

    # Get all chunks for one complaint ID, ordered by chunk_index
    cur.execute(
        f"""
        SELECT chunk_text
        FROM {table_name}
        WHERE representative_complaint_id = %s
        ORDER BY chunk_index;
        """,
        (str(complaint_id),)
    )

    # Fetch all chunk texts from the database result
    chunks = [r[0] for r in cur.fetchall()]

    # Join all chunks together into one text
    text = "\n".join(chunks)

    # Return only the first 3000 characters.
    # This keeps the LLM prompt shorter and avoids sending very long complaints.
    return text[:3000]


# ---------------------------------------------------------
# 13. Function to judge retrieval relevance using an LLM
# ---------------------------------------------------------

def judge_with_llm(question, retrieved_text):
    """
    This function gives the user question and retrieved complaints to GPT.
    GPT then decides whether the retrieved complaints are relevant or not.
    """

    # Create the prompt that will be sent to GPT
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

    # Send the prompt to the OpenAI model
    # temperature=0 makes the output more stable and less random.
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0
    )

    # Extract the text response from the model
    text = response.output_text.strip()

    # Try to convert the response into JSON
    try:
        data = json.loads(text)

        # Extract the label, for example: y, partial, or n
        label = data.get("label", "").strip().lower()

        # Extract the short reason written by the LLM
        reason = data.get("reason", "").strip()

    except:
        # If GPT does not return valid JSON, mark it as a parse error
        label = "parse_error"
        reason = text

    # Make sure the label is one of the expected labels
    if label not in ["y", "partial", "n"]:
        label = "parse_error"

    # Return the final label and reason
    return label, reason


# ---------------------------------------------------------
# 14. Main evaluation loop
# ---------------------------------------------------------

# Go through each remaining question one by one
for i, row in remaining_df.iterrows():

    # Get the current question
    question = row["question"]

    # Get the embedding for the current question
    embedding = question_embeddings[i]


    # ---------------------------------------------------------
    # 15. Retrieve top 3 most similar chunks from PostgreSQL
    # ---------------------------------------------------------

    # This SQL query searches the recursive_chunks table.
    # embedding <=> %s calculates the vector distance between:
    # 1. the stored chunk embedding
    # 2. the current question embedding
    #
    # The smaller the distance, the more similar the chunk is to the question.
    #
    # LIMIT 3 means we retrieve the top 3 closest results.
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

    # Fetch the top 3 retrieved results
    retrieved = cur.fetchall()


    # ---------------------------------------------------------
    # 16. Prepare retrieved complaints text for the LLM
    # ---------------------------------------------------------

    # This variable will contain the formatted retrieved complaints.
    retrieved_text = ""

    # Go through each retrieved complaint
    for rank, r in enumerate(retrieved, start=1):

        # Get the complaint ID from the retrieved row
        complaint_id = str(r[0])

        # Reconstruct the full complaint text from its chunks
        full_text = get_full_complaint_text(complaint_id)

        # Add this retrieved complaint to the text that will be sent to the LLM
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


    # ---------------------------------------------------------
    # 17. Ask the LLM to judge the retrieved complaints
    # ---------------------------------------------------------

    # The LLM receives:
    # 1. the user question
    # 2. the top 3 retrieved complaints
    #
    # Then it returns:
    # y, partial, or n
    label, reason = judge_with_llm(question, retrieved_text)


    # ---------------------------------------------------------
    # 18. Save the evaluation result for this question
    # ---------------------------------------------------------

    # Store the result in the results list
    results.append({
        "method": "recursive",                          # The chunking method being evaluated
        "question_index": len(results) + 1,             # The current question number
        "question": question,                           # The evaluated question
        "question_type": row["question_type"],          # Type/category of generated question
        "source_complaint_id": row["source_complaint_id"],  # Original complaint used to generate the question
        "llm_primary_label": label,                     # LLM judgment: y, partial, n, or parse_error
        "llm_reason": reason                            # Short reason from the LLM
    })


    # Save the results to CSV after every question.
    # This is useful because if the code stops, previous results are not lost.
    pd.DataFrame(results).to_csv(output_path, index=False)


    # Print progress
    print(f"Done {len(results)}/100 | Label: {label} | Reason: {reason}")


    # Pause for half a second between API calls
    time.sleep(0.5)


# ---------------------------------------------------------
# 19. Close the database connection
# ---------------------------------------------------------

# Close the cursor
cur.close()

# Close the PostgreSQL connection
conn.close()


# ---------------------------------------------------------
# 20. Create the final results DataFrame
# ---------------------------------------------------------

final_df = pd.DataFrame(results)


# ---------------------------------------------------------
# 21. Print how many results got each label
# ---------------------------------------------------------

print("\nFinal results:")

# This shows the count of y, partial, n, and parse_error labels
print(final_df["llm_primary_label"].value_counts())


# ---------------------------------------------------------
# 22. Calculate the weighted relevance score
# ---------------------------------------------------------

# Scoring logic:
# y       = 1 point
# partial = 0.5 point
# n       = 0 point
#
# The final score is the average weighted relevance score.
score = (
    (final_df["llm_primary_label"] == "y").sum()
    + 0.5 * (final_df["llm_primary_label"] == "partial").sum()
) / len(final_df)


# Print the final weighted relevance score
print("Weighted relevance score:", round(score, 4))