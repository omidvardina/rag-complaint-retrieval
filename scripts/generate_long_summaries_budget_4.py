# GPT
# import os
# import time
# import psycopg2
# import tiktoken
# from openai import OpenAI

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# TABLE_NAME = "fixed_size_chunks"

# BUDGET_USD = 4.50
# SAFETY_STOP_USD = 4.00

# MIN_COMPLAINT_CHARS = 2000

# INPUT_PRICE_PER_1M = 0.40
# OUTPUT_PRICE_PER_1M = 1.60

# MODEL_NAME = "gpt-4.1-mini"

# PROMPT_TEMPLATE = """
# You are summarizing consumer financial complaints for a complaint retrieval system.

# Produce a concise summary of 3-4 sentences.

# Focus on:
# - the consumer’s main issue,
# - relevant events leading to the complaint,
# - any actions taken by the company, including the company behavior or problem being reported.

# Preserve important financial, legal, fraud, billing, debt collection, account management, or credit reporting details when they are central to the complaint.

# Do not include unnecessary background information, greetings, opinions, or repetitive details.

# Do not invent information that is not explicitly stated.

# Write in clear, neutral language.

# Complaint:
# {complaint}
# """

# def get_narrative_only(text):
#     if text is None:
#         return ""

#     marker = "Complaint narrative:"

#     if marker in text:
#         return text.split(marker, 1)[1].strip()

#     return text.strip()


# def prepare_text_for_summary(narrative, max_chars=6000):
#     if len(narrative) <= max_chars:
#         return narrative

#     first_part = narrative[:4000]
#     last_part = narrative[-2000:]

#     return (
#         first_part
#         + "\n\n[...middle part omitted...]\n\n"
#         + last_part
#     )


# def estimate_cost(input_tokens, output_tokens):
#     input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
#     output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
#     return input_cost + output_cost


# encoding = tiktoken.encoding_for_model("gpt-4.1-mini")

# conn = psycopg2.connect(
#     dbname="ragdb",
#     user="postgres",
#     password="postgres",
#     host="localhost",
#     port=5433
# )

# cur = conn.cursor()

# cur.execute(f"""
# WITH complaint_lengths AS (
#     SELECT
#         representative_complaint_id,
#         SUM(LENGTH(chunk_text)) AS total_text_length
#     FROM {TABLE_NAME}
#     GROUP BY representative_complaint_id
# )
# SELECT representative_complaint_id
# FROM complaint_lengths
# WHERE total_text_length > %s
# AND representative_complaint_id NOT IN (
#     SELECT representative_complaint_id
#     FROM complaint_summaries
# )
# ORDER BY total_text_length DESC;
# """, (MIN_COMPLAINT_CHARS,))

# complaint_ids = [row[0] for row in cur.fetchall()]

# print(f"Found {len(complaint_ids)} long complaints without summaries.")
# print(f"Budget limit: ${BUDGET_USD}")
# print(f"Safety stop: ${SAFETY_STOP_USD}")

# total_estimated_cost = 0.0
# saved_count = 0
# skipped_count = 0

# for idx, complaint_id in enumerate(complaint_ids, start=1):

#     cur.execute(f"""
#     SELECT chunk_text
#     FROM {TABLE_NAME}
#     WHERE representative_complaint_id = %s
#     ORDER BY chunk_index;
#     """, (str(complaint_id),))

#     chunks = [row[0] for row in cur.fetchall()]

#     if not chunks:
#         skipped_count += 1
#         continue

#     full_text = "\n".join(chunks)
#     narrative = get_narrative_only(full_text)

#     if len(narrative) <= MIN_COMPLAINT_CHARS:
#         skipped_count += 1
#         continue

#     summary_input = prepare_text_for_summary(narrative)

#     prompt = PROMPT_TEMPLATE.format(
#         complaint=summary_input
#     )

#     input_tokens = len(encoding.encode(prompt))

#     estimated_output_tokens = 180

#     estimated_this_cost = estimate_cost(
#         input_tokens,
#         estimated_output_tokens
#     )

#     if total_estimated_cost + estimated_this_cost >= SAFETY_STOP_USD:
#         print("\nStopping before exceeding safety budget.")
#         break

#     try:
#         response = client.responses.create(
#             model=MODEL_NAME,
#             input=prompt,
#             temperature=0
#         )

#         summary = response.output_text.strip()

#         output_tokens = len(encoding.encode(summary))

#         actual_estimated_cost = estimate_cost(
#             input_tokens,
#             output_tokens
#         )

#         total_estimated_cost += actual_estimated_cost

#         cur.execute("""
#         INSERT INTO complaint_summaries (
#             representative_complaint_id,
#             summary,
#             summary_model
#         )
#         VALUES (%s, %s, %s)
#         ON CONFLICT (representative_complaint_id)
#         DO NOTHING;
#         """, (
#             str(complaint_id),
#             summary,
#             MODEL_NAME
#         ))

#         conn.commit()

#         saved_count += 1

#         print(
#             f"{saved_count} saved | "
#             f"Complaint {complaint_id} | "
#             f"Input tokens: {input_tokens} | "
#             f"Output tokens: {output_tokens} | "
#             f"Estimated total cost: ${total_estimated_cost:.4f}"
#         )

#         time.sleep(0.3)

#     except Exception as e:
#         print(f"Error on complaint {complaint_id}: {e}")
#         skipped_count += 1
#         time.sleep(1)

# cur.close()
# conn.close()

# print("\nDone.")
# print("Saved summaries:", saved_count)
# print("Skipped complaints:", skipped_count)
# print(f"Final estimated cost: ${total_estimated_cost:.4f}")


# Ollama
import time
import psycopg2
import requests

TABLE_NAME = "fixed_size_chunks"
MODEL_NAME = "qwen3:8b"

MIN_COMPLAINT_CHARS = 2000
MAX_COMPLAINTS_TO_PROCESS = 500

PROMPT_TEMPLATE = """
You are summarizing consumer financial complaints for a complaint retrieval system.

Produce a concise summary of 3-4 sentences.

Focus on:
- the consumer’s main issue,
- relevant events leading to the complaint,
- any actions taken by the company, including the company behavior or problem being reported.

Preserve important financial, legal, fraud, billing, debt collection, account management, or credit reporting details when they are central to the complaint.

Do not include unnecessary background information, greetings, opinions, or repetitive details.

Do not invent information that is not explicitly stated.

Write in clear, neutral language.

Complaint:
{complaint}
"""

def get_narrative_only(text):
    if text is None:
        return ""

    marker = "Complaint narrative:"

    if marker in text:
        return text.split(marker, 1)[1].strip()

    return text.strip()


def prepare_text_for_summary(narrative, max_chars=6000):
    if len(narrative) <= max_chars:
        return narrative

    first_part = narrative[:4000]
    last_part = narrative[-2000:]

    return (
        first_part
        + "\n\n[...middle part omitted...]\n\n"
        + last_part
    )


conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

cur = conn.cursor()

cur.execute(f"""
WITH complaint_lengths AS (
    SELECT
        representative_complaint_id,
        SUM(LENGTH(chunk_text)) AS total_text_length
    FROM {TABLE_NAME}
    GROUP BY representative_complaint_id
)
SELECT representative_complaint_id
FROM complaint_lengths
WHERE total_text_length > %s
AND representative_complaint_id NOT IN (
    SELECT representative_complaint_id
    FROM complaint_summaries
)
ORDER BY total_text_length DESC
LIMIT %s;
""", (MIN_COMPLAINT_CHARS, MAX_COMPLAINTS_TO_PROCESS))

complaint_ids = [row[0] for row in cur.fetchall()]

print(f"Found {len(complaint_ids)} long complaints without summaries.")
print(f"Using local model: {MODEL_NAME}")

saved_count = 0
skipped_count = 0

for idx, complaint_id in enumerate(complaint_ids, start=1):

    cur.execute(f"""
    SELECT chunk_text
    FROM {TABLE_NAME}
    WHERE representative_complaint_id = %s
    ORDER BY chunk_index;
    """, (str(complaint_id),))

    chunks = [row[0] for row in cur.fetchall()]

    if not chunks:
        skipped_count += 1
        continue

    full_text = "\n".join(chunks)
    narrative = get_narrative_only(full_text)

    if len(narrative) <= MIN_COMPLAINT_CHARS:
        skipped_count += 1
        continue

    summary_input = prepare_text_for_summary(narrative)

    prompt = PROMPT_TEMPLATE.format(
        complaint=summary_input
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "think": False
            },
            timeout=180
        )

        response.raise_for_status()

        summary = response.json()["response"].strip()

        cur.execute("""
        INSERT INTO complaint_summaries (
            representative_complaint_id,
            summary,
            summary_model
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (representative_complaint_id)
        DO NOTHING;
        """, (
            str(complaint_id),
            summary,
            MODEL_NAME
        ))

        conn.commit()

        saved_count += 1

        print(
            f"{saved_count}/{len(complaint_ids)} saved | "
            f"Complaint {complaint_id}"
        )

        time.sleep(0.3)

    except Exception as e:
        print(f"Error on complaint {complaint_id}: {e}")
        skipped_count += 1
        time.sleep(1)

cur.close()
conn.close()

print("\nDone.")
print("Saved summaries:", saved_count)
print("Skipped complaints:", skipped_count)