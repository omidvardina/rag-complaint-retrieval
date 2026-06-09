import time
import psycopg2
import requests

TABLE_NAME = "fixed_size_chunks"
MODEL_NAME = "qwen3:8b"

DEMO_COMPLAINT_IDS = [
    "1424373",
    "1818060",
    "1995184",
    "1795329",
    "1799277"
]

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
    parts = text.split(marker)

    if len(parts) > 1:
        return parts[-1].strip()

    return text.strip()

def prepare_text_for_summary(narrative, max_chars=6000):
    if len(narrative) <= max_chars:
        return narrative

    return narrative[:4000] + "\n\n[...middle part omitted...]\n\n" + narrative[-2000:]

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

cur = conn.cursor()

for idx, complaint_id in enumerate(DEMO_COMPLAINT_IDS, start=1):

    cur.execute(f"""
    SELECT chunk_text
    FROM {TABLE_NAME}
    WHERE representative_complaint_id = %s
    ORDER BY chunk_index;
    """, (complaint_id,))

    chunks = [row[0] for row in cur.fetchall()]

    if not chunks:
        print(f"{idx}/{len(DEMO_COMPLAINT_IDS)} skipped | Complaint {complaint_id} not found")
        continue

    full_text = "\n".join(chunks)
    narrative = get_narrative_only(full_text)
    summary_input = prepare_text_for_summary(narrative)

    prompt = PROMPT_TEMPLATE.format(complaint=summary_input)

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
        DO UPDATE SET
            summary = EXCLUDED.summary,
            summary_model = EXCLUDED.summary_model;
        """, (
            complaint_id,
            summary,
            MODEL_NAME
        ))

        conn.commit()

        print("=" * 80)
        print(f"{idx}/{len(DEMO_COMPLAINT_IDS)} saved | Complaint {complaint_id}")
        print(summary)

        time.sleep(0.3)

    except Exception as e:
        print(f"Error on complaint {complaint_id}: {e}")
        time.sleep(1)

cur.close()
conn.close()

print("\nDone.")