import psycopg2

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS complaint_summaries (
    representative_complaint_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    summary_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()

cur.close()
conn.close()

print("complaint_summaries table created successfully.")