import streamlit as st
import psycopg2
import requests
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

st.markdown("""
<style>
div[data-testid="stButton"] button {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    height: 50px;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="RAG Complaint Retrieval",
    layout="wide"
)

st.title("RAG Complaint Retrieval System")

# Initialize chat history
# Chat history is stored in Streamlit session state.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

# 2
# Load cross encoder model
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

model = load_model()

def get_connection():
    conn = psycopg2.connect(
        dbname="ragdb",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5433
    )
    register_vector(conn)
    return conn

method_options = {
    "Fixed-size chunking": "fixed_size_chunks",
    "Recursive chunking": "recursive_chunks",
    "Token-aware semantic chunking": "token_aware_semantic_chunks"
}

st.sidebar.title("Search Settings")

selected_method_name = st.sidebar.selectbox(
    "Retrieval method",
    list(method_options.keys()),
    help=(
        "Based on our evaluation, fixed-size and recursive chunking performed best overall. "
        "Token-aware semantic chunking was weaker in this test."
    )
)

# 3
# Enable feature
generate_overview = st.sidebar.checkbox(
    "Generate AI overview",
    value=False,
    help="AI overview uses a local Ollama model and may take some time."
)

table_name = method_options[selected_method_name]

# top_k = st.sidebar.slider(
#     "Number of results",
#     min_value=1,
#     max_value=10,
#     value=5
# )

top_k = 5

# 2
# Reranking UI controls
use_reranking = st.sidebar.checkbox(
    "Use cross-encoder reranking",
    value=False,
    help="Cross-encoder reranking may take a couple of minutes to complete."
)
if use_reranking:
    candidate_k = 30
    # candidate_k = st.sidebar.slider(
    #     "Reranking candidate pool",
    #     min_value=10,
    #     max_value=50,
    #     value=30,
    #     step=10
    # )
else:
    candidate_k = top_k



@st.cache_data
def load_filter_options(table_name):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(f"""
        SELECT DISTINCT product
        FROM {table_name}
        WHERE product IS NOT NULL
        ORDER BY product;
    """)
    products = [row[0] for row in cur.fetchall()]

    cur.execute(f"""
        SELECT DISTINCT state
        FROM {table_name}
        WHERE state IS NOT NULL
        ORDER BY state;
    """)
    states = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return products, states

def get_narrative_only(text):
    if text is None:
        return ""

    marker = "Complaint narrative:"

    parts = text.split(marker)

    if len(parts) > 1:
        return parts[-1].strip()

    return text.strip()

def get_reconstructed_complaint(cur, table_name, complaint_id, max_chars=3000):
    cur.execute(
        f"""
        SELECT chunk_text
        FROM {table_name}
        WHERE representative_complaint_id = %s
        ORDER BY chunk_index;
        """,
        (str(complaint_id),)
    )

    chunks = [row[0] for row in cur.fetchall()]
    full_text = "\n".join(chunks)
    narrative_only = get_narrative_only(full_text)

    return narrative_only[:max_chars]

products, states = load_filter_options(table_name)

selected_product = st.sidebar.selectbox(
    "Filter by product",
    ["All"] + products
)

selected_state = st.sidebar.selectbox(
    "Filter by state",
    ["All"] + states
)

selected_company = st.sidebar.text_input(
    "Filter by company (optional)"
)

# 3
# AI Overview
def generate_ai_overview(user_question, final_results):
    retrieved_context = ""

    # Build retrieved context
    # This collects information from retrieved complaints
    for rank, item in enumerate(final_results, start=1):
        # Prefer summary if available
        text_for_context = item["summary"] if item["summary"] else item["complaint_preview"]

        retrieved_context += f"""
        Complaint {rank}
        Complaint ID: {item["complaint_id"]}
        Product: {item["product"]}
        Issue: {item["issue"]}
        Company: {item["company"]}
        State: {item["state"]}
        Text:
        {text_for_context}
        """

        # Create prompt
        prompt = f"""
        You are generating an overview for a complaint retrieval system.

        User question:
        {user_question}

        Retrieved complaints:
        {retrieved_context}

        Write a concise answer for the user.

        Requirements:
        - Focus on what the retrieved complaints suggest overall.
        - Describe common issues and patterns.
        - Use cautious language like "the retrieved complaints suggest" or "some complaints indicate".
        - Do not mention the number of complaints.
        - Do not list complaints individually.
        - Do not invent information.
        - If the retrieved complaints are weak or limited, say so.
        - Keep the answer to 3-5 sentences.
        - Write in professional, user-friendly language.
        - Avoid vague phrases like "security, transparency, and customer support" unless they are directly supported by the complaints.
        - Be specific about the actual issues mentioned, such as unauthorized charges, delayed investigations, denied disputes, continued billing, fees, or credit reporting problems.
        - Do not emphasize specific companies unless they appear consistently across most retrieved complaints and are directly relevant to the user question.
        """



    # Call Ollama
    # Qwen generates the overview.
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False,
            "think": False
        },
        timeout=180
    )

    return response.json()["response"].strip()


def display_results(final_results, use_reranking):
    st.subheader("Supporting Complaints")

    for rank, item in enumerate(final_results, start=1):

        match_score = (1 - item["distance"]) * 100

        header = (
            f"Result {rank} | Complaint ID: {item['complaint_id']} | "
            f"Vector Score: {match_score:.2f}%"
        )

        if use_reranking:
            header += f" | Cross-Encoder Rank: #{rank}"

        with st.expander(header):

            st.write("**Product:**", item["product"])
            st.write("**Issue:**", item["issue"])
            st.write("**Company:**", item["company"])
            st.write("**State:**", item["state"])

            st.write("**Complaint Summary:**")
            if item["summary"]:
                st.write(item["summary"])
            else:
                st.info("Summary not generated yet for this complaint.")

            with st.expander("View complaint preview"):
                st.write(item["complaint_preview"])

            with st.expander(
                f"View retrieved chunk — chunk {item['chunk_index'] + 1} of {item['chunk_count_for_complaint']}"
            ):
                st.write(get_narrative_only(item["retrieved_chunk"]))
                
# Display all turns in the conversation
def display_chat_turns(chat):
    turns = chat.get("turns", [
        {
            "question": chat["question"],
            "overview": chat["overview"],
            "use_reranking": chat["use_reranking"],
            "results": chat["results"]
        }
    ])

    for i, turn in enumerate(turns, start=1):
        st.markdown("---")
        st.subheader(f"Question {i}")
        st.write(turn["question"])

        if turn["overview"]:
            st.subheader("AI-Generated Overview")
            st.write(turn["overview"])

        display_results(
            turn["results"],
            turn["use_reranking"]
        )



main_col, history_col = st.columns([4, 1])

with main_col:
    with st.form("search_form"):
        user_question = st.text_input(
            "Enter your question:",
            key="question_input"
        )
        search_clicked = st.form_submit_button("Search")


# Detect whether this is a follow-up
def is_followup_chat():
    return "selected_chat" in st.session_state



# Build the memory-enhanced query
def build_memory_enhanced_query(user_question, max_memory_questions=3):
    if is_followup_chat():
        memory = st.session_state.selected_chat.get("memory", [])
        recent_memory = memory[-max_memory_questions:]

        memory_text = "\n".join(recent_memory)

        enhanced_query = f"""
        Previous questions in this chat:
        {memory_text}

        Current question:
        {user_question}
        """
        return enhanced_query.strip()
        
    return user_question

# Load a selected previous chat
def select_previous_chat(chat):
    st.session_state.selected_chat = chat
    st.session_state.question_input = ""

    if "current_chat" in st.session_state:
        del st.session_state.current_chat

with history_col:
    with st.container(border=True):
        st.subheader("Chat History")

        if not st.session_state.chat_history:
            st.info("No previous searches yet.")
        else:
            # Display saved chats in the right-side panel
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                short_title = chat["question"]

                if len(short_title) > 35:
                    short_title = short_title[:35] + "..."

                st.button(
                    short_title,
                    key=f"history_{i}",
                    on_click=select_previous_chat,
                    args=(chat,),
                    use_container_width=True
                )



# chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.chat_history = []
    if "selected_chat" in st.session_state:
        del st.session_state.selected_chat
    st.rerun()


with main_col:
    if "current_chat" in st.session_state and not search_clicked:
        current_chat = st.session_state.current_chat

        st.subheader("Search Results")
        # st.write("**Question:**", current_chat["question"])

        # if current_chat["overview"]:
        #     st.subheader("AI-Generated Overview")
        #     st.write(current_chat["overview"])

        display_chat_turns(current_chat)
        # display_results(
        #     current_chat["results"],
        #     current_chat["use_reranking"]
        # )

    elif "selected_chat" in st.session_state and not search_clicked:
        selected_chat = st.session_state.selected_chat

        st.subheader("Selected Previous Search")
        # st.write("**Question:**", selected_chat["question"])

        # if selected_chat["overview"]:
        #     st.subheader("AI-Generated Overview")
        #     st.write(selected_chat["overview"])

        display_chat_turns(selected_chat)
        # display_results(
        #     selected_chat["results"],
        #     selected_chat["use_reranking"]
        # )

# 2
# Stage 1 message
with main_col:
    if search_clicked and user_question:
        # st.write("Follow-up:", is_followup_chat())
        if use_reranking:
            st.write(f"Stage 1: Retrieving top {candidate_k} candidates...")
        else:
            st.write("Searching database...")

        # Embed the enhanced query
        enhanced_query = build_memory_enhanced_query(user_question)

        query_embedding = model.encode(
            enhanced_query,
            normalize_embeddings=True
        )

        conn = get_connection()
        cur = conn.cursor()
        
        # 2
        # Retrieve larger candidate set
        # Without reranking: retrieve top 5 -> show top 5
        # With reranking: retrieve top 30 -> rerank -> show best 5
        search_limit = candidate_k if use_reranking else top_k


        # 1
        # Retrieve summaries from database
        # Joins the retrieval table with the complaint_summaries table
        # Retrieves the summary if one exists
        # Makes the summary available for display and AI overview generation
        sql_query = f"""
        SELECT
            c.representative_complaint_id,
            c.product,
            c.issue,
            c.company,
            c.state,
            LEFT(c.chunk_text, 1200),
            c.chunk_index,
            c.chunk_count_for_complaint,
            s.summary,
            c.embedding <=> %s AS distance
        FROM {table_name} c
        LEFT JOIN complaint_summaries s
            ON c.representative_complaint_id = s.representative_complaint_id
        WHERE 1=1
        """

        params = [query_embedding]

        if selected_product != "All":
            sql_query += " AND c.product = %s"
            params.append(selected_product)

        if selected_state != "All":
            sql_query += " AND c.state = %s"
            params.append(selected_state)

        if selected_company:
            sql_query += " AND c.company ILIKE %s"
            params.append(f"%{selected_company}%")

        sql_query += """
        ORDER BY c.embedding <=> %s
        LIMIT %s;
        """

        params.append(query_embedding)
        params.append(search_limit)

        cur.execute(sql_query, params)
        results = cur.fetchall()

        if len(results) == 0:
            st.warning(
                "No results found with current filters. Try another retrieval method, "
                "removing one filter, or using a broader company name."
            )

        else:
            final_results = []

            # 2
            # Entire reranking pipeline
            if use_reranking:
                st.write("Stage 2: Cross-encoder reranking...")

                cross_encoder = load_cross_encoder()

                pairs = []
                result_items = []

                for r in results:
                    (
                        complaint_id,
                        product,
                        issue,
                        company,
                        state,
                        retrieved_chunk,
                        chunk_index,
                        chunk_count_for_complaint,
                        summary,
                        distance
                    ) = r

                    complaint_text = get_reconstructed_complaint(
                        cur,
                        table_name,
                        complaint_id,
                        max_chars=3000
                    )
                    
                    pairs.append((user_question, complaint_text))

                    # Reranking mode
                    result_items.append({
                        "complaint_id": complaint_id,
                        "product": product,
                        "issue": issue,
                        "company": company,
                        "state": state,
                        "retrieved_chunk": retrieved_chunk,
                        "chunk_index": chunk_index,
                        "chunk_count_for_complaint": chunk_count_for_complaint,
                        "summary": summary, # Store summary inside retrieved results
                        "distance": distance,
                        "complaint_preview": complaint_text
                    })

                # Score candidates
                cross_scores = cross_encoder.predict(pairs)
                # Attach scores
                for item, score in zip(result_items, cross_scores):
                    item["cross_encoder_score"] = float(score)
                # Reorder candidates
                final_results = sorted(
                    result_items,
                    key=lambda x: x["cross_encoder_score"],
                    reverse=True
                )[:top_k]

            else:
                for r in results:
                    (
                        complaint_id,
                        product,
                        issue,
                        company,
                        state,
                        retrieved_chunk,
                        chunk_index,
                        chunk_count_for_complaint,
                        summary,
                        distance
                    ) = r

                    complaint_preview = get_reconstructed_complaint(
                        cur,
                        table_name,
                        complaint_id,
                        max_chars=3000
                    )

                    # 1
                    # Normal retrieval mode
                    final_results.append({
                        "complaint_id": complaint_id,
                        "product": product,
                        "issue": issue,
                        "company": company,
                        "state": state,
                        "retrieved_chunk": retrieved_chunk,
                        "chunk_index": chunk_index,
                        "chunk_count_for_complaint": chunk_count_for_complaint,
                        "summary": summary, # Store summary inside retrieved results
                        "distance": distance,
                        "complaint_preview": complaint_preview,
                        "cross_encoder_score": None
                    })

            # 2
            # Reranking display
            mode_text = "with cross-encoder reranking" if use_reranking else "using vector search"
            st.success(f"Found {len(final_results)} results {mode_text} using {selected_method_name}")


            # Generate overview
            if generate_overview:
                with st.spinner("Generating AI overview..."):
                    overview = generate_ai_overview(
                        user_question,
                        final_results
                    )
                # Display overview
                st.subheader("AI-Generated Overview")
                st.write(overview)
            else:
                overview = None
            # chat history
            # Save follow-up as a new turn
            if is_followup_chat():
                selected_chat = st.session_state.selected_chat
                selected_chat["memory"].append(user_question)

                new_turn = {
                    "question": user_question,
                    "overview": overview,
                    "retrieval_method": selected_method_name,
                    "use_reranking": use_reranking,
                    "results": final_results
                }

                if "turns" not in selected_chat:
                    selected_chat["turns"] = [
                        {
                            "question": selected_chat["question"],
                            "overview": selected_chat["overview"],
                            "retrieval_method": selected_chat["retrieval_method"],
                            "use_reranking": selected_chat["use_reranking"],
                            "results": selected_chat["results"]
                        }
                    ]

                selected_chat["turns"].append(new_turn)

                selected_chat["overview"] = overview
                selected_chat["results"] = final_results

                st.session_state.current_chat = selected_chat
            else:

                # Create and save a new chat after search
                new_chat = {
                    "question": user_question,
                    "overview": overview,
                    "retrieval_method": selected_method_name,
                    "use_reranking": use_reranking,
                    "results": final_results,
                    "memory": [user_question]
                }

                st.session_state.chat_history.append(new_chat)
                st.session_state.current_chat = new_chat


            if "selected_chat" in st.session_state:
                del st.session_state.selected_chat

            st.rerun()

        cur.close()
        conn.close()



