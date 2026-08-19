import streamlit as st
import fitz
import faiss
import numpy as np
import re
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Intelligent Document Q&A",
    page_icon="📄"
)

st.title("📄 Intelligent Document Q&A")

st.write(
    "Upload multiple PDFs and ask questions about "
    "their contents."
)


# ==================================================
# CONVERSATION MEMORY
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# STEP 21:
# EVALUATION HISTORY
# ==================================================

if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []


# ==================================================
# STEP 7:
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(uploaded_file):

    pdf_bytes = uploaded_file.read()

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    documents = []

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        blocks = page.get_text("blocks")

        blocks = sorted(
            blocks,
            key=lambda block: (
                block[1],
                block[0]
            )
        )

        page_lines = []

        for block in blocks:

            text = block[4].strip()

            if text:
                page_lines.append(text)

        page_text = "\n\n".join(
            page_lines
        )

        documents.append({
            "page": page_number,
            "text": page_text,
            "filename": uploaded_file.name
        })

    pdf.close()

    return documents


# ==================================================
# STEP 19:
# CLEAN TEXT
# ==================================================

def clean_text(text):

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()


# ==================================================
# STEP 19:
# BETTER CHUNKING
# ==================================================

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=120,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ],

        length_function=len
    )

    chunks = []

    for document in documents:

        cleaned_text = clean_text(
            document["text"]
        )

        if not cleaned_text:
            continue

        page_chunks = text_splitter.split_text(
            cleaned_text
        )

        for chunk in page_chunks:

            chunk = chunk.strip()

            if len(chunk) < 50:
                continue

            chunks.append({
                "page": document["page"],
                "text": chunk,
                "filename": document["filename"]
            })

    return chunks


# ==================================================
# STEP 9:
# EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


def create_embeddings(chunks):

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


# ==================================================
# STEP 10:
# FAISS
# ==================================================

def create_faiss_index(embeddings):

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ==================================================
# STEP 20:
# CROSS-ENCODER RERANKER
# ==================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


# ==================================================
# STEP 11 + STEP 16:
# FAISS RETRIEVAL
# ==================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    top_k=5,
    max_distance=1.50
):

    model = load_embedding_model()

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        distance = float(distance)

        if distance <= max_distance:

            retrieved_chunks.append({
                "page": chunks[index_position]["page"],
                "text": chunks[index_position]["text"],
                "filename": chunks[index_position]["filename"],
                "distance": distance
            })

    return retrieved_chunks


# ==================================================
# STEP 20:
# RERANK CHUNKS
# ==================================================

def rerank_chunks(
    question,
    retrieved_chunks,
    top_k=3
):

    if not retrieved_chunks:
        return []

    reranker = load_reranker()

    pairs = []

    for chunk in retrieved_chunks:

        pairs.append([
            question,
            chunk["text"]
        ])

    scores = reranker.predict(pairs)

    reranked_chunks = []

    for chunk, score in zip(
        retrieved_chunks,
        scores
    ):

        new_chunk = chunk.copy()

        new_chunk["rerank_score"] = float(
            score
        )

        reranked_chunks.append(
            new_chunk
        )

    reranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_chunks[:top_k]


# ==================================================
# GEMINI CLIENT
# ==================================================

api_key = st.secrets[
    "GEMINI_API_KEY"
]

client = genai.Client(
    api_key=api_key
)


# ==================================================
# STEP 18:
# CHAT HISTORY
# ==================================================

def format_chat_history():

    if not st.session_state.chat_history:

        return "No previous conversation."

    history = []

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            history.append(
                f"User: {message['content']}"
            )

        else:

            history.append(
                f"Assistant: {message['content']}"
            )

    return "\n".join(history)


# ==================================================
# STEP 12 + STEP 14 + STEP 18:
# GEMINI ANSWER
# ==================================================

def generate_answer(
    question,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "I could not find the answer "
            "in the provided document."
        )

    context_parts = []

    for result in retrieved_chunks:

        context_parts.append(
            f"Document: {result['filename']}\n"
            f"Page: {result['page']}\n"
            f"Content:\n{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    chat_history = format_chat_history()

    prompt = f"""
You are a strict document question-answering assistant.

Answer the user's question ONLY using information
from the provided document context.

Use previous conversation only to understand
references such as "it", "they", "them",
"that project", or "which one".

Rules:

1. Use ONLY the document context.
2. Do NOT use general knowledge.
3. Do NOT make assumptions.
4. Do NOT invent facts.
5. If the answer is not present in the context,
   respond exactly:

I could not find the answer in the provided document.

6. Keep the answer clear and concise.

PREVIOUS CONVERSATION:
{chat_history}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text.strip()


# ==================================================
# STEP 21:
# CALCULATE EVALUATION METRICS
# ==================================================

def calculate_evaluation_metrics(
    retrieved_chunks,
    reranked_chunks,
    generation_time
):

    # ----------------------------------------------
    # Number of retrieved candidates
    # ----------------------------------------------

    retrieval_count = len(
        retrieved_chunks
    )

    # ----------------------------------------------
    # Number after reranking
    # ----------------------------------------------

    reranked_count = len(
        reranked_chunks
    )

    # ----------------------------------------------
    # Average FAISS distance
    # ----------------------------------------------

    if retrieved_chunks:

        average_distance = (

            sum(
                chunk["distance"]
                for chunk in retrieved_chunks
            )
            /
            len(retrieved_chunks)

        )

    else:

        average_distance = 0.0

    # ----------------------------------------------
    # Best reranker score
    # ----------------------------------------------

    if reranked_chunks:

        best_rerank_score = max(

            chunk["rerank_score"]

            for chunk in reranked_chunks

        )

    else:

        best_rerank_score = 0.0

    # ----------------------------------------------
    # Average reranker score
    # ----------------------------------------------

    if reranked_chunks:

        average_rerank_score = (

            sum(
                chunk["rerank_score"]
                for chunk in reranked_chunks
            )
            /
            len(reranked_chunks)

        )

    else:

        average_rerank_score = 0.0

    return {

        "retrieval_count": retrieval_count,

        "reranked_count": reranked_count,

        "average_distance": average_distance,

        "best_rerank_score": best_rerank_score,

        "average_rerank_score": average_rerank_score,

        "generation_time": generation_time

    }


# ==================================================
# STEP 17:
# MULTIPLE PDF UPLOAD
# ==================================================

uploaded_files = st.file_uploader(

    "Upload your PDFs",

    type=["pdf"],

    accept_multiple_files=True

)


# ==================================================
# PROCESS DOCUMENTS
# ==================================================

if uploaded_files:

    st.success(

        f"{len(uploaded_files)} PDF(s) "
        f"uploaded successfully!"

    )

    # --------------------------------------------------
    # PDF EXTRACTION
    # --------------------------------------------------

    all_documents = []

    with st.spinner(
        "Extracting text from PDFs..."
    ):

        for uploaded_file in uploaded_files:

            documents = extract_pdf_text(
                uploaded_file
            )

            all_documents.extend(
                documents
            )

    st.success(

        f"Extracted text from "
        f"{len(all_documents)} page(s) "
        f"across {len(uploaded_files)} PDF(s)."

    )

    # --------------------------------------------------
    # UPLOADED FILES
    # --------------------------------------------------

    with st.expander(
        "📚 Uploaded Documents"
    ):

        for uploaded_file in uploaded_files:

            st.write(
                f"📄 {uploaded_file.name}"
            )

    # --------------------------------------------------
    # EXTRACTED TEXT
    # --------------------------------------------------

    with st.expander(
        "📄 View extracted text"
    ):

        for document in all_documents:

            st.markdown(

                f"### {document['filename']} "
                f"— Page {document['page']}"

            )

            st.text(
                document["text"]
            )

    # --------------------------------------------------
    # CHUNKING
    # --------------------------------------------------

    chunks = create_chunks(
        all_documents
    )

    st.success(

        f"Created {len(chunks)} "
        f"improved text chunks."

    )

    if chunks:

        average_length = (

            sum(
                len(chunk["text"])
                for chunk in chunks
            )
            /
            len(chunks)

        )

        st.info(

            f"📊 Average chunk length: "
            f"{average_length:.0f} characters"

        )

    # --------------------------------------------------
    # VIEW CHUNKS
    # --------------------------------------------------

    with st.expander(
        "🧩 View text chunks"
    ):

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            st.markdown(
                f"### Chunk {i}"
            )

            st.caption(

                f"📄 {chunk['filename']} "
                f"— Page {chunk['page']}"

            )

            st.write(
                chunk["text"]
            )

    # --------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------

    with st.spinner(
        "Creating embeddings..."
    ):

        embeddings = create_embeddings(
            chunks
        )

    st.success(

        f"Created embeddings with "
        f"dimension {embeddings.shape[1]}."

    )

    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    with st.spinner(
        "Creating FAISS vector database..."
    ):

        index = create_faiss_index(
            embeddings
        )

    st.success(

        f"FAISS database created with "
        f"{index.ntotal} vectors."

    )

    # --------------------------------------------------
    # QUESTION
    # --------------------------------------------------

    question = st.text_input(

        "Ask a question about your documents:"

    )

    if st.button("🔍 Ask"):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            # ==========================================
            # FAISS RETRIEVAL
            # ==========================================

            with st.spinner(
                "Searching documents..."
            ):

                retrieval_start = time.perf_counter()

                retrieved_chunks = retrieve_chunks(

                    question,

                    index,

                    chunks,

                    top_k=5,

                    max_distance=1.50

                )

                retrieval_time = (
                    time.perf_counter()
                    - retrieval_start
                )

            # ==========================================
            # NO RESULTS
            # ==========================================

            if not retrieved_chunks:

                answer = (
                    "I could not find the answer "
                    "in the provided document."
                )

                st.warning(answer)

                st.session_state.chat_history.append({

                    "role": "user",

                    "content": question

                })

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": answer

                })

            else:

                # ==========================================
                # FAISS RESULTS
                # ==========================================

                st.subheader(
                    "🔎 FAISS Retrieved Chunks"
                )

                for i, result in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"### Candidate {i}"
                    )

                    st.caption(

                        f"📄 {result['filename']} "
                        f"— Page {result['page']}"

                    )

                    st.write(
                        result["text"]
                    )

                    st.caption(

                        f"FAISS Distance: "
                        f"{result['distance']:.4f}"

                    )

                # ==========================================
                # RERANK
                # ==========================================

                with st.spinner(
                    "🎯 Reranking relevant chunks..."
                ):

                    rerank_start = time.perf_counter()

                    reranked_chunks = rerank_chunks(

                        question,

                        retrieved_chunks,

                        top_k=3

                    )

                    rerank_time = (
                        time.perf_counter()
                        - rerank_start
                    )

                # ==========================================
                # RERANKED RESULTS
                # ==========================================

                st.subheader(
                    "🎯 Reranked Context"
                )

                for i, result in enumerate(
                    reranked_chunks,
                    start=1
                ):

                    st.markdown(

                        f"### Reranked Result {i}"

                    )

                    st.caption(

                        f"📄 {result['filename']} "
                        f"— Page {result['page']}"

                    )

                    st.write(
                        result["text"]
                    )

                    st.caption(

                        f"Reranker Score: "
                        f"{result['rerank_score']:.4f}"

                    )

                # ==========================================
                # GEMINI
                # ==========================================

                with st.spinner(
                    "🤖 Gemini is generating the answer..."
                ):

                    generation_start = time.perf_counter()

                    answer = generate_answer(

                        question,

                        reranked_chunks

                    )

                    generation_time = (
                        time.perf_counter()
                        - generation_start
                    )

                # ==========================================
                # EVALUATION
                # ==========================================

                metrics = calculate_evaluation_metrics(

                    retrieved_chunks,

                    reranked_chunks,

                    generation_time

                )

                metrics["retrieval_time"] = (
                    retrieval_time
                )

                metrics["rerank_time"] = (
                    rerank_time
                )

                metrics["question"] = question

                st.session_state.evaluation_history.append(
                    metrics
                )

                # ==========================================
                # SAVE CHAT
                # ==========================================

                st.session_state.chat_history.append({

                    "role": "user",

                    "content": question

                })

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": answer

                })

                # ==========================================
                # FINAL ANSWER
                # ==========================================

                st.subheader(
                    "🤖 Answer"
                )

                st.write(
                    answer
                )

                # ==========================================
                # SOURCES
                # ==========================================

                st.subheader(
                    "📚 Sources"
                )

                source_pages = sorted(

                    set(

                        (
                            result["filename"],
                            result["page"]
                        )

                        for result
                        in reranked_chunks

                    )

                )

                for filename, page in source_pages:

                    st.write(

                        f"📄 **{filename}** "
                        f"— Page {page}"

                    )

                # ==========================================
                # STEP 21:
                # RAG EVALUATION
                # ==========================================

                st.divider()

                st.subheader(
                    "📊 RAG Evaluation"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "FAISS Candidates",
                        metrics["retrieval_count"]
                    )

                with col2:

                    st.metric(
                        "Reranked Chunks",
                        metrics["reranked_count"]
                    )

                with col3:

                    st.metric(
                        "Generation Time",
                        f"{metrics['generation_time']:.2f}s"
                    )

                col4, col5, col6 = st.columns(3)

                with col4:

                    st.metric(
                        "Avg FAISS Distance",
                        f"{metrics['average_distance']:.3f}"
                    )

                with col5:

                    st.metric(
                        "Best Reranker Score",
                        f"{metrics['best_rerank_score']:.3f}"
                    )

                with col6:

                    st.metric(
                        "Avg Reranker Score",
                        f"{metrics['average_rerank_score']:.3f}"
                    )

                with st.expander(
                    "⚙️ Detailed Timing"
                ):

                    st.write(
                        f"FAISS retrieval time: "
                        f"{metrics['retrieval_time']:.4f} seconds"
                    )

                    st.write(
                        f"Reranking time: "
                        f"{metrics['rerank_time']:.4f} seconds"
                    )

                    st.write(
                        f"Gemini generation time: "
                        f"{metrics['generation_time']:.4f} seconds"
                    )


# ==================================================
# STEP 21:
# EVALUATION HISTORY
# ==================================================

if st.session_state.evaluation_history:

    st.divider()

    st.subheader(
        "📈 Evaluation History"
    )

    evaluation_rows = []

    for item in st.session_state.evaluation_history:

        evaluation_rows.append({

            "Question": item["question"],

            "FAISS Candidates":
                item["retrieval_count"],

            "Reranked Chunks":
                item["reranked_count"],

            "Avg FAISS Distance":
                round(
                    item["average_distance"],
                    3
                ),

            "Best Reranker Score":
                round(
                    item["best_rerank_score"],
                    3
                ),

            "Generation Time (s)":
                round(
                    item["generation_time"],
                    2
                )

        })

    st.dataframe(
        evaluation_rows,
        use_container_width=True
    )


# ==================================================
# STEP 18:
# CONVERSATION HISTORY
# ==================================================

if st.session_state.chat_history:

    st.divider()

    st.subheader(
        "💬 Conversation History"
    )

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            st.markdown(
                f"**👤 You:** "
                f"{message['content']}"
            )

        else:

            st.markdown(
                f"**🤖 Assistant:** "
                f"{message['content']}"
            )

    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.chat_history = []

        st.rerun()