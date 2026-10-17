"""
Databricks App-ready backend for Streamlit UI.
Based on your 09_final_application notebook, fixed for Databricks Apps:
- no writes to /Workspace
- app reads local app files
- runtime writes go to /tmp
- every UI question is tracked in MLflow
"""

import os
import json
import uuid
import requests
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import mlflow


# =========================
# PROJECT CONFIGURATION
# =========================

PROJECT_NAME = "domain_specific_rag_movie_analysis"
PROJECT_TITLE = "Agentic Domain-Specific RAG System for Film Analysis"
DOMAIN_NAME = "Narrative, thematic, and character analysis of curated film documents"

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

COLLECTION_NAME = "movie_analysis_chunks"
TOP_K = 5
TEMPERATURE = 0

GOOGLE_RESULTS_LIMIT = 5
YOUTUBE_RESULTS_LIMIT = 5
MEMORY_LOOKBACK_TURNS = 5

APP_DIR = Path(__file__).resolve().parent

RUNTIME_DIR = Path("/tmp/rag_chatbot_ui")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = APP_DIR / ".env"
CHUNKS_CSV_PATH = APP_DIR / "chunks_metadata.csv"

MEMORY_DIR = RUNTIME_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

SESSION_LOG_CSV_PATH = MEMORY_DIR / "session_logs.csv"
SESSION_SUMMARY_JSON_PATH = MEMORY_DIR / "session_summary.json"

CHROMA_PERSIST_DIR = str(RUNTIME_DIR / "chroma_db")

MLFLOW_EXPERIMENT_NAME = "/Shared/rag-chatbot-ui"

print("PROJECT_NAME =", PROJECT_NAME)
print("PROJECT_TITLE =", PROJECT_TITLE)
print("DOMAIN_NAME =", DOMAIN_NAME)
print("APP_DIR =", APP_DIR)
print("ENV_PATH =", ENV_PATH)
print("CHUNKS_CSV_PATH =", CHUNKS_CSV_PATH)
print("SESSION_LOG_CSV_PATH =", SESSION_LOG_CSV_PATH)
print("SESSION_SUMMARY_JSON_PATH =", SESSION_SUMMARY_JSON_PATH)
print("CHROMA_PERSIST_DIR =", CHROMA_PERSIST_DIR)
print("MLFLOW_EXPERIMENT_NAME =", MLFLOW_EXPERIMENT_NAME)


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

if not ENV_PATH.exists():
    raise FileNotFoundError(
        f".env file not found at: {ENV_PATH}\n"
        "Put a .env file next to app.py and rag_backend.py in your Databricks App source."
    )

load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not found in .env file.")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# =========================
# INITIALIZE MODELS
# =========================

embeddings_model = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
    api_key=OPENAI_API_KEY
)

print("Embeddings model and chat model initialized successfully.")


# =========================
# INITIALIZE MLFLOW
# =========================

import os
import mlflow

# Try to use Databricks tracking if available, fall back to local
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

try:
    if DATABRICKS_HOST:
        mlflow.set_tracking_uri("databricks")
    else:
        # Use local MLflow backend for development
        mlflow.set_tracking_uri(f"file:///{RUNTIME_DIR}/mlflow")
except Exception as e:
    print(f"MLflow tracking URI setup warning: {e}")

if MLFLOW_EXPERIMENT_ID:
    try:
        mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
        print("Using MLflow Experiment ID:", MLFLOW_EXPERIMENT_ID)
    except Exception as e:
        print(f"MLflow experiment setup warning: {e}")
        try:
            mlflow.set_experiment(experiment_name="rag-chatbot-ui")
        except Exception as e2:
            print(f"MLflow experiment name setup warning: {e2}")
else:
    try:
        mlflow.set_experiment(experiment_name="rag-chatbot-ui")
        print("Using local MLflow experiment: rag-chatbot-ui")
    except Exception as e:
        print(f"MLflow setup warning (continuing without MLflow): {e}")

# =========================
# CONNECT TO CHROMADB
# =========================

persist_path = Path(CHROMA_PERSIST_DIR)
persist_path.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
collection_count = collection.count()
print("Initial Chroma collection count:", collection_count)


# =========================
# CHROMA REBUILD HELPERS
# =========================

def load_chunks_dataframe(chunks_csv_path: Path) -> pd.DataFrame:
    if not chunks_csv_path.exists():
        raise FileNotFoundError(
            f"chunks_metadata.csv not found: {chunks_csv_path}\n"
            "Upload chunks_metadata.csv into the Databricks App source folder."
        )

    chunks_df = pd.read_csv(chunks_csv_path)

    required_columns = [
        "chunk_id",
        "doc_id",
        "file_name",
        "source_path",
        "chunk_index",
        "chunk_text",
        "chunk_char_count",
        "chunk_token_count",
        "chunk_size",
        "chunk_overlap",
        "domain_tag",
        "created_at"
    ]

    missing_cols = [c for c in required_columns if c not in chunks_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in chunks metadata: {missing_cols}")

    chunks_df = chunks_df.copy()
    chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str).str.strip()

    empty_chunks = chunks_df[chunks_df["chunk_text"] == ""]
    if len(empty_chunks) > 0:
        raise ValueError("Empty chunk_text values found in chunks metadata.")

    return chunks_df


def rebuild_collection_from_chunks(chunks_df: pd.DataFrame) -> int:
    global collection

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    texts = chunks_df["chunk_text"].tolist()
    ids = chunks_df["chunk_id"].astype(str).tolist()

    metadata_list = []
    for _, row in chunks_df.iterrows():
        metadata_list.append({
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "file_name": str(row["file_name"]),
            "source_path": str(row["source_path"]),
            "chunk_index": int(row["chunk_index"]),
            "chunk_char_count": int(row["chunk_char_count"]),
            "chunk_token_count": int(row["chunk_token_count"]),
            "chunk_size": int(row["chunk_size"]),
            "chunk_overlap": int(row["chunk_overlap"]),
            "domain_tag": str(row["domain_tag"]),
            "created_at": str(row["created_at"])
        })

    vectors = embeddings_model.embed_documents(texts)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadata_list,
        embeddings=vectors
    )

    return collection.count()


# =========================
# ENSURE CHROMA COLLECTION IS READY
# =========================

if collection_count == 0:
    print("Chroma collection is empty. Rebuilding from chunks_metadata.csv ...")
    chunks_df = load_chunks_dataframe(CHUNKS_CSV_PATH)
    collection_count = rebuild_collection_from_chunks(chunks_df)
    print("Rebuilt Chroma collection count:", collection_count)

if collection_count == 0:
    raise ValueError("Chroma collection is still empty after rebuild.")

print("Chroma collection ready.")
print("Collection count:", collection_count)


# =========================
# INTERNAL RETRIEVAL TOOL
# =========================

def internal_pdf_search(question: str, top_k: int = TOP_K) -> List[Document]:
    query_vector = embeddings_model.embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    documents = []
    result_docs = results.get("documents", [[]])[0]
    result_metas = results.get("metadatas", [[]])[0]

    for text, meta in zip(result_docs, result_metas):
        documents.append(
            Document(
                page_content=text,
                metadata=meta
            )
        )

    return documents


# =========================
# EXTERNAL TOOLS
# =========================

def google_web_search(question: str, num_results: int = GOOGLE_RESULTS_LIMIT) -> List[Dict[str, Any]]:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": question,
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "")
        })

    return results


def youtube_search(question: str, max_results: int = YOUTUBE_RESULTS_LIMIT) -> List[Dict[str, Any]]:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": question,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("items", [])[:max_results]:
        video_id = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})

        results.append({
            "title": snippet.get("title", ""),
            "link": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
            "snippet": snippet.get("description", "")
        })

    return results


# =========================
# EVIDENCE FORMATTERS
# =========================

def format_internal_docs(docs: List[Document]) -> str:
    if not docs:
        return "No internal PDF evidence found."

    blocks = []
    for doc in docs:
        blocks.append(
            f"[Internal Source: {doc.metadata.get('doc_id')} | {doc.metadata.get('chunk_id')}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)


def format_external_results(results: List[Dict[str, Any]], source_name: str) -> str:
    if not results:
        return f"No {source_name} results found."

    blocks = []
    for idx, item in enumerate(results, start=1):
        blocks.append(
            f"[{source_name} Result {idx}]\n"
            f"Title: {item.get('title', '')}\n"
            f"Link: {item.get('link', '')}\n"
            f"Snippet: {item.get('snippet', '')}"
        )

    return "\n\n".join(blocks)


# =========================
# AGENT DECISION MODEL
# =========================

class ToolDecision(BaseModel):
    action: Literal["INTERNAL_ONLY", "GOOGLE_SEARCH", "YOUTUBE_SEARCH", "GOOGLE_AND_YOUTUBE"] = Field(
        description="Which evidence path should be used next."
    )
    reason: str = Field(description="Why this action was selected.")


# =========================
# MEMORY + AGENT PROMPTS
# =========================

decision_llm = llm.with_structured_output(ToolDecision)

decision_prompt = ChatPromptTemplate.from_template("""
You are an intelligent routing agent in an advanced RAG system.

Your job is to decide whether internal documents are enough OR whether external tools are REQUIRED.

Conversation History:
{chat_history}

STRICT ROUTING RULES:

1. If the user asks for:
   - a link
   - a URL
   - a website
   - a trailer
   - a YouTube video
   - something to watch
   - latest/current/news info

   then INTERNAL_ONLY is NOT allowed.

2. If the user explicitly asks for:
   - "youtube"
   - "video"
   - "trailer"
   - "watch"

   then choose YOUTUBE_SEARCH.

3. If the user explicitly asks for:
   - "link"
   - "url"
   - "website"
   - "official page"

   then choose GOOGLE_SEARCH unless video is more appropriate.

4. If the user asks for latest/current/external information that would not reasonably exist in the internal PDFs,
   choose GOOGLE_SEARCH or GOOGLE_AND_YOUTUBE.

5. INTERNAL_ONLY should be used ONLY when:
   - the answer is clearly available in internal documents
   - and the user is NOT asking for external content.

Current Question:
{question}

Internal Retrieved Evidence:
{internal_context}

Return ONLY one action:
INTERNAL_ONLY / GOOGLE_SEARCH / YOUTUBE_SEARCH / GOOGLE_AND_YOUTUBE
""")

final_prompt = ChatPromptTemplate.from_template("""
You are a professional domain-specific agentic RAG assistant.

Use the available evidence to answer the user's question.

Conversation History:
{chat_history}

Guidelines:
- Prefer internal PDF evidence when it is sufficient.
- Use external evidence whenever the user explicitly asks for links, videos, trailers, websites, or current/latest information.
- Be accurate, grounded, and concise.
- Do not mention raw chunk IDs in the main answer.
- Do not expose technical retrieval identifiers to the user.
- Write the answer in a natural, polished way.
- If external links are available, provide them directly.
- Never tell the user to search manually if tool results already exist.
- If YouTube results exist and the user asked for a trailer/video, provide the best matching video link first.
- If Google results exist and the user asked for a link/website, provide the most relevant link first.
- If evidence is still insufficient after actual searching, say clearly that you could not find a reliable result.

Current Question:
{question}

Internal Evidence:
{internal_context}

Google/Web Evidence:
{google_context}

YouTube Evidence:
{youtube_context}
""")

final_chain = final_prompt | llm | StrOutputParser()


# =========================
# MEMORY STORAGE HELPERS
# =========================

def initialize_session_log() -> None:
    if not SESSION_LOG_CSV_PATH.exists():
        df = pd.DataFrame(columns=[
            "session_id",
            "turn_id",
            "timestamp",
            "user_question",
            "decision_action",
            "decision_reason",
            "answer",
            "num_internal_sources",
            "num_google_results",
            "num_youtube_results",
            "internal_chunk_ids",
            "google_links",
            "youtube_links"
        ])
        df.to_csv(SESSION_LOG_CSV_PATH, index=False)


def load_session_logs() -> pd.DataFrame:
    initialize_session_log()
    return pd.read_csv(SESSION_LOG_CSV_PATH)


def save_session_logs(df: pd.DataFrame) -> None:
    df.to_csv(SESSION_LOG_CSV_PATH, index=False)


def create_session_id() -> str:
    return str(uuid.uuid4())


def get_chat_history_for_session(session_id: str, max_turns: int = MEMORY_LOOKBACK_TURNS) -> str:
    logs_df = load_session_logs()
    session_df = logs_df[logs_df["session_id"] == session_id].copy()

    if session_df.empty:
        return "No previous conversation history."

    session_df = session_df.sort_values("timestamp").tail(max_turns)

    history_blocks = []
    for _, row in session_df.iterrows():
        history_blocks.append(
            f"User: {row['user_question']}\nAssistant: {row['answer']}"
        )

    return "\n\n".join(history_blocks)


def update_session_summary() -> Dict[str, Any]:
    logs_df = load_session_logs()

    summary = {
        "total_sessions": int(logs_df["session_id"].nunique()) if not logs_df.empty else 0,
        "total_turns": int(len(logs_df)),
        "last_updated": datetime.now(UTC).isoformat()
    }

    with open(SESSION_SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =========================
# CORE APPLICATION FUNCTION
# =========================

def force_route_from_question(question: str) -> Optional[str]:
    q = question.lower().strip()

    youtube_terms = ["youtube", "video", "trailer", "watch", "clip", "teaser"]
    google_terms = ["link", "url", "website", "official page", "official site"]
    latest_terms = ["latest", "current", "today", "news", "recent", "now"]

    if any(term in q for term in youtube_terms):
        return "YOUTUBE_SEARCH"

    if any(term in q for term in google_terms):
        return "GOOGLE_SEARCH"

    if any(term in q for term in latest_terms):
        return "GOOGLE_SEARCH"

    return None

def answer_with_memory_agentic_rag(question: str, session_id: Optional[str] = None, top_k: int = TOP_K) -> Dict[str, Any]:
    if session_id is None:
        session_id = create_session_id()

    turn_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).isoformat()

    # Load memory
    chat_history = get_chat_history_for_session(
        session_id=session_id,
        max_turns=MEMORY_LOOKBACK_TURNS
    )

    # Internal retrieval
    internal_docs = internal_pdf_search(question, top_k=top_k)
    internal_context = format_internal_docs(internal_docs)

    # Rule-based forced routing first
    forced_action = force_route_from_question(question)

    if forced_action is not None:
        decision = ToolDecision(
            action=forced_action,
            reason=f"Forced routing triggered by user intent in question: {forced_action}"
        )
    else:
        decision = decision_llm.invoke(
            decision_prompt.format_messages(
                chat_history=chat_history,
                question=question,
                internal_context=internal_context
            )
        )

    google_results = []
    youtube_results = []

    # Tool execution based on decision
    if decision.action in ["GOOGLE_SEARCH", "GOOGLE_AND_YOUTUBE"]:
        google_results = google_web_search(question, num_results=GOOGLE_RESULTS_LIMIT)

    if decision.action in ["YOUTUBE_SEARCH", "GOOGLE_AND_YOUTUBE"]:
        youtube_results = youtube_search(question, max_results=YOUTUBE_RESULTS_LIMIT)

    # Smart fallback if the first chosen tool returns nothing
    if decision.action == "YOUTUBE_SEARCH" and not youtube_results:
        google_results = google_web_search(question, num_results=GOOGLE_RESULTS_LIMIT)

    if decision.action == "GOOGLE_SEARCH" and not google_results:
        youtube_results = youtube_search(question, max_results=YOUTUBE_RESULTS_LIMIT)

    google_context = format_external_results(google_results, "Google/Web")
    youtube_context = format_external_results(youtube_results, "YouTube")

    # Final answer
    final_answer = final_chain.invoke({
        "chat_history": chat_history,
        "question": question,
        "internal_context": internal_context,
        "google_context": google_context if google_results else "No Google/Web evidence used.",
        "youtube_context": youtube_context if youtube_results else "No YouTube evidence used."
    })

    # Internal source packaging
    internal_sources = []
    for rank, doc in enumerate(internal_docs, start=1):
        internal_sources.append({
            "rank": rank,
            "source_type": "internal_pdf",
            "doc_id": doc.metadata.get("doc_id"),
            "file_name": doc.metadata.get("file_name"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "preview": doc.page_content[:250]
        })

    # Save memory
    logs_df = load_session_logs()

    new_row = {
        "session_id": session_id,
        "turn_id": turn_id,
        "timestamp": timestamp,
        "user_question": question,
        "decision_action": decision.action,
        "decision_reason": decision.reason,
        "answer": final_answer,
        "num_internal_sources": len(internal_sources),
        "num_google_results": len(google_results),
        "num_youtube_results": len(youtube_results),
        "internal_chunk_ids": json.dumps([s["chunk_id"] for s in internal_sources]),
        "google_links": json.dumps([r.get("link", "") for r in google_results]),
        "youtube_links": json.dumps([r.get("link", "") for r in youtube_results])
    }

    logs_df = pd.concat([logs_df, pd.DataFrame([new_row])], ignore_index=True)
    save_session_logs(logs_df)
    session_summary = update_session_summary()

    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "question": question,
        "decision_action": decision.action,
        "decision_reason": decision.reason,
        "answer": final_answer,
        "internal_sources": internal_sources,
        "google_results": google_results,
        "youtube_results": youtube_results,
        "chat_history_used": chat_history,
        "session_summary": session_summary
    }


# =========================
# MLFLOW TRACKED APPLICATION WRAPPER
# =========================

def answer_with_mlflow_tracking(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    start_time = datetime.now(UTC)

    with mlflow.start_run(run_name=f"UI - {question[:40]}"):
        result = answer_with_memory_agentic_rag(
            question=question,
            session_id=session_id
        )

        end_time = datetime.now(UTC)
        latency = (end_time - start_time).total_seconds()

        mlflow.log_param("question", question)
        mlflow.log_param("session_id", result["session_id"])
        mlflow.log_param("decision_action", result["decision_action"])

        mlflow.log_metric("num_internal_sources", len(result["internal_sources"]))
        mlflow.log_metric("num_google_results", len(result["google_results"]))
        mlflow.log_metric("num_youtube_results", len(result["youtube_results"]))
        mlflow.log_metric("latency_seconds", latency)

        artifact_data = {
            "question": result["question"],
            "decision_action": result["decision_action"],
            "decision_reason": result["decision_reason"],
            "answer": result["answer"],
            "internal_sources": result["internal_sources"],
            "google_results": result["google_results"],
            "youtube_results": result["youtube_results"],
            "chat_history": result["chat_history_used"]
        }

        artifact_path = "/tmp/final_application_result.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact_data, f, indent=2)

        mlflow.log_artifact(artifact_path)

        result["latency_seconds"] = latency
        return result


# =========================
# APPLICATION ENTRYPOINTS
# =========================

def start_new_session() -> str:
    return create_session_id()


def ask_application(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    return answer_with_mlflow_tracking(question=question, session_id=session_id)


def show_system_status() -> pd.DataFrame:
    logs_df = load_session_logs()

    status = {
        "project_name": PROJECT_NAME,
        "domain_name": DOMAIN_NAME,
        "collection_name": COLLECTION_NAME,
        "collection_count": int(collection.count()),
        "total_sessions": int(logs_df["session_id"].nunique()) if not logs_df.empty else 0,
        "total_turns": int(len(logs_df)),
        "session_log_exists": SESSION_LOG_CSV_PATH.exists(),
        "session_summary_exists": SESSION_SUMMARY_JSON_PATH.exists(),
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME
    }

    return pd.DataFrame([status])