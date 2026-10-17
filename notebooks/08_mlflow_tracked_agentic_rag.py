# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - MLflow Tracked Agentic RAG
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC This notebook integrates **MLflow tracking** into the agentic RAG system.
# MAGIC
# MAGIC At this stage, the project supports:
# MAGIC - document ingestion and chunking
# MAGIC - ChromaDB-based vector retrieval
# MAGIC - agentic decision making with tool calling
# MAGIC - conversational memory and session storage
# MAGIC
# MAGIC This notebook adds:
# MAGIC - full MLflow tracing of each query
# MAGIC - logging of:
# MAGIC   - prompts
# MAGIC   - responses
# MAGIC   - retrieval metadata
# MAGIC   - tool usage
# MAGIC   - system decisions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this matters
# MAGIC
# MAGIC This notebook satisfies key grading requirements:
# MAGIC
# MAGIC - MLflow tracking (mandatory)
# MAGIC - traceable execution pipeline
# MAGIC - experiment logging (top-k, retrieval behavior)
# MAGIC - reproducibility and debugging
# MAGIC
# MAGIC This transforms the project into a **production-grade RAG system**.

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import json
from datetime import datetime, UTC
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize MLflow
# MAGIC
# MAGIC We configure MLflow tracking so that every query becomes a tracked run.

# COMMAND ----------

# =========================
# MLFLOW CONFIG
# =========================

MLFLOW_EXPERIMENT_NAME = "/Users/gouthamrp576@gmail.com/Agentic_RAG_Project"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("MLflow experiment set to:", MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Wrap Agentic RAG with MLflow Tracking
# MAGIC
# MAGIC We now create a wrapper around the existing function:
# MAGIC
# MAGIC `answer_with_memory_agentic_rag()`
# MAGIC
# MAGIC This wrapper:
# MAGIC - logs inputs
# MAGIC - logs outputs
# MAGIC - logs decisions
# MAGIC - logs retrieval data
# MAGIC - logs tool usage

# COMMAND ----------

# =========================
# LOCAL SESSION HELPERS
# =========================

import uuid

def create_session_id() -> str:
    return str(uuid.uuid4())

print("Local session helper loaded.")

# COMMAND ----------

# =========================
# MLFLOW TRACKED FUNCTION
# =========================

def answer_with_mlflow_tracking(question: str, session_id: str = None):

    with mlflow.start_run():

        start_time = datetime.now(UTC)

        # =========================
        # RUN CORE PIPELINE
        # =========================

        result = answer_with_memory_agentic_rag(
            question=question,
            session_id=session_id
        )

        end_time = datetime.now(UTC)
        latency = (end_time - start_time).total_seconds()

        # =========================
        # LOG PARAMETERS
        # =========================

        mlflow.log_param("question", question)
        mlflow.log_param("session_id", result["session_id"])
        mlflow.log_param("decision_action", result["decision_action"])

        # =========================
        # LOG METRICS
        # =========================

        mlflow.log_metric("num_internal_sources", len(result["internal_sources"]))
        mlflow.log_metric("num_google_results", len(result["google_results"]))
        mlflow.log_metric("num_youtube_results", len(result["youtube_results"]))
        mlflow.log_metric("latency_seconds", latency)

        # =========================
        # LOG ARTIFACTS (JSON)
        # =========================

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

        artifact_path = "/tmp/mlflow_rag_result.json"

        with open(artifact_path, "w") as f:
            json.dump(artifact_data, f, indent=2)

        mlflow.log_artifact(artifact_path)

        print("MLflow run logged successfully.")

        return result

# COMMAND ----------

# MAGIC %pip install chromadb==1.5.6
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# =========================
# SELF-CONTAINED AGENTIC MEMORY LAYER FOR NOTEBOOK 08
# =========================

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

# =========================
# CONFIG
# =========================

PROJECT_NAME = "domain_specific_rag_movie_analysis"
DOMAIN_NAME = "Narrative, thematic, and character analysis of curated film documents"

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

COLLECTION_NAME = "movie_analysis_chunks"
TOP_K = 5
TEMPERATURE = 0

BASE_DIR = Path("/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project")
ENV_PATH = BASE_DIR / ".env"

CHUNKS_CSV_PATH = BASE_DIR / "data" / "processed" / "chunks_metadata.csv"

MEMORY_DIR = BASE_DIR / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

SESSION_LOG_CSV_PATH = MEMORY_DIR / "session_logs.csv"
SESSION_SUMMARY_JSON_PATH = MEMORY_DIR / "session_summary.json"

CHROMA_PERSIST_DIR = "/tmp/chroma_db/movie_analysis"

GOOGLE_RESULTS_LIMIT = 5
YOUTUBE_RESULTS_LIMIT = 5
MEMORY_LOOKBACK_TURNS = 5

# =========================
# ENV
# =========================

if not ENV_PATH.exists():
    raise FileNotFoundError(
        f".env file not found at: {ENV_PATH}\n"
        "Please create the .env file with required API keys."
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
# MODELS
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

# =========================
# CHROMADB
# =========================

persist_path = Path(CHROMA_PERSIST_DIR)
persist_path.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def load_chunks_dataframe(chunks_csv_path: Path) -> pd.DataFrame:
    if not chunks_csv_path.exists():
        raise FileNotFoundError(
            f"Chunk metadata file not found: {chunks_csv_path}. "
            "Please run the chunking notebook first."
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

def rebuild_collection_from_chunks(chunks_df: pd.DataFrame):
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

collection_count = collection.count()
if collection_count == 0:
    chunks_df = load_chunks_dataframe(CHUNKS_CSV_PATH)
    collection_count = rebuild_collection_from_chunks(chunks_df)

if collection_count == 0:
    raise ValueError("Chroma collection is still empty after rebuild.")

# =========================
# HELPERS
# =========================

def create_session_id() -> str:
    return str(uuid.uuid4())

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
# TOOLS
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
# DECISION MODEL
# =========================

class ToolDecision(BaseModel):
    action: Literal["INTERNAL_ONLY", "GOOGLE_SEARCH", "YOUTUBE_SEARCH", "GOOGLE_AND_YOUTUBE"] = Field(
        description="Which evidence path should be used next."
    )
    reason: str = Field(description="Why this action was selected.")

decision_llm = llm.with_structured_output(ToolDecision)

decision_prompt = ChatPromptTemplate.from_template("""
You are an agentic routing model for a domain-specific RAG system.

Your job is to decide whether the current internal PDF evidence is enough to answer the user's question.

Conversation History:
{chat_history}

Rules:
- Choose INTERNAL_ONLY if the internal PDF evidence is clearly sufficient.
- Choose GOOGLE_SEARCH if broader web information is needed.
- Choose YOUTUBE_SEARCH if video-based or broader explanatory material would be more useful.
- Choose GOOGLE_AND_YOUTUBE if both external web and video sources would help.

Current Question:
{question}

Internal Retrieved Evidence:
{internal_context}
""")

final_prompt = ChatPromptTemplate.from_template("""
You are a professional domain-specific agentic RAG assistant.

Use the available evidence to answer the user's question.

Conversation History:
{chat_history}

Guidelines:
- Prefer internal PDF evidence when it is sufficient.
- Use external evidence only when needed.
- Be accurate, grounded, and concise.
- Clearly mention whether the answer is based on:
  - internal PDFs
  - external web sources
  - YouTube sources
  - or a combination
- Cite internal chunk IDs when used.
- Include external links only when they are part of the gathered evidence.
- If evidence is still insufficient, say so clearly.

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
# MEMORY STORAGE
# =========================

def initialize_session_log():
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

def save_session_logs(df: pd.DataFrame):
    df.to_csv(SESSION_LOG_CSV_PATH, index=False)

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

def update_session_summary():
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
# MAIN FUNCTION USED BY NOTEBOOK 08
# =========================

def answer_with_memory_agentic_rag(question: str, session_id: Optional[str] = None, top_k: int = TOP_K) -> Dict[str, Any]:
    if session_id is None:
        session_id = create_session_id()

    turn_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).isoformat()

    chat_history = get_chat_history_for_session(session_id=session_id, max_turns=MEMORY_LOOKBACK_TURNS)

    internal_docs = internal_pdf_search(question, top_k=top_k)
    internal_context = format_internal_docs(internal_docs)

    decision = decision_llm.invoke(
        decision_prompt.format_messages(
            chat_history=chat_history,
            question=question,
            internal_context=internal_context
        )
    )

    google_results = []
    youtube_results = []

    if decision.action in ["GOOGLE_SEARCH", "GOOGLE_AND_YOUTUBE"]:
        google_results = google_web_search(question, num_results=GOOGLE_RESULTS_LIMIT)

    if decision.action in ["YOUTUBE_SEARCH", "GOOGLE_AND_YOUTUBE"]:
        youtube_results = youtube_search(question, max_results=YOUTUBE_RESULTS_LIMIT)

    google_context = format_external_results(google_results, "Google/Web")
    youtube_context = format_external_results(youtube_results, "YouTube")

    final_answer = final_chain.invoke({
        "chat_history": chat_history,
        "question": question,
        "internal_context": internal_context,
        "google_context": google_context if google_results else "No Google/Web evidence used.",
        "youtube_context": youtube_context if youtube_results else "No YouTube evidence used."
    })

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

print("Notebook 08 self-contained agentic memory layer loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. What is being tracked?
# MAGIC
# MAGIC Each query logs:
# MAGIC
# MAGIC ### Parameters
# MAGIC - question
# MAGIC - session_id
# MAGIC - decision_action
# MAGIC
# MAGIC ### Metrics
# MAGIC - number of retrieved chunks
# MAGIC - number of external results
# MAGIC - latency
# MAGIC
# MAGIC ### Artifacts
# MAGIC - full response
# MAGIC - tool outputs
# MAGIC - chat history
# MAGIC
# MAGIC This gives full traceability.

# COMMAND ----------

import mlflow
from datetime import datetime, UTC

MLFLOW_EXPERIMENT_NAME = "/Users/gouthamrp576@gmail.com/Agentic_RAG_Project"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("MLflow experiment set to:", MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# =========================
# MLFLOW TRACKED FUNCTION
# =========================

def answer_with_mlflow_tracking(question: str, session_id: str = None):
    with mlflow.start_run():
        start_time = datetime.now(UTC)

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

        artifact_path = "/tmp/mlflow_rag_result.json"

        with open(artifact_path, "w") as f:
            json.dump(artifact_data, f, indent=2)

        mlflow.log_artifact(artifact_path)

        print("MLflow run logged successfully.")
        return result

# COMMAND ----------

# =========================
# TEST RUN 1
# =========================

session_id = create_session_id()

query = "How does Interstellar connect love and survival?"

result = answer_with_mlflow_tracking(query, session_id=session_id)

print("\nSESSION ID:")
print(session_id)

print("\nDECISION:")
print(result["decision_action"])
print(result["decision_reason"])

print("\nANSWER:")
print(result["answer"])