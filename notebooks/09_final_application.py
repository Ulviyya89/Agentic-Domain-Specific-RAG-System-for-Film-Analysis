# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Final Application
# MAGIC
# MAGIC ## Project Title
# MAGIC **Agentic Domain-Specific RAG Assistant with Memory, Tool Calling, ChromaDB, and MLflow Tracing**
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook is the final integrated application layer of the project.
# MAGIC
# MAGIC It combines all previously implemented components into one professional workflow:
# MAGIC - internal document retrieval from ChromaDB
# MAGIC - agentic decision-making
# MAGIC - external fallback tools
# MAGIC - conversation memory
# MAGIC - persistent session storage
# MAGIC - MLflow tracking
# MAGIC
# MAGIC This notebook is intended to serve as:
# MAGIC - the final runnable application
# MAGIC - the main notebook for presentation
# MAGIC - the main notebook for submission
# MAGIC - the integrated end-to-end system entry point
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## System Capabilities
# MAGIC The application supports:
# MAGIC - answering questions from internal PDF knowledge
# MAGIC - deciding when internal evidence is insufficient
# MAGIC - searching external sources when needed
# MAGIC - remembering previous turns in a session
# MAGIC - storing interaction history
# MAGIC - logging queries and execution metadata in MLflow
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Final Workflow
# MAGIC 1. User asks a question
# MAGIC 2. System searches internal PDF knowledge first
# MAGIC 3. System decides whether external tools are needed
# MAGIC 4. System optionally calls:
# MAGIC    - Google/Web search
# MAGIC    - YouTube search
# MAGIC 5. System synthesizes the final answer
# MAGIC 6. System stores memory in the session log
# MAGIC 7. System logs execution details in MLflow

# COMMAND ----------

# MAGIC %pip install -U "numpy<2" pandas python-dotenv requests pydantic==2.11.7 pydantic-core==2.33.2 typing-extensions==4.15.0 chromadb==1.5.6 openai langchain langchain-openai mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This notebook uses:
# MAGIC - `chromadb` for vector database access
# MAGIC - `langchain` and `langchain-openai` for prompts and model integration
# MAGIC - `pandas` for structured memory/session storage
# MAGIC - `python-dotenv` for loading environment variables
# MAGIC - `requests` for external tool calls
# MAGIC - `mlflow` for execution tracking
# MAGIC - `pydantic` for structured model decisions
# MAGIC
# MAGIC This notebook is intentionally self-contained and does not depend on notebook runtime state from earlier notebooks.

# COMMAND ----------

# =========================
# STANDARD LIBRARIES
# =========================

import os
import json
import uuid
import requests
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Literal, Optional

# =========================
# VERSION CHECK
# =========================

from importlib.metadata import version

print("numpy:", version("numpy"))
print("typing-extensions:", version("typing-extensions"))
print("pydantic:", version("pydantic"))
print("pydantic-core:", version("pydantic-core"))
print("chromadb:", version("chromadb"))
print("mlflow:", version("mlflow"))

# =========================
# DATA + ENVIRONMENT
# =========================

import pandas as pd
from dotenv import load_dotenv

# =========================
# PYDANTIC
# =========================

from pydantic import BaseModel, Field

# =========================
# CHROMA + OPENAI + LANGCHAIN CORE
# =========================

import chromadb
from chromadb.config import Settings

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# MLFLOW
# =========================

import mlflow

print("All imports successful.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import required libraries
# MAGIC
# MAGIC This section imports the full application stack:
# MAGIC - environment handling
# MAGIC - vector database access
# MAGIC - model integration
# MAGIC - agentic decision structures
# MAGIC - memory and storage handling
# MAGIC - MLflow tracking

# COMMAND ----------

# =========================
# PROJECT CONFIGURATION
# =========================

PROJECT_NAME = "domain_specific_rag_movie_analysis"
PROJECT_TITLE = "Agentic Domain-Specific RAG Assistant with Memory, Tool Calling, ChromaDB, and MLflow Tracing"
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

# Temporary runtime Chroma path
CHROMA_PERSIST_DIR = "/tmp/chroma_db/movie_analysis"

GOOGLE_RESULTS_LIMIT = 5
YOUTUBE_RESULTS_LIMIT = 5
MEMORY_LOOKBACK_TURNS = 5

MLFLOW_EXPERIMENT_NAME = "/Users/gouthamrp576@gmail.com/Agentic_RAG_Project"

print("PROJECT_NAME =", PROJECT_NAME)
print("PROJECT_TITLE =", PROJECT_TITLE)
print("DOMAIN_NAME =", DOMAIN_NAME)
print("COLLECTION_NAME =", COLLECTION_NAME)
print("SESSION_LOG_CSV_PATH =", SESSION_LOG_CSV_PATH)
print("SESSION_SUMMARY_JSON_PATH =", SESSION_SUMMARY_JSON_PATH)
print("CHROMA_PERSIST_DIR =", CHROMA_PERSIST_DIR)
print("MLFLOW_EXPERIMENT_NAME =", MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define application configuration
# MAGIC
# MAGIC This section stores the core settings for the final application:
# MAGIC - model names
# MAGIC - retrieval depth
# MAGIC - memory paths
# MAGIC - ChromaDB location
# MAGIC - external tool limits
# MAGIC - MLflow experiment name
# MAGIC
# MAGIC This keeps the final notebook centralized and maintainable.

# COMMAND ----------


# =========================
# LOAD ENVIRONMENT VARIABLES
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

print("OPENAI_API_KEY loaded successfully.")
print("SERPAPI_API_KEY loaded successfully.")
print("YOUTUBE_API_KEY loaded successfully.")

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Initialize models
# MAGIC
# MAGIC The final application uses:
# MAGIC - OpenAI embeddings for internal retrieval
# MAGIC - OpenAI chat model for:
# MAGIC   - tool routing
# MAGIC   - answer generation
# MAGIC   - memory-aware conversational behavior

# COMMAND ----------

# =========================
# INITIALIZE MLFLOW
# =========================

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("MLflow experiment set to:", MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Initialize MLflow
# MAGIC
# MAGIC All final application queries will be logged into MLflow.
# MAGIC
# MAGIC This supports:
# MAGIC - traceability
# MAGIC - evaluation
# MAGIC - debugging
# MAGIC - submission evidence

# COMMAND ----------

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

# COMMAND ----------

# =========================
# CHROMA REBUILD HELPERS
# =========================

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

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Ensure the vector database is ready
# MAGIC
# MAGIC If the current runtime does not have a populated Chroma collection, the notebook automatically rebuilds it from `chunks_metadata.csv`.
# MAGIC
# MAGIC This makes the final application resilient across runtime restarts.

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Define the evidence tools
# MAGIC
# MAGIC The final application can use three evidence sources:
# MAGIC - internal PDF retrieval
# MAGIC - Google/Web search
# MAGIC - YouTube search
# MAGIC
# MAGIC The model will decide when to use them.

# COMMAND ----------

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

# COMMAND ----------

# =========================
# AGENT DECISION MODEL
# =========================

class ToolDecision(BaseModel):
    action: Literal["INTERNAL_ONLY", "GOOGLE_SEARCH", "YOUTUBE_SEARCH", "GOOGLE_AND_YOUTUBE"] = Field(
        description="Which evidence path should be used next."
    )
    reason: str = Field(description="Why this action was selected.")

# COMMAND ----------

# =========================
# MEMORY + AGENT PROMPTS
# =========================

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Define the agentic prompts
# MAGIC
# MAGIC The final application has two prompt stages:
# MAGIC - routing prompt for tool decisions
# MAGIC - synthesis prompt for final answer generation
# MAGIC
# MAGIC Both are memory-aware and evidence-aware.

# COMMAND ----------

# =========================
# MEMORY STORAGE HELPERS
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Define the memory store
# MAGIC
# MAGIC The final application keeps:
# MAGIC - short-term conversational memory in prompts
# MAGIC - persistent interaction logs in CSV
# MAGIC - session summary metadata in JSON
# MAGIC
# MAGIC This makes the application stateful and auditable.

# COMMAND ----------

# =========================
# CORE APPLICATION FUNCTION
# =========================

def answer_with_memory_agentic_rag(question: str, session_id: Optional[str] = None, top_k: int = TOP_K) -> Dict[str, Any]:
    if session_id is None:
        session_id = create_session_id()

    turn_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).isoformat()

    # Load memory
    chat_history = get_chat_history_for_session(session_id=session_id, max_turns=MEMORY_LOOKBACK_TURNS)

    # Internal retrieval
    internal_docs = internal_pdf_search(question, top_k=top_k)
    internal_context = format_internal_docs(internal_docs)

    # Agentic decision
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

# COMMAND ----------

# =========================
# MLFLOW TRACKED APPLICATION WRAPPER
# =========================

def answer_with_mlflow_tracking(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    with mlflow.start_run():
        start_time = datetime.now(UTC)

        result = answer_with_memory_agentic_rag(
            question=question,
            session_id=session_id
        )

        end_time = datetime.now(UTC)
        latency = (end_time - start_time).total_seconds()

        # Log params
        mlflow.log_param("question", question)
        mlflow.log_param("session_id", result["session_id"])
        mlflow.log_param("decision_action", result["decision_action"])

        # Log metrics
        mlflow.log_metric("num_internal_sources", len(result["internal_sources"]))
        mlflow.log_metric("num_google_results", len(result["google_results"]))
        mlflow.log_metric("num_youtube_results", len(result["youtube_results"]))
        mlflow.log_metric("latency_seconds", latency)

        # Log artifact
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

        return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Add MLflow tracking to the application
# MAGIC
# MAGIC The final application logs:
# MAGIC - question
# MAGIC - session ID
# MAGIC - routing decision
# MAGIC - internal retrieval counts
# MAGIC - external tool counts
# MAGIC - latency
# MAGIC - full result artifact
# MAGIC
# MAGIC This makes the system submission-ready and traceable.

# COMMAND ----------

# =========================
# APPLICATION ENTRYPOINTS
# =========================

def start_new_session() -> str:
    session_id = create_session_id()
    print("New session created:", session_id)
    return session_id


def ask_application(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    result = answer_with_mlflow_tracking(question=question, session_id=session_id)

    print("\n" + "=" * 100)
    print("SESSION ID:")
    print(result["session_id"])

    print("\nQUESTION:")
    print(result["question"])

    print("\nDECISION:")
    print(result["decision_action"])
    print(result["decision_reason"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nINTERNAL SOURCES:")
    if result["internal_sources"]:
        for s in result["internal_sources"]:
            print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']} | {s['file_name']}")
    else:
        print("No internal sources used.")

    print("\nGOOGLE RESULTS:")
    if result["google_results"]:
        for item in result["google_results"][:3]:
            print("-", item.get("title", ""), "|", item.get("link", ""))
    else:
        print("No Google/Web results used.")

    print("\nYOUTUBE RESULTS:")
    if result["youtube_results"]:
        for item in result["youtube_results"][:3]:
            print("-", item.get("title", ""), "|", item.get("link", ""))
    else:
        print("No YouTube results used.")

    print("=" * 100 + "\n")

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Define the final application entrypoints
# MAGIC
# MAGIC This final notebook exposes two main entrypoints:
# MAGIC
# MAGIC - `start_new_session()` → starts a new conversation session
# MAGIC - `ask_application(question, session_id)` → runs the full application and displays results
# MAGIC
# MAGIC These are the main functions you should use during final presentation or submission checks.

# COMMAND ----------

# =========================
# SYSTEM STATUS HELPER
# =========================

def show_system_status():
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

    status_df = pd.DataFrame([status])
    return status_df

# COMMAND ----------

# =========================
# SYSTEM STATUS CHECK
# =========================

show_system_status()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Show system status
# MAGIC
# MAGIC This summary provides a quick application health check:
# MAGIC - vector collection size
# MAGIC - number of sessions
# MAGIC - number of turns
# MAGIC - memory file status
# MAGIC - MLflow experiment name

# COMMAND ----------

# =========================
# FINAL APPLICATION TEST - NEW SESSION
# =========================

session_id = start_new_session()

result_1 = ask_application(
    question="How does Interstellar connect love and survival?",
    session_id=session_id
)

# COMMAND ----------

# =========================
# FINAL APPLICATION TEST - FOLLOW-UP
# =========================

result_2 = ask_application(
    question="How is that different from Memento?",
    session_id=session_id
)

# COMMAND ----------

# =========================
# FINAL APPLICATION TEST - FOLLOW-UP
# =========================

result_2 = ask_application(
    question="How is that different from Memento?",
    session_id=session_id
)

# COMMAND ----------

# =========================
# FINAL APPLICATION TEST - OUTSIDE CORPUS
# =========================

result_3 = ask_application(
    question="What are some well-known outside interpretations of Fight Club beyond the internal PDFs?",
    session_id=session_id
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Validate final application behavior
# MAGIC
# MAGIC These tests demonstrate:
# MAGIC - internal retrieval
# MAGIC - follow-up question handling with memory
# MAGIC - external fallback when the internal corpus is not enough
# MAGIC
# MAGIC This is the final proof that the application is integrated and working.

# COMMAND ----------

# =========================
# VIEW SESSION LOGS
# =========================

logs_df = load_session_logs()
logs_df.tail(10)

# COMMAND ----------

# =========================
# VIEW SESSION SUMMARY
# =========================

if SESSION_SUMMARY_JSON_PATH.exists():
    with open(SESSION_SUMMARY_JSON_PATH, "r") as f:
        summary_json = json.load(f)

    summary_df = pd.DataFrame([summary_json])
    summary_df
else:
    print("Session summary file not found yet.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Inspect persistent memory outputs
# MAGIC
# MAGIC The final application saves:
# MAGIC - full session logs in CSV
# MAGIC - compact session summary in JSON
# MAGIC
# MAGIC This is the persistent memory layer of the system.

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Summary
# MAGIC
# MAGIC This notebook is the final integrated application layer of the project.
# MAGIC
# MAGIC ## What the final application now supports
# MAGIC - internal semantic retrieval from ChromaDB
# MAGIC - agentic tool calling
# MAGIC - Google/Web fallback
# MAGIC - YouTube fallback
# MAGIC - conversation memory
# MAGIC - persistent session storage
# MAGIC - MLflow query tracking
# MAGIC - clean application entrypoints
# MAGIC - source traceability
# MAGIC
# MAGIC ## Why this notebook is important
# MAGIC This is the final professional application notebook for:
# MAGIC - submission
# MAGIC - presentation
# MAGIC - live execution
# MAGIC - system validation
# MAGIC
# MAGIC ## Main application entrypoints
# MAGIC - `start_new_session()`
# MAGIC - `ask_application(question, session_id)`
# MAGIC - `show_system_status()`
# MAGIC
# MAGIC ## Application status
# MAGIC At this point, the project has a complete end-to-end implementation of:
# MAGIC - domain-specific RAG
# MAGIC - agentic behavior
# MAGIC - memory
# MAGIC - database-backed session storage
# MAGIC - observability through MLflow

# COMMAND ----------

