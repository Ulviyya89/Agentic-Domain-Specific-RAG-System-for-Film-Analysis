# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Memory and Session Store
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook adds **memory** to the agentic RAG system.
# MAGIC
# MAGIC At this stage, the project already supports:
# MAGIC - document ingestion
# MAGIC - chunking
# MAGIC - ChromaDB vector retrieval
# MAGIC - LangChain-based prompting and answer generation
# MAGIC - agentic tool calling with internal and external evidence
# MAGIC
# MAGIC The next step is to make the system **stateful**.
# MAGIC
# MAGIC This notebook will:
# MAGIC - maintain short-term conversational memory for follow-up questions
# MAGIC - store each conversation turn in a structured session log
# MAGIC - save memory records in persistent storage
# MAGIC - support retrieval of past conversation history
# MAGIC - combine memory with agentic RAG behavior
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC A basic RAG system answers each query independently.
# MAGIC
# MAGIC A stronger conversational system should also:
# MAGIC - remember previous turns
# MAGIC - understand follow-up questions
# MAGIC - preserve session history
# MAGIC - store interaction data for later analysis
# MAGIC
# MAGIC This notebook provides that memory layer.

# COMMAND ----------

# MAGIC %pip install -U "numpy<2" pandas python-dotenv requests pydantic==2.11.7 pydantic-core==2.33.2 typing-extensions==4.15.0 chromadb==1.5.6 openai langchain langchain-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This notebook uses:
# MAGIC - `chromadb` for direct vector database access
# MAGIC - `langchain` and `langchain-openai` for prompts and chat model orchestration
# MAGIC - `pandas` for session storage
# MAGIC - `python-dotenv` for loading environment variables
# MAGIC - `requests` for external tool calls
# MAGIC - `pydantic` for structured model decisions
# MAGIC
# MAGIC This version intentionally avoids `langchain_chroma` to keep the environment stable.

# COMMAND ----------

# =========================
# STANDARD LIBRARIES
# =========================

import os
import json
import uuid
import requests
from pathlib import Path
from datetime import datetime
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

print("All imports successful.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import required libraries
# MAGIC
# MAGIC This section imports the components required for:
# MAGIC - direct ChromaDB access
# MAGIC - OpenAI embeddings
# MAGIC - chat-based reasoning
# MAGIC - memory management
# MAGIC - session storage
# MAGIC - external tool calls

# COMMAND ----------

# =========================
# PROJECT CONFIGURATION
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

# Same temporary Chroma location used earlier
CHROMA_PERSIST_DIR = "/tmp/chroma_db/movie_analysis"

GOOGLE_RESULTS_LIMIT = 5
YOUTUBE_RESULTS_LIMIT = 5
MEMORY_LOOKBACK_TURNS = 5

print("PROJECT_NAME =", PROJECT_NAME)
print("DOMAIN_NAME =", DOMAIN_NAME)
print("COLLECTION_NAME =", COLLECTION_NAME)
print("SESSION_LOG_CSV_PATH =", SESSION_LOG_CSV_PATH)
print("SESSION_SUMMARY_JSON_PATH =", SESSION_SUMMARY_JSON_PATH)
print("CHROMA_PERSIST_DIR =", CHROMA_PERSIST_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project configuration
# MAGIC
# MAGIC This section stores the configuration for:
# MAGIC - models
# MAGIC - retrieval depth
# MAGIC - memory storage
# MAGIC - ChromaDB path
# MAGIC - external tool limits
# MAGIC
# MAGIC ### Memory-related outputs
# MAGIC - `session_logs.csv` stores every conversation turn
# MAGIC - `session_summary.json` stores compact session-level summary information

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
# INITIALIZE EMBEDDINGS AND CHAT MODEL
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
# MAGIC ## 5. Initialize embeddings and chat model
# MAGIC
# MAGIC The same OpenAI stack is reused here so that:
# MAGIC - retrieval remains consistent
# MAGIC - memory-aware reasoning stays aligned with the rest of the project

# COMMAND ----------

# =========================
# CHROMADB CONNECTION
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
# REBUILD CHROMADB IF EMPTY
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


if collection_count == 0:
    print("Chroma collection is empty. Rebuilding from chunks_metadata.csv ...")
    chunks_df = load_chunks_dataframe(CHUNKS_CSV_PATH)
    collection_count = rebuild_collection_from_chunks(chunks_df)
    print("Rebuilt Chroma collection count:", collection_count)

if collection_count == 0:
    raise ValueError(
        "Chroma collection is still empty after rebuild. "
        "Please verify chunks_metadata.csv and embedding setup."
    )

print("Chroma collection ready.")
print("Collection count:", collection_count)

# COMMAND ----------

# =========================
# INTERNAL RETRIEVAL TOOL
# =========================

def internal_pdf_search(question: str, top_k: int = TOP_K) -> List[Document]:
    """
    Search the internal ChromaDB knowledge base directly.
    """
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


# Quick validation
test_docs = internal_pdf_search("How does Interstellar connect love and survival?", top_k=3)
print("Internal retrieval test results:", len(test_docs))
for d in test_docs:
    print(d.metadata.get("chunk_id"), "|", d.metadata.get("doc_id"))

# COMMAND ----------

# =========================
# GOOGLE / WEB SEARCH TOOL
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

# COMMAND ----------

# =========================
# YOUTUBE SEARCH TOOL
# =========================

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

# =========================
# YOUTUBE SEARCH TOOL
# =========================

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
# MEMORY-AWARE PROMPTS
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
# MAGIC ## 10. Create memory-aware prompts
# MAGIC
# MAGIC The prompts now include:
# MAGIC - conversation history
# MAGIC - internal evidence
# MAGIC - external evidence
# MAGIC
# MAGIC This allows the agent to answer follow-up questions more intelligently.

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
        print("Initialized session log file:", SESSION_LOG_CSV_PATH)


def load_session_logs() -> pd.DataFrame:
    initialize_session_log()
    return pd.read_csv(SESSION_LOG_CSV_PATH)


def save_session_logs(df: pd.DataFrame):
    df.to_csv(SESSION_LOG_CSV_PATH, index=False)


def create_new_session_id() -> str:
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
        "last_updated": datetime.utcnow().isoformat()
    }

    with open(SESSION_SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Define memory storage helpers
# MAGIC
# MAGIC This section creates the persistent session store.
# MAGIC
# MAGIC ### What is stored for each turn
# MAGIC - session ID
# MAGIC - turn ID
# MAGIC - timestamp
# MAGIC - user question
# MAGIC - model decision
# MAGIC - final answer
# MAGIC - source usage statistics
# MAGIC - source IDs and links
# MAGIC
# MAGIC This makes the chatbot stateful and auditable.

# COMMAND ----------

# =========================
# MEMORY-ENABLED AGENTIC RAG FUNCTION
# =========================

def answer_with_memory_agentic_rag(question: str, session_id: Optional[str] = None, top_k: int = TOP_K) -> Dict[str, Any]:
    if session_id is None:
        session_id = create_new_session_id()

    turn_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Load memory
    chat_history = get_chat_history_for_session(session_id=session_id, max_turns=MEMORY_LOOKBACK_TURNS)

    # Internal retrieval
    internal_docs = internal_pdf_search(question, top_k=top_k)
    internal_context = format_internal_docs(internal_docs)

    # Decision
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Build the memory-enabled agentic RAG function
# MAGIC
# MAGIC This is the main function of the notebook.
# MAGIC
# MAGIC ### What it does
# MAGIC - reads recent conversation history
# MAGIC - performs internal retrieval
# MAGIC - lets the model decide whether external tools are needed
# MAGIC - synthesizes the final answer
# MAGIC - saves the conversation turn in persistent storage

# COMMAND ----------

# =========================
# FORMATTERS
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

print("Formatter functions loaded successfully.")

# COMMAND ----------

# =========================
# TEST SESSION - TURN 1
# =========================

session_id = create_new_session_id()

query_1 = "How does Interstellar connect love and survival?"
result_1 = answer_with_memory_agentic_rag(query_1, session_id=session_id)

print("SESSION ID:", result_1["session_id"])
print("QUESTION:", result_1["question"])
print("\nDECISION:", result_1["decision_action"])
print(result_1["decision_reason"])
print("\nANSWER:")
print(result_1["answer"])

# COMMAND ----------

# =========================
# TEST SESSION - TURN 2 (FOLLOW-UP)
# =========================

query_2 = "How is that different from Memento?"
result_2 = answer_with_memory_agentic_rag(query_2, session_id=session_id)

print("SESSION ID:", result_2["session_id"])
print("QUESTION:", result_2["question"])
print("\nCHAT HISTORY USED:")
print(result_2["chat_history_used"])
print("\nDECISION:", result_2["decision_action"])
print(result_2["decision_reason"])
print("\nANSWER:")
print(result_2["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Validate follow-up memory behavior
# MAGIC
# MAGIC This is the key memory test.
# MAGIC
# MAGIC The second query is intentionally short and ambiguous:
# MAGIC **"How is that different from Memento?"**
# MAGIC
# MAGIC A system without memory may fail here.  
# MAGIC A memory-enabled system should use prior conversation turns to interpret the follow-up correctly.

# COMMAND ----------

# =========================
# LOAD SESSION LOGS
# =========================

logs_df = load_session_logs()
logs_df.tail(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Inspect stored session logs
# MAGIC
# MAGIC This table shows the persistent memory records saved by the notebook.
# MAGIC
# MAGIC ### What it proves
# MAGIC - each turn is stored
# MAGIC - each session has an ID
# MAGIC - tool decisions are recorded
# MAGIC - answers are stored
# MAGIC - source usage is preserved

# COMMAND ----------

# =========================
# INSPECT SESSION SUMMARY
# =========================

with open(SESSION_SUMMARY_JSON_PATH, "r") as f:
    summary_json = json.load(f)

summary_df = pd.DataFrame([summary_json])
summary_df

# COMMAND ----------

# =========================
# OPTIONAL INTERACTIVE MEMORY CHAT
# =========================

def start_memory_chat(session_id: Optional[str] = None):
    if session_id is None:
        session_id = create_new_session_id()
        print("New session created:", session_id)
    else:
        print("Continuing session:", session_id)

    return session_id


def ask_memory_chat(question: str, session_id: str):
    result = answer_with_memory_agentic_rag(question, session_id=session_id)

    print("\nSESSION ID:")
    print(result["session_id"])

    print("\nQUESTION:")
    print(result["question"])

    print("\nDECISION:")
    print(result["decision_action"])
    print(result["decision_reason"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nINTERNAL SOURCES:")
    for s in result["internal_sources"]:
        print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']}")

    if result["google_results"]:
        print("\nGOOGLE RESULTS:")
        for item in result["google_results"][:3]:
            print("-", item.get("title", ""), "|", item.get("link", ""))

    if result["youtube_results"]:
        print("\nYOUTUBE RESULTS:")
        for item in result["youtube_results"][:3]:
            print("-", item.get("title", ""), "|", item.get("link", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Optional interactive memory chat helpers
# MAGIC
# MAGIC These helpers allow manual testing of:
# MAGIC - new session creation
# MAGIC - follow-up questions in the same session
# MAGIC - persistent memory behavior

# COMMAND ----------

