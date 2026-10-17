# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Agentic RAG with Tool Calling
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook upgrades the existing RAG pipeline into an **agentic RAG system**.
# MAGIC
# MAGIC The current project already supports:
# MAGIC - document ingestion
# MAGIC - chunking
# MAGIC - ChromaDB vector retrieval
# MAGIC - LangChain-based retrieval-augmented answering
# MAGIC
# MAGIC This notebook adds:
# MAGIC - tool calling
# MAGIC - model-based decision making
# MAGIC - external fallback search
# MAGIC - multi-source evidence synthesis
# MAGIC
# MAGIC The system will now behave like this:
# MAGIC
# MAGIC 1. search the internal PDF knowledge base first
# MAGIC 2. check whether internal evidence is sufficient
# MAGIC 3. if needed, use external tools
# MAGIC 4. gather evidence from:
# MAGIC    - internal ChromaDB
# MAGIC    - Google/Web
# MAGIC    - YouTube
# MAGIC 5. generate a final grounded answer
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC This notebook implements the **agentic feature** of the project.
# MAGIC
# MAGIC The model is no longer limited to:
# MAGIC - only internal retrieval
# MAGIC - only one fixed pipeline path
# MAGIC
# MAGIC Instead, it can decide when to:
# MAGIC - answer from PDFs
# MAGIC - search outside sources
# MAGIC - combine multiple evidence sources
# MAGIC
# MAGIC This is a much stronger final architecture for:
# MAGIC - tool calling
# MAGIC - adaptive retrieval
# MAGIC - broader coverage
# MAGIC - production-style AI behavior

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-openai langchain-community langchain-chroma chromadb openai pandas python-dotenv requests pydantic

# COMMAND ----------

# MAGIC %pip install "pydantic>=2.0"

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-openai langchain-community langchain-chroma

# COMMAND ----------

# MAGIC %pip install -U typing_extensions

# COMMAND ----------

# MAGIC %pip install -U chromadb

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Literal

import requests
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# Same temporary Chroma path used in earlier notebooks
CHROMA_PERSIST_DIR = "/tmp/chroma_db/movie_analysis"

# External tool limits
GOOGLE_RESULTS_LIMIT = 5
YOUTUBE_RESULTS_LIMIT = 5

print("PROJECT_NAME =", PROJECT_NAME)
print("DOMAIN_NAME =", DOMAIN_NAME)
print("CHAT_MODEL =", CHAT_MODEL)
print("EMBED_MODEL =", EMBED_MODEL)
print("COLLECTION_NAME =", COLLECTION_NAME)
print("TOP_K =", TOP_K)
print("CHROMA_PERSIST_DIR =", CHROMA_PERSIST_DIR)
print("GOOGLE_RESULTS_LIMIT =", GOOGLE_RESULTS_LIMIT)
print("YOUTUBE_RESULTS_LIMIT =", YOUTUBE_RESULTS_LIMIT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project configuration
# MAGIC
# MAGIC This section stores the main configuration used by the agentic RAG system.
# MAGIC
# MAGIC ### Internal retrieval settings
# MAGIC - embedding model
# MAGIC - Chroma collection name
# MAGIC - top-k retrieval
# MAGIC
# MAGIC ### External tool settings
# MAGIC - number of web results
# MAGIC - number of YouTube results
# MAGIC
# MAGIC This makes the notebook easier to tune and debug.

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
    raise ValueError(
        "OPENAI_API_KEY not found in .env file.\n"
        "Please add: OPENAI_API_KEY=your_openai_api_key"
    )

if not SERPAPI_API_KEY:
    raise ValueError(
        "SERPAPI_API_KEY not found in .env file.\n"
        "Please add: SERPAPI_API_KEY=your_serpapi_key"
    )

if not YOUTUBE_API_KEY:
    raise ValueError(
        "YOUTUBE_API_KEY not found in .env file.\n"
        "Please add: YOUTUBE_API_KEY=your_youtube_api_key"
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("OPENAI_API_KEY loaded successfully.")
print("SERPAPI_API_KEY loaded successfully.")
print("YOUTUBE_API_KEY loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load environment variables
# MAGIC
# MAGIC This section loads all credentials required for the agentic RAG system.
# MAGIC
# MAGIC The notebook will stop early if any required key is missing.
# MAGIC
# MAGIC This is important because the full agentic version depends on:
# MAGIC - OpenAI
# MAGIC - Google/Web search
# MAGIC - YouTube search

# COMMAND ----------

# =========================
# INITIALIZE EMBEDDINGS AND CHAT MODEL
# =========================

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
    api_key=OPENAI_API_KEY
)

print("Embeddings and chat model initialized successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Initialize embeddings and chat model
# MAGIC
# MAGIC The system uses:
# MAGIC - embeddings for internal semantic retrieval
# MAGIC - chat model for agent decisions and final answer generation
# MAGIC
# MAGIC The same chat model is used for:
# MAGIC - deciding whether to use tools
# MAGIC - synthesizing the final answer

# COMMAND ----------

# =========================
# LOAD OR REBUILD CHROMA VECTOR STORE
# =========================

def build_documents_from_chunks_csv(chunks_csv_path: Path) -> List[Document]:
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

    documents = []
    for _, row in chunks_df.iterrows():
        documents.append(
            Document(
                page_content=row["chunk_text"],
                metadata={
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
                }
            )
        )

    return documents


vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

collection_count = vector_store._collection.count()
print("Initial Chroma collection count:", collection_count)

if collection_count == 0:
    print("Chroma collection is empty. Rebuilding from chunks_metadata.csv ...")
    documents = build_documents_from_chunks_csv(CHUNKS_CSV_PATH)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    collection_count = vector_store._collection.count()
    print("Rebuilt Chroma collection count:", collection_count)

if collection_count == 0:
    raise ValueError(
        "Chroma collection is still empty after rebuild. "
        "Please verify chunks_metadata.csv and embedding setup."
    )

print("Chroma vector store ready.")
print("Collection name:", COLLECTION_NAME)
print("Collection count:", collection_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load or rebuild the internal vector store
# MAGIC
# MAGIC This section ensures that the internal PDF knowledge base is available.
# MAGIC
# MAGIC ### Why rebuild logic is included
# MAGIC The ChromaDB path currently uses temporary runtime storage, so the collection may disappear between sessions.
# MAGIC
# MAGIC To make the notebook robust, it rebuilds automatically from `chunks_metadata.csv` if needed.

# COMMAND ----------

# =========================
# CREATE INTERNAL RETRIEVER
# =========================

internal_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Internal retriever created successfully.")
print("Retriever top_k =", TOP_K)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Create the internal retriever
# MAGIC
# MAGIC This retriever searches the internal PDF knowledge base stored in ChromaDB.
# MAGIC
# MAGIC This is the first source the agent should check before deciding whether external tools are needed.

# COMMAND ----------

# =========================
# INTERNAL PDF SEARCH TOOL
# =========================

def internal_pdf_search(question: str, top_k: int = TOP_K) -> List[Document]:
    """
    Search the internal ChromaDB knowledge base.
    """
    runtime_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    return runtime_retriever.invoke(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Define the internal PDF search tool
# MAGIC
# MAGIC This wraps the internal retriever as a callable tool.
# MAGIC
# MAGIC The agent will treat the internal knowledge base as one of its available evidence sources.

# COMMAND ----------

# =========================
# GOOGLE / WEB SEARCH TOOL
# =========================

def google_web_search(question: str, num_results: int = GOOGLE_RESULTS_LIMIT) -> List[Dict[str, Any]]:
    """
    Search the web using SerpAPI Google Search.
    """
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

# MAGIC %md
# MAGIC ## 10. Define the Google/Web search tool
# MAGIC
# MAGIC This external tool is used when the internal PDF knowledge base does not contain enough evidence.
# MAGIC
# MAGIC ### What it returns
# MAGIC For each result:
# MAGIC - title
# MAGIC - link
# MAGIC - short snippet

# COMMAND ----------

# =========================
# YOUTUBE SEARCH TOOL
# =========================

def youtube_search(question: str, max_results: int = YOUTUBE_RESULTS_LIMIT) -> List[Dict[str, Any]]:
    """
    Search YouTube videos using the YouTube Data API.
    """
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
# MAGIC ## 11. Define the YouTube search tool
# MAGIC
# MAGIC This external tool is used when the system needs broader explanatory material, video analysis, interviews, or public commentary.
# MAGIC
# MAGIC ### What it returns
# MAGIC For each result:
# MAGIC - title
# MAGIC - link
# MAGIC - short description snippet

# COMMAND ----------

# =========================
# FORMATTERS FOR EVIDENCE
# =========================

def format_internal_docs(docs: List[Document]) -> str:
    if not docs:
        return "No internal PDF evidence found."

    blocks = []
    for doc in docs:
        block = (
            f"[Internal Source: {doc.metadata.get('doc_id')} | {doc.metadata.get('chunk_id')}]\n"
            f"{doc.page_content}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def format_external_results(results: List[Dict[str, Any]], source_name: str) -> str:
    if not results:
        return f"No {source_name} results found."

    blocks = []
    for idx, item in enumerate(results, start=1):
        block = (
            f"[{source_name} Result {idx}]\n"
            f"Title: {item.get('title', '')}\n"
            f"Link: {item.get('link', '')}\n"
            f"Snippet: {item.get('snippet', '')}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Define evidence formatting helpers
# MAGIC
# MAGIC These helpers format:
# MAGIC - internal retrieved chunks
# MAGIC - Google/Web search results
# MAGIC - YouTube search results
# MAGIC
# MAGIC This makes all evidence easy to inject into prompts.

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

# MAGIC %md
# MAGIC ## 13. Define the tool decision schema
# MAGIC
# MAGIC The model should decide whether to:
# MAGIC - answer using internal evidence only
# MAGIC - call Google/Web search
# MAGIC - call YouTube search
# MAGIC - call both external tools
# MAGIC
# MAGIC This structured schema ensures the decision is clear and machine-readable.

# COMMAND ----------

# =========================
# TOOL DECISION CHAIN
# =========================

decision_llm = llm.with_structured_output(ToolDecision)

decision_prompt = ChatPromptTemplate.from_template("""
You are an agentic routing model for a domain-specific RAG system.

Your job is to decide whether the current internal PDF evidence is enough to answer the user's question.

Rules:
- Choose INTERNAL_ONLY if the internal PDF evidence is clearly sufficient.
- Choose GOOGLE_SEARCH if broader web information is needed.
- Choose YOUTUBE_SEARCH if video-based or broader explanatory material would be more useful.
- Choose GOOGLE_AND_YOUTUBE if both external web and video sources would help.

Question:
{question}

Internal Retrieved Evidence:
{internal_context}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Build the model-based decision step
# MAGIC
# MAGIC This is the key agentic feature.
# MAGIC
# MAGIC The model now decides:
# MAGIC - whether internal evidence is sufficient
# MAGIC - whether external tools should be called
# MAGIC
# MAGIC This is not hardcoded branching.  
# MAGIC The model makes the decision based on the question and the internal evidence.

# COMMAND ----------

# =========================
# FINAL SYNTHESIS PROMPT
# =========================

final_prompt = ChatPromptTemplate.from_template("""
You are a professional domain-specific agentic RAG assistant.

Use the available evidence to answer the user's question.

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

Question:
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
# MAGIC ## 15. Define the final synthesis prompt
# MAGIC
# MAGIC This prompt combines all available evidence and produces the final answer.
# MAGIC
# MAGIC ### Evidence sources
# MAGIC - internal PDFs
# MAGIC - Google/Web search
# MAGIC - YouTube search
# MAGIC
# MAGIC This is where the system behaves like an agentic assistant rather than a simple retriever.

# COMMAND ----------

# =========================
# AGENTIC RAG FUNCTION
# =========================

def answer_with_agentic_rag(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Agentic RAG pipeline:
    1. Search internal ChromaDB
    2. Let model decide if tools are needed
    3. Call tools if needed
    4. Generate final answer from combined evidence
    """
    # Step 1: internal retrieval
    internal_docs = internal_pdf_search(question, top_k=top_k)
    internal_context = format_internal_docs(internal_docs)

    # Step 2: model decides tool usage
    decision = decision_llm.invoke(
        decision_prompt.format_messages(
            question=question,
            internal_context=internal_context
        )
    )

    google_results = []
    youtube_results = []

    # Step 3: external tools if needed
    if decision.action in ["GOOGLE_SEARCH", "GOOGLE_AND_YOUTUBE"]:
        google_results = google_web_search(question, num_results=GOOGLE_RESULTS_LIMIT)

    if decision.action in ["YOUTUBE_SEARCH", "GOOGLE_AND_YOUTUBE"]:
        youtube_results = youtube_search(question, max_results=YOUTUBE_RESULTS_LIMIT)

    google_context = format_external_results(google_results, "Google/Web")
    youtube_context = format_external_results(youtube_results, "YouTube")

    # Step 4: final synthesis
    final_answer = final_chain.invoke({
        "question": question,
        "internal_context": internal_context,
        "google_context": google_context if google_results else "No Google/Web evidence used.",
        "youtube_context": youtube_context if youtube_results else "No YouTube evidence used."
    })

    # Step 5: package sources
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

    return {
        "question": question,
        "decision_action": decision.action,
        "decision_reason": decision.reason,
        "answer": final_answer,
        "internal_sources": internal_sources,
        "google_results": google_results,
        "youtube_results": youtube_results
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Build the full agentic RAG function
# MAGIC
# MAGIC This is the main function of the notebook.
# MAGIC
# MAGIC ### Flow
# MAGIC 1. retrieve internal evidence
# MAGIC 2. let the model decide whether external tools are needed
# MAGIC 3. call selected tools
# MAGIC 4. synthesize the final answer from all available evidence
# MAGIC
# MAGIC This is the full agentic workflow.

# COMMAND ----------

# =========================
# TEST QUERY 1 - INTERNAL DOMAIN QUESTION
# =========================

query_1 = "How does Interstellar connect love and survival?"
result_1 = answer_with_agentic_rag(query_1)

print("QUESTION:")
print(result_1["question"])

print("\nDECISION:")
print(result_1["decision_action"])
print(result_1["decision_reason"])

print("\nANSWER:")
print(result_1["answer"])

print("\nINTERNAL SOURCES:")
for s in result_1["internal_sources"]:
    print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']}")

# COMMAND ----------

# =========================
# TEST QUERY 2 - CROSS-DOCUMENT ANALYSIS
# =========================

query_2 = "Compare how memory is represented in Memento and Interstellar."
result_2 = answer_with_agentic_rag(query_2)

print("QUESTION:")
print(result_2["question"])

print("\nDECISION:")
print(result_2["decision_action"])
print(result_2["decision_reason"])

print("\nANSWER:")
print(result_2["answer"])

print("\nINTERNAL SOURCES:")
for s in result_2["internal_sources"]:
    print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']}")

# COMMAND ----------

# =========================
# TEST QUERY 3 - OUTSIDE KNOWLEDGE BASE
# =========================

query_3 = "What are some well-known academic interpretations of nihilism in Fight Club outside the provided PDFs?"
result_3 = answer_with_agentic_rag(query_3)

print("QUESTION:")
print(result_3["question"])

print("\nDECISION:")
print(result_3["decision_action"])
print(result_3["decision_reason"])

print("\nANSWER:")
print(result_3["answer"])

print("\nGOOGLE RESULTS:")
for item in result_3["google_results"][:3]:
    print("-", item.get("title", ""), "|", item.get("link", ""))

print("\nYOUTUBE RESULTS:")
for item in result_3["youtube_results"][:3]:
    print("-", item.get("title", ""), "|", item.get("link", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Validate different agent behaviors
# MAGIC
# MAGIC These tests help verify:
# MAGIC - purely internal retrieval
# MAGIC - cross-document reasoning
# MAGIC - external fallback behavior
# MAGIC
# MAGIC This is important for demonstrating that the system is truly agentic.

# COMMAND ----------

# =========================
# BATCH VALIDATION SUMMARY
# =========================

test_questions = [
    "How does Interstellar connect love and survival?",
    "What role does memory play in Memento?",
    "How is identity represented in Fight Club?",
    "What are some external perspectives on dream theory related to Inception?",
    "Are there YouTube explainers on the Joker's ideology in The Dark Knight?"
]

batch_rows = []

for q in test_questions:
    res = answer_with_agentic_rag(q)
    batch_rows.append({
        "question": q,
        "decision_action": res["decision_action"],
        "answer_preview": res["answer"][:250],
        "num_internal_sources": len(res["internal_sources"]),
        "num_google_results": len(res["google_results"]),
        "num_youtube_results": len(res["youtube_results"])
    })

batch_results_df = pd.DataFrame(batch_rows)
batch_results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Batch validation table
# MAGIC
# MAGIC This table shows how the agent behaves across multiple query types.
# MAGIC
# MAGIC It helps verify:
# MAGIC - whether the model is using only internal evidence when possible
# MAGIC - whether it uses external tools when needed
# MAGIC - whether source routing is functioning correctly

# COMMAND ----------

# =========================
# OPTIONAL INTERACTIVE HELPER
# =========================

def ask_agentic_question():
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return

    result = answer_with_agentic_rag(question)

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
# MAGIC ## 19. Optional interactive helper
# MAGIC
# MAGIC This function allows manual testing of the full agentic system during development or demo rehearsal.

# COMMAND ----------

# MAGIC %md
# MAGIC # Overall Summary
# MAGIC
# MAGIC This notebook completes the **agentic RAG stage** of the project.
# MAGIC
# MAGIC ## What is complete after this notebook
# MAGIC - internal PDF retrieval is available through ChromaDB
# MAGIC - the model decides whether internal evidence is sufficient
# MAGIC - Google/Web search tool is available as fallback
# MAGIC - YouTube search tool is available as fallback
# MAGIC - the model can choose between:
# MAGIC   - internal only
# MAGIC   - Google/Web
# MAGIC   - YouTube
# MAGIC   - both external tools
# MAGIC - final answers are synthesized from combined evidence
# MAGIC - source traceability is preserved
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC This notebook transforms the project from a standard RAG pipeline into an **agentic AI application** with:
# MAGIC - tool calling
# MAGIC - adaptive evidence gathering
# MAGIC - model-based routing
# MAGIC - broader coverage beyond the local PDF corpus
# MAGIC

# COMMAND ----------

