# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Build ChromaDB Vector Database
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook builds the vector database layer of the RAG pipeline using **ChromaDB**.
# MAGIC
# MAGIC At this stage, the project already has:
# MAGIC - ingested PDF documents
# MAGIC - extracted document text
# MAGIC - chunk-level metadata
# MAGIC
# MAGIC The next step is to convert those chunks into a searchable vector database.
# MAGIC
# MAGIC This notebook will:
# MAGIC - load chunk metadata
# MAGIC - validate the chunk data
# MAGIC - convert each chunk into a LangChain `Document`
# MAGIC - generate embeddings using OpenAI
# MAGIC - store text, metadata, and embeddings in ChromaDB
# MAGIC - validate retrieval using sample queries
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC This is the stage where the knowledge base becomes searchable by meaning instead of exact keywords.
# MAGIC
# MAGIC By storing chunk embeddings in ChromaDB, the system can retrieve relevant chunks for questions about:
# MAGIC - themes
# MAGIC - character motivations
# MAGIC - symbolic meaning
# MAGIC - narrative structure
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why ChromaDB
# MAGIC ChromaDB is used because it:
# MAGIC - stores vectors and metadata together
# MAGIC - integrates cleanly with LangChain
# MAGIC - supports persistent local storage
# MAGIC - makes source traceability easier

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-openai langchain-community langchain-chroma chromadb openai pandas numpy tiktoken python-dotenv

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This notebook requires:
# MAGIC - `langchain` for orchestration compatibility
# MAGIC - `langchain-openai` for embeddings
# MAGIC - `langchain-chroma` for Chroma integration
# MAGIC - `chromadb` as the vector database
# MAGIC - `openai` for embedding model access
# MAGIC - `pandas` for metadata handling
# MAGIC - `numpy` and `tiktoken` for utility support
# MAGIC - `python-dotenv` for loading environment variables from a `.env` file
# MAGIC
# MAGIC If Databricks asks for a Python restart after installation, restart and rerun the notebook from the top.

# COMMAND ----------

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import required libraries
# MAGIC
# MAGIC This section imports the libraries needed to:
# MAGIC - load chunk metadata
# MAGIC - convert rows into LangChain `Document` objects
# MAGIC - load environment variables
# MAGIC - initialize embeddings
# MAGIC - build the ChromaDB vector store

# COMMAND ----------

# =========================
# PROJECT CONFIGURATION
# =========================

from pathlib import Path

PROJECT_NAME = "domain_specific_rag_movie_analysis"
DOMAIN_NAME = "Narrative, thematic, and character analysis of curated film documents"

EMBED_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "movie_analysis_chunks"

BASE_DIR = Path("/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_CSV_PATH = PROCESSED_DIR / "chunks_metadata.csv"

# Use writable temp directory in current Databricks runtime
CHROMA_PERSIST_DIR = "/tmp/chroma_db/movie_analysis"

REBUILD_COLLECTION = True

print("PROJECT_NAME =", PROJECT_NAME)
print("DOMAIN_NAME =", DOMAIN_NAME)
print("EMBED_MODEL =", EMBED_MODEL)
print("COLLECTION_NAME =", COLLECTION_NAME)
print("CHUNKS_CSV_PATH =", CHUNKS_CSV_PATH)
print("CHROMA_PERSIST_DIR =", CHROMA_PERSIST_DIR)
print("REBUILD_COLLECTION =", REBUILD_COLLECTION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project configuration
# MAGIC
# MAGIC This section stores the key settings for the vector database stage.
# MAGIC
# MAGIC ### Main settings
# MAGIC - embedding model
# MAGIC - Chroma collection name
# MAGIC - input chunk metadata path
# MAGIC - persistent Chroma storage path
# MAGIC - rebuild behavior
# MAGIC
# MAGIC ### Why this matters
# MAGIC Keeping configuration in one place makes the notebook easier to maintain and explain.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure API Access Using `.env`
# MAGIC
# MAGIC This project uses OpenAI embeddings to convert chunk text into vectors.
# MAGIC
# MAGIC To keep credentials separate from notebook code, the API key is stored in a `.env` file at:
# MAGIC
# MAGIC `/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project/.env`
# MAGIC

# COMMAND ----------


# =========================
# API KEY CONFIGURATION (.env)
# =========================

ENV_PATH = Path("/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project/.env")

if not ENV_PATH.exists():
    raise FileNotFoundError(
        f".env file not found at: {ENV_PATH}\n"
        "Please create the .env file and add:\n"
        "OPENAI_API_KEY=your_openai_api_key"
    )

load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in .env file.\n"
        "Please make sure your .env file contains:\n"
        "OPENAI_API_KEY=your_openai_api_key"
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print("OPENAI_API_KEY loaded successfully from .env file.")

# COMMAND ----------

# =========================
# LOAD CHUNK METADATA
# =========================

if not CHUNKS_CSV_PATH.exists():
    raise FileNotFoundError(
        f"Chunk metadata file not found: {CHUNKS_CSV_PATH}. "
        "Please run the chunking notebook first."
    )

chunks_df = pd.read_csv(CHUNKS_CSV_PATH)

print("Loaded chunk rows:", len(chunks_df))
print("Columns:", list(chunks_df.columns))
chunks_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load chunk metadata
# MAGIC
# MAGIC This notebook starts from the output of the chunking stage.
# MAGIC
# MAGIC ### Expected columns
# MAGIC - `chunk_id`
# MAGIC - `doc_id`
# MAGIC - `file_name`
# MAGIC - `source_path`
# MAGIC - `chunk_index`
# MAGIC - `chunk_text`
# MAGIC - `chunk_char_count`
# MAGIC - `chunk_token_count`
# MAGIC - `chunk_size`
# MAGIC - `chunk_overlap`
# MAGIC - `domain_tag`
# MAGIC - `created_at`
# MAGIC
# MAGIC These chunk records will be stored in ChromaDB as searchable vector documents.

# COMMAND ----------

# =========================
# VALIDATE CHUNK METADATA
# =========================

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
    raise ValueError(f"Missing required columns: {missing_cols}")

chunks_df = chunks_df.copy()

chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str).str.strip()
chunks_df["chunk_id"] = chunks_df["chunk_id"].astype(str)
chunks_df["doc_id"] = chunks_df["doc_id"].astype(str)
chunks_df["file_name"] = chunks_df["file_name"].astype(str)
chunks_df["source_path"] = chunks_df["source_path"].astype(str)
chunks_df["domain_tag"] = chunks_df["domain_tag"].astype(str)
chunks_df["created_at"] = chunks_df["created_at"].astype(str)

chunks_df["chunk_index"] = chunks_df["chunk_index"].astype(int)
chunks_df["chunk_char_count"] = chunks_df["chunk_char_count"].astype(int)
chunks_df["chunk_token_count"] = chunks_df["chunk_token_count"].astype(int)
chunks_df["chunk_size"] = chunks_df["chunk_size"].astype(int)
chunks_df["chunk_overlap"] = chunks_df["chunk_overlap"].astype(int)

empty_chunks = chunks_df[chunks_df["chunk_text"] == ""]
duplicate_chunk_ids = chunks_df["chunk_id"].duplicated().sum()

print("Empty chunk count:", len(empty_chunks))
print("Duplicate chunk_id count:", duplicate_chunk_ids)

if len(empty_chunks) > 0:
    raise ValueError("Empty chunk_text values found. Please fix chunk generation before building ChromaDB.")

if duplicate_chunk_ids > 0:
    raise ValueError("Duplicate chunk_id values found. chunk_id must be unique.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate chunk metadata
# MAGIC
# MAGIC Before inserting data into ChromaDB, the chunk dataset is validated.
# MAGIC
# MAGIC ### Checks performed
# MAGIC - confirms that all required columns exist
# MAGIC - removes invalid empty text
# MAGIC - normalizes column types
# MAGIC - checks for duplicate chunk IDs
# MAGIC
# MAGIC ### Why this matters
# MAGIC A clean vector database depends on clean metadata.  
# MAGIC If source metadata is inconsistent, retrieval traceability will break later.

# COMMAND ----------

# =========================
# CONVERT CHUNKS TO DOCUMENT OBJECTS
# =========================

documents: List[Document] = []

for _, row in chunks_df.iterrows():
    doc = Document(
        page_content=row["chunk_text"],
        metadata={
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "file_name": row["file_name"],
            "source_path": row["source_path"],
            "chunk_index": row["chunk_index"],
            "chunk_char_count": row["chunk_char_count"],
            "chunk_token_count": row["chunk_token_count"],
            "chunk_size": row["chunk_size"],
            "chunk_overlap": row["chunk_overlap"],
            "domain_tag": row["domain_tag"],
            "created_at": row["created_at"]
        }
    )
    documents.append(doc)

print("Total LangChain Document objects created:", len(documents))
print("Sample metadata:", documents[0].metadata)
print("Sample text preview:")
print(documents[0].page_content[:400])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Convert chunk rows into LangChain `Document` objects
# MAGIC
# MAGIC This is one of the most important conceptual steps.
# MAGIC
# MAGIC Right now the chunk data is stored in a table.  
# MAGIC LangChain works more naturally when each chunk is represented as a `Document` object.
# MAGIC
# MAGIC ### Each chunk becomes
# MAGIC - `page_content` → actual chunk text
# MAGIC - `metadata` → source and traceability information
# MAGIC
# MAGIC ### Why this matters
# MAGIC This allows retrieval to return:
# MAGIC - text
# MAGIC - metadata
# MAGIC together in one object, which is much cleaner than maintaining a separate mapping file.

# COMMAND ----------

# =========================
# INITIALIZE EMBEDDINGS
# =========================

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY
)

print("Embeddings object initialized successfully.")
print("Embedding model:", EMBED_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Initialize the embedding model
# MAGIC
# MAGIC This section creates the embeddings wrapper used to convert chunk text into vectors.
# MAGIC
# MAGIC These embeddings are what make semantic retrieval possible.

# COMMAND ----------

# =========================
# PREPARE CHROMA PERSIST DIRECTORY
# =========================

import shutil
from pathlib import Path

persist_path = Path(CHROMA_PERSIST_DIR)

if REBUILD_COLLECTION and persist_path.exists():
    shutil.rmtree(persist_path)
    print("Existing Chroma persistence directory removed.")

persist_path.mkdir(parents=True, exist_ok=True)
print("Chroma persistence directory ready:", persist_path)

# COMMAND ----------

# =========================
# BUILD CHROMA VECTOR STORE
# =========================

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=COLLECTION_NAME
)

print("ChromaDB vector store created successfully.")
print("Collection name:", COLLECTION_NAME)
print("Persist directory:", CHROMA_PERSIST_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Build the ChromaDB vector store
# MAGIC
# MAGIC This is the main vector database creation step.
# MAGIC
# MAGIC ### What happens here
# MAGIC - every chunk document is embedded
# MAGIC - embeddings are stored in ChromaDB
# MAGIC - metadata remains attached to each chunk
# MAGIC - the collection is persisted for reuse
# MAGIC
# MAGIC This completes the semantic retrieval layer of the project.

# COMMAND ----------

# =========================
# VALIDATE COLLECTION SIZE
# =========================

collection_count = vector_store._collection.count()

print("Chroma collection record count:", collection_count)

if collection_count != len(documents):
    raise ValueError(
        f"Collection count mismatch. Expected {len(documents)}, got {collection_count}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Validate collection size
# MAGIC
# MAGIC This sanity check confirms that all chunk documents were inserted into the vector database.
# MAGIC
# MAGIC The total number of Chroma records should match the number of chunk documents created.

# COMMAND ----------

# =========================
# CREATE RETRIEVER
# =========================

TOP_K = 5

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Retriever created successfully.")
print("Retriever top_k =", TOP_K)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Create the retriever
# MAGIC
# MAGIC The vector store is now converted into a retriever.
# MAGIC
# MAGIC ### What the retriever does
# MAGIC For any user query, it:
# MAGIC - embeds the query
# MAGIC - searches the vector database
# MAGIC - returns the top relevant chunk documents

# COMMAND ----------

# =========================
# TEST RETRIEVAL
# =========================

test_query = "How does Interstellar connect love and survival?"
retrieved_docs = retriever.invoke(test_query)

print("Retrieved documents:", len(retrieved_docs))

for i, doc in enumerate(retrieved_docs, start=1):
    print("=" * 100)
    print(f"Result {i}")
    print("doc_id:", doc.metadata.get("doc_id"))
    print("chunk_id:", doc.metadata.get("chunk_id"))
    print("file_name:", doc.metadata.get("file_name"))
    print("chunk_index:", doc.metadata.get("chunk_index"))
    print("chunk_text preview:")
    print(doc.page_content[:500])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Test retrieval quality
# MAGIC
# MAGIC This section performs a sample semantic retrieval test.
# MAGIC
# MAGIC ### What is being checked
# MAGIC - relevant chunks are returned
# MAGIC - metadata remains attached
# MAGIC - retrieval is meaningful for a domain-specific question
# MAGIC
# MAGIC This is an important checkpoint before building the full answer-generation chain.

# COMMAND ----------

# =========================
# MULTI-QUERY VALIDATION
# =========================

test_queries = [
    "How does Interstellar connect love and survival?",
    "What role does memory play in Memento?",
    "How is identity represented in Fight Club?",
    "How does Inception treat the boundary between dream and reality?",
    "What makes the Joker disruptive in The Dark Knight?"
]

validation_rows = []

for q in test_queries:
    docs = retriever.invoke(q)
    validation_rows.append({
        "query": q,
        "retrieved_count": len(docs),
        "top_chunk_ids": [d.metadata.get("chunk_id") for d in docs]
    })

validation_df = pd.DataFrame(validation_rows)
validation_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Run multi-query validation
# MAGIC
# MAGIC A good vector database should work across multiple question types.
# MAGIC
# MAGIC This table helps verify:
# MAGIC - retrieval consistency
# MAGIC - metadata visibility
# MAGIC - domain relevance

# COMMAND ----------

# =========================
# SAVE BUILD SUMMARY
# =========================

build_summary = {
    "project_name": PROJECT_NAME,
    "domain_name": DOMAIN_NAME,
    "collection_name": COLLECTION_NAME,
    "embed_model": EMBED_MODEL,
    "chunk_rows_loaded": int(len(chunks_df)),
    "documents_indexed": int(len(documents)),
    "collection_count": int(collection_count),
    "persist_directory": CHROMA_PERSIST_DIR,
    "built_at": datetime.utcnow().isoformat()
}

summary_path = PROCESSED_DIR / "chromadb_build_summary.json"

with open(summary_path, "w") as f:
    json.dump(build_summary, f, indent=2)

print("ChromaDB build summary saved to:", summary_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Save build summary
# MAGIC
# MAGIC This section saves a JSON summary of the ChromaDB build.
# MAGIC
# MAGIC This is useful for:
# MAGIC - reproducibility
# MAGIC - debugging
# MAGIC - reporting
# MAGIC - MLflow integration later

# COMMAND ----------

# =========================
# FINAL SUMMARY TABLE
# =========================

summary_df = pd.DataFrame([build_summary])
summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Final vector database summary
# MAGIC
# MAGIC This table gives a final summary of:
# MAGIC - collection name
# MAGIC - embedding model
# MAGIC - number of indexed chunks
# MAGIC - persistence path
# MAGIC
# MAGIC This is useful for reporting and presentation.

# COMMAND ----------

