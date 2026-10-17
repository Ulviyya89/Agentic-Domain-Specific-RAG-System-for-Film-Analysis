# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Chunk and Store Metadata
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook performs the **chunking stage** of the RAG pipeline.
# MAGIC
# MAGIC After document ingestion, the project already has full extracted text for each PDF document. However, long documents are not suitable for direct retrieval or direct use in LLM prompting.
# MAGIC
# MAGIC This notebook converts each document into smaller retrieval units called **chunks**.
# MAGIC
# MAGIC The notebook is responsible for:
# MAGIC - loading document-level extracted text
# MAGIC - cleaning and normalizing the text
# MAGIC - splitting each document into overlapping chunks
# MAGIC - assigning chunk-level identifiers
# MAGIC - storing chunk text together with source metadata
# MAGIC - saving the final chunk dataset for embedding and vector storage
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC Retrieval-Augmented Generation works best when the knowledge base is split into smaller, meaningful, searchable units.
# MAGIC
# MAGIC If chunks are too large:
# MAGIC - retrieval becomes less precise
# MAGIC - context windows are wasted
# MAGIC - irrelevant text is pulled into the answer
# MAGIC
# MAGIC If chunks are too small:
# MAGIC - important context may be lost
# MAGIC - meaning may become fragmented
# MAGIC
# MAGIC This notebook creates the chunk-level structure that the retriever will later use.

# COMMAND ----------

# MAGIC %pip install -U pandas tiktoken

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This notebook mainly uses:
# MAGIC
# MAGIC - `pandas` → to load and store structured metadata
# MAGIC - `tiktoken` → optional token-aware inspection
# MAGIC - core Python string logic for chunking
# MAGIC
# MAGIC Although chunking can be implemented in many ways, this notebook starts with a clean and explainable baseline approach.

# COMMAND ----------

import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd
import tiktoken

# COMMAND ----------

# =========================
# PROJECT PATH CONFIG
# =========================

BASE_DIR = Path("/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DOCUMENTS_CSV_PATH = PROCESSED_DIR / "documents_metadata.csv"
CHUNKS_CSV_PATH = PROCESSED_DIR / "chunks_metadata.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("DOCUMENTS_CSV_PATH =", DOCUMENTS_CSV_PATH)
print("CHUNKS_CSV_PATH =", CHUNKS_CSV_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project paths
# MAGIC
# MAGIC This section defines the paths for:
# MAGIC - document-level input metadata
# MAGIC - chunk-level output metadata
# MAGIC
# MAGIC ### Input
# MAGIC - `documents_metadata.csv`
# MAGIC
# MAGIC ### Output
# MAGIC - `chunks_metadata.csv`
# MAGIC
# MAGIC This notebook assumes that the ingestion notebook has already completed successfully.

# COMMAND ----------

# =========================
# CHUNKING CONFIGURATION
# =========================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOKEN_ENCODING = "cl100k_base"

print("CHUNK_SIZE =", CHUNK_SIZE)
print("CHUNK_OVERLAP =", CHUNK_OVERLAP)
print("TOKEN_ENCODING =", TOKEN_ENCODING)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define chunking settings
# MAGIC
# MAGIC This notebook uses a baseline **character-based chunking strategy** with overlap.
# MAGIC
# MAGIC ### Parameters
# MAGIC - `CHUNK_SIZE = 1200`  
# MAGIC   Approximate number of characters in each chunk
# MAGIC
# MAGIC - `CHUNK_OVERLAP = 200`  
# MAGIC   Number of characters shared between adjacent chunks
# MAGIC
# MAGIC ### Why overlap is used
# MAGIC Overlap preserves context across chunk boundaries.  
# MAGIC Without overlap, important ideas that span chunk edges may become fragmented during retrieval.
# MAGIC
# MAGIC ### Why this is a good baseline
# MAGIC Character-based chunking is:
# MAGIC - simple
# MAGIC - reproducible
# MAGIC - easy to explain
# MAGIC - effective for a first production-style RAG baseline
# MAGIC
# MAGIC Later, this can be compared against token-based or recursive chunking in MLflow experiments.

# COMMAND ----------

# =========================
# LOAD DOCUMENT METADATA
# =========================

if not DOCUMENTS_CSV_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {DOCUMENTS_CSV_PATH}. "
        "Please run the ingestion notebook first."
    )

documents_df = pd.read_csv(DOCUMENTS_CSV_PATH)

print("Loaded documents:", len(documents_df))
print("Columns:", list(documents_df.columns))
documents_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load document-level metadata
# MAGIC
# MAGIC This notebook starts from the document-level output produced by ingestion.
# MAGIC
# MAGIC ### Expected columns
# MAGIC The input should contain fields such as:
# MAGIC - `doc_id`
# MAGIC - `file_name`
# MAGIC - `source_path`
# MAGIC - `raw_text`
# MAGIC - `char_count`
# MAGIC - `domain_tag`
# MAGIC - `ingested_at`
# MAGIC
# MAGIC ### Why this matters
# MAGIC The chunking stage depends on having the full extracted text available for each document.

# COMMAND ----------

# =========================
# VALIDATE DOCUMENT DATA
# =========================

required_columns = [
    "doc_id",
    "file_name",
    "source_path",
    "raw_text",
    "char_count",
    "domain_tag",
    "ingested_at"
]

missing_cols = [c for c in required_columns if c not in documents_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

documents_df = documents_df.copy()
documents_df["raw_text"] = documents_df["raw_text"].fillna("").astype(str)

empty_docs = documents_df[documents_df["raw_text"].str.strip() == ""]
duplicate_doc_ids = documents_df["doc_id"].duplicated().sum()

print("Empty raw_text documents:", len(empty_docs))
print("Duplicate doc_id count:", duplicate_doc_ids)

if duplicate_doc_ids > 0:
    raise ValueError("Duplicate doc_id values found. Please fix document IDs before chunking.")

# COMMAND ----------

# =========================
# TEXT CLEANING HELPERS
# =========================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving readable sentence flow.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_document_text(text: str) -> str:
    """
    Basic text cleanup for extracted PDF content.
    """
    text = normalize_whitespace(text)
    return text

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Define text cleaning helpers
# MAGIC
# MAGIC PDF extraction often produces text with:
# MAGIC - inconsistent spaces
# MAGIC - broken line breaks
# MAGIC - formatting noise
# MAGIC
# MAGIC This section normalizes the extracted text before chunking.
# MAGIC
# MAGIC ### Why this matters
# MAGIC Chunking noisy text leads to:
# MAGIC - inconsistent chunk sizes
# MAGIC - poor retrieval quality
# MAGIC - messy prompt context later
# MAGIC
# MAGIC So text cleaning is an important preprocessing step.

# COMMAND ----------

# =========================
# CHUNKING FUNCTION
# =========================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a document into overlapping character-based chunks.
    """
    text = clean_document_text(text)

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start += chunk_size - overlap

    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Define the chunking function
# MAGIC
# MAGIC This function splits each document into overlapping chunks.
# MAGIC
# MAGIC ### How it works
# MAGIC - the document text is first cleaned
# MAGIC - the text is divided into windows of size `CHUNK_SIZE`
# MAGIC - adjacent windows overlap by `CHUNK_OVERLAP`
# MAGIC - each chunk is stored as a separate retrieval unit
# MAGIC
# MAGIC ### Why this matters
# MAGIC This chunking strategy helps preserve local context while still allowing precise retrieval.

# COMMAND ----------

# =========================
# TOKEN COUNT HELPER
# =========================

encoding = tiktoken.get_encoding(TOKEN_ENCODING)

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Define token counting helper
# MAGIC
# MAGIC This helper is used to estimate token counts for each chunk.
# MAGIC
# MAGIC ### Why token counts matter
# MAGIC Even though this notebook uses character-based chunking, token counts are still useful for:
# MAGIC - understanding chunk size relative to LLM limits
# MAGIC - later evaluation and tuning
# MAGIC - MLflow experiment comparison

# COMMAND ----------

# =========================
# CREATE CHUNK RECORDS
# =========================

chunk_records: List[Dict] = []

for _, row in documents_df.iterrows():
    doc_id = row["doc_id"]
    file_name = row["file_name"]
    source_path = row["source_path"]
    domain_tag = row["domain_tag"]
    raw_text = row["raw_text"]

    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = f"{doc_id}_chunk_{idx}"

        chunk_records.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "file_name": file_name,
            "source_path": source_path,
            "chunk_index": idx,
            "chunk_text": chunk,
            "chunk_char_count": len(chunk),
            "chunk_token_count": count_tokens(chunk),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "domain_tag": domain_tag,
            "created_at": datetime.utcnow().isoformat()
        })

print("Total chunk records created:", len(chunk_records))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Create chunk-level records
# MAGIC
# MAGIC This is the main chunk generation step.
# MAGIC
# MAGIC ### For each document
# MAGIC The notebook:
# MAGIC 1. reads the full extracted text
# MAGIC 2. splits it into chunks
# MAGIC 3. assigns chunk IDs
# MAGIC 4. stores each chunk with metadata
# MAGIC
# MAGIC ### Metadata stored for each chunk
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
# MAGIC ### Why this matters
# MAGIC These chunk-level records are the direct input to:
# MAGIC - embeddings
# MAGIC - ChromaDB storage
# MAGIC - retrieval
# MAGIC - source tracing

# COMMAND ----------

# =========================
# CREATE CHUNKS DATAFRAME
# =========================

chunks_df = pd.DataFrame(chunk_records)

print("Chunks DataFrame created successfully.")
print("Shape:", chunks_df.shape)

chunks_df[[
    "chunk_id",
    "doc_id",
    "file_name",
    "chunk_index",
    "chunk_char_count",
    "chunk_token_count"
]].head()

# COMMAND ----------

# =========================
# VALIDATE CHUNK OUTPUT
# =========================

duplicate_chunk_ids = chunks_df["chunk_id"].duplicated().sum()
empty_chunks = chunks_df[chunks_df["chunk_text"].str.strip() == ""]

print("Duplicate chunk_id count:", duplicate_chunk_ids)
print("Empty chunk count:", len(empty_chunks))

if duplicate_chunk_ids > 0:
    raise ValueError("Duplicate chunk_id values found. Chunk IDs must be unique.")

if len(empty_chunks) > 0:
    raise ValueError("Empty chunks detected. Please inspect text cleaning and chunking logic.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validate chunk output
# MAGIC
# MAGIC This section ensures that:
# MAGIC - every chunk ID is unique
# MAGIC - no empty chunk text exists
# MAGIC
# MAGIC ### Why this matters
# MAGIC Chunk-level consistency is essential for clean embedding generation and source traceability later.

# COMMAND ----------

# =========================
# CHUNK DISTRIBUTION SUMMARY
# =========================

summary_by_doc = (
    chunks_df.groupby(["doc_id", "file_name"])
    .agg(
        num_chunks=("chunk_id", "count"),
        total_chunk_chars=("chunk_char_count", "sum"),
        avg_chunk_chars=("chunk_char_count", "mean"),
        avg_chunk_tokens=("chunk_token_count", "mean")
    )
    .reset_index()
)

summary_by_doc

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Review chunk distribution by document
# MAGIC
# MAGIC This summary table shows how many chunks were created for each document.
# MAGIC
# MAGIC ### Why this matters
# MAGIC It helps identify:
# MAGIC - very short documents
# MAGIC - very long documents
# MAGIC - imbalance in chunk counts
# MAGIC - unusually large or small chunk outputs

# COMMAND ----------

# =========================
# PREVIEW SAMPLE CHUNKS
# =========================

for i in range(min(3, len(chunks_df))):
    row = chunks_df.iloc[i]
    print("=" * 100)
    print("chunk_id:", row["chunk_id"])
    print("doc_id:", row["doc_id"])
    print("file_name:", row["file_name"])
    print("chunk_index:", row["chunk_index"])
    print("chunk_char_count:", row["chunk_char_count"])
    print("chunk_token_count:", row["chunk_token_count"])
    print("chunk_text preview:")
    print(row["chunk_text"][:800])
    print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Preview sample chunks
# MAGIC
# MAGIC This step prints a few generated chunks for manual inspection.
# MAGIC
# MAGIC ### Why this matters
# MAGIC Manual inspection helps confirm that:
# MAGIC - chunk boundaries are reasonable
# MAGIC - chunk text remains readable
# MAGIC - context is preserved
# MAGIC - chunk metadata is correct

# COMMAND ----------

# =========================
# SAVE CHUNK METADATA CSV
# =========================

chunks_df.to_csv(CHUNKS_CSV_PATH, index=False)

print("Chunk metadata saved successfully.")
print("Saved to:", CHUNKS_CSV_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Save chunk metadata to CSV
# MAGIC
# MAGIC This section saves the chunk dataset for later use.
# MAGIC
# MAGIC ### Output
# MAGIC - `chunks_metadata.csv`
# MAGIC
# MAGIC ### Why this matters
# MAGIC This file becomes the main input to the embedding and vector database notebook.

# COMMAND ----------

# =========================
# OPTIONAL: SAVE TO SPARK TABLE
# =========================

spark.sql("DROP TABLE IF EXISTS movie_chunks_metadata")

spark_chunks_df = spark.createDataFrame(chunks_df)
spark_chunks_df.write.mode("overwrite").saveAsTable("movie_chunks_metadata")

print("Spark table 'movie_chunks_metadata' created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Optional Spark table creation
# MAGIC
# MAGIC This section writes the chunk-level data to a Spark table.
# MAGIC
# MAGIC ### Table name
# MAGIC - `movie_chunks_metadata`
# MAGIC
# MAGIC ### Why this is useful
# MAGIC Saving chunk metadata as a Spark table supports:
# MAGIC - Databricks-native workflows
# MAGIC - easy reloading across notebooks
# MAGIC - cleaner downstream integration

# COMMAND ----------

# =========================
# FINAL SUMMARY
# =========================

final_summary = {
    "documents_loaded": int(documents_df["doc_id"].nunique()),
    "total_chunks_created": int(len(chunks_df)),
    "average_chunk_char_count": float(chunks_df["chunk_char_count"].mean()) if len(chunks_df) > 0 else 0,
    "average_chunk_token_count": float(chunks_df["chunk_token_count"].mean()) if len(chunks_df) > 0 else 0,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP
}

final_summary_df = pd.DataFrame([final_summary])
final_summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Final chunking summary
# MAGIC
# MAGIC This final summary provides a compact overview of the chunking stage.
# MAGIC
# MAGIC It is useful for:
# MAGIC - reporting
# MAGIC - presentation slides
# MAGIC - MLflow comparison later
# MAGIC - tuning decisions such as chunk size and overlap