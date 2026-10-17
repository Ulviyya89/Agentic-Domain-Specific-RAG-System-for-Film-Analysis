# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Ingest Documents
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook performs the **document ingestion stage** of the project.
# MAGIC
# MAGIC The purpose of this stage is to load the raw PDF documents from the project folder, extract their text content, and convert them into a structured document-level dataset that can be used in later stages such as chunking, embedding, and retrieval.
# MAGIC
# MAGIC This notebook is responsible for:
# MAGIC - locating source PDF files
# MAGIC - extracting text from each document
# MAGIC - assigning document-level identifiers
# MAGIC - storing the extracted text with metadata
# MAGIC - saving the results in a reusable structured format
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC A RAG system cannot directly work on raw PDF files.
# MAGIC
# MAGIC Before any chunking, embedding, or retrieval can happen, the documents must first be converted into machine-readable text.
# MAGIC
# MAGIC This notebook creates that first structured layer of the knowledge base.

# COMMAND ----------

# MAGIC %pip install -U pymupdf pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This notebook mainly depends on:
# MAGIC
# MAGIC - `pymupdf` → for reading PDF files and extracting text
# MAGIC - `pandas` → for organizing extracted document metadata into tabular format
# MAGIC
# MAGIC These libraries are sufficient for the ingestion stage.

# COMMAND ----------

import os
import re
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import required libraries
# MAGIC
# MAGIC This section imports the core Python libraries needed for document ingestion.
# MAGIC
# MAGIC ### Main roles
# MAGIC - `fitz` → open and parse PDF files
# MAGIC - `Path` → handle folder and file paths
# MAGIC - `datetime` → record ingestion timestamps
# MAGIC - `pandas` → store extracted document metadata in DataFrame form

# COMMAND ----------

# =========================
# PROJECT PATH CONFIG
# =========================

BASE_DIR = Path("/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project")
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"

DOCUMENTS_CSV_PATH = PROCESSED_DIR / "documents_metadata.csv"

RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("RAW_DOCS_DIR =", RAW_DOCS_DIR)
print("PROCESSED_DIR =", PROCESSED_DIR)
print("DOCUMENTS_CSV_PATH =", DOCUMENTS_CSV_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project paths
# MAGIC
# MAGIC This section sets the folder paths used in the ingestion process.
# MAGIC
# MAGIC ### Main folders
# MAGIC - `raw_pdfs/` → contains the original source PDF files
# MAGIC - `processed/` → stores extracted document-level metadata
# MAGIC
# MAGIC ### Main output file
# MAGIC - `documents_metadata.csv` → structured record of all ingested documents

# COMMAND ----------

# =========================
# VALIDATE SOURCE DOCUMENTS
# =========================

pdf_files = sorted(list(RAW_DOCS_DIR.glob("*.pdf")))

print("Number of PDF files found:", len(pdf_files))
for pdf_file in pdf_files:
    print("-", pdf_file.name)

if len(pdf_files) == 0:
    raise FileNotFoundError(
        f"No PDF files found in {RAW_DOCS_DIR}. "
        "Please place your curated PDF documents in the raw_pdfs folder before running ingestion."
    )

# COMMAND ----------

# =========================
# HELPER FUNCTIONS
# =========================

def normalize_doc_id(file_name: str) -> str:
    """
    Convert a file name into a clean document ID.
    Example:
    'The Dark Knight.pdf' -> 'the_dark_knight'
    """
    name = file_name.replace(".pdf", "").strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from all pages of a PDF file using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        page_text = page.get_text()
        if page_text:
            pages_text.append(page_text)

    return "\n".join(pages_text).strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define helper functions
# MAGIC
# MAGIC Two helper functions are used in this notebook.
# MAGIC
# MAGIC ### `normalize_doc_id`
# MAGIC This converts a file name into a standardized document identifier.
# MAGIC
# MAGIC ### `extract_text_from_pdf`
# MAGIC This reads the entire PDF file page by page and returns the full extracted text as a single string.
# MAGIC
# MAGIC ### Why this matters
# MAGIC These functions make the ingestion process:
# MAGIC - cleaner
# MAGIC - reusable
# MAGIC - easier to debug

# COMMAND ----------

# =========================
# INGEST DOCUMENTS
# =========================

documents = []

for pdf_path in pdf_files:
    print(f"Reading: {pdf_path.name}")

    raw_text = extract_text_from_pdf(pdf_path)
    doc_id = normalize_doc_id(pdf_path.name)

    document_record = {
        "doc_id": doc_id,
        "file_name": pdf_path.name,
        "source_path": str(pdf_path),
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "domain_tag": "film_narrative_analysis",
        "ingested_at": datetime.utcnow().isoformat()
    }

    documents.append(document_record)

print("\nTotal documents ingested:", len(documents))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ingest the PDF documents
# MAGIC
# MAGIC This is the main ingestion loop.
# MAGIC
# MAGIC ### What happens here
# MAGIC For each PDF file:
# MAGIC 1. the notebook reads the file
# MAGIC 2. extracts the full text
# MAGIC 3. creates a normalized document ID
# MAGIC 4. stores document metadata in a structured dictionary
# MAGIC
# MAGIC ### Metadata stored
# MAGIC - `doc_id`
# MAGIC - `file_name`
# MAGIC - `source_path`
# MAGIC - `raw_text`
# MAGIC - `char_count`
# MAGIC - `domain_tag`
# MAGIC - `ingested_at`
# MAGIC
# MAGIC ### Why this matters
# MAGIC This creates a clean document-level dataset that later notebooks can use for chunking and retrieval preparation.

# COMMAND ----------

# =========================
# CREATE DOCUMENT DATAFRAME
# =========================

documents_df = pd.DataFrame(documents)

print("Documents DataFrame created successfully.")
print("Shape:", documents_df.shape)

documents_df[["doc_id", "file_name", "char_count", "domain_tag"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Convert ingested records into a DataFrame
# MAGIC
# MAGIC This section converts the list of extracted document records into a Pandas DataFrame.
# MAGIC
# MAGIC ### Why use a DataFrame
# MAGIC A DataFrame makes it easy to:
# MAGIC - inspect document metadata
# MAGIC - validate extraction results
# MAGIC - save output files
# MAGIC - reuse the data in later notebooks

# COMMAND ----------

# =========================
# BASIC VALIDATION CHECKS
# =========================

empty_docs = documents_df[documents_df["raw_text"].str.strip() == ""]
duplicate_doc_ids = documents_df["doc_id"].duplicated().sum()

print("Empty extracted documents:", len(empty_docs))
print("Duplicate doc_id count:", duplicate_doc_ids)

if len(empty_docs) > 0:
    print("\nWARNING: Some documents have empty extracted text.")
    print(empty_docs[["doc_id", "file_name"]])

if duplicate_doc_ids > 0:
    raise ValueError("Duplicate doc_id values found. Please check source file naming.")

# COMMAND ----------

# =========================
# PREVIEW EXTRACTED TEXT
# =========================

for i in range(min(3, len(documents_df))):
    row = documents_df.iloc[i]
    print("=" * 80)
    print("Document:", row["file_name"])
    print("doc_id:", row["doc_id"])
    print("Text preview:")
    print(row["raw_text"][:1000])
    print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Preview extracted text
# MAGIC
# MAGIC This section prints a short preview of the extracted content.
# MAGIC
# MAGIC ### Why this matters
# MAGIC PDF text extraction can sometimes produce:
# MAGIC - broken formatting
# MAGIC - empty outputs
# MAGIC - strange spacing
# MAGIC - incomplete text
# MAGIC
# MAGIC Previewing the extracted content early helps catch such issues before chunking begins.

# COMMAND ----------

# =========================
# SAVE DOCUMENT METADATA
# =========================

documents_df.to_csv(DOCUMENTS_CSV_PATH, index=False)

print("Document metadata saved successfully.")
print("Saved to:", DOCUMENTS_CSV_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save document metadata
# MAGIC
# MAGIC This section saves the ingested document-level dataset as a CSV file.
# MAGIC
# MAGIC ### Output
# MAGIC - `documents_metadata.csv`
# MAGIC
# MAGIC ### Why this is important
# MAGIC This file becomes the input to the next notebook, where the extracted text will be chunked and converted into smaller retrieval units.

# COMMAND ----------

# =========================
# OPTIONAL: SAVE TO SPARK TABLE
# =========================

spark_documents_df = spark.createDataFrame(documents_df)
spark_documents_df.write.mode("overwrite").saveAsTable("movie_documents_metadata")

print("Spark table 'movie_documents_metadata' created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Optional Spark table creation
# MAGIC
# MAGIC This section writes the ingested document metadata into a Spark table.
# MAGIC
# MAGIC ### Table name
# MAGIC - `movie_documents_metadata`
# MAGIC
# MAGIC ### Why this is useful
# MAGIC Saving the data as a Spark table makes it easier to:
# MAGIC - query in Databricks
# MAGIC - reuse across notebooks
# MAGIC - integrate with larger pipelines later
# MAGIC
# MAGIC This is optional, but recommended for a more professional Databricks workflow.

# COMMAND ----------

# =========================
# FINAL SUMMARY TABLE
# =========================

summary_df = documents_df[["doc_id", "file_name", "char_count", "domain_tag"]].copy()
summary_df["char_count"] = summary_df["char_count"].astype(int)

summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Final ingestion summary
# MAGIC
# MAGIC This table gives a clean summary of the ingested document collection.
# MAGIC
# MAGIC It is useful for:
# MAGIC - validating the knowledge base
# MAGIC - checking document sizes
# MAGIC - presentation preparation
# MAGIC - confirming ingestion completeness