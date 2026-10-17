# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Project Setup
# MAGIC
# MAGIC ## Project Title
# MAGIC **Domain-Specific RAG for Narrative, Thematic, and Character Analysis of Curated Film Documents**
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook initializes the full project environment for a production-style domain-specific RAG system in Databricks.
# MAGIC
# MAGIC The goal of this notebook is to prepare the workspace, install required libraries, define the project architecture, configure paths and environment variables, and create the base folder structure for the rest of the pipeline.
# MAGIC
# MAGIC This project will use:
# MAGIC
# MAGIC - **Databricks** as the development platform  
# MAGIC - **LangChain** as the orchestration framework  
# MAGIC - **ChromaDB** as the vector database  
# MAGIC - **OpenAI embeddings and chat models** for retrieval and answer generation  
# MAGIC - **MLflow** for tracking, observability, and tracing  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC A professional RAG system should not begin with scattered code. It should begin with a controlled project setup that ensures:
# MAGIC
# MAGIC - all dependencies are installed
# MAGIC - project paths are standardized
# MAGIC - environment settings are clear
# MAGIC - architecture decisions are fixed
# MAGIC - the rest of the notebooks can build on a stable foundation
# MAGIC
# MAGIC This notebook creates that foundation.

# COMMAND ----------

# MAGIC %pip install -U \
# MAGIC langchain \
# MAGIC langchain-openai \
# MAGIC langchain-community \
# MAGIC langchain-chroma \
# MAGIC chromadb \
# MAGIC openai \
# MAGIC pymupdf \
# MAGIC pandas \
# MAGIC numpy \
# MAGIC mlflow \
# MAGIC tiktoken \
# MAGIC pydantic \
# MAGIC tqdm

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This project requires a set of libraries across multiple layers of the RAG pipeline.
# MAGIC
# MAGIC ### Core libraries
# MAGIC - `langchain` → orchestration framework
# MAGIC - `langchain-openai` → OpenAI integration for LangChain
# MAGIC - `langchain-community` → community integrations
# MAGIC - `langchain-chroma` → Chroma vector database integration
# MAGIC - `chromadb` → vector database
# MAGIC - `openai` → model access
# MAGIC
# MAGIC ### Data processing libraries
# MAGIC - `pymupdf` → PDF text extraction
# MAGIC - `pandas` → tabular data handling
# MAGIC - `numpy` → numerical operations
# MAGIC
# MAGIC ### Tracking and utilities
# MAGIC - `mlflow` → experiment tracking and tracing
# MAGIC - `tiktoken` → token-aware operations
# MAGIC - `pydantic` → validation
# MAGIC - `tqdm` → progress tracking
# MAGIC
# MAGIC ### Why this matters
# MAGIC This notebook sets up the full technical stack for:
# MAGIC - ingestion
# MAGIC - chunking
# MAGIC - embeddings
# MAGIC - vector storage
# MAGIC - orchestration
# MAGIC - tracking

# COMMAND ----------

import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import core Python libraries
# MAGIC
# MAGIC This section imports the basic Python libraries required for project configuration and file handling.
# MAGIC
# MAGIC ### Main roles
# MAGIC - `os` and `Path` → folder and file management
# MAGIC - `json` → configuration serialization
# MAGIC - `datetime` → timestamp generation
# MAGIC - `pandas` and `numpy` → data handling
# MAGIC
# MAGIC At this stage, the purpose is not yet to build the RAG pipeline itself, but to prepare a clean project workspace.

# COMMAND ----------

# =========================
# PROJECT IDENTITY
# =========================

PROJECT_NAME = "domain_specific_rag_movie_analysis"
PROJECT_TITLE = "Domain-Specific RAG for Narrative, Thematic, and Character Analysis of Curated Film Documents"
DOMAIN_NAME = "Narrative, thematic, and character analysis of curated film documents"

KNOWLEDGE_BASE_DESCRIPTION = """
A curated film document collection focused on narrative analysis, thematic interpretation,
character motivation, symbolic meaning, and story structure.
"""

print("PROJECT_NAME =", PROJECT_NAME)
print("PROJECT_TITLE =", PROJECT_TITLE)
print("DOMAIN_NAME =", DOMAIN_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project identity
# MAGIC
# MAGIC This section defines the official identity of the project.
# MAGIC
# MAGIC ### Why this matters
# MAGIC A professional project should have a clearly defined:
# MAGIC - project name
# MAGIC - domain
# MAGIC - purpose
# MAGIC - knowledge base description
# MAGIC
# MAGIC This becomes useful later for:
# MAGIC - reports
# MAGIC - presentations
# MAGIC - configuration files
# MAGIC - MLflow experiment naming

# COMMAND ----------

# =========================
# MODEL CONFIGURATION
# =========================

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
TEMPERATURE = 0

print("Embedding model =", EMBED_MODEL)
print("Chat model =", CHAT_MODEL)
print("Chunk size =", CHUNK_SIZE)
print("Chunk overlap =", CHUNK_OVERLAP)
print("Top-k =", TOP_K)
print("Temperature =", TEMPERATURE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define model and retrieval settings
# MAGIC
# MAGIC This section stores the core configuration for the RAG pipeline.
# MAGIC
# MAGIC ### Main settings
# MAGIC - `EMBED_MODEL` → model used to convert chunks and queries into embeddings
# MAGIC - `CHAT_MODEL` → model used to generate final answers
# MAGIC - `CHUNK_SIZE` → maximum size of text chunks
# MAGIC - `CHUNK_OVERLAP` → overlap between consecutive chunks
# MAGIC - `TOP_K` → number of retrieved chunks for each query
# MAGIC - `TEMPERATURE` → chat generation randomness control
# MAGIC
# MAGIC ### Why this matters
# MAGIC These values are important because they affect:
# MAGIC - retrieval quality
# MAGIC - answer grounding
# MAGIC - chunk granularity
# MAGIC - reproducibility
# MAGIC
# MAGIC They will also be tracked later in MLflow experiments.

# COMMAND ----------

# =========================
# ENVIRONMENT / SECRETS
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")
    print("Recommended: configure it using Databricks secrets or environment variables.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("OPENAI_API_KEY is available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configure API access
# MAGIC
# MAGIC This project uses OpenAI models for:
# MAGIC - embeddings
# MAGIC - final answer generation
# MAGIC
# MAGIC ### Important implementation note
# MAGIC In a production-style Databricks project, the API key should **not** be hardcoded in the notebook.
# MAGIC
# MAGIC It should be managed using:
# MAGIC - Databricks secrets
# MAGIC - environment variables
# MAGIC - secure runtime configuration
# MAGIC
# MAGIC ### Why this matters
# MAGIC Secure credential handling is an important part of making the project professional and production-aware.

# COMMAND ----------

# =========================
# PROJECT ROOT PATHS
# =========================

BASE_DIR = Path("/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project")
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
EXPORTS_DIR = BASE_DIR / "exports"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

ALL_DIRS = [
    BASE_DIR,
    DATA_DIR,
    RAW_DOCS_DIR,
    PROCESSED_DIR,
    CHROMA_DIR,
    EXPORTS_DIR,
    CONFIG_DIR,
    LOGS_DIR,
    REPORTS_DIR
]

for d in ALL_DIRS:
    d.mkdir(parents=True, exist_ok=True)

print("Project folders created successfully.")
for d in ALL_DIRS:
    print("-", d)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create the project folder structure
# MAGIC
# MAGIC This section creates a clean workspace layout for the project.
# MAGIC
# MAGIC ### Folder roles
# MAGIC
# MAGIC - `raw_pdfs/` → original source documents
# MAGIC - `processed/` → extracted text, chunk metadata, and processed tables
# MAGIC - `chroma_db/` → Chroma vector database files
# MAGIC - `exports/` → exported outputs if needed
# MAGIC - `config/` → saved configuration files
# MAGIC - `logs/` → runtime and pipeline logs
# MAGIC - `reports/` → final report artifacts
# MAGIC
# MAGIC ### Why this matters
# MAGIC A professional project should separate:
# MAGIC - raw inputs
# MAGIC - processed data
# MAGIC - vector storage
# MAGIC - configuration
# MAGIC - outputs
# MAGIC
# MAGIC This makes the project easier to manage and present.

# COMMAND ----------

# =========================
# DATA FILE PATHS
# =========================

DOCUMENTS_CSV_PATH = PROCESSED_DIR / "documents_metadata.csv"
CHUNKS_CSV_PATH = PROCESSED_DIR / "chunks_metadata.csv"
PROJECT_CONFIG_PATH = CONFIG_DIR / "project_config.json"

print("DOCUMENTS_CSV_PATH =", DOCUMENTS_CSV_PATH)
print("CHUNKS_CSV_PATH =", CHUNKS_CSV_PATH)
print("PROJECT_CONFIG_PATH =", PROJECT_CONFIG_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Define core data artifact paths
# MAGIC
# MAGIC This section defines the main file paths that later notebooks will use.
# MAGIC
# MAGIC ### Planned artifacts
# MAGIC - `documents_metadata.csv` → document-level information
# MAGIC - `chunks_metadata.csv` → chunk-level metadata
# MAGIC - `project_config.json` → project configuration snapshot
# MAGIC
# MAGIC ### Why this matters
# MAGIC Defining file paths early avoids confusion and reduces hardcoded values later in the project.

# COMMAND ----------

# =========================
# SAVE PROJECT CONFIG
# =========================

project_config = {
    "project_name": PROJECT_NAME,
    "project_title": PROJECT_TITLE,
    "domain_name": DOMAIN_NAME,
    "knowledge_base_description": KNOWLEDGE_BASE_DESCRIPTION.strip(),
    "embed_model": EMBED_MODEL,
    "chat_model": CHAT_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
    "temperature": TEMPERATURE,
    "base_dir": str(BASE_DIR),
    "data_dir": str(DATA_DIR),
    "raw_docs_dir": str(RAW_DOCS_DIR),
    "processed_dir": str(PROCESSED_DIR),
    "chroma_dir": str(CHROMA_DIR),
    "exports_dir": str(EXPORTS_DIR),
    "config_dir": str(CONFIG_DIR),
    "logs_dir": str(LOGS_DIR),
    "reports_dir": str(REPORTS_DIR),
    "created_at": datetime.utcnow().isoformat()
}

with open(PROJECT_CONFIG_PATH, "w") as f:
    json.dump(project_config, f, indent=2)

print("Project configuration saved to:", PROJECT_CONFIG_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save project configuration
# MAGIC
# MAGIC This section writes the current setup into a JSON configuration file.
# MAGIC
# MAGIC ### Why this is useful
# MAGIC A saved config file helps with:
# MAGIC - reproducibility
# MAGIC - report writing
# MAGIC - debugging
# MAGIC - MLflow parameter logging
# MAGIC - consistent notebook-to-notebook configuration
# MAGIC
# MAGIC This is a small but important professional practice.

# COMMAND ----------

# =========================
# OPTIONAL: DISPLAY CONFIG
# =========================

config_df = pd.DataFrame(
    [{"parameter": k, "value": str(v)} for k, v in project_config.items()]
)

config_df

# COMMAND ----------

# =========================
# PROJECT ARCHITECTURE SUMMARY
# =========================

architecture_steps = [
    "1. Setup and configuration",
    "2. Document ingestion and PDF text extraction",
    "3. Chunking and metadata creation",
    "4. ChromaDB vector database construction",
    "5. LangChain-based retrieval and answering pipeline",
    "6. MLflow tracking and tracing",
    "7. Memory and tool-calling integration",
    "8. Final Databricks demo notebook",
    "9. Reporting and submission"
]

arch_df = pd.DataFrame({"Pipeline Stage": architecture_steps})
arch_df

# COMMAND ----------

# =========================
# VALIDATE RAW DOCUMENT FOLDER
# =========================

pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))

print("Number of PDF files currently available:", len(pdf_files))
for f in pdf_files:
    print("-", f.name)

if len(pdf_files) == 0:
    print("\nNOTE: No PDF files found yet in raw_pdfs/.")
    print("Next step: upload or place the curated film PDFs into the raw_pdfs folder before running ingestion.")

# COMMAND ----------

# =========================
# OPTIONAL: CREATE EMPTY SCHEMAS
# =========================

documents_columns = [
    "doc_id",
    "file_name",
    "source_path",
    "raw_text",
    "char_count",
    "domain_tag",
    "ingested_at"
]

chunks_columns = [
    "chunk_id",
    "doc_id",
    "file_name",
    "chunk_index",
    "chunk_text",
    "chunk_char_count",
    "chunk_size",
    "chunk_overlap",
    "created_at"
]

documents_schema_df = pd.DataFrame(columns=documents_columns)
chunks_schema_df = pd.DataFrame(columns=chunks_columns)

print("Document schema columns:")
print(documents_schema_df.columns.tolist())

print("\nChunk schema columns:")
print(chunks_schema_df.columns.tolist())