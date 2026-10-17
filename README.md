# Agentic Domain-Specific RAG System for Film Analysis

A Lambton College Natural Language Processing project for narrative, thematic, and character analysis of curated film documents.

## Overview

The project combines retrieval-augmented generation with conversational memory, tool calling, and experiment tracking. It includes a Streamlit interface, a LangChain/OpenAI backend, ChromaDB document retrieval, Google search through SerpAPI, YouTube search, and MLflow tracking.

The supplied film documents cover Fight Club, Inception, Interstellar, Memento, and The Dark Knight.

## Contents

- `app/`: Streamlit application, backend, dependency list, chunk metadata, and Databricks Apps configuration
- `notebooks/`: nine exported Databricks notebooks covering setup, ingestion, chunking, ChromaDB, RAG, tools, memory, tracking, and the final application
- `config/`: project settings; replace `YOUR_DATABRICKS_USER` with your workspace user before running the notebooks
- `data/raw_pdfs/`: five supplied film reference documents
- `data/processed/`: document and chunk metadata
- `Technical _report/`: technical report and project presentation

## Application setup

The source targets Databricks and includes a local-development fallback. Use a Python environment compatible with the dependencies in `app/requirements.txt` (Python 3.11 is a starting point; dependency compatibility has not been verified).

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r app/requirements.txt`.
3. Copy `app/.env.example` to `app/.env` and supply your own OpenAI, SerpAPI, and YouTube API keys.
4. Start the application: `streamlit run app/app.py`.

The backend initializes its local vector store and may request embeddings on startup. OpenAI and other external services require internet access and may incur usage charges. Databricks Apps deployment uses `app/app.yaml` and needs an MLflow experiment resource named `experiment`.

For the notebook workflow, import the exported Python notebooks into Databricks and run them in numeric order after configuring workspace paths and credentials.

## Project credit

This repository is prepared from Ulviyya Rahimova's local college project folder. The supplied source includes shared project materials. Preserve the original contributors' credit; this repository does not claim sole authorship of collaborative work. Third-party film documents remain the property of their respective owners.

## Verification and exclusions

All supplied Python source files passed syntax parsing during upload preparation. The application, dependency installation, Databricks deployment, and live AI responses have not been tested as part of this upload.

API-key files, virtual environments, saved conversations, and generated vector databases are excluded from version control. Personal Databricks workspace paths were replaced with placeholders. The original local project folder is unchanged.
