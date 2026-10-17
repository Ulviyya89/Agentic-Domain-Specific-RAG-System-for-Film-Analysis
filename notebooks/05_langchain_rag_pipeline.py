# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - LangChain RAG Pipeline
# MAGIC
# MAGIC ## Objective
# MAGIC This notebook builds the full **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** on top of the ChromaDB vector store.
# MAGIC
# MAGIC At this stage, the project already has:
# MAGIC - source PDF documents
# MAGIC - extracted document text
# MAGIC - chunk-level metadata
# MAGIC - a ChromaDB-based vector database
# MAGIC
# MAGIC The next step is to build the final question-answering pipeline.
# MAGIC
# MAGIC This notebook will:
# MAGIC - load the existing ChromaDB vector store
# MAGIC - rebuild the ChromaDB collection automatically if it is empty
# MAGIC - create a retriever
# MAGIC - define a grounded prompt template
# MAGIC - initialize the chat model
# MAGIC - build a LangChain RAG chain
# MAGIC - answer questions using retrieved context
# MAGIC - return source chunk information for traceability
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why this notebook matters
# MAGIC This is the notebook where the project becomes a real working chatbot.
# MAGIC
# MAGIC The system now:
# MAGIC 1. receives a user question
# MAGIC 2. retrieves relevant chunks from the vector database
# MAGIC 3. injects those chunks into a prompt
# MAGIC 4. generates a grounded answer
# MAGIC 5. preserves source traceability through chunk metadata
# MAGIC
# MAGIC This is the main orchestration stage of the entire project.

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-openai langchain-community langchain-chroma chromadb openai pandas python-dotenv

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries
# MAGIC
# MAGIC This notebook uses:
# MAGIC - `langchain` for orchestration
# MAGIC - `langchain-openai` for OpenAI chat and embeddings
# MAGIC - `langchain-chroma` for Chroma vector database integration
# MAGIC - `chromadb` for vector storage and retrieval
# MAGIC - `openai` for model access
# MAGIC - `pandas` for result formatting
# MAGIC - `python-dotenv` for loading environment variables from the `.env` file
# MAGIC
# MAGIC If Databricks requests a Python restart after `%pip install`, rerun the notebook from the top.

# COMMAND ----------

import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import required libraries
# MAGIC
# MAGIC This section imports the core LangChain components used to build the final RAG pipeline.
# MAGIC
# MAGIC ### Main components
# MAGIC - `Chroma` → loads the persisted vector database
# MAGIC - `OpenAIEmbeddings` → query embedding model
# MAGIC - `ChatOpenAI` → chat model for final answer generation
# MAGIC - `ChatPromptTemplate` → structured prompt creation
# MAGIC - `RunnableLambda`, `RunnablePassthrough` → chain composition
# MAGIC - `StrOutputParser` → converts model output into plain text

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

# Same temporary Chroma path used in notebook 04
CHROMA_PERSIST_DIR = "/tmp/chroma_db/movie_analysis"

print("PROJECT_NAME =", PROJECT_NAME)
print("DOMAIN_NAME =", DOMAIN_NAME)
print("EMBED_MODEL =", EMBED_MODEL)
print("CHAT_MODEL =", CHAT_MODEL)
print("COLLECTION_NAME =", COLLECTION_NAME)
print("TOP_K =", TOP_K)
print("TEMPERATURE =", TEMPERATURE)
print("CHUNKS_CSV_PATH =", CHUNKS_CSV_PATH)
print("CHROMA_PERSIST_DIR =", CHROMA_PERSIST_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define project configuration
# MAGIC
# MAGIC This section stores the key settings used by the RAG chain.
# MAGIC
# MAGIC ### Main settings
# MAGIC - embedding model
# MAGIC - chat model
# MAGIC - Chroma collection name
# MAGIC - retrieval depth (`TOP_K`)
# MAGIC - generation temperature
# MAGIC - vector database persistence location
# MAGIC
# MAGIC ### Why this matters
# MAGIC Centralized configuration makes the notebook easier to manage, debug, and present.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure API Access Using `.env`
# MAGIC
# MAGIC This project uses OpenAI for:
# MAGIC - query embedding
# MAGIC - final answer generation
# MAGIC
# MAGIC The API key is stored in a `.env` file located at:
# MAGIC
# MAGIC `/Workspace/Users/YOUR_DATABRICKS_USER/Agentic_RAG_Professional_Project/.env`
# MAGIC

# COMMAND ----------


# =========================
# API KEY CONFIGURATION (.env)
# =========================

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
# MAGIC ## 5. Initialize embeddings and chat model
# MAGIC
# MAGIC This section initializes the two model components used in the RAG pipeline.
# MAGIC
# MAGIC ### Embeddings
# MAGIC The embedding model is used for:
# MAGIC - embedding queries
# MAGIC - retrieving relevant chunks from ChromaDB
# MAGIC
# MAGIC ### Chat model
# MAGIC The chat model is used for:
# MAGIC - generating the final grounded answer
# MAGIC
# MAGIC ### Why low temperature is used
# MAGIC A low temperature helps keep the answer:
# MAGIC - deterministic
# MAGIC - consistent
# MAGIC - grounded in retrieved evidence

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load or rebuild the Chroma vector store
# MAGIC
# MAGIC This notebook should not assume that the persisted Chroma collection is always available.
# MAGIC
# MAGIC In Databricks, temporary storage may be reset between sessions.  
# MAGIC To make the pipeline more robust, this section:
# MAGIC
# MAGIC - tries to load the existing Chroma collection
# MAGIC - checks whether it contains records
# MAGIC - rebuilds it from `chunks_metadata.csv` if the collection is empty
# MAGIC
# MAGIC This makes the notebook more reliable and easier to rerun without manual recovery steps.

# COMMAND ----------

# =========================
# LOAD CHROMA VECTOR STORE OR REBUILD IF EMPTY
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


# Step 1: try loading existing Chroma collection
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

collection_count = vector_store._collection.count()
print("Initial Chroma collection count:", collection_count)

# Step 2: rebuild if empty
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

# Final validation
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
# MAGIC ## 7. Prepare the Chroma vector store for retrieval
# MAGIC
# MAGIC This section ensures the vector store is available before retrieval begins.
# MAGIC
# MAGIC ### Why rebuild logic is included
# MAGIC The Chroma persistence path currently uses temporary runtime storage.
# MAGIC
# MAGIC That means:
# MAGIC - the collection may disappear after session resets
# MAGIC - the notebook should be able to recover automatically
# MAGIC
# MAGIC ### Professional benefit
# MAGIC This makes the pipeline more robust and reduces manual recovery work during development or demo preparation.

# COMMAND ----------

# =========================
# CREATE RETRIEVER
# =========================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Retriever created successfully.")
print("Retriever top_k =", TOP_K)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Create the retriever
# MAGIC
# MAGIC The retriever is the component responsible for:
# MAGIC - embedding the user query
# MAGIC - searching the vector database
# MAGIC - returning the most relevant chunk documents
# MAGIC
# MAGIC This is the retrieval step of Retrieval-Augmented Generation.

# COMMAND ----------

# =========================
# TEST RETRIEVER
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
    print(doc.page_content[:400])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Validate retriever output
# MAGIC
# MAGIC Before building the full RAG chain, the retriever is tested directly.
# MAGIC
# MAGIC ### What is checked
# MAGIC - relevant chunks are returned
# MAGIC - metadata remains attached
# MAGIC - retrieval behaves correctly for a domain-specific query

# COMMAND ----------

# =========================
# DEFINE CONTEXT FORMATTER
# =========================

def format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved documents into a formatted context block.
    """
    blocks = []

    for doc in docs:
        block = (
            f"[Source: {doc.metadata.get('doc_id')} | {doc.metadata.get('chunk_id')}]\n"
            f"{doc.page_content}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Define the context formatter
# MAGIC
# MAGIC The retriever returns a list of `Document` objects.  
# MAGIC This helper function converts those documents into one structured context block.
# MAGIC
# MAGIC ### Why this matters
# MAGIC The language model does not directly read Python objects.  
# MAGIC It needs a well-formatted text prompt containing:
# MAGIC - retrieved content
# MAGIC - visible source labels

# COMMAND ----------

# =========================
# DEFINE PROMPT TEMPLATE
# =========================

prompt = ChatPromptTemplate.from_template("""
You are a professional narrative-analysis assistant.

Use only the retrieved context to answer the user's question.
If the answer is not clearly supported by the retrieved context, say that the documents do not provide enough evidence.
Be concise, accurate, analytical, and grounded.
Always cite the source chunk IDs you relied on.

Question:
{question}

Retrieved Context:
{context}
""")

print("Prompt template created successfully.")

# COMMAND ----------

# =========================
# BUILD LANGCHAIN RAG CHAIN
# =========================

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("LangChain RAG chain created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Build the LangChain RAG chain
# MAGIC
# MAGIC This is the orchestration layer of the project.
# MAGIC
# MAGIC ### Chain flow
# MAGIC - input question
# MAGIC - retrieve relevant chunks
# MAGIC - format chunks into context
# MAGIC - combine question and context in a structured prompt
# MAGIC - generate final answer with the chat model
# MAGIC
# MAGIC This is the core LangChain-based RAG workflow.

# COMMAND ----------

# =========================
# DEFINE ANSWER FUNCTION
# =========================

def answer_with_rag(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Run the full RAG pipeline and return answer + source metadata.
    """
    runtime_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    runtime_chain = (
        {
            "context": runtime_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieved_docs = runtime_retriever.invoke(question)
    answer = runtime_chain.invoke(question)

    sources = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        sources.append({
            "rank": rank,
            "doc_id": doc.metadata.get("doc_id"),
            "file_name": doc.metadata.get("file_name"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "chunk_char_count": doc.metadata.get("chunk_char_count"),
            "chunk_token_count": doc.metadata.get("chunk_token_count"),
            "preview": doc.page_content[:250]
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Define a reusable answer function
# MAGIC
# MAGIC This helper function makes the RAG pipeline easier to use for:
# MAGIC - testing
# MAGIC - demos
# MAGIC - evaluation
# MAGIC - later MLflow logging
# MAGIC
# MAGIC ### Function output
# MAGIC It returns:
# MAGIC - original question
# MAGIC - final answer
# MAGIC - retrieved source metadata

# COMMAND ----------

# =========================
# TEST QUERY 1
# =========================

query_1 = "How does Interstellar connect love and survival?"
result_1 = answer_with_rag(query_1, top_k=5)

print("QUESTION:")
print(result_1["question"])

print("\nANSWER:")
print(result_1["answer"])

print("\nSOURCES:")
for s in result_1["sources"]:
    print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']} | chunk_index={s['chunk_index']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Test the full RAG pipeline
# MAGIC
# MAGIC This cell validates the end-to-end RAG workflow.
# MAGIC
# MAGIC ### What is being tested
# MAGIC - query handling
# MAGIC - retrieval
# MAGIC - context formatting
# MAGIC - grounded answer generation
# MAGIC - source traceability

# COMMAND ----------

# =========================
# TEST QUERY 2
# =========================

query_2 = "Compare how memory is represented in Memento and Interstellar."
result_2 = answer_with_rag(query_2, top_k=5)

print("QUESTION:")
print(result_2["question"])

print("\nANSWER:")
print(result_2["answer"])

print("\nSOURCES:")
for s in result_2["sources"]:
    print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']} | chunk_index={s['chunk_index']}")

# COMMAND ----------

# =========================
# TEST QUERY 3
# =========================

query_3 = "What makes the Joker a destabilizing force in The Dark Knight?"
result_3 = answer_with_rag(query_3, top_k=5)

print("QUESTION:")
print(result_3["question"])

print("\nANSWER:")
print(result_3["answer"])

print("\nSOURCES:")
for s in result_3["sources"]:
    print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']} | chunk_index={s['chunk_index']}")

# COMMAND ----------

# =========================
# BATCH VALIDATION
# =========================

test_questions = [
    "How does Interstellar connect love and survival?",
    "What role does memory play in Memento?",
    "How is identity represented in Fight Club?",
    "How does Inception blur the boundary between dream and reality?",
    "What makes the Joker disruptive in The Dark Knight?"
]

batch_results = []

for q in test_questions:
    res = answer_with_rag(q, top_k=5)
    batch_results.append({
        "question": q,
        "answer_preview": res["answer"][:250],
        "source_chunk_ids": [s["chunk_id"] for s in res["sources"]]
    })

batch_results_df = pd.DataFrame(batch_results)
batch_results_df

# COMMAND ----------

# =========================
# EDGE-CASE TEST
# =========================

edge_query = "What does the knowledge base say about climate regulation in Europe?"
edge_result = answer_with_rag(edge_query, top_k=5)

print("QUESTION:")
print(edge_result["question"])

print("\nANSWER:")
print(edge_result["answer"])

print("\nSOURCES:")
for s in edge_result["sources"]:
    print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']}")

# COMMAND ----------

# =========================
# OPTIONAL SIMPLE CHAT HELPER
# =========================

def ask_question():
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return

    result = answer_with_rag(question, top_k=TOP_K)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"- Rank {s['rank']} | {s['doc_id']} | {s['chunk_id']} | {s['file_name']}")