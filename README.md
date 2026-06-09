# RAG Complaint Retrieval System

A Retrieval-Augmented Generation (RAG) system for searching and analyzing consumer complaint narratives using semantic search, vector databases, reranking, and large language models.

## Features

* Semantic search using BAAI/bge-small-en-v1.5 embeddings
* PostgreSQL + pgvector vector database
* Multiple chunking strategies:

  * Fixed-size chunking
  * Recursive chunking
  * Token-aware semantic chunking
* Metadata filtering (Product, State, Company)
* Cross-encoder reranking
* AI-generated complaint overviews using a local Ollama model
* Streamlit user interface

## Project Structure

* `app/` — Streamlit application
* `notebooks/` — Data preprocessing, chunking, and embedding generation
* `scripts/` — Data loading and utility scripts
* `evaluation/` — Retrieval and reranking evaluation experiments
* `sql/` — Database schema and setup files

## Technologies

* Python
* Streamlit
* PostgreSQL
* pgvector
* Sentence Transformers
* Cross Encoder Reranking
* Ollama
* Pandas
* NumPy

## Dataset

This project uses the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database.

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
