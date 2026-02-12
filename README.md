# 📚 Wikipedia FAISS Indexer

A high-performance semantic search engine built by indexing Wikipedia content using vector embeddings and FAISS.

This project transforms large-scale Wikipedia dumps into searchable vector indexes, enabling fast similarity search, semantic retrieval, and AI-assisted querying.

---

## 🚀 Overview

This system:

1. Parses Wikipedia dump files
2. Cleans and chunks article text
3. Converts text into vector embeddings
4. Stores vectors inside a FAISS index
5. Enables fast similarity-based retrieval

The result is a scalable semantic search engine over Wikipedia data.

---

## 🧠 Key Features

- 📦 Wikipedia dump parsing
- ✂️ Smart text chunking
- 🔎 Embedding generation
- ⚡ FAISS vector indexing
- 📊 Fast similarity search
- 💾 Persistent index storage
- 🔄 Reloadable indexes
- 🧩 Modular architecture

---

## 🏗️ Architecture

Wikipedia Dump → Text Cleaning → Chunking → Embeddings → FAISS Index → Query Search

### Components

- **Parser** – Extracts text from Wikipedia XML dump
- **Chunker** – Splits text into manageable segments
- **Embedder** – Converts text into numerical vectors
- **Indexer** – Builds FAISS index
- **Query Engine** – Returns top-k similar results

---

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Data Source

Download a Wikipedia dump from:

https://dumps.wikimedia.org/

Example:
- enwiki-latest-pages-articles.xml.bz2

Place the file inside the `data/` directory.

---

## ▶️ Building the Index

Run:

```bash
python indexer.py
```

This will:

- Parse Wikipedia dump
- Generate embeddings
- Create FAISS index
- Save index locally

---

## 🔎 Querying the Index

```bash
python wiki_pr_faiss_xgboost_train.py
"<Wikipedia article>"
```

Returns the most semantically similar Wikipedia chunks.

---

## 🧮 FAISS Index Type

Depending on configuration, the project may use:

- IndexFlatL2 (exact search)
- IndexIVFFlat (approximate search)
- HNSW (efficient ANN search)

Index type can be configured at line 48.
Wikipedia dump type can be configued at line 24-25

---

## 📊 Performance Considerations

- Larger chunk size → fewer vectors, less precision
- Smaller chunk size → more vectors, better semantic granularity
- IVF/HNSW recommended for large-scale indexing

---

## 💡 Use Cases

- RAG (Retrieval-Augmented Generation)
- Offline semantic search
- LLM knowledge base
- AI research experiments
- Personal knowledge engine

---

## 🧠 Roadmap

- [ ] Multi-language support
- [ ] Metadata filtering
- [ ] Web interface
- [ ] Distributed indexing
- [ ] Incremental updates
- [ ] GPU FAISS support

---

## ⚠️ Disclaimer

Wikipedia content is licensed under Creative Commons Attribution-ShareAlike.

This project is for research and educational purposes.

---

## 📜 License

GPLv3 license

---

## 🔥 Future Vision

This project can evolve into:

- A local AI search engine
- A private LLM knowledge backend
- A scalable retrieval system for custom corpora
- A foundation for semantic AI applications

---

⭐ If you find this project useful, consider starring the repository.
