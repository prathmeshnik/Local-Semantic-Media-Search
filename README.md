# Local Semantic Image Search

A high-performance, privacy-focused semantic image search engine that runs entirely on your local machine. No cloud APIs, no data leakage.

## Overview

This project uses the state-of-the-art **Qwen3-VL-Embedding-2B** Vision-Language Model (VLM) to map your images and text queries into a shared vector space. This allows you to search your photo collection using natural language descriptions (e.g., "sunset at the beach", "people wearing red", "close up of a cat") instead of relying on filenames or manual tags.

## Core Features

- **Semantic Intelligence:** Powered by `Qwen/Qwen3-VL-Embedding-2B` for deep visual understanding.
- **Fast Local Search:** ChromaDB provides sub-100ms vector lookups for your local collection.
- **Privacy First:** All processing happens locally. Your images never leave your device.
- **Optimized for Performance:** Pre-computed thumbnails ensure lightning-fast UI responses.
- **RESTful API:** Clean FastAPI backend for easy integration with Obsidian, custom UIs, or other scripts.

## Tech Stack

- **Model:** Qwen3-VL-Embedding-2B (Hugging Face Transformers)
- **Database:** ChromaDB (Persistent Vector Store)
- **Backend:** FastAPI & Uvicorn
- **Processing:** Pillow (PIL), PyTorch
- **Hardware:** CPU/GPU (CUDA/MPS) support

## Project Structure

```text
image-db/
├── vault/             # Put your raw images here
├── model/             # Place your Qwen3-VL-Embedding-2B model files here
├── image/             # Python virtual environment
├── .db/               # Persistent ChromaDB vector store
├── .cache/
│   └── thumbnails/    # Pre-computed image thumbnails
├── indexer.py         # Image ingestion and vectorization script
├── api.py             # FastAPI search server
├── embedding_utils.py # Specialized Qwen3-VL embedding logic
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

## Quick Start

1. **Install Dependencies:** `pip install fastapi uvicorn chromadb transformers accelerate Pillow torch torchvision`
2. **Index Images:** Run the `indexer.py` script pointing to your vault.
3. **Start Search:** Launch `api.py` and query via REST.

---
*For a deep dive into the architecture and implementation details, see [GEMINI.md](./GEMINI.md).*
