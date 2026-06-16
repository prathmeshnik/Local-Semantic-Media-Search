# Local Semantic Media Search

A high-performance, privacy-focused semantic search engine for images and videos that runs entirely on your local machine. No cloud APIs, no data leakage.

## Overview

This project uses the state-of-the-art **Qwen3-VL-Embedding-2B** Vision-Language Model (VLM) to map your media and text queries into a shared vector space. This allows you to search your collection using natural language descriptions (e.g., "sunset at the beach", "dog playing with a ball", "city skyline at night") instead of relying on filenames or manual tags.

## Core Features

- **Semantic Media Intelligence:** Powered by `Qwen/Qwen3-VL-Embedding-2B` for deep visual understanding of both images and video frames.
- **Video Support:** Automatically extracts representative frames from videos for indexing and provides an integrated player for playback.
- **Incremental Indexing:** Smart "memory" via `mtime` tracking—only re-indexes files if they have been modified since the last scan.
- **Fast Local Search:** ChromaDB provides sub-100ms vector lookups for your collection.
- **Privacy First:** All processing happens locally. Your media never leaves your device.
- **Optimized UI:** Pre-computed thumbnails and a dedicated file-serving endpoint for high-resolution viewing and smooth video streaming.

## Tech Stack

- **Model:** Qwen3-VL-Embedding-2B (Hugging Face Transformers)
- **Database:** ChromaDB (Persistent Vector Store)
- **Backend:** FastAPI & Uvicorn
- **Processing:** OpenCV (for video), Pillow (PIL), PyTorch
- **Frontend:** Pure HTML5/JS (Modern Light/Dark UI)

## Project Structure

```text
image-db/
├── vault/                         # Put your raw images and videos here
├── Qwen/Qwen3-VL-Embedding-2B/    # Auto-downloaded model weights
├── .db/                           # Persistent ChromaDB vector store
├── .cache/
│   └── thumbnails/                # Pre-computed thumbnails for search results
├── indexer.py                     # Media ingestion, frame extraction, and vectorization script
├── api.py                         # FastAPI search server and file streamer
├── embedding_utils.py             # Specialized Qwen3-VL embedding logic
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

*Note: The `Qwen/` directory will be created and populated automatically from Hugging Face on the first run.*

## Quick Start

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Index Media:** Run `python indexer.py vault` to scan your collection.
3. **Start Search Engine:** Launch `python api.py`.
4. **Search:** Open `http://127.0.0.1:8000` in your browser.

---

