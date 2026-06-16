from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import chromadb
import uvicorn
import os
import mimetypes
from huggingface_hub import snapshot_download

from embedding_utils import Qwen3VLEmbedder

# Configuration
DB_PATH = "./.db"
THUMBNAIL_PATH = "./.cache/thumbnails"
MODEL_ID = "./Qwen/Qwen3-VL-Embedding-2B"
HF_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DISTANCE_THRESHOLD = 0.7  # Cosine distance: 0 is identical, 1 is opposite. Lower is better.

def ensure_model():
    """Checks if the local model directory exists and contains files. If not, downloads from HF."""
    model_path = Path(MODEL_ID)
    if not model_path.exists() or not any(model_path.iterdir()):
        print(f"Model not found in {MODEL_ID}. Downloading {HF_MODEL_ID} from Hugging Face...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=MODEL_ID,
            local_dir_use_symlinks=False
        )

app = FastAPI(title="Local Semantic Image Search API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local use, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Global State
class SearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        
        print(f"Using device: {self.device}")
        
        # Initialize Embedder
        print(f"Loading model {MODEL_ID}...")
        self.embedder = Qwen3VLEmbedder(
            MODEL_ID, 
            device=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )

        # Initialize DB
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_collection("images")

    def search(self, query_text: str, top_k: int = 12):
        # Embed the text query
        query_vector = self.embedder.embed([{"text": query_text}]).cpu().numpy().tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "distances"]
        )


        filtered_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # In cosine space for Chroma, distance is 1 - similarity. 
            # If distance > threshold, it's not a good match.
            if distance > DISTANCE_THRESHOLD:
                continue
                
            filtered_results.append({
                "id": results["ids"][0][i],
                "score": round(1 - distance, 4), # Similarity score
                "metadata": results["metadatas"][0][i]
            })
            
        return filtered_results

# Initialize Search Engine (Singleton-ish for FastAPI)
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    ensure_model()
    engine = SearchEngine()

# Mount Static Files for Thumbnails
if not os.path.exists(THUMBNAIL_PATH):
    os.makedirs(THUMBNAIL_PATH, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_PATH), name="thumbnails")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/search")
async def search(q: str = Query(..., description="The semantic search query"), 
                 limit: int = Query(12, description="Number of results to return")):
    if engine is None:
        return {"error": "Engine not initialized"}
    
    results = engine.search(q, top_k=limit)
    return {"results": results}

@app.get("/file")
async def get_file(path: str = Query(..., description="Full path to the image or video")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    mime_type, _ = mimetypes.guess_type(path)
    return FileResponse(path, media_type=mime_type)

@app.get("/health")
async def health():
    return {"status": "ok", "device": engine.device if engine else "not_ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
