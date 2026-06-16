import os
import argparse
from pathlib import Path
from PIL import Image
import torch
import chromadb
from tqdm import tqdm
import hashlib
from pillow_heif import register_heif_opener

from embedding_utils import Qwen3VLEmbedder

# Register HEIF opener for Pillow
register_heif_opener()

# Configuration
DB_PATH = "./.db"
THUMBNAIL_PATH = "./.cache/thumbnails"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tiff", ".tif"}
MODEL_ID = "./model"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class ImageIndexer:
    def __init__(self, vault_path):
        self.vault_path = Path(vault_path).resolve()
        self.device = get_device()
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
        self.collection = self.chroma_client.get_or_create_collection(
            name="images", 
            metadata={"hnsw:space": "cosine"}
        )

        # Ensure thumbnail dir exists
        os.makedirs(THUMBNAIL_PATH, exist_ok=True)

    def get_relative_path(self, full_path):
        return str(full_path.relative_to(self.vault_path))

    def generate_thumbnail(self, img, rel_path):
        # Create a unique filename based on the relative path to avoid collisions
        thumb_name = hashlib.md5(rel_path.encode()).hexdigest() + ".webp"
        thumb_file = Path(THUMBNAIL_PATH) / thumb_name
        
        if not thumb_file.exists():
            img_copy = img.copy()
            img_copy.thumbnail((300, 300))
            img_copy.save(thumb_file, "WEBP", quality=80)
        
        return str(thumb_file)

    def index(self, batch_size=8):
        print(f"Scanning {self.vault_path} for images...")
        all_files = []
        for ext in IMAGE_EXTS:
            all_files.extend(self.vault_path.rglob(f"*{ext}"))
        
        existing_ids = set(self.collection.get(include=[])["ids"])
        to_index = [f for f in all_files if self.get_relative_path(f) not in existing_ids]

        if not to_index:
            print("No new images to index.")
            return

        print(f"Indexing {len(to_index)} new images...")
        
        for i in range(0, len(to_index), batch_size):
            batch_files = to_index[i : i + batch_size]
            input_items = []
            ids = []
            metadatas = []
            
            for fpath in batch_files:
                try:
                    rel_path = self.get_relative_path(fpath)
                    img = Image.open(fpath)
                    img.load() 
                    img = img.convert("RGB")
                    
                    thumb_path = self.generate_thumbnail(img, rel_path)
                    
                    input_items.append({"image": img})
                    ids.append(rel_path)
                    metadatas.append({
                        "full_path": str(fpath),
                        "thumbnail_path": thumb_path,
                        "filename": fpath.name
                    })
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
                    continue

            if not input_items:
                continue

            # Vectorize using the specialized embedder
            embeddings = self.embedder.embed(input_items)
            embeddings_list = embeddings.cpu().numpy().tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            print(f"  Processed {min(i + batch_size, len(to_index))}/{len(to_index)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index images for semantic search.")
    parser.add_argument("path", help="Path to the image vault")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for indexing")
    args = parser.parse_args()

    indexer = ImageIndexer(args.path)
    indexer.index(batch_size=args.batch)
