import os
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR = "D:/chatbot/models"
DOC_PATH = "data/docs.pkl"
INDEX_PATH = "data/index.faiss"
CHUNK_PATH = "data/chunks.pkl"

if not os.path.exists(DOC_PATH):
    raise FileNotFoundError(f"{DOC_PATH} not found. Run scrape.py first.")

print("🔧 Loading embedding model...")
embed_model = SentenceTransformer(os.path.join(MODEL_DIR, "all-MiniLM-L6-v2"))

print("📄 Loading scraped documents...")
with open(DOC_PATH, "rb") as f:
    docs = pickle.load(f)


def split_into_chunks(text):
    lines = text.split("\n")
    chunks, current = [], ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(#{1,6}\s*|[A-Z][\w\s]{4,40}[:\-])", line) or line.istitle():
            if len(current.strip()) >= 100:
                chunks.append(current.strip())
            current = line
        else:
            current += " " + line
    if len(current.strip()) >= 100:
        chunks.append(current.strip())
    return chunks

print("🧱 Splitting documents into chunks...")
all_chunks = []
for doc in docs:
    all_chunks.extend(split_into_chunks(doc))

print(f"✅ Total chunks created: {len(all_chunks)}")

print("📐 Computing embeddings...")
embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

print("💾 Saving index and chunks...")
os.makedirs("data", exist_ok=True)
faiss.write_index(index, INDEX_PATH)
with open(CHUNK_PATH, "wb") as f:
    pickle.dump(all_chunks, f)
print("✅ Index building complete.")
