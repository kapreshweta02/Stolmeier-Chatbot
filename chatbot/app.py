import os
import sys
import pickle
import torch
import faiss
import numpy as np
import re
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from flask_cors import CORS

if sys.version_info >= (3, 13):
    print("⚠️ Python 3.13 detected. Multiprocessing is limited.")
torch.set_num_threads(1)

MODEL_DIR = "D:/chatbot/models"
DOC_PATH = "data/chunks.pkl"
INDEX_PATH = "data/index.faiss"

if not os.path.exists(DOC_PATH) or not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("Model files missing. Run build_index.py first.")

print("🧠 Loading models...")
embed_model = SentenceTransformer(os.path.join(MODEL_DIR, "all-MiniLM-L6-v2"))
answer_generator = pipeline(
    "text2text-generation",
    model=os.path.join(MODEL_DIR, "sshleifer-distilbart-cnn-12-6"),
    device=-1,
    truncation=True
)

with open(DOC_PATH, "rb") as f:
    docs = pickle.load(f)
index = faiss.read_index(INDEX_PATH)


def clean_text(text):
    return text.strip().replace("Continue reading", "").replace("By doodle", "").replace("Free Consultation", "")


def retrieve_context(query, top_k=10, min_len=40):
    vec = embed_model.encode([query])
    D, I = index.search(np.array(vec), top_k)
    seen, contexts = set(), []
    for idx in I[0]:
        if idx < len(docs):
            text = clean_text(docs[idx])
            if len(text) >= min_len and text not in seen:
                contexts.append(text)
                seen.add(text)
    return contexts


def is_contact_query(q):
    q = q.lower()
    return any(k in q for k in ["contact", "phone", "email", "address", "location", "reach", "get in touch"])


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("query", "").strip()
    if not question:
        return jsonify({"answer": "❌ Please ask a question."})

    contexts = retrieve_context(question)
    if not contexts:
        return jsonify({"answer": "❌ Sorry, no relevant information found."})

    if is_contact_query(question):
        combined = " ".join(contexts)
        phones = re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", combined)
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", combined)
        addresses = re.findall(r"\d{3,5}\s[\w\s.,#-]*San Antonio,\s*TX\s*\d{5}", combined, re.IGNORECASE)
        reply = []
        if phones: reply.append(f"📞 {phones[0]}")
        if emails: reply.append(f"📧 {emails[0]}")
        if addresses: reply.append(f"📍 {addresses[0]}")
        return jsonify({"answer": " | ".join(reply[:3]) if reply else "❌ Sorry, contact info not found."})

    prompt = (
        "You are a helpful legal assistant for Stolmeier Law. Use the following website content to answer clearly and accurately:\n\n"
        + "\n\n---\n\n".join(contexts[:6]) +
        f"\n\nQuestion: {question}\nAnswer:"
    )
    try:
        result = answer_generator(prompt, max_length=250, min_length=50, do_sample=False)
        answer = result[0]["generated_text"].strip()
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        answer = contexts[0]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=False, port=8000, use_reloader=False)

