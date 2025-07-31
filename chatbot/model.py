# ✅ FILE: model.py
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline
from sentence_transformers import SentenceTransformer

# Set where to store downloaded models
BASE_DIR = "D:/chatbot/models"
os.makedirs(BASE_DIR, exist_ok=True)

print("⬇️ Downloading SentenceTransformer model...")
SentenceTransformer('all-MiniLM-L6-v2').save(os.path.join(BASE_DIR, 'all-MiniLM-L6-v2'))

print("⬇️ Downloading DistilBERT (MNLI) intent model...")
AutoTokenizer.from_pretrained('typeform/distilbert-base-uncased-mnli').save_pretrained(os.path.join(BASE_DIR, 'distilbert-base-uncased-mnli'))
AutoModelForSequenceClassification.from_pretrained('typeform/distilbert-base-uncased-mnli').save_pretrained(os.path.join(BASE_DIR, 'distilbert-base-uncased-mnli'))

print("⬇️ Downloading DistilBART summarizer...")
pipeline('summarization', model='sshleifer/distilbart-cnn-12-6').model.save_pretrained(os.path.join(BASE_DIR, 'sshleifer-distilbart-cnn-12-6'))
pipeline('summarization', model='sshleifer/distilbart-cnn-12-6').tokenizer.save_pretrained(os.path.join(BASE_DIR, 'sshleifer-distilbart-cnn-12-6'))

print("✅ All models downloaded to D:/chatbot/models")
