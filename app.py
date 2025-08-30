from flask import Flask, render_template, request
import fitz
import os
import json
import uuid
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize DocIndex
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384

class DocIndex:
    def __init__(self):
        self.model = SentenceTransformer(EMB_NAME)
        self.index = None
        self.metadata = []

    def _new_index(self):
        return faiss.IndexFlatIP(EMB_DIM)

    def _chunk_text(self, text, page_no):
        words = text.split()
        step, size = 180, 220
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+size]).strip()
            if len(chunk) < 200:
                continue
            chunks.append({"id": str(uuid.uuid4()), "text": chunk, "page": page_no})
        return chunks

    def ingest_pdf(self, path):
        doc = fitz.open(path)
        all_chunks = []
        for pno, page in enumerate(doc, start=1):
            text = " ".join(page.get_text().split())
            all_chunks.extend(self._chunk_text(text, pno))
        vecs = self.model.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype="float32")
        if self.index is None:
            self.index = self._new_index()
        self.index.add(vecs)
        self.metadata.extend(all_chunks)

    def search(self, query, top_k=5):
        if self.index is None:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        D, I = self.index.search(qv, top_k)
        hits = []
        for d, i in zip(D[0], I[0]):
            if i < 0:
                continue
            m = self.metadata[i]
            m = {**m, "score": float(d)}
            hits.append(m)
        return hits

idx = DocIndex()

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files.get("pdf")
    if not file:
        return "No file uploaded", 400
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    idx.ingest_pdf(filepath)
    return f"<h2>PDF uploaded and indexed: {file.filename}</h2>"

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return "No question provided", 400
    hits = idx.search(question)
    if not hits:
        return "<h2>No answers found. Make sure you uploaded a PDF first!</h2>"
    response = "<h2>Answers:</h2>"
    for h in hits:
        response += f"<p><strong>Page {h['page']}:</strong> {h['text'][:300]}...</p>"
    return response

if __name__ == "__main__":
    print("🚀 StudyMate AI running at http://127.0.0.1:5000")
    app.run(debug=True)
