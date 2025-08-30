import json
import os
import uuid
import faiss
import numpy as np
import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMB_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMB_DIM = 384  # adjust for your embedding model

class DocIndex:
    def __init__(self, workdir="studymate_index"):
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True)
        self.index_path = self.workdir / "faiss.index"
        self.meta_path = self.workdir / "meta.jsonl"
        self.model = SentenceTransformer(EMB_NAME)
        self.index = None
        self.metadata = []

    def _new_index(self):
        # cosine similarity via inner product on L2-normalized vectors
        return faiss.IndexFlatIP(EMB_DIM)

    def _chunk_page_text(self, text, page_no, title):
        words = text.split()
        step, size = 180, 220  # adjust for chunk sizes
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+size]).strip()
            if len(chunk) < 200:
                continue
            yield {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "page": page_no,
                "title": title
            }

    def ingest_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        title = Path(pdf_path).stem
        chunks = []
        for pno in range(len(doc)):
            page = doc[pno]
            text = page.get_text("text")
            text = " ".join(text.split())
            for ch in self._chunk_page_text(text, pno+1, title):
                chunks.append(ch)

        texts = [c["text"] for c in chunks]
        vecs = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        vecs = np.asarray(vecs, dtype="float32")

        if self.index is None:
            self.index = self._new_index()
        self.index.add(vecs)
        self.metadata.extend(chunks)

    def save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            self.metadata = [json.loads(l) for l in open(self.meta_path, encoding="utf-8").read().splitlines()]
        return self

    def search(self, query, top_k=20):
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


# ---------------- DEMO USAGE ----------------
if __name__ == "__main__":
    pdf_path = "sample.pdf"  # 🔹 replace with your PDF file

    idx = DocIndex()

    print("📥 Ingesting PDF...")
    idx.ingest_pdf(pdf_path)
    idx.save()

    print("✅ Index built and saved.")

    # Reload and search
    idx = DocIndex().load()
    query = "What is machine learning?"   # 🔹 test query
    print(f"\n🔎 Searching for: {query}\n")

    results = idx.search(query, top_k=3)
    for r in results:
        print(f"[Page {r['page']}] {r['text'][:100]}... (score={r['score']:.4f})")
