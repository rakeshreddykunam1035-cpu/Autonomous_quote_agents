"""
rag.py -- Minimal retrieval layer over the underwriting guidelines KB.

Uses TF-IDF + cosine similarity (no external API / network / vector DB
service required) so this runs fully offline. If you later have access to
an LLM API, swap `generate_justification`'s templating for an actual
completion call using the retrieved chunks as context -- the retrieval
half of this file does not need to change.
"""
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rag_kb", "underwriting_guidelines.md")


def _load_chunks(path: str = KB_PATH):
    """Split the guideline doc into numbered-clause chunks (e.g. '2.1 ...')."""
    with open(path, "r") as f:
        text = f.read()

    # Split on lines that start a new numbered clause like "2.1 " or "## Section"
    raw_chunks = re.split(r"\n(?=\d+\.\d+\s)", text)
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        clause_match = re.match(r"^(\d+\.\d+)", chunk)
        clause_id = clause_match.group(1) if clause_match else None
        chunks.append({"id": clause_id, "text": chunk.replace("\n", " ")})
    return chunks


class GuidelineRetriever:
    def __init__(self, path: str = KB_PATH):
        self.chunks = _load_chunks(path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 2):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i], "score": round(float(sims[i]), 4)}
            for i in top_idx if sims[i] > 0
        ]


_retriever = None


def get_retriever() -> GuidelineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GuidelineRetriever()
    return _retriever


def generate_justification(query: str, top_k: int = 2) -> dict:
    """
    Retrieve the most relevant guideline clause(s) for a decision and
    return both the raw clauses (for audit) and a plain-English blurb
    an agent/underwriter can read. Templated for now -- swap in an LLM
    call here if you want free-form generation instead of a template.
    """
    retriever = get_retriever()
    hits = retriever.retrieve(query, top_k=top_k)

    if not hits:
        return {"justification": "No matching guideline found for this decision.",
                "citations": []}

    citation_ids = [h["id"] for h in hits if h["id"]]
    lead = hits[0]["text"]
    justification = f"Per guideline {hits[0]['id']}: {lead}"

    return {"justification": justification, "citations": citation_ids, "retrieved": hits}


if __name__ == "__main__":
    r = get_retriever()
    print(f"Loaded {len(r.chunks)} guideline clauses\n")
    for q in ["discount for low risk driver", "when to escalate to underwriter",
              "high risk premium adjustment"]:
        print("Query:", q)
        for hit in r.retrieve(q):
            print(" ", hit["id"], "-", hit["text"][:90], "...", f"(score={hit['score']})")
        print()
