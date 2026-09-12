"""
Semantic matching module using sentence-transformers.

Features:
- Module-level singleton loader for 'all-MiniLM-L6-v2'.
- Chunking for long text (~300 words) with embedding averaging.
- Cosine similarity calculation mapped to [0, 1].
- Vectorized batch encoding for CPU efficiency across multiple resumes.
"""

from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# Suppress noisy symlink warnings on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ==============================================================================
# 1. Module-Level Singleton for SentenceTransformer
# ==============================================================================

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Returns the module-level singleton instance of SentenceTransformer.
    Loads 'all-MiniLM-L6-v2' once and caches it for all subsequent operations.

    Args:
        model_name: HuggingFace model identifier. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer singleton model instance.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        with _MODEL_LOCK:
            if _MODEL_INSTANCE is None:
                _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


# ==============================================================================
# 2. Text Chunking & Embedding Utilities
# ==============================================================================

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
    """
    Splits text into chunks of approximately `chunk_size` words with `overlap`.

    Args:
        text: Input text string.
        chunk_size: Target number of words per chunk (default ~300).
        overlap: Overlap in words between consecutive chunks (default 30).

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    stride = max(1, chunk_size - overlap)
    
    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks


def get_text_embedding(
    text: str,
    chunk_size: int = 300,
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """
    Computes a single normalized embedding for a text document.
    If the document exceeds `chunk_size` words, chunks are encoded
    and averaged, followed by L2 normalization.

    Args:
        text: Input text.
        chunk_size: Target words per chunk (default ~300).
        model: Optional SentenceTransformer model (uses singleton if None).

    Returns:
        1D numpy array of shape (embedding_dim,) normalized to unit length.
    """
    if model is None:
        model = get_model()

    chunks = chunk_text(text, chunk_size=chunk_size)
    get_dim_fn = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", lambda: 384))
    dim = get_dim_fn()

    if not chunks:
        return np.zeros(dim, dtype=np.float32)

    if len(chunks) == 1:
        emb = model.encode(chunks[0], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    # Multiple chunks: encode all and average
    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    avg_emb = np.mean(chunk_embs, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 1e-12:
        avg_emb = avg_emb / norm
    return np.asarray(avg_emb, dtype=np.float32)


# ==============================================================================
# 3. Semantic Similarity Scoring (Mapped to [0, 1])
# ==============================================================================

def semantic_score(
    jd_text: str,
    resume_text: str,
    chunk_size: int = 300,
    model: Optional[SentenceTransformer] = None,
) -> float:
    """
    Computes the semantic match score between a Job Description and a resume.
    Chunks long text into ~300-word sections, averages chunk embeddings,
    and returns cosine similarity mapped to [0, 1].

    Mapping formula:
        mapped_score = (cos_sim + 1.0) / 2.0

    Args:
        jd_text: Raw Job Description text.
        resume_text: Raw resume text.
        chunk_size: Words per chunk (default 300).
        model: Optional SentenceTransformer model.

    Returns:
        Similarity score as a float rounded to 4 decimals between 0.0 and 1.0.
    """
    if not jd_text or not jd_text.strip() or not resume_text or not resume_text.strip():
        return 0.0

    if model is None:
        model = get_model()

    jd_emb = get_text_embedding(jd_text, chunk_size=chunk_size, model=model)
    res_emb = get_text_embedding(resume_text, chunk_size=chunk_size, model=model)

    # Cosine similarity between unit-normalized vectors
    cos_sim = float(np.dot(jd_emb, res_emb))

    # Map cosine similarity [-1, 1] to [0, 1]
    mapped_score = (cos_sim + 1.0) / 2.0
    mapped_score = float(np.clip(mapped_score, 0.0, 1.0))

    return round(mapped_score, 4)


# ==============================================================================
# 4. Batch Encoding for Fast CPU Scoring Across Multiple Resumes
# ==============================================================================

def batch_semantic_scores(
    jd_text: str,
    resumes: Union[Dict[str, str], List[str]],
    chunk_size: int = 300,
    batch_size: int = 32,
    model: Optional[SentenceTransformer] = None,
) -> Union[Dict[str, float], List[float]]:
    """
    Batch-encodes all resumes in a single model.encode(list) call for high
    throughput on CPU across multiple resumes (e.g. 18+ candidates).

    Args:
        jd_text: Job Description text.
        resumes: Either a dictionary {filename: text} or a list of texts.
        chunk_size: Words per chunk (default ~300).
        batch_size: Batch size for model.encode (default 32).
        model: Optional SentenceTransformer model.

    Returns:
        Dictionary {filename: score} if dict was passed, or list of scores if list was passed.
    """
    if not jd_text or not jd_text.strip() or not resumes:
        if isinstance(resumes, dict):
            return {k: 0.0 for k in resumes}
        return [0.0] * len(resumes)

    if model is None:
        model = get_model()

    # Determine input type
    is_dict_input = isinstance(resumes, dict)
    resume_keys: List[str] = list(resumes.keys()) if is_dict_input else [str(i) for i in range(len(resumes))]
    resume_texts: List[str] = list(resumes.values()) if is_dict_input else list(resumes)

    # 1. Prepare chunks across all resumes for a single model.encode call
    all_chunks: List[str] = []
    resume_chunk_indices: List[List[int]] = []

    for text in resume_texts:
        chunks = chunk_text(text, chunk_size=chunk_size)
        if not chunks:
            chunks = [""]  # placeholder for empty text
        
        start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        end_idx = len(all_chunks)
        resume_chunk_indices.append(list(range(start_idx, end_idx)))

    # 2. Also append JD chunks into the batch for maximum efficiency
    jd_chunks = chunk_text(jd_text, chunk_size=chunk_size) or [""]
    jd_start_idx = len(all_chunks)
    all_chunks.extend(jd_chunks)
    jd_indices = list(range(jd_start_idx, len(all_chunks)))

    # 3. Single vectorized model.encode call
    all_embeddings = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)

    # 4. Extract and average JD embedding
    jd_chunk_embs = all_embeddings[jd_indices]
    jd_avg = np.mean(jd_chunk_embs, axis=0)
    jd_norm = np.linalg.norm(jd_avg)
    jd_vec = jd_avg / jd_norm if jd_norm > 1e-12 else jd_avg

    # 5. Extract and aggregate each resume's embedding
    resume_vectors: List[np.ndarray] = []
    for indices in resume_chunk_indices:
        embs = all_embeddings[indices]
        avg = np.mean(embs, axis=0)
        norm = np.linalg.norm(avg)
        res_vec = avg / norm if norm > 1e-12 else avg
        resume_vectors.append(res_vec)

    resume_matrix = np.vstack(resume_vectors)  # shape: (num_resumes, embedding_dim)

    # 6. Vectorized cosine similarity computation
    cos_sims = np.dot(resume_matrix, jd_vec)  # shape: (num_resumes,)
    mapped_scores = (cos_sims + 1.0) / 2.0
    mapped_scores = np.clip(mapped_scores, 0.0, 1.0)
    rounded_scores = [round(float(s), 4) for s in mapped_scores]

    if is_dict_input:
        return dict(zip(resume_keys, rounded_scores))
    return rounded_scores


# ==============================================================================
# Verification Test Runner
# ==============================================================================

def main():
    """Runs a verification test comparing sample JD against sample resumes."""
    from parser import extract_jd, extract_resumes

    base_dir = Path(__file__).resolve().parent
    jd_path = base_dir / "sample_data" / "job_description.txt"
    resumes_dir = base_dir / "sample_data" / "resumes"

    print("=" * 70)
    print("SEMANTIC MATCH MODULE - VERIFICATION TEST")
    print("=" * 70)

    # 1. Load JD and Resumes
    print("\n[1] Loading model and documents...")
    model = get_model()
    get_dim_fn = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", lambda: 384))
    print(f"    Model loaded: {MODEL_NAME} (embedding dim: {get_dim_fn()})")

    jd_text = extract_jd(jd_path)
    print(f"    Loaded JD: {jd_path.name} ({len(jd_text.split())} words, {len(jd_text)} chars)")

    resumes = extract_resumes(resumes_dir)
    print(f"    Loaded {len(resumes)} resumes from {resumes_dir.name}/")

    # 2. Compute individual semantic scores (for sanity check)
    print("\n[2] Individual semantic_score results (JD vs each resume):")
    print("-" * 70)
    print(f"{'Resume Filename':<30} | {'Words':<8} | {'Raw Cosine':<12} | {'Semantic Score [0,1]'}")
    print("-" * 70)

    for filename, resume_text in resumes.items():
        # Get raw cosine similarity for sanity check display
        jd_emb = get_text_embedding(jd_text)
        res_emb = get_text_embedding(resume_text)
        raw_cos = float(np.dot(jd_emb, res_emb))
        
        score = semantic_score(jd_text, resume_text)
        words = len(resume_text.split())
        print(f"{filename:<30} | {words:<8} | {raw_cos:<12.4f} | {score:<10.4f} ({score:.2%})")

    # 3. Test Batch Encoding (speed & consistency check)
    print("\n[3] Testing batch_semantic_scores across all resumes at once:")
    batch_scores = batch_semantic_scores(jd_text, resumes)
    for filename, score in batch_scores.items():
        print(f"    Batch score for {filename:<26} -> {score:.4f} ({score:.2%})")

    print("\n" + "=" * 70)
    print("Verification completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
