# 📄 Nexora Resume Ranker

An AI-powered candidate resume ranking and evaluation system built for the **Nexora Hackathon**. Combines deep semantic text embeddings with domain-specific keyword extraction, rule-based explainability, and structural completeness scoring.

---

## 🚀 Key Features

1. **PDF Parsing (`parser.py`)**:
   - Multi-page text extraction using `pdfplumber`.
   - Supports file paths, byte streams, and uploaded in-memory buffers.

2. **Keyword Matching (`keyword_match.py`)**:
   - Predefined technical taxonomy of **168+ canonical skills** with alias normalization (`js -> javascript`, `k8s -> kubernetes`, `postgres -> postgresql`).
   - Symbol-aware boundary regex matching (`c++`, `c#`, `.net`, `ci/cd`, `node.js`).
   - **RapidFuzz** fuzzy matching (threshold $\ge 85.0$) for typo tolerance (`pytrch -> pytorch`, `kubernets -> kubernetes`).
   - Automated separation into **Required Skills** (from requirements sections & cue phrases) and **Preferred Skills**.
   - Weighted score: $0.7 \times \text{Required} + 0.3 \times \text{Preferred}$.

3. **Semantic Matching (`semantic_match.py`)**:
   - Thread-safe singleton loader for `all-MiniLM-L6-v2`.
   - Document chunking (~300 words) with embedding averaging and $L_2$ normalization.
   - Vectorized batch CPU encoding across all resumes in a single pass.
   - Cosine similarity mapped linearly to $[0.0, 1.0]$.

4. **Hybrid Ranking (`ranker.py`)**:
   - Combines qualitative context and strict technical requirements:
     $$\text{final\_score} = 0.5 \times \text{semantic\_score} + 0.5 \times \text{keyword\_score}$$
   - Returns a sorted pandas DataFrame with clean score spread without artificial clustering.

5. **Structural Completeness Check (`resume_quality.py`)**:
   - Validates the presence of 6 essential sections: **Education**, **Skills**, **Experience**, **Projects**, **Certifications**, and **Contact Information** (email/phone).
   - Formats score as `X/6`.

6. **Deterministic Explainability (`explain.py`)**:
   - 2-3 sentence audit trail explaining why each candidate achieved their rank.
   - 100% rule-based without black-box LLM dependencies.

7. **Streamlit Web Dashboard (`app.py`)**:
   - File uploaders for JD and multi-file resumes (up to 20 PDFs).
   - Interactive sortable results table with visual score bars.
   - Score distribution bar chart across all candidates.
   - Top-3 candidate explanation expanders with metric cards.
   - Pre-loaded sample demo toggle for live presentations.

---

## 📂 Project Structure

```
Nexora-hackathon/
├── app.py                # Streamlit web application
├── parser.py             # PDF & text extraction utilities
├── keyword_match.py      # Taxonomy, alias normalization & fuzzy matching
├── semantic_match.py     # SentenceTransformer embeddings & batch encoding
├── ranker.py             # Hybrid ranking engine
├── resume_quality.py     # 6-section completeness analysis
├── explain.py            # Rule-based candidate explanations
├── test_parser.py        # Parser verification script
├── requirements.txt      # Project dependencies
└── sample_data/          # Sample JD and resumes for testing
    ├── job_description.txt
    └── resumes/
        ├── alice_ml_engineer.pdf
        ├── bob_frontend_dev.pdf
        └── carol_data_scientist.pdf
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tsrinath2007/nexora-hack.git
   cd nexora-hack
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Or run standalone module verification tests from the CLI:
```bash
# Verify PDF extraction
python test_parser.py

# Verify keyword extraction & scoring
python keyword_match.py

# Verify semantic embeddings & batch scoring
python semantic_match.py

# Verify hybrid ranking & score spread
python ranker.py

# Verify top-3 candidate explanations
python explain.py

# Verify resume completeness check
python resume_quality.py
```
