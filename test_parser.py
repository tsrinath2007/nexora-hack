import argparse
import sys
from pathlib import Path
from parser import extract_jd, extract_resumes, extract_text_from_pdf


def generate_sample_pdf(file_path: Path, title: str, lines: list[str]) -> None:
    """Helper to generate a minimal standard PDF for testing without heavy dependencies."""
    content_lines = [f"({title}) Tj", "0 -20 Td"]
    for line in lines:
        # Escape parenthesis in PDF text string
        escaped_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"({escaped_line}) Tj")
        content_lines.append("0 -15 Td")
    
    stream_data = "BT /F1 12 Tf 50 720 Td " + " ".join(content_lines) + " ET"
    stream_bytes = stream_data.encode("latin1")
    stream_length = len(stream_bytes)

    body = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {stream_length} >>\nstream\n{stream_data}\nendstream\nendobj\n"
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "xref\n0 6\n"
        "0000000000 65535 f \n"
        "0000000010 00000 n \n"
        "0000000060 00000 n \n"
        "0000000117 00000 n \n"
        "0000000247 00000 n \n"
        "0000000350 00000 n \n"
        "trailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF\n"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(body.encode("latin1"))


def create_sample_files_if_needed(sample_dir: Path) -> tuple[Path, Path]:
    """Creates sample JD and resumes if none exist, so tests run immediately."""
    jd_file = sample_dir / "job_description.txt"
    resumes_dir = sample_dir / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    if not jd_file.exists():
        jd_content = (
            "Job Title: Senior Machine Learning Engineer\n"
            "Company: Nexora Tech\n\n"
            "Responsibilities:\n"
            "- Design and implement machine learning and NLP pipelines.\n"
            "- Build scalable Streamlit and FastAPI applications.\n"
            "- Experience with sentence-transformers, embeddings, and vector search.\n"
            "- Fine-tune LLMs and develop automated evaluation frameworks.\n\n"
            "Requirements:\n"
            "- Strong proficiency in Python, PyTorch, scikit-learn, and Pandas.\n"
            "- 3+ years experience with Information Retrieval and NLP.\n"
            "- Familiarity with Git, Docker, and CI/CD."
        )
        jd_file.write_text(jd_content, encoding="utf-8")

    resume_1 = resumes_dir / "alice_ml_engineer.pdf"
    if not resume_1.exists():
        generate_sample_pdf(
            resume_1,
            "Alice Smith - Machine Learning Engineer",
            [
                "Email: alice@example.com | Phone: 555-0101",
                "Skills: Python, PyTorch, NLP, sentence-transformers, scikit-learn, Pandas",
                "Experience: 4 years building NLP search pipelines, vector embeddings, and LLM apps.",
                "Education: M.S. in Computer Science."
            ]
        )

    resume_2 = resumes_dir / "bob_frontend_dev.pdf"
    if not resume_2.exists():
        generate_sample_pdf(
            resume_2,
            "Bob Jones - Frontend Developer",
            [
                "Email: bob@example.com | Phone: 555-0102",
                "Skills: JavaScript, TypeScript, React, Next.js, HTML, CSS, Figma",
                "Experience: 3 years developing responsive web applications and design systems.",
                "Education: B.S. in Software Engineering."
            ]
        )

    resume_3 = resumes_dir / "carol_data_scientist.pdf"
    if not resume_3.exists():
        generate_sample_pdf(
            resume_3,
            "Carol Danvers - Data Scientist",
            [
                "Email: carol@example.com | Phone: 555-0103",
                "Skills: Python, SQL, Pandas, NumPy, scikit-learn, Data Analysis, Tableau",
                "Experience: 2 years analyzing large scale datasets and training predictive ML models.",
                "Education: B.S. in Statistics."
            ]
        )

    return jd_file, resumes_dir


def test_parsing(jd_path: Path, resumes_folder: Path) -> None:
    print("=" * 60)
    print("RESUME RANKER - PARSER VERIFICATION TEST")
    print("=" * 60)

    # 1. Test JD Extraction
    print(f"\n[1] Testing Job Description Parsing:")
    print(f"    Target JD Path: {jd_path}")
    try:
        jd_text = extract_jd(jd_path)
        jd_char_len = len(jd_text)
        jd_word_count = len(jd_text.split())
        print(f"    [SUCCESS] Extracted JD text length: {jd_char_len} characters ({jd_word_count} words)")
        print(f"    Preview (first 100 chars):\n    ---\n    {repr(jd_text[:100])}\n    ---")
    except Exception as e:
        print(f"    [ERROR] Failed to extract JD: {e}")
        sys.exit(1)

    # 2. Test Resumes Extraction
    print(f"\n[2] Testing Resumes Extraction from Folder:")
    print(f"    Resumes Folder: {resumes_folder}")
    try:
        resumes_dict = extract_resumes(resumes_folder)
        print(f"    Found {len(resumes_dict)} PDF resume(s).\n")
        
        if not resumes_dict:
            print("    [WARNING] No PDF files found in resumes folder.")
            return

        print(f"    {'Resume Filename':<30} | {'Characters':<12} | {'Words':<8} | {'Status'}")
        print("    " + "-" * 62)

        for filename, raw_text in resumes_dict.items():
            char_len = len(raw_text)
            word_count = len(raw_text.split())
            status = "OK" if char_len > 0 else "EMPTY"
            print(f"    {filename:<30} | {char_len:<12} | {word_count:<8} | {status}")

        print("\n" + "=" * 60)
        print("All parsing tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"    [ERROR] Failed to extract resumes: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test resume_ranker parser module.")
    parser.add_argument("--jd", type=str, default=None, help="Path to Job Description file (.pdf or .txt)")
    parser.add_argument("--resumes", type=str, default=None, help="Path to folder containing resume PDFs")
    args = parser.parse_args()

    default_sample_dir = Path(__file__).resolve().parent / "sample_data"

    if args.jd:
        jd_path = Path(args.jd)
    else:
        jd_path, _ = create_sample_files_if_needed(default_sample_dir)

    if args.resumes:
        resumes_folder = Path(args.resumes)
    else:
        _, resumes_folder = create_sample_files_if_needed(default_sample_dir)

    test_parsing(jd_path, resumes_folder)


if __name__ == "__main__":
    main()
