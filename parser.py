import io
from pathlib import Path
from typing import Dict, Any, Union
import pdfplumber


def extract_text_from_pdf(source: Union[str, Path, Any]) -> str:
    """
    Extracts all text from a given PDF file or buffer using pdfplumber.
    Supports file paths (str/Path), file-like objects (BytesIO), and Streamlit UploadedFiles.
    
    Args:
        source: Path to the PDF file or file-like object.
        
    Returns:
        Extracted raw text content as a string.
    """
    if hasattr(source, "getvalue"):
        file_obj = io.BytesIO(source.getvalue())
    elif hasattr(source, "read") and not isinstance(source, (str, Path)):
        file_obj = source
    elif isinstance(source, bytes):
        file_obj = io.BytesIO(source)
    else:
        pdf_path = Path(source)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        file_obj = str(pdf_path)

    extracted_pages = []
    with pdfplumber.open(file_obj) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text.strip())

    return "\n\n".join(extracted_pages)


def extract_jd(source: Union[str, Path, Any]) -> str:
    """
    Extracts raw Job Description text from a file (.pdf, .txt, .md) or uploaded buffer.
    
    Args:
        source: Path to the Job Description file or UploadedFile.
        
    Returns:
        Extracted raw JD text.
    """
    if hasattr(source, "name") and hasattr(source, "getvalue"):
        filename = getattr(source, "name", "").lower()
        if filename.endswith(".pdf"):
            return extract_text_from_pdf(source)
        raw_bytes = source.getvalue()
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="ignore").strip()

    jd_path = Path(source)
    if not jd_path.is_file():
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    if jd_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(jd_path)
    
    # For plain text, markdown, or similar text files
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return jd_path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    # Fallback with ignore errors if needed
    return jd_path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_resumes(folder_path: str | Path) -> Dict[str, str]:
    """
    Extracts text from all PDF resume files found in the specified directory.
    
    Args:
        folder_path: Directory path containing resume PDF files.
        
    Returns:
        A dictionary mapping filename (e.g. 'candidate1.pdf') to extracted raw text.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Resume folder not found: {folder}")

    resumes: Dict[str, str] = {}
    
    # Iterate through all files in folder with .pdf extension (case-insensitive)
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            try:
                resumes[file_path.name] = extract_text_from_pdf(file_path)
            except Exception as e:
                print(f"Warning: Failed to parse '{file_path.name}': {e}")
                resumes[file_path.name] = ""

    return resumes
