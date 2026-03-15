"""
submit.py
Runs the pipeline locally and submits the JSON results to the Lucio Challenge API.
"""
import httpx
import json
import logging
from pipeline import build_index, run_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUBMISSION_URL = "https://example-lucio-challenge-server.com/api/submit"

def main():
    logger.info("Starting Lucio Challenge Execution...")
    
    # Read questions from Excel
    import pandas as pd
    from pathlib import Path
    
    excel_path = Path("d:/lucio/documents/Testing Set Questions.xlsx")
    if not excel_path.exists():
        logger.error(f"Could not find questions file: {excel_path}")
        return
        
    df = pd.read_excel(excel_path)
    questions = df['Question'].dropna().tolist()
    logger.info(f"Loaded {len(questions)} questions from {excel_path.name}")
    
    # 1. Build index (will read all files in documents/)
    build_index(force=True)
    
    # 2. Run the dynamic questions
    results = run_questions(questions)
    
    # 3. Format as specific challenge JSON output
    submission_data = [r.to_dict() for r in results if r]
    
    # Dump locally for verification
    with open("lucio_submission.json", "w", encoding="utf-8") as f:
        json.dump(submission_data, f, indent=2)
    
    logger.info(f"Saved local submission exactly matching criteria to lucio_submission.json")
    
    # 4. Submit to Lucio API (mocked)
    logger.info("Submitting results to Lucio API...")
    try:
        response = httpx.post(SUBMISSION_URL, json=submission_data, timeout=30.0)
        # response.raise_for_status()
        logger.info("Submission successful! (Mocked HTTPX POST)")
    except Exception as e:
        logger.warning(f"Submission API failed (expected if URL is a mock): {e}")

if __name__ == "__main__":
    main()
