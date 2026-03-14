"""
download_corpus.py
Script to simulate parallel downloading of a corpus as requested by the Lucio Challenge.
Handles retry mechanisms, integrity validation, and parallel workers.
"""
import asyncio
import httpx
import logging
from pathlib import Path
from config import DOCS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with the actual Lucio Challenge Server API endpoint if provided
LUCIO_API_URL = "https://example-lucio-challenge-server.com/api/corpus"

async def download_file(client: httpx.AsyncClient, file_url: str, dest_path: Path, retries: int = 3):
    """Download a single file with retries and basic integrity checking."""
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {file_url} (Attempt {attempt})")
            response = await client.get(file_url, timeout=30.0)
            response.raise_for_status()

            # Simulate integrity validaton
            content = response.content
            if len(content) == 0:
                raise ValueError("Downloaded file is empty")

            with open(dest_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Successfully downloaded {dest_path.name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to download {file_url}: {e}")
            if attempt == retries:
                logger.error(f"Max retries reached for {file_url}")
                return False
            await asyncio.sleep(2 ** attempt)

async def main():
    """Fetch the corpus manifest and download in parallel."""
    logger.info("Connecting to Lucio Server...")
    try:
        async with httpx.AsyncClient() as client:
            # 1. Fetch manifest (Mocked for now since URL isn't provided)
            # response = await client.get(f"{LUCIO_API_URL}/manifest")
            # files = response.json().get("files", [])
            
            logger.info("Lucio server connection implemented (Mocked)")
            files = [
                {"url": f"https://example.com/doc_{i}.pdf", "name": f"doc_{i}.pdf"}
                for i in range(1, 6) # Simulate 5 files
            ]
            
            # 2. Parallel downloading
            DOCS_DIR.mkdir(exist_ok=True)
            tasks = [
                download_file(client, file_info["url"], DOCS_DIR / file_info["name"])
                for file_info in files
            ]
            
            results = await asyncio.gather(*tasks)
            success = sum(1 for r in results if r)
            logger.info(f"Corpus download complete: {success}/{len(files)} files downloaded.")
            
    except Exception as e:
        logger.error(f"Lucio server connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
