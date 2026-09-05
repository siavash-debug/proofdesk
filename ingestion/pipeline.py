"""Entry point tying together OCR -> clean -> chunk -> embed for /data PDFs."""


def run_ingestion(data_dir: str = "data") -> None:
    """Process every PDF in data_dir through the full ingestion pipeline."""
    raise NotImplementedError
