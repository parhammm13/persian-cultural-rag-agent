import json
import logging
from pathlib import Path
from typing import Any, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


class PageDataCleaner:
    def __init__(self, base_dir: str | Path, output_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_dir / "cleaned_pages_data.json"

    def process_data(self) -> List[Dict[str, Any]]:
        """
        Read page folders, combine metadata/content/text,
        remove duplicate pages, and save normalized records.
        """

        logging.info("Starting data cleaning process...")

        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.base_dir}"
            )

        seen_ids = set()
        cleaned_pages = []

        duplicate_count = 0
        invalid_count = 0
        error_count = 0

        for folder in sorted(self.base_dir.iterdir()):
            if not folder.is_dir():
                continue

            metadata_path = folder / "page_metadata.json"
            content_path = folder / "page_content.json"
            text_path = folder / "page_text.txt"

            if not metadata_path.exists() or not content_path.exists():
                logging.warning(
                    f"Required files missing in folder: {folder.name}"
                )
                continue

            try:
                # Read metadata
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Read structured page content
                with content_path.open("r", encoding="utf-8") as f:
                    page_content = json.load(f)

                # Read full flattened page text
                page_text = ""

                if text_path.exists():
                    page_text = text_path.read_text(
                        encoding="utf-8"
                    ).strip()

                # Extract page ID
                page_id = str(metadata.get("Id", "")).strip()

                if not page_id:
                    invalid_count += 1

                    logging.warning(
                        f"Missing page ID in folder: {folder.name}"
                    )
                    continue

                # Deduplication
                if page_id in seen_ids:
                    duplicate_count += 1
                    continue

                seen_ids.add(page_id)

                # Extract metadata fields
                source_row = metadata.get("source_row", {})

                if not isinstance(source_row, dict):
                    source_row = {}

                node_type = source_row.get(
                    "node_type",
                    "page",
                )

                # Extract top-level page information
                page_url = page_content.get("page_url")
                title = page_content.get("title")

                # Keep structured content,
                # but remove fields already stored elsewhere.
                cleaned_page_content = {
                    key: value
                    for key, value in page_content.items()
                    if key not in {
                        "page_url",
                        "title",
                        "full_text",
                    }
                }

                record = {
                    "unique_id": f"page_{page_id}",
                    "id": page_id,
                    "node_type": node_type,
                    "page_url": page_url,
                    "title": title,
                    "text": page_text,
                    "page_content": cleaned_page_content,
                }

                cleaned_pages.append(record)

            except Exception:
                error_count += 1

                logging.exception(
                    f"Error processing folder: {folder.name}"
                )

        logging.info(
            f"Duplicate pages removed: {duplicate_count}"
        )

        logging.info(
            f"Invalid pages skipped: {invalid_count}"
        )

        logging.info(
            f"Processing errors: {error_count}"
        )

        logging.info(
            f"Total cleaned pages: {len(cleaned_pages)}"
        )

        # Save output
        with self.output_file.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                cleaned_pages,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logging.info(
            f"Cleaned file saved to: "
            f"{self.output_file.resolve()}"
        )

        return cleaned_pages


if __name__ == "__main__":
    BASE_DIR = Path(
        r"D:\Summer code\portfolio project"
        r"\persian-cultural-rag-agent"
        r"\Data\pages_no_photos"
    )

    OUTPUT_DIR = Path(
        r"D:\Summer code\portfolio project"
        r"\persian-cultural-rag-agent"
        r"\Data\processed"
    )

    cleaner = PageDataCleaner(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
    )

    cleaned_data = cleaner.process_data()   