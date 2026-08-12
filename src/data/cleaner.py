import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# CLEANER
# ============================================================

class PageDataCleaner:
    def __init__(
        self,
        base_dir: str | Path,
        output_dir: str | Path,
    ):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # فقط خروجی نهایی
        self.output_file = (
            self.output_dir
            / "cleaned_data.json"
        )

    # ========================================================
    # NORMALIZE TITLE
    # ========================================================

    @staticmethod
    def normalize_title(title: Any) -> str:
        """
        Normalize page titles.

        Examples:
            "الگو : مسجد" -> "الگو:مسجد"
            "فهرست   آثار" -> "فهرست آثار"
        """

        if title is None:
            return ""

        title = str(title).strip()

        # Normalize spaces around colon
        title = re.sub(
            r"\s*:\s*",
            ":",
            title,
        )

        # Remove repeated whitespace
        title = re.sub(
            r"\s+",
            " ",
            title,
        )

        return title

    # ========================================================
    # REMOVE RULES
    # ========================================================

    @classmethod
    def should_remove(cls, title: Any) -> bool:
        """
        Remove:
        - Template pages
        - List pages
        """

        title = cls.normalize_title(title)

        # Template pages
        if title.startswith("الگو:"):
            return True

        # List pages
        if title == "فهرست":
            return True

        if title.startswith("فهرست "):
            return True

        if title.startswith("فهرست‌"):
            return True

        return False

    # ========================================================
    # PROCESS DATA
    # ========================================================

    def process_data(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Pipeline:

        1. Read page folders
        2. Combine metadata/content/text
        3. Remove duplicate pages
        4. Remove template/list pages
        5. Save final cleaned_data.json
        """

        logging.info(
            "Starting data cleaning process..."
        )

        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: "
                f"{self.base_dir}"
            )

        seen_ids = set()
        cleaned_pages = []

        duplicate_count = 0
        invalid_count = 0
        error_count = 0
        removed_count = 0

        removed_titles = set()

        # ====================================================
        # READ PAGE FOLDERS
        # ====================================================

        for folder in sorted(
            self.base_dir.iterdir()
        ):
            if not folder.is_dir():
                continue

            metadata_path = (
                folder
                / "page_metadata.json"
            )

            content_path = (
                folder
                / "page_content.json"
            )

            text_path = (
                folder
                / "page_text.txt"
            )

            # ------------------------------------------------
            # Required files
            # ------------------------------------------------

            if (
                not metadata_path.exists()
                or not content_path.exists()
            ):
                logging.warning(
                    "Required files missing in folder: "
                    f"{folder.name}"
                )

                continue

            try:
                # ============================================
                # READ METADATA
                # ============================================

                with metadata_path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    metadata = json.load(f)

                # ============================================
                # READ STRUCTURED CONTENT
                # ============================================

                with content_path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    page_content = json.load(f)

                # ============================================
                # READ PAGE TEXT
                # ============================================

                page_text = ""

                if text_path.exists():
                    page_text = (
                        text_path.read_text(
                            encoding="utf-8"
                        )
                        .strip()
                    )

                # ============================================
                # PAGE ID
                # ============================================

                page_id = str(
                    metadata.get(
                        "Id",
                        "",
                    )
                ).strip()

                if not page_id:
                    invalid_count += 1

                    logging.warning(
                        "Missing page ID in folder: "
                        f"{folder.name}"
                    )

                    continue

                # ============================================
                # DEDUPLICATION
                # ============================================

                if page_id in seen_ids:
                    duplicate_count += 1
                    continue

                seen_ids.add(page_id)

                # ============================================
                # METADATA
                # ============================================

                source_row = metadata.get(
                    "source_row",
                    {},
                )

                if not isinstance(
                    source_row,
                    dict,
                ):
                    source_row = {}

                node_type = source_row.get(
                    "node_type",
                    "page",
                )

                # ============================================
                # PAGE INFORMATION
                # ============================================

                page_url = page_content.get(
                    "page_url"
                )

                title = page_content.get(
                    "title"
                )

                # ============================================
                # REMOVE TEMPLATE / LIST PAGE
                # ============================================

                if self.should_remove(title):
                    removed_count += 1

                    if title:
                        removed_titles.add(
                            str(title).strip()
                        )

                    continue

                # ============================================
                # STRUCTURED PAGE CONTENT
                # ============================================

                cleaned_page_content = {
                    key: value
                    for key, value
                    in page_content.items()
                    if key not in {
                        "page_url",
                        "title",
                        "full_text",
                    }
                }

                # ============================================
                # FINAL RECORD
                # ============================================

                record = {
                    "unique_id": f"page_{page_id}",
                    "id": page_id,
                    "node_type": node_type,
                    "page_url": page_url,
                    "title": title,
                    "text": page_text,
                    "page_content": cleaned_page_content,
                }

                cleaned_pages.append(
                    record
                )

            except Exception:
                error_count += 1

                logging.exception(
                    "Error processing folder: "
                    f"{folder.name}"
                )

        # ====================================================
        # REPORT
        # ====================================================

        print()
        print("=" * 70)
        print("CLEANING REPORT")
        print("=" * 70)

        print(
            "Duplicate pages removed:",
            duplicate_count,
        )

        print(
            "Invalid pages skipped:",
            invalid_count,
        )

        print(
            "Template/List pages removed:",
            removed_count,
        )

        print(
            "Processing errors:",
            error_count,
        )

        print(
            "Final pages:",
            len(cleaned_pages),
        )

        print()

        print("Removed titles:")

        if removed_titles:
            for title in sorted(
                removed_titles
            ):
                print(title)
        else:
            print(
                "No template/list pages found."
            )

        print("=" * 70)
        print()

        # ====================================================
        # SAVE FINAL JSON
        # ====================================================

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
            "Final cleaned file saved to: "
            f"{self.output_file.resolve()}"
        )

        return cleaned_pages


# ============================================================
# RUN
# ============================================================

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