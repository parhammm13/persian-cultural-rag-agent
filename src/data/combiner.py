from pathlib import Path
import json

# ۱. تعریف مسیرها
base_dir = Path(r"D:\Summer code\portfolio project\persian-cultural-rag-agent\Data\pages_no_photos")
output_dir = Path(r"D:\Summer code\portfolio project\persian-cultural-rag-agent\Data\processed")
output_file = output_dir / "final_pages_data.json"

# ساخت پوشه خروجی در صورت عدم وجود
output_dir.mkdir(parents=True, exist_ok=True)

if not base_dir.exists():
    raise FileNotFoundError(f"پوشه ورودی یافت نشد:\n{base_dir}")

combined_data = []

# ۲. پیمایش و ویرایش ساختار داده
for folder in base_dir.iterdir():
    if folder.is_dir():
        metadata_path = folder / "page_metadata.json"
        content_path = folder / "page_content.json"
        text_path = folder / "page_text.txt"

        if metadata_path.exists() and content_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            with open(content_path, "r", encoding="utf-8") as f:
                page_content = json.load(f)

            page_text = ""
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as f:
                    page_text = f.read()

            # استخراج داده‌های متا
            page_id = metadata.get("Id")
            node_type = metadata.get("source_row", {}).get("node_type")

            # جدا کردن page_url و title از page_content و حذف full_text
            page_url = page_content.pop("page_url", None)
            title = page_content.pop("title", None)
            page_content.pop("full_text", None)  # حذف full_text از داخل page_content

            # ساخت رکورد جدید با ساختار مورد نظر شما
            record = {
                "unique_id": f"page_{page_id}",
                "id": page_id,
                "node_type": node_type,
                "page_url": page_url,
                "title": title,
                "text": page_text,
                "page_content": page_content  # شامل lead و sections
            }
            combined_data.append(record)

# ۳. ذخیره خروجی
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"پردازش انجام شد. خروجی در مسیر زیر ذخیره گردید:\n{output_file.resolve()}")