from pathlib import Path
import gradio as gr
import pandas as pd

PATH = Path("Data/evaluation/v1/human_review.csv")


# تبدیل امن مقادیر به متن جهت جلوگیری از خطای NaN در گرادیو
def safe_str(val):
    if pd.isna(val):
        return ""
    return str(val)


# بارگذاری اولیه داده‌ها
def load_data():
    if not PATH.exists():
        raise FileNotFoundError(f"فایل در مسیر {PATH} یافت نشد.")
    df = pd.read_csv(PATH, encoding="utf-8-sig")

    # تبدیل ستون‌ها به object برای جلوگیری از خطای ناسازگاری نوع داده
    for col in ["review_status", "human_notes", "question", "reference_answer"]:
        if col in df.columns:
            df[col] = df[col].astype(object)
        else:
            df[col] = ""

    return df


# ذخیره داده‌ها
def save_data(df):
    df.to_csv(PATH, index=False, encoding="utf-8-sig")


# نمایش داده‌های نمونه فعلی
def show_sample(idx, df):
    if df.empty or idx < 0 or idx >= len(df):
        return ("نمونه‌ای یافت نشد",) + ("",) * 11

    row = df.iloc[idx]
    status = safe_str(row.get("review_status", ""))
    status_text = f" [وضعیت فعلی: {status}]" if status else " [بررسی نشده]"

    header_info = f"### کاندید {idx + 1} از {len(df)}{status_text}"

    q_val = safe_str(row.get("question", ""))
    a_val = safe_str(row.get("reference_answer", ""))
    supp_val = safe_str(row.get("supporting_text", ""))
    cat_val = safe_str(row.get("query_category", ""))
    diff_val = safe_str(row.get("difficulty", ""))
    page_val = safe_str(row.get("page_title", ""))
    parent_val = safe_str(row.get("parent_ids", ""))
    child_val = safe_str(row.get("child_ids", ""))
    note_val = safe_str(row.get("human_notes", ""))

    # ترتیب دقیقاً مطابق با لیست outputs در gradio
    return (
        header_info,     # 1. header
        q_val,           # 2. q_display
        a_val,           # 3. ans_display
        cat_val,         # 4. cat_display
        diff_val,        # 5. diff_display
        page_val,        # 6. page_display
        parent_val,      # 7. parent_display
        child_val,       # 8. child_display
        supp_val,        # 9. supp_display
        q_val,           # 10. edit_q
        a_val,           # 11. edit_ans
        note_val,        # 12. note_input
    )


# اعمال اکشن (تایید/رد/ویرایش) و رفتن به نمونه بعدی
def process_action(action, idx, edited_q, edited_a, note):
    df = load_data()
    if idx < 0 or idx >= len(df):
        return idx, *show_sample(idx, df)

    if action == "accept":
        df.at[idx, "review_status"] = "accepted"
    elif action == "reject":
        df.at[idx, "review_status"] = "rejected"
    elif action == "edit":
        df.at[idx, "question"] = edited_q
        df.at[idx, "reference_answer"] = edited_a
        df.at[idx, "review_status"] = "edited"

    df.at[idx, "human_notes"] = note if note else ""
    save_data(df)

    # رفتن به نمونه بعدی
    next_idx = min(idx + 1, len(df) - 1)
    return next_idx, *show_sample(next_idx, df)


# تغییر نمونه به قبلی/بعدی بدون ثبت تغییرات جدید
def navigate(idx, step):
    df = load_data()
    new_idx = max(0, min(idx + step, len(df) - 1))
    return new_idx, *show_sample(new_idx, df)


# استایل‌دهی جهت راست‌به‌چپ (RTL)
css = """
.container { direction: rtl; text-align: right; font-family: Tahoma, Arial, sans-serif; }
textarea, input { direction: rtl; text-align: right; }
"""

with gr.Blocks(css=css, title="ابزار ارزیابی و بررسی داده") as demo:
    df_init = load_data()
    idx_state = gr.State(value=0)

    with gr.Column(elem_classes="container"):
        gr.Markdown("# 📋 ابزار ارزیابی و بازبینی داده‌های فارسی")

        header = gr.Markdown("### در حال بارگذاری...")

        with gr.Row():
            with gr.Column(scale=2):
                q_display = gr.Textbox(label="سوال (Question)", interactive=False, lines=2)
                ans_display = gr.Textbox(
                    label="پاسخ مرجع (Reference Answer)", interactive=False, lines=4
                )
                supp_display = gr.Textbox(
                    label="متن پشتیبان (Supporting Text)", interactive=False, lines=5
                )

            with gr.Column(scale=1):
                cat_display = gr.Textbox(label="دسته (Category)", interactive=False)
                diff_display = gr.Textbox(label="سختی (Difficulty)", interactive=False)
                page_display = gr.Textbox(label="عنوان صفحه (Page Title)", interactive=False)
                parent_display = gr.Textbox(label="Parent IDs", interactive=False)
                child_display = gr.Textbox(label="Child IDs", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### ✏️ بخش ویرایش و ثبت نظر")

        with gr.Row():
            edit_q = gr.Textbox(label="ویرایش سوال (در صورت نیاز)", lines=2)
            edit_ans = gr.Textbox(label="ویرایش پاسخ (در صورت نیاز)", lines=3)

        note_input = gr.Textbox(label="یادداشت (اختیاری)", placeholder="توضیحات یا علت رد...")

        with gr.Row():
            btn_accept = gr.Button("✅ تایید (Accept)", variant="primary")
            btn_edit = gr.Button("✏️ ثبت ویرایش (Save Edit)", variant="secondary")
            btn_reject = gr.Button("❌ رد (Reject)", variant="stop")

        with gr.Row():
            btn_prev = gr.Button("⬅️ قبلی")
            btn_next = gr.Button("بعدی ➡️")

    # خروجی‌های مشترک
    outputs = [
        header,
        q_display,
        ans_display,
        cat_display,
        diff_display,
        page_display,
        parent_display,
        child_display,
        supp_display,
        edit_q,
        edit_ans,
        note_input,
    ]

    # رویداد بارگذاری اولیه
    demo.load(
        fn=lambda: (0, *show_sample(0, df_init)),
        outputs=[idx_state, *outputs],
    )

    # دکمه‌های عملیاتی
    btn_accept.click(
        fn=lambda idx, eq, ea, n: process_action("accept", idx, eq, ea, n),
        inputs=[idx_state, edit_q, edit_ans, note_input],
        outputs=[idx_state, *outputs],
    )

    btn_reject.click(
        fn=lambda idx, eq, ea, n: process_action("reject", idx, eq, ea, n),
        inputs=[idx_state, edit_q, edit_ans, note_input],
        outputs=[idx_state, *outputs],
    )

    btn_edit.click(
        fn=lambda idx, eq, ea, n: process_action("edit", idx, eq, ea, n),
        inputs=[idx_state, edit_q, edit_ans, note_input],
        outputs=[idx_state, *outputs],
    )

    btn_next.click(
        fn=lambda idx: navigate(idx, 1),
        inputs=[idx_state],
        outputs=[idx_state, *outputs],
    )

    btn_prev.click(
        fn=lambda idx: navigate(idx, -1),
        inputs=[idx_state],
        outputs=[idx_state, *outputs],
    )

if __name__ == "__main__":
    demo.launch()