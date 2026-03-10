"""
Telegram Bot chat với AI (OpenAI, Deepseek, ...) và truy vấn Supabase.
Chạy: python telegram_ai_bot.py
Cấu hình: copy .env.example thành .env và điền token/API key.
"""

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None  # type: ignore
    Client = None  # type: ignore

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

# Load biến môi trường từ .env
load_dotenv()

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Cấu hình từ .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
_default_model = "deepseek-chat" if (OPENAI_BASE_URL and "deepseek" in OPENAI_BASE_URL.lower()) else "gpt-4o-mini"
AI_MODEL = os.getenv("AI_MODEL", _default_model).strip() or _default_model

# Supabase
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip() or None
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip() or None

# Cache schema thật từ DB (tự động lấy khi cần)
_cached_schema: Optional[str] = None

# RAG: Supabase Storage + keyword search (không cần embedding API)
SUPABASE_RAG_BUCKET = (os.getenv("SUPABASE_RAG_BUCKET") or "documents").strip()
SUPABASE_RAG_TABLE = (os.getenv("SUPABASE_RAG_TABLE") or "rag_chunks").strip()
RAG_CHUNK_SIZE = max(100, min(2000, int(os.getenv("RAG_CHUNK_SIZE", "800"))))
RAG_CHUNK_OVERLAP = max(0, min(200, int(os.getenv("RAG_CHUNK_OVERLAP", "100"))))
RAG_TOP_K = max(1, min(20, int(os.getenv("RAG_TOP_K", "8"))))

# Lưu lịch sử hội thoại theo user (chat_id -> list messages)
user_conversations: Dict[int, List[dict]] = {}
MAX_HISTORY = 20

# Lưu lịch sử /query riêng (chat_id -> list of {question, sql, answer})
query_history: Dict[int, List[dict]] = {}
MAX_QUERY_HISTORY = 10


# ======================= CLIENTS =======================

def get_openai_client() -> OpenAI:
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def get_supabase_client() -> Optional[Any]:
    if not create_client or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.warning("Không tạo được Supabase client: %s", e)
        return None


# ======================= DB SCHEMA + SQL =======================

def fetch_db_schema(sb: Any) -> str:
    global _cached_schema
    if _cached_schema:
        return _cached_schema
    try:
        r = sb.rpc("get_schema_info", {}).execute()
        rows = (r.data or []) if hasattr(r, "data") else []
        if not rows:
            return "(Không lấy được schema. Chạy QUERY_SETUP.sql trong Supabase SQL Editor.)"
        tables: Dict[str, List[str]] = {}
        for row in rows:
            tbl = row.get("table_name", "")
            col = row.get("column_name", "")
            dtype = row.get("data_type", "")
            nullable = row.get("is_nullable", "")
            desc = f"{col} ({dtype}{', nullable' if nullable == 'YES' else ''})"
            tables.setdefault(tbl, []).append(desc)
        lines = []
        for tbl, cols in sorted(tables.items()):
            lines.append(f"TABLE {tbl}: {', '.join(cols)}")
        _cached_schema = "\n".join(lines)
        return _cached_schema
    except Exception as e:
        logger.warning("fetch_db_schema: %s", e)
        return f"(Lỗi lấy schema: {e}. Chạy QUERY_SETUP.sql trong Supabase SQL Editor.)"


def execute_sql(sb: Any, sql: str) -> Tuple[List[dict], Optional[str]]:
    try:
        r = sb.rpc("execute_readonly_sql", {"query": sql}).execute()
        data = r.data if hasattr(r, "data") else []
        if isinstance(data, list):
            return data, None
        return [], None
    except Exception as e:
        return [], str(e)


def refresh_schema_cache() -> None:
    global _cached_schema
    _cached_schema = None


# ======================= RAG: chunking, storage, keyword search =======================


def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> List[str]:
    if not text or not text.strip():
        return []
    text = text.strip().replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_br = chunk.rfind("\n")
            if last_br > chunk_size // 2:
                chunk = chunk[: last_br + 1]
                end = start + last_br + 1
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap if overlap > 0 else end
    return chunks


def _list_storage_files(sb: Any, bucket: str, prefix: str = "") -> List[str]:
    out: List[str] = []
    try:
        opts = {"limit": 1000}
        path = prefix if prefix else ""
        resp = sb.storage.from_(bucket).list(path, opts)
        if hasattr(resp, "data"):
            resp = resp.data
        if not resp or not isinstance(resp, list):
            return out
        for item in resp:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            fpath = f"{prefix}/{name}" if prefix else name
            is_file = "." in name
            is_folder = isinstance(item.get("metadata"), dict) and (item.get("metadata") or {}).get("mimetype") == "application/folder"
            if is_file and not is_folder:
                out.append(fpath)
            elif is_folder or not is_file:
                sub = _list_storage_files(sb, bucket, fpath)
                out.extend(sub)
            else:
                out.append(fpath)
    except Exception as e:
        logger.warning("Storage list %s/%s: %s", bucket, prefix, e)
    return out


def _decode_file_content(data: bytes, path: str) -> Optional[str]:
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _extract_pdf_text(data: bytes) -> Optional[str]:
    """Trích xuất toàn bộ text từ file PDF (dùng PyMuPDF)."""
    if not fitz:
        return None
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text and text.strip():
                pages.append(text.strip())
        doc.close()
        if pages:
            return "\n\n".join(pages)
    except Exception as e:
        logger.warning("PDF extract error: %s", e)
    return None


def rag_index_storage(sb: Any, bucket: str) -> Tuple[int, str]:
    paths = _list_storage_files(sb, bucket)
    text_ext = (".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".html", ".htm", ".xml", ".yaml", ".yml", ".rst")
    total_chunks = 0
    errors = []
    indexed_files = 0
    for path in paths:
        is_pdf = path.lower().endswith(".pdf")
        is_text = any(path.lower().endswith(ext) for ext in text_ext)
        if not is_pdf and not is_text:
            continue
        try:
            raw = sb.storage.from_(bucket).download(path)
            if not raw:
                continue
            data = bytes(raw) if not isinstance(raw, bytes) else raw
            if is_pdf:
                if not fitz:
                    errors.append(f"{path}: cần cài PyMuPDF (pip install PyMuPDF)")
                    continue
                text = _extract_pdf_text(data)
                if not text:
                    errors.append(f"{path}: không trích xuất được text từ PDF")
                    continue
            else:
                text = _decode_file_content(data, path)
                if not text:
                    errors.append(f"{path}: không decode được text")
                    continue
            chunks = chunk_text(text)
            if not chunks:
                continue
            rows = [{"source": path, "content": c} for c in chunks]
            sb.table(SUPABASE_RAG_TABLE).insert(rows).execute()
            total_chunks += len(rows)
            indexed_files += 1
        except Exception as e:
            errors.append(f"{path}: {e}")
    msg = f"Đã index {indexed_files} file, {total_chunks} chunk (tổng {len(paths)} file trong bucket)."
    if errors:
        msg += " Lỗi: " + "; ".join(errors[:5])
        if len(errors) > 5:
            msg += f" (+{len(errors) - 5} lỗi khác)"
    return total_chunks, msg


def rag_keyword_search(sb: Any, keywords: List[str], top_k: int = RAG_TOP_K) -> List[dict]:
    """Tìm chunk bằng từ khóa (RPC search_rag_chunks)."""
    if not keywords:
        return []
    try:
        r = sb.rpc("search_rag_chunks", {"keywords": keywords, "match_count": top_k}).execute()
        return (r.data or []) if hasattr(r, "data") else []
    except Exception as e:
        logger.warning("RAG keyword search: %s", e)
        return []


def extract_keywords_from_question(client: OpenAI, question: str) -> List[str]:
    """Dùng AI trích xuất từ khóa tìm kiếm từ câu hỏi."""
    try:
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Trích xuất 3-8 từ khóa quan trọng từ câu hỏi để tìm kiếm trong tài liệu. "
                    "Trả về CHỈ các từ khóa, mỗi từ cách nhau bằng dấu phẩy, không giải thích. "
                    "Bao gồm cả từ gốc và biến thể (ví dụ: 'bảo hành' → 'bảo hành, warranty'). "
                    "Ưu tiên danh từ, tên riêng, thuật ngữ chuyên ngành."
                )},
                {"role": "user", "content": question},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        keywords = [k.strip() for k in raw.split(",") if k.strip()]
        return keywords[:10]
    except Exception as e:
        logger.warning("extract_keywords: %s", e)
        return question.split()[:5]


# ======================= TELEGRAM HANDLERS =======================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [
        "Chào! Gửi tin nhắn để chat với AI.",
        "/clear - Xóa lịch sử hội thoại.",
        "/model - Xem model đang dùng.",
    ]
    if get_supabase_client():
        lines.append("/query <câu hỏi> - Truy vấn CSDL bằng ngôn ngữ tự nhiên.")
        lines.append("/tables - Xem cấu trúc CSDL (bảng, cột).")
        lines.append("/refresh - Cập nhật lại cache schema.")
    if get_supabase_client() and SUPABASE_RAG_BUCKET:
        lines.append("/rag_index - Index file trong Supabase Storage vào RAG.")
        lines.append("/ask <câu hỏi> - Trả lời dựa trên tài liệu đã index (RAG).")
    await update.message.reply_text("\n".join(lines))


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if chat_id in user_conversations:
        del user_conversations[chat_id]
    if chat_id in query_history:
        del query_history[chat_id]
    await update.message.reply_text("Đã xóa lịch sử hội thoại và lịch sử truy vấn.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    base = OPENAI_BASE_URL or "api.openai.com"
    await update.message.reply_text(f"Model: {AI_MODEL}\nBase URL: {base}")


async def cmd_tables(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sb = get_supabase_client()
    if not sb:
        await update.message.reply_text("Chưa cấu hình Supabase.")
        return
    await update.message.chat.send_action("typing")
    schema = fetch_db_schema(sb)
    if len(schema) > 4000:
        for i in range(0, len(schema), 4000):
            await update.message.reply_text(schema[i : i + 4000])
    else:
        await update.message.reply_text(schema)


async def cmd_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    refresh_schema_cache()
    await update.message.reply_text("Đã xóa cache schema. Lần truy vấn sau sẽ đọc lại từ DB.")


async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Truy vấn CSDL bằng ngôn ngữ tự nhiên (Text-to-SQL) với lịch sử hội thoại."""
    sb = get_supabase_client()
    if not sb:
        await update.message.reply_text("Chưa cấu hình Supabase. Thêm SUPABASE_URL và SUPABASE_KEY vào .env.")
        return
    query_text = context.args or []
    if not query_text:
        await update.message.reply_text(
            "Dùng: /query <câu hỏi>\n"
            "Ví dụ:\n"
            "  /query Doanh thu của Med Bắc Ninh năm 2025\n"
            "  /query Top 10 sản phẩm bán chạy nhất\n"
            "  /query Tổng số đơn hàng tháng 3/2025"
        )
        return
    user_question = " ".join(query_text).strip()
    if not user_question:
        await update.message.reply_text("Vui lòng nhập câu hỏi sau /query.")
        return

    await update.message.chat.send_action("typing")

    chat_id = update.effective_chat.id

    # Bước 1: Lấy schema thật từ DB
    schema = fetch_db_schema(sb)

    # Bước 2: Xây lịch sử query trước đó làm context
    history = query_history.get(chat_id, [])[-MAX_QUERY_HISTORY:]
    history_for_sql = []
    history_for_summary = []
    for h in history:
        history_for_sql.append({"role": "user", "content": h["question"]})
        history_for_sql.append({"role": "assistant", "content": h["sql"]})
        history_for_summary.append({"role": "user", "content": h["question"]})
        history_for_summary.append({"role": "assistant", "content": h["answer"]})

    # Bước 3: AI sinh câu SQL từ câu hỏi + schema + lịch sử
    sql_system = (
        "Bạn là chuyên gia SQL PostgreSQL. Nhiệm vụ: chuyển câu hỏi tiếng Việt thành MỘT câu SQL SELECT.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "- CHỈ trả về câu SQL thuần, không markdown, không giải thích, không ```.\n"
        "- Chỉ dùng SELECT. KHÔNG INSERT/UPDATE/DELETE/DROP.\n"
        "- LUÔN bọc tên bảng và tên cột bằng dấu ngoặc kép (double quotes) để giữ đúng chữ hoa/thường. "
        "Ví dụ: SELECT \"revenue\" FROM \"Revenue\" WHERE \"BranchName\" ILIKE '%abc%'.\n"
        "- Dùng ILIKE thay LIKE để tìm kiếm không phân biệt hoa thường.\n"
        "- Khi tìm theo tên (vd: 'Med Bắc Ninh'), dùng ILIKE '%...%'.\n"
        "- Khi câu hỏi yêu cầu tổng, đếm, trung bình... → LUÔN dùng SUM, COUNT, AVG, GROUP BY "
        "trên TOÀN BỘ dữ liệu (KHÔNG thêm LIMIT). Đây là quy tắc quan trọng nhất.\n"
        "- Chỉ thêm LIMIT khi câu hỏi yêu cầu liệt kê danh sách (top N, N dòng đầu...).\n"
        "- Dùng alias tiếng Việt cho cột kết quả khi có thể (AS \"Doanh thu\", AS \"Số lượng\").\n"
        "- Nếu người dùng hỏi tiếp nối (ví dụ: 'so sánh với năm 2024', 'còn chi nhánh khác thì sao'), "
        "hãy dựa vào lịch sử hội thoại để hiểu ngữ cảnh và sinh SQL phù hợp.\n\n"
        f"SCHEMA CƠ SỞ DỮ LIỆU:\n{schema}"
    )

    try:
        client = get_openai_client()

        sql_messages = [{"role": "system", "content": sql_system}]
        sql_messages.extend(history_for_sql)
        sql_messages.append({"role": "user", "content": user_question})

        resp1 = client.chat.completions.create(
            model=AI_MODEL,
            messages=sql_messages,
        )
        raw_sql = (resp1.choices[0].message.content or "").strip()
        if raw_sql.startswith("```"):
            raw_sql = re.sub(r"^```\w*\n?", "", raw_sql)
            raw_sql = re.sub(r"\n?```\s*$", "", raw_sql)
        raw_sql = raw_sql.strip().rstrip(";")

        logger.info("Text-to-SQL: %s -> %s", user_question, raw_sql)

        # Bước 4: Chạy SQL qua RPC
        data, err = execute_sql(sb, raw_sql)
        if err:
            await update.message.reply_text(f"Lỗi SQL: {err}\n\nCâu SQL đã sinh:\n{raw_sql}")
            return
        if not data:
            await update.message.reply_text(f"Không có kết quả.\n\nSQL: {raw_sql}")
            return

        # Bước 5: AI tổng hợp kết quả thành câu trả lời tự nhiên
        data_str = json.dumps(data[:50], ensure_ascii=False, default=str)
        if len(data_str) > 6000:
            data_str = data_str[:6000] + "..."

        summary_system = (
            "Bạn là trợ lý phân tích dữ liệu. Dựa vào kết quả truy vấn SQL bên dưới, "
            "hãy trả lời câu hỏi của người dùng bằng tiếng Việt, rõ ràng, dễ hiểu. "
            "Nếu có số liệu, format cho dễ đọc (phân cách hàng nghìn, đơn vị). "
            "Nếu có nhiều dòng, trình bày dạng danh sách ngắn gọn. "
            "Hãy tận dụng lịch sử hội thoại trước đó (nếu có) để đưa ra so sánh hoặc nhận xét thêm."
        )
        summary_user = (
            f"Câu hỏi: {user_question}\n\n"
            f"SQL đã chạy:\n{raw_sql}\n\n"
            f"Kết quả ({len(data)} dòng):\n{data_str}"
        )

        await update.message.chat.send_action("typing")

        summary_messages = [{"role": "system", "content": summary_system}]
        summary_messages.extend(history_for_summary)
        summary_messages.append({"role": "user", "content": summary_user})

        resp2 = client.chat.completions.create(
            model=AI_MODEL,
            messages=summary_messages,
        )
        answer = (resp2.choices[0].message.content or "").strip()

        # Lưu vào lịch sử query
        if chat_id not in query_history:
            query_history[chat_id] = []
        query_history[chat_id].append({
            "question": user_question,
            "sql": raw_sql,
            "answer": answer,
        })

        if len(answer) > 4000:
            for i in range(0, len(answer), 4000):
                await update.message.reply_text(answer[i : i + 4000])
        else:
            await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("cmd_query: %s", e)
        await update.message.reply_text(f"Có lỗi: {str(e)}")


async def cmd_rag_index(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sb = get_supabase_client()
    if not sb:
        await update.message.reply_text("Chưa cấu hình Supabase (SUPABASE_URL, SUPABASE_KEY).")
        return
    await update.message.reply_text("Đang quét Storage (PDF + text files) và tạo embedding... Vui lòng đợi.")
    try:
        total, msg = rag_index_storage(sb, SUPABASE_RAG_BUCKET)
        await update.message.reply_text(msg)
    except Exception as e:
        logger.exception("rag_index: %s", e)
        await update.message.reply_text(f"Lỗi: {str(e)}. Kiểm tra bucket '{SUPABASE_RAG_BUCKET}', bảng '{SUPABASE_RAG_TABLE}' và RPC '{RAG_MATCH_RPC}'.")


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trả lời câu hỏi dựa trên tài liệu đã index (RAG)."""
    sb = get_supabase_client()
    if not sb:
        await update.message.reply_text("Chưa cấu hình Supabase.")
        return
    question = context.args or []
    if not question:
        await update.message.reply_text("Dùng: /ask <câu hỏi>\nVí dụ: /ask chính sách bảo hành là gì?")
        return
    user_question = " ".join(question).strip()
    if not user_question:
        await update.message.reply_text("Vui lòng nhập câu hỏi sau /ask.")
        return

    await update.message.chat.send_action("typing")

    try:
        client = get_openai_client()
        keywords = extract_keywords_from_question(client, user_question)
        logger.info("RAG keywords: %s", keywords)

        chunks = rag_keyword_search(sb, keywords, top_k=RAG_TOP_K)
        if not chunks:
            await update.message.reply_text(
                "Không tìm thấy tài liệu liên quan. Chạy /rag_index để index file trong Storage trước."
            )
            return
        context_parts = []
        for i, row in enumerate(chunks, 1):
            content = (row.get("content") or "").strip()
            source = (row.get("source") or "").strip()
            if content:
                context_parts.append(f"[{i}] (nguồn: {source})\n{content}")
        context_text = "\n\n---\n\n".join(context_parts)
        system = (
            "Bạn trả lời câu hỏi CHỈ dựa trên ngữ cảnh tài liệu được cung cấp bên dưới. "
            "Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ. Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt."
        )
        user_msg = f"Ngữ cảnh tài liệu:\n\n{context_text}\n\nCâu hỏi: {user_question}"
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        reply = (resp.choices[0].message.content or "").strip()
        if len(reply) > 4000:
            for i in range(0, len(reply), 4000):
                await update.message.reply_text(reply[i : i + 4000])
        else:
            await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("cmd_ask: %s", e)
        await update.message.reply_text(f"Có lỗi: {str(e)}")


# ======================= CHAT TỰ DO =======================

def get_messages_for_user(chat_id: int) -> List[dict]:
    if chat_id not in user_conversations:
        return []
    return user_conversations[chat_id][-MAX_HISTORY:]


def add_to_conversation(chat_id: int, role: str, content: str) -> None:
    if chat_id not in user_conversations:
        user_conversations[chat_id] = []
    user_conversations[chat_id].append({"role": role, "content": content})


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    if not user_text or not user_text.strip():
        return

    await update.message.chat.send_action("typing")

    try:
        client = get_openai_client()
        history = get_messages_for_user(chat_id)
        messages = [{"role": "system", "content": "Bạn là trợ lý hữu ích. Trả lời ngắn gọn, rõ ràng."}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
        )
        reply = response.choices[0].message.content

        add_to_conversation(chat_id, "user", user_text)
        add_to_conversation(chat_id, "assistant", reply)

        if len(reply) > 4000:
            for i in range(0, len(reply), 4000):
                await update.message.reply_text(reply[i : i + 4000])
        else:
            await update.message.reply_text(reply)

    except Exception as e:
        logger.exception("Lỗi khi gọi AI: %s", e)
        await update.message.reply_text(
            f"Có lỗi khi gọi AI: {str(e)}\nKiểm tra API key và .env."
        )


# ======================= MAIN =======================

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Thiếu TELEGRAM_BOT_TOKEN trong .env")
    if not OPENAI_API_KEY:
        raise SystemExit("Thiếu OPENAI_API_KEY trong .env")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("tables", cmd_tables))
    app.add_handler(CommandHandler("refresh", cmd_refresh))
    app.add_handler(CommandHandler("query", cmd_query))
    app.add_handler(CommandHandler("rag_index", cmd_rag_index))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot đang chạy...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
