"""
Telegram Bot chat với AI (OpenAI, Deepseek, ...) và truy vấn Supabase.
Chạy: python telegram_ai_bot.py
Cấu hình: copy .env.example thành .env và điền token/API key.
"""

import asyncio
import functools
import json
import os
import re
import logging
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    TypeHandler,
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


async def run_blocking(func, *args, **kwargs):
    """Chạy hàm đồng bộ trong thread pool (Python 3.8 không có asyncio.to_thread)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# Cấu hình từ .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
_default_model = "deepseek-chat" if (OPENAI_BASE_URL and "deepseek" in OPENAI_BASE_URL.lower()) else "gpt-4o-mini"
AI_MODEL = os.getenv("AI_MODEL", _default_model).strip() or _default_model

# Embedding (RAG): dùng API OpenAI riêng để embed tài liệu (chat vẫn dùng OPENAI_API_KEY + BASE_URL)
OPENAI_EMBEDDING_API_KEY = (os.getenv("OPENAI_EMBEDDING_API_KEY") or "").strip() or None
OPENAI_EMBEDDING_BASE_URL = (os.getenv("OPENAI_EMBEDDING_BASE_URL") or "").strip() or "https://api.openai.com/v1"
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()

# Supabase
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip() or None
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip() or None

# Cache schema thật từ DB (tự động lấy khi cần)
_cached_schema: Optional[str] = None

# RAG: Storage + embedding (OpenAI) + vector search
SUPABASE_RAG_BUCKET = (os.getenv("SUPABASE_RAG_BUCKET") or "documents").strip()
SUPABASE_RAG_TABLE = (os.getenv("SUPABASE_RAG_TABLE") or "rag_chunks").strip()
SUPABASE_CHAT_LOG_TABLE = (os.getenv("SUPABASE_CHAT_LOG_TABLE") or "telegram_chat_logs").strip() or "telegram_chat_logs"

# Google Calendar — nhắc lịch hàng ngày (Service Account Workspace hoặc OAuth + cột gcal_refresh_token)
GCALENDAR_TZ = (
    (os.getenv("GCALENDAR_TZ") or os.getenv("MS_CALENDAR_TZ") or "Asia/Ho_Chi_Minh").strip()
    or "Asia/Ho_Chi_Minh"
)
GOOGLE_SERVICE_ACCOUNT_JSON = (os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip() or None
GOOGLE_OAUTH_CLIENT_ID = (os.getenv("GOOGLE_OAUTH_CLIENT_ID") or "").strip() or None
GOOGLE_OAUTH_CLIENT_SECRET = (os.getenv("GOOGLE_OAUTH_CLIENT_SECRET") or "").strip() or None
DAILY_CALENDAR_HOUR = max(0, min(23, int(os.getenv("DAILY_CALENDAR_HOUR", "7"))))
DAILY_CALENDAR_MINUTE = max(0, min(59, int(os.getenv("DAILY_CALENDAR_MINUTE", "0"))))
SUPABASE_USER_TABLE = (os.getenv("SUPABASE_USER_TABLE") or "user").strip() or "user"
RAG_CHUNK_SIZE = max(100, min(2000, int(os.getenv("RAG_CHUNK_SIZE", "800"))))
RAG_CHUNK_OVERLAP = max(0, min(200, int(os.getenv("RAG_CHUNK_OVERLAP", "100"))))
RAG_TOP_K = max(1, min(20, int(os.getenv("RAG_TOP_K", "8"))))
RAG_EMBEDDING_BATCH = max(1, min(100, int(os.getenv("RAG_EMBEDDING_BATCH", "50"))))

# Lưu lịch sử hội thoại theo user (chat_id -> list messages)
user_conversations: Dict[int, List[dict]] = {}
MAX_HISTORY = 20
MAX_DB_HISTORY = max(4, min(60, int(os.getenv("MAX_DB_HISTORY", "24"))))

# Lưu lịch sử /query riêng (chat_id -> list of {question, sql, answer})
query_history: Dict[int, List[dict]] = {}
MAX_QUERY_HISTORY = 10

# Bật/tắt chế độ thinking (reasoning) theo user: chat_id -> True/False
user_thinking: Dict[int, bool] = {}

# Cache service account JSON cho Google Calendar
_cached_sa: Optional[Dict[str, Any]] = None
_ORIGINAL_BOT_SEND_MESSAGE = Bot.send_message


# ======================= CLIENTS =======================

def get_openai_client() -> OpenAI:
    """Client cho chat (dùng OPENAI_API_KEY, có thể trỏ Deepseek)."""
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def get_embedding_client() -> Optional[OpenAI]:
    """Client chỉ cho embedding (OpenAI). Dùng OPENAI_EMBEDDING_API_KEY riêng."""
    if not OPENAI_EMBEDDING_API_KEY:
        return None
    return OpenAI(
        api_key=OPENAI_EMBEDDING_API_KEY,
        base_url=OPENAI_EMBEDDING_BASE_URL or "https://api.openai.com/v1",
    )


def get_supabase_client() -> Optional[Any]:
    if not create_client or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.warning("Không tạo được Supabase client: %s", e)
        return None


def save_chat_log(
    *,
    direction: str,
    chat_id: Optional[int],
    message_text: str,
    message_type: str = "text",
    telegram_user_id: Optional[int] = None,
    telegram_username: Optional[str] = None,
    telegram_full_name: Optional[str] = None,
    update_id: Optional[int] = None,
) -> None:
    if not chat_id:
        return
    sb = get_supabase_client()
    if not sb:
        return
    payload = {
        "chat_id": int(chat_id),
        "direction": (direction or "").strip().lower(),
        "message_text": (message_text or "").strip(),
        "message_type": (message_type or "text").strip().lower(),
        "telegram_user_id": int(telegram_user_id) if telegram_user_id else None,
        "telegram_username": (telegram_username or "").strip() or None,
        "telegram_full_name": (telegram_full_name or "").strip() or None,
        "update_id": int(update_id) if update_id is not None else None,
    }
    try:
        sb.table(SUPABASE_CHAT_LOG_TABLE).insert(payload).execute()
    except Exception as e:
        logger.warning("Không lưu được chat log: %s", e)


def _extract_message_payload(update: Update) -> Tuple[str, str]:
    msg = update.effective_message
    if not msg:
        return "", "unknown"
    if msg.text:
        return msg.text, "text"
    if msg.caption:
        return msg.caption, "caption"
    if msg.sticker:
        return "[sticker]", "sticker"
    if msg.photo:
        return "[photo]", "photo"
    if msg.video:
        return "[video]", "video"
    if msg.document:
        return "[document]", "document"
    if msg.voice:
        return "[voice]", "voice"
    if msg.audio:
        return "[audio]", "audio"
    if msg.location:
        return "[location]", "location"
    if msg.contact:
        return "[contact]", "contact"
    return "[unsupported]", "other"


async def capture_incoming_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    chat = update.effective_chat
    user = update.effective_user
    chat_id = chat.id if chat else None
    if chat_id is None:
        return
    text, message_type = _extract_message_payload(update)
    save_chat_log(
        direction="incoming",
        chat_id=chat_id,
        message_text=text,
        message_type=message_type,
        telegram_user_id=user.id if user else None,
        telegram_username=user.username if user else None,
        telegram_full_name=user.full_name if user else None,
        update_id=update.update_id,
    )


async def patched_send_message(self: Bot, *args, **kwargs):
    sent = await _ORIGINAL_BOT_SEND_MESSAGE(self, *args, **kwargs)
    chat_id = kwargs.get("chat_id")
    if chat_id is None and args:
        chat_id = args[0]
    text = kwargs.get("text")
    if text is None and len(args) > 1:
        text = args[1]
    save_chat_log(
        direction="outgoing",
        chat_id=int(chat_id) if chat_id is not None else None,
        message_text=str(text or ""),
        message_type="text",
        telegram_user_id=None,
        telegram_username="bot",
        telegram_full_name="bot",
        update_id=None,
    )
    return sent


# ======================= GOOGLE CALENDAR =======================

GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _load_service_account_dict() -> Optional[Dict[str, Any]]:
    global _cached_sa
    if _cached_sa is not None:
        return _cached_sa
    raw = (GOOGLE_SERVICE_ACCOUNT_JSON or "").strip()
    if not raw:
        return None
    try:
        if raw.startswith("{"):
            _cached_sa = json.loads(raw)
        else:
            with open(raw, encoding="utf-8") as f:
                _cached_sa = json.load(f)
    except Exception as e:
        logger.error("GOOGLE_SERVICE_ACCOUNT_JSON không đọc được: %s", e)
        return None
    return _cached_sa


def _build_google_calendar_service_account(user_email: str) -> Tuple[Any, Optional[str]]:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    info = _load_service_account_dict()
    if not info:
        return None, "Thiếu GOOGLE_SERVICE_ACCOUNT_JSON (đường dẫn file hoặc chuỗi JSON)."

    creds = service_account.Credentials.from_service_account_info(info, scopes=GOOGLE_CALENDAR_SCOPES)
    creds = creds.with_subject(user_email.strip())
    svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return svc, None


def _build_google_calendar_oauth(refresh_token: str) -> Tuple[Any, Optional[str]]:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    cid = (GOOGLE_OAUTH_CLIENT_ID or "").strip()
    csec = (GOOGLE_OAUTH_CLIENT_SECRET or "").strip()
    if not cid or not csec:
        return None, "Thiếu GOOGLE_OAUTH_CLIENT_ID hoặc GOOGLE_OAUTH_CLIENT_SECRET."

    creds = Credentials(
        token=None,
        refresh_token=refresh_token.strip(),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=csec,
        scopes=GOOGLE_CALENDAR_SCOPES,
    )
    svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return svc, None


def fetch_calendar_events_for_day(
    user_email: str,
    day: date,
    display_tz: str = "Asia/Ho_Chi_Minh",
    oauth_refresh_token: Optional[str] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    tz = ZoneInfo(display_tz)
    start_local = datetime.combine(day, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    time_min = start_local.isoformat()
    time_max = end_local.isoformat()

    try:
        service: Any = None
        err: Optional[str] = None
        rt = (oauth_refresh_token or "").strip()
        if rt:
            service, err = _build_google_calendar_oauth(rt)
        else:
            if not (user_email or "").strip():
                return None, "Thiếu useremail khi dùng Service Account (Google Workspace)."
            service, err = _build_google_calendar_service_account(user_email)

        if err or not service:
            return None, err or "Không tạo được Google Calendar client."

        ev = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=100,
                timeZone=display_tz,
            )
            .execute()
        )
        return list(ev.get("items") or []), None
    except Exception as e:
        logger.exception("fetch_calendar_events_for_day: %s", e)
        return None, str(e)


def _parse_google_start_end(
    ev: Dict[str, Any],
    display_tz: str,
) -> Tuple[bool, datetime, datetime, str]:
    start = ev.get("start") or {}
    end = ev.get("end") or {}
    loc = (ev.get("location") or "").strip()

    if "date" in start and "dateTime" not in start:
        try:
            d0 = datetime.strptime(str(start.get("date")), "%Y-%m-%d").date()
        except ValueError:
            d0 = datetime.now(ZoneInfo(display_tz)).date()
        tz = ZoneInfo(display_tz)
        st = datetime.combine(d0, datetime.min.time(), tzinfo=tz)
        en_d = str(end.get("date") or start.get("date"))
        try:
            d1 = datetime.strptime(en_d, "%Y-%m-%d").date()
        except ValueError:
            d1 = d0
        en = datetime.combine(d1, datetime.min.time(), tzinfo=tz)
        return True, st, en, loc

    st_s = str(start.get("dateTime") or "")
    en_s = str(end.get("dateTime") or "")
    if st_s.endswith("Z"):
        st_s = st_s[:-1] + "+00:00"
    if en_s.endswith("Z"):
        en_s = en_s[:-1] + "+00:00"
    try:
        st = datetime.fromisoformat(st_s)
    except ValueError:
        st = datetime.now(ZoneInfo(display_tz))
    try:
        en = datetime.fromisoformat(en_s)
    except ValueError:
        en = st
    if st.tzinfo is None:
        st = st.replace(tzinfo=ZoneInfo(display_tz))
    if en.tzinfo is None:
        en = en.replace(tzinfo=ZoneInfo(display_tz))
    st = st.astimezone(ZoneInfo(display_tz))
    en = en.astimezone(ZoneInfo(display_tz))
    return False, st, en, loc


def format_day_schedule(
    events: List[Dict[str, Any]],
    day: date,
    display_tz: str = "Asia/Ho_Chi_Minh",
) -> str:
    if not events:
        return f"Lịch ngày {day.strftime('%d/%m/%Y')}: không có sự kiện trên Google Calendar."

    lines: List[str] = [f"Lịch ngày {day.strftime('%d/%m/%Y')} (giờ {display_tz}):"]
    for ev in events:
        subj = (ev.get("summary") or "(Không tiêu đề)").strip()
        is_all, st, en, loc = _parse_google_start_end(ev, display_tz)
        if is_all:
            piece = f"• Cả ngày: {subj}"
        else:
            piece = f"• {st.strftime('%H:%M')}–{en.strftime('%H:%M')}: {subj}"
        if loc:
            piece += f" — {loc}"
        lines.append(piece)

    return "\n".join(lines)


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


# ======================= RAG: embedding + vector search =======================


def get_embeddings(client: OpenAI, texts: List[str], batch_size: int = RAG_EMBEDDING_BATCH) -> List[List[float]]:
    """Gọi API embedding (OpenAI), trả về list vector 1536 chiều. Batch để tránh quá tải."""
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            r = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            for e in (r.data or []):
                emb = getattr(e, "embedding", None)
                if emb is not None:
                    out.append(emb)
                else:
                    out.append([])
        except Exception as e:
            logger.warning("get_embeddings batch %s: %s", i, e)
            for _ in batch:
                out.append([])
    return out


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


def _embedding_to_text(emb: List[float]) -> str:
    """Chuyển list float sang chuỗi '[a,b,c,...]' để gửi RPC vector."""
    return "[" + ",".join(str(x) for x in emb) + "]"


def rag_vector_search(sb: Any, embedding: List[float], top_k: int = RAG_TOP_K) -> List[dict]:
    """Tìm chunk theo độ tương đồng vector (RPC search_rag_by_embedding)."""
    if not embedding or len(embedding) != 1536:
        return []
    try:
        r = sb.rpc(
            "search_rag_by_embedding",
            {"query_embedding_text": _embedding_to_text(embedding), "match_count": top_k},
        ).execute()
        return (r.data or []) if hasattr(r, "data") else []
    except Exception as e:
        logger.warning("RAG vector search: %s", e)
        return []


def rag_index_storage(sb: Any, bucket: str, embedding_client: Optional[OpenAI] = None) -> Tuple[int, str]:
    """Quét Storage, chunk, embed (nếu có API), rồi insert vào rag_chunks."""
    paths = _list_storage_files(sb, bucket)
    text_ext = (".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".html", ".htm", ".xml", ".yaml", ".yml", ".rst")
    all_chunks: List[Tuple[str, str]] = []
    errors = []
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
            for c in chunks:
                all_chunks.append((path, c))
        except Exception as e:
            errors.append(f"{path}: {e}")

    if not all_chunks:
        return 0, f"Không có file text/PDF nào trong bucket (đã quét {len(paths)} file). " + ("Lỗi: " + "; ".join(errors[:3]) if errors else "")

    contents = [c for _, c in all_chunks]
    embeddings: List[List[float]] = []
    if embedding_client:
        embeddings = get_embeddings(embedding_client, contents)
        if len(embeddings) != len(contents):
            errors.append("Số embedding không khớp số chunk; kiểm tra OPENAI_EMBEDDING_API_KEY và EMBEDDING_MODEL.")
    else:
        embeddings = [[] for _ in contents]

    # Xóa dữ liệu cũ (re-index toàn bộ)
    try:
        sb.rpc("truncate_rag_chunks", {}).execute()
    except Exception as e:
        logger.warning("RAG clear old chunks: %s", e)

    batch_size = 100
    inserted = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embs = embeddings[i : i + batch_size]
        rows = []
        for j, (source, content) in enumerate(batch):
            row = {"source": source, "content": content}
            if j < len(embs) and embs[j]:
                row["embedding"] = _embedding_to_text(embs[j])
            rows.append(row)
        try:
            sb.table(SUPABASE_RAG_TABLE).insert(rows).execute()
            inserted += len(rows)
        except Exception as e:
            errors.append(f"insert batch: {e}")

    msg = f"Đã index {len(paths)} file, {inserted} chunk (embedding: {'có' if embedding_client else 'không'})."
    if errors:
        msg += " Lỗi: " + "; ".join(errors[:5])
        if len(errors) > 5:
            msg += f" (+{len(errors) - 5} lỗi khác)"
    return inserted, msg


def rag_keyword_search(sb: Any, keywords: List[str], top_k: int = RAG_TOP_K) -> List[dict]:
    """Tìm chunk bằng từ khóa (fallback khi không có embedding)."""
    if not keywords:
        return []
    try:
        r = sb.rpc("search_rag_chunks", {"keywords": keywords, "match_count": top_k}).execute()
        return (r.data or []) if hasattr(r, "data") else []
    except Exception as e:
        logger.warning("RAG keyword search: %s", e)
        return []


def extract_keywords_from_question(client: OpenAI, question: str) -> List[str]:
    """Trích từ khóa từ câu hỏi (fallback khi không dùng embedding)."""
    try:
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Trích xuất 3-8 từ khóa quan trọng từ câu hỏi để tìm kiếm trong tài liệu. "
                    "Trả về CHỈ các từ khóa, mỗi từ cách nhau bằng dấu phẩy, không giải thích."
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


def get_user_calendar_profile(sb: Any, chat_id: int) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """
    Tìm profile lịch theo telegram chat_id.
    Trả về: (email, refresh_token, display_name, error)
    """
    try:
        r = (
            sb.table(SUPABASE_USER_TABLE)
            .select("useremail,gcal_refresh_token,Username,telegram_ID")
            .eq("telegram_ID", str(chat_id))
            .limit(1)
            .execute()
        )
        rows = r.data or []
        if not rows:
            return (
                None,
                None,
                "",
                "Không thấy bạn trong bảng user (telegram_ID).\n\n"
                "Chưa có lệnh đăng ký tự động — cần thêm tay trên Supabase:\n"
                "1) Gõ /id để xem Chat ID.\n"
                "2) Dashboard → Table Editor → bảng user → Insert row.\n"
                "3) Điền telegram_ID = đúng số từ /id (dạng text), thêm useremail; "
                "nếu dùng OAuth Google thì điền gcal_refresh_token.",
            )
        row = rows[0] or {}
        email = (row.get("useremail") or "").strip() or None
        refresh = (row.get("gcal_refresh_token") or "").strip() or None
        name = (row.get("Username") or "").strip()
        if not refresh and not (GOOGLE_SERVICE_ACCOUNT_JSON and email):
            return (
                email,
                refresh,
                name,
                "Thiếu kết nối Google Calendar. Cần một trong hai:\n\n"
                "• Workspace: trong .env có GOOGLE_SERVICE_ACCOUNT_JSON; "
                "Admin Google ủy quyền domain cho service account (scope calendar.readonly); "
                "trên Supabase dòng của bạn có useremail đúng email @côngty.\n\n"
                "• Gmail cá nhân: trong .env có GOOGLE_OAUTH_CLIENT_ID và GOOGLE_OAUTH_CLIENT_SECRET (OAuth Web); "
                "chạy get_gcal_refresh_token.py (redirect URI trong Console khớp GOOGLE_OAUTH_REDIRECT_URI), "
                "dán refresh token vào cột gcal_refresh_token trên Supabase.",
            )
        return email, refresh, name, None
    except Exception as e:
        logger.exception("get_user_calendar_profile: %s", e)
        return None, None, "", str(e)


def resolve_day_keyword(raw: str) -> Tuple[Optional[date], Optional[str]]:
    tz = ZoneInfo(GCALENDAR_TZ)
    today = datetime.now(tz).date()
    val = (raw or "").strip().lower()
    if val in ("nay", "hôm nay", "hom nay", "today"):
        return today, None
    if val in ("mai", "ngày mai", "ngay mai", "tomorrow"):
        return today + timedelta(days=1), None
    return None, "Dùng: /lich nay hoặc /lich mai"


def summarize_schedule_with_ai(question: str, schedule_text: str) -> str:
    """Nhờ AI duyệt/tóm tắt lịch theo câu hỏi người dùng (đồng bộ — chạy trong thread qua run_blocking)."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý lịch làm việc. Dựa vào lịch trong ngày được cung cấp, "
                    "hãy trả lời ngắn gọn bằng tiếng Việt. "
                    "Nếu không có lịch thì nói rõ không có cuộc họp/sự kiện."
                ),
            },
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nDữ liệu lịch:\n{schedule_text}",
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def answer_calendar_question(
    update: Update,
    day_raw: str,
    custom_question: Optional[str] = None,
) -> bool:
    """Trả lời lịch theo ngày. Trả True nếu đã xử lý xong."""
    if not gcalendar_ready():
        await update.message.reply_text("Google Calendar chưa sẵn sàng. Kiểm tra GOOGLE_* trong .env.")
        return True
    sb = get_supabase_client()
    if not sb:
        await update.message.reply_text("Chưa cấu hình Supabase.")
        return True

    target_day, day_err = resolve_day_keyword(day_raw)
    if day_err or not target_day:
        await update.message.reply_text(day_err or "Tham số ngày không hợp lệ.")
        return True

    chat_id = update.effective_chat.id
    email, refresh, name, profile_err = get_user_calendar_profile(sb, chat_id)
    if profile_err:
        await update.message.reply_text(profile_err)
        return True

    await update.message.chat.send_action("typing")
    try:
        events, err = await run_blocking(
            fetch_calendar_events_for_day,
            email or "",
            target_day,
            GCALENDAR_TZ,
            refresh,
        )
    except Exception as e:
        logger.exception("answer_calendar_question fetch_calendar: %s", e)
        await update.message.reply_text(f"Lấy lịch lỗi: {e}")
        return True
    if err:
        await update.message.reply_text(f"Lấy lịch lỗi: {err}")
        return True

    assert format_day_schedule is not None
    base_schedule = format_day_schedule(events or [], target_day, GCALENDAR_TZ)
    question = custom_question or f"Hãy cho tôi biết lịch {day_raw} gồm những gì quan trọng."
    try:
        answer = await run_blocking(summarize_schedule_with_ai, question, base_schedule)
    except Exception as e:
        logger.exception("answer_calendar_question summarize: %s", e)
        answer = base_schedule

    if name:
        answer = f"Chào {name},\n\n{answer}"
    if len(answer) > 4000:
        for i in range(0, len(answer), 4000):
            await update.message.reply_text(answer[i : i + 4000])
    else:
        await update.message.reply_text(answer)
    return True


# ======================= TELEGRAM HANDLERS =======================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [
        "Chào! Gửi tin nhắn để chat với AI.",
        "/clear - Xóa lịch sử hội thoại.",
        "/model - Xem model đang dùng.",
        "/think - Bật/tắt chế độ suy nghĩ (reasoning): AI sẽ trình bày bước suy luận trước khi trả lời.",
    ]
    if gcalendar_ready():
        lines.append(
            f"(Bot sẽ gửi lịch Google Calendar mỗi ngày khoảng {DAILY_CALENDAR_HOUR:02d}:{DAILY_CALENDAR_MINUTE:02d} giờ {GCALENDAR_TZ} nếu bạn có trong Supabase.)"
        )
        lines.append("/lich <nay|mai> - Xem lịch Google Calendar đã qua AI duyệt.")
    if get_supabase_client():
        lines.append("/query <câu hỏi> - Truy vấn CSDL bằng ngôn ngữ tự nhiên.")
        lines.append("/tables - Xem cấu trúc CSDL (bảng, cột).")
        lines.append("/refresh - Cập nhật lại cache schema.")
        lines.append("/id - Xem Chat ID (để thêm vào bảng user / lịch Google Calendar).")
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


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chat ID dùng cho cột telegram_ID trong bảng user (Supabase)."""
    cid = update.effective_chat.id
    chat_type = update.effective_chat.type
    user = update.effective_user
    # Không dùng Markdown/HTML: username có dấu _ hoặc ký tự đặc biệt sẽ làm Telegram báo lỗi parse entity.
    parts = [f"Chat ID: {cid}"]
    if user and user.username:
        parts.append(f"Username: @{user.username}")
    if chat_type == "private":
        parts.append(
            "Trong chat riêng, số trên là ID của bạn — dán vào cột telegram_ID (text) "
            "khi thêm dòng trong bảng user trên Supabase."
        )
    else:
        parts.append(
            "Đây là ID nhóm/kênh. Lịch cá nhân thường cấu hình bằng Chat ID lấy từ tin nhắn riêng với bot."
        )
    await update.message.reply_text("\n".join(parts))


async def cmd_think(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bật/tắt chế độ thinking: AI suy nghĩ từng bước trước khi trả lời."""
    chat_id = update.effective_chat.id
    current = user_thinking.get(chat_id, False)
    user_thinking[chat_id] = not current
    new_state = user_thinking[chat_id]
    if new_state:
        await update.message.reply_text(
            "Đã bật chế độ **Suy nghĩ** (reasoning).\n"
            "Từ giờ mỗi khi bạn chat, AI sẽ trình bày phần suy luận trước, rồi mới đưa ra câu trả lời.\n"
            "Gõ /think lần nữa để tắt.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "Đã tắt chế độ Suy nghĩ. Chat sẽ trả lời trực tiếp.\nGõ /think để bật lại."
        )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    base = OPENAI_BASE_URL or "api.openai.com"
    await update.message.reply_text(f"Model: {AI_MODEL}\nBase URL: {base}")


async def cmd_lich(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tra lịch hôm nay/ngày mai và nhờ AI tóm tắt trước khi trả lời."""
    day_raw = " ".join(context.args or []).strip() if context.args else "nay"
    await answer_calendar_question(update, day_raw)


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
    emb_client = get_embedding_client()
    if not emb_client:
        await update.message.reply_text(
            "Chưa cấu hình OPENAI_EMBEDDING_API_KEY trong .env. "
            "RAG cần API key OpenAI riêng để embed tài liệu (xem hướng dẫn trong .env.example)."
        )
        return
    await update.message.reply_text("Đang quét Storage, tạo embedding và lưu... Vui lòng đợi.")
    try:
        total, msg = rag_index_storage(sb, SUPABASE_RAG_BUCKET, embedding_client=emb_client)
        await update.message.reply_text(msg)
    except Exception as e:
        logger.exception("rag_index: %s", e)
        await update.message.reply_text(f"Lỗi: {str(e)}. Kiểm tra bucket '{SUPABASE_RAG_BUCKET}', bảng '{SUPABASE_RAG_TABLE}' và QUERY_SETUP.sql (pgvector, search_rag_by_embedding).")


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trả lời câu hỏi dựa trên tài liệu đã index (RAG). Ưu tiên tìm theo embedding."""
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
        chat_client = get_openai_client()
        emb_client = get_embedding_client()
        chunks: List[dict] = []

        if emb_client:
            # Tìm theo embedding (chuẩn)
            q_emb = get_embeddings(emb_client, [user_question], batch_size=1)
            if q_emb and q_emb[0]:
                chunks = rag_vector_search(sb, q_emb[0], top_k=RAG_TOP_K)
                logger.info("RAG vector search: %s chunks", len(chunks))
        if not chunks and chat_client:
            # Fallback: tìm theo từ khóa
            keywords = extract_keywords_from_question(chat_client, user_question)
            chunks = rag_keyword_search(sb, keywords, top_k=RAG_TOP_K)
            logger.info("RAG keyword fallback: %s", keywords)

        if not chunks:
            await update.message.reply_text(
                "Không tìm thấy tài liệu liên quan. Chạy /rag_index (cần OPENAI_EMBEDDING_API_KEY) để index file trước."
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
        resp = chat_client.chat.completions.create(
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
    if chat_id in user_conversations and user_conversations[chat_id]:
        return user_conversations[chat_id][-MAX_HISTORY:]
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        res = (
            sb.table(SUPABASE_CHAT_LOG_TABLE)
            .select("direction,message_text,message_type")
            .eq("chat_id", chat_id)
            .order("created_at", desc=True)
            .limit(MAX_DB_HISTORY * 2)
            .execute()
        )
        rows = list(res.data or [])
    except Exception as e:
        logger.warning("Không đọc được lịch sử chat từ Supabase: %s", e)
        return []

    rows.reverse()
    messages: List[dict] = []
    for row in rows:
        direction = (row.get("direction") or "").strip().lower()
        content = (row.get("message_text") or "").strip()
        message_type = (row.get("message_type") or "").strip().lower()
        if not content:
            continue
        if message_type not in {"text", "caption", "other", "unknown"} and not content.startswith("["):
            continue
        if direction == "incoming":
            role = "user"
        elif direction == "outgoing":
            role = "assistant"
        else:
            continue
        messages.append({"role": role, "content": content})
    return messages[-MAX_HISTORY:]


def add_to_conversation(chat_id: int, role: str, content: str) -> None:
    if chat_id not in user_conversations:
        user_conversations[chat_id] = []
    user_conversations[chat_id].append({"role": role, "content": content})


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    if not user_text or not user_text.strip():
        return

    # Hỏi tự nhiên về lịch: "nay có lịch gì", "mai có họp không", ...
    lowered = user_text.lower()
    if "lịch" in lowered and ("nay" in lowered or "hôm nay" in lowered or "hom nay" in lowered):
        handled = await answer_calendar_question(update, "nay", custom_question=user_text.strip())
        if handled:
            return
    if "lịch" in lowered and ("mai" in lowered or "ngày mai" in lowered or "ngay mai" in lowered):
        handled = await answer_calendar_question(update, "mai", custom_question=user_text.strip())
        if handled:
            return

    await update.message.chat.send_action("typing")

    try:
        client = get_openai_client()
        history = get_messages_for_user(chat_id)
        thinking_on = user_thinking.get(chat_id, False)
        if thinking_on:
            system = (
                "Bạn là trợ lý hữu ích. Khi trả lời, LUÔN làm theo đúng format sau:\n\n"
                "**Suy nghĩ:**\n(Trình bày từng bước suy luận, phân tích câu hỏi, cân nhắc các khả năng trước khi kết luận. Dùng tiếng Việt.)\n\n"
                "**Trả lời:**\n(Câu trả lời ngắn gọn, rõ ràng dựa trên phần suy nghĩ trên.)\n\n"
                "Nếu câu hỏi đơn giản thì phần Suy nghĩ có thể ngắn, nhưng luôn có đủ hai phần."
            )
        else:
            system = "Bạn là trợ lý hữu ích. Trả lời ngắn gọn, rõ ràng."
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        if not history or history[-1].get("role") != "user" or history[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
        )
        reply = response.choices[0].message.content

        add_to_conversation(chat_id, "user", user_text)
        add_to_conversation(chat_id, "assistant", reply)

        use_md = thinking_on and len(reply) <= 4000
        if len(reply) > 4000:
            for i in range(0, len(reply), 4000):
                await update.message.reply_text(reply[i : i + 4000])
        else:
            await update.message.reply_text(
                reply, parse_mode="Markdown" if use_md else None
            )

    except Exception as e:
        logger.exception("Lỗi khi gọi AI: %s", e)
        await update.message.reply_text(
            f"Có lỗi khi gọi AI: {str(e)}\nKiểm tra API key và .env."
        )


# ======================= NHẮC LỊCH GOOGLE (hàng ngày) =======================


def gcalendar_ready() -> bool:
    """Cần file JSON service account (Workspace), hoặc OAuth client + refresh token từng user trong DB."""
    if not fetch_calendar_events_for_day or not format_day_schedule:
        return False
    sa = bool(GOOGLE_SERVICE_ACCOUNT_JSON)
    oauth = bool(GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET)
    return sa or oauth


def parse_telegram_chat_id(raw: Any) -> Optional[int]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


async def daily_calendar_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue: gửi lịch Google Calendar trong ngày cho từng user có telegram_ID + useremail."""
    if not gcalendar_ready():
        logger.warning(
            "Nhắc lịch: thiếu google_calendar hoặc cấu hình Google (GOOGLE_SERVICE_ACCOUNT_JSON hoặc OAuth client)."
        )
        return
    sb = get_supabase_client()
    if not sb:
        logger.warning("Nhắc lịch: chưa cấu hình Supabase.")
        return

    try:
        res = sb.table(SUPABASE_USER_TABLE).select("*").execute()
        rows = res.data or []
    except Exception as e:
        logger.exception("Nhắc lịch: đọc bảng %s: %s", SUPABASE_USER_TABLE, e)
        return

    tz = ZoneInfo(GCALENDAR_TZ)
    today_local = datetime.now(tz).date()
    bot = context.application.bot

    for row in rows:
        email = (row.get("useremail") or "").strip()
        chat_id = parse_telegram_chat_id(row.get("telegram_ID"))
        refresh = (row.get("gcal_refresh_token") or "").strip() or None
        if not chat_id:
            continue
        if refresh:
            pass
        elif GOOGLE_SERVICE_ACCOUNT_JSON and email:
            pass
        else:
            continue

        try:
            events, err = await run_blocking(
                fetch_calendar_events_for_day,
                email,
                today_local,
                GCALENDAR_TZ,
                refresh,
            )
        except Exception as e:
            logger.exception("Nhắc lịch fetch_calendar %s: %s", chat_id, e)
            err = str(e)
            events = None
        if err:
            who = email or (f"gcal_refresh…{refresh[:6]}" if refresh else "?")
            logger.warning("Lịch %s: %s", who, err)
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "Không lấy được lịch Google Calendar hôm nay. "
                        "Kiểm tra: Workspace + service account ủy quyền domain, hoặc cột gcal_refresh_token + OAuth."
                    ),
                )
            except Exception as se:
                logger.warning("Gửi lỗi lịch tới %s: %s", chat_id, se)
            continue

        assert format_day_schedule is not None
        text = format_day_schedule(events or [], today_local, GCALENDAR_TZ)
        name = (row.get("Username") or "").strip()
        if name:
            text = f"Chào {name},\n\n{text}"
        try:
            if len(text) > 4000:
                for i in range(0, len(text), 4000):
                    await bot.send_message(chat_id=chat_id, text=text[i : i + 4000])
            else:
                await bot.send_message(chat_id=chat_id, text=text)
        except Exception as se:
            logger.warning("Gửi lịch tới %s: %s", chat_id, se)


async def post_init_schedule(application: Application) -> None:
    """Lên lịch 7h sáng (có thể đổi bằng DAILY_CALENDAR_HOUR/MINUTE, GCALENDAR_TZ)."""
    jq = application.job_queue
    if not jq:
        logger.warning(
            "Không có JobQueue. Cài: pip install 'python-telegram-bot[job-queue]' (đã khai trong requirements.txt)."
        )
        return
    if not gcalendar_ready():
        logger.info("Nhắc lịch Google: tắt (thiếu GOOGLE_* / google_calendar).")
        return
    tz = ZoneInfo(GCALENDAR_TZ)
    when = dtime(hour=DAILY_CALENDAR_HOUR, minute=DAILY_CALENDAR_MINUTE, tzinfo=tz)
    jq.run_daily(daily_calendar_reminder, time=when, name="daily_gcalendar")
    logger.info(
        "Đã bật nhắc lịch hàng ngày lúc %02d:%02d (%s).",
        DAILY_CALENDAR_HOUR,
        DAILY_CALENDAR_MINUTE,
        GCALENDAR_TZ,
    )


# ======================= MAIN =======================

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Thiếu TELEGRAM_BOT_TOKEN trong .env")
    if not OPENAI_API_KEY:
        raise SystemExit("Thiếu OPENAI_API_KEY trong .env")

    Bot.send_message = patched_send_message

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init_schedule)
        .build()
    )
    app.add_handler(TypeHandler(Update, capture_incoming_update), group=-1)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("lich", cmd_lich))
    app.add_handler(CommandHandler("think", cmd_think))
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
