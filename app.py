import io
import json
import math
from datetime import datetime, timezone
from dateutil import tz
from collections import defaultdict

import pandas as pd
import streamlit as st

# ====== CONFIG ======
st.set_page_config(page_title="AI analysis TTG", layout="wide")
LOCAL_TZ = tz.gettz("Asia/Ho_Chi_Minh")

# ====== UTILS ======
def to_dt(ts):
    """Robust timestamp to datetime (local timezone)."""
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    try:
        # ts is usually in seconds; sometimes string
        ts = float(ts)
        # If someone accidentally passes ms, heuristically downscale
        if ts > 1e12:  # ms
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(LOCAL_TZ)
    except Exception:
        return None

def safe_get_text(msg):
    """Extract text content from ChatGPT export message blocks."""
    if not msg:
        return ""
    # v1: content could be {"content_type":"text","parts":[...]} (old)
    c = msg.get("content")
    if isinstance(c, dict):
        parts = c.get("parts")
        if isinstance(parts, list):
            return "\n".join([str(p) for p in parts if p is not None])
        # newer: content is {"content_type":"text","text": "..."} or tool calls
        if "text" in c and isinstance(c["text"], str):
            return c["text"]
    # v2: some exports put text directly at msg["content"] (string)
    if isinstance(c, str):
        return c
    return ""

def parse_conversation_json(blob, account_label=None):
    """
    Accepts either a dict loaded from conversations.json or a single conversation dict.
    Returns a flat dataframe with columns:
    account, conversation_id, conversation_title, message_id, role, author, text, create_time
    """
    rows = []
    data = blob

    # Some exports wrap in {"conversations": [...]}
    conversations = None
    if isinstance(data, dict) and "conversations" in data and isinstance(data["conversations"], list):
        conversations = data["conversations"]
    else:
        # Might be a single conversation or a top-level dict with many fields
        if isinstance(data, list):
            conversations = data
        else:
            # single conversation?
            conversations = [data]

    for conv in conversations or []:
        conv_id = conv.get("id") or conv.get("conversation_id")
        title = conv.get("title") or conv.get("conversation_title") or ""

        # Newer export: conv["mapping"] (graph of messages)
        if isinstance(conv.get("mapping"), dict):
            for node_id, node in conv["mapping"].items():
                msg = node.get("message") or {}
                if not msg:
                    continue
                role = (msg.get("author") or {}).get("role")
                author_name = (msg.get("author") or {}).get("name")
                text = safe_get_text(msg)
                ts = msg.get("create_time")
                dt = to_dt(ts)

                if role and text:
                    rows.append({
                        "account": account_label,
                        "conversation_id": conv_id,
                        "conversation_title": title,
                        "message_id": msg.get("id") or node_id,
                        "role": role,
                        "author": author_name,
                        "text": text,
                        "create_time": dt
                    })

        # Newer-ish export style: conv["messages"] = [...]
        elif isinstance(conv.get("messages"), list):
            for m in conv["messages"]:
                role = (m.get("author") or {}).get("role") or m.get("role")
                author_name = (m.get("author") or {}).get("name")
                text = safe_get_text(m)
                ts = m.get("create_time") or m.get("timestamp")  # sometimes "timestamp"
                dt = to_dt(ts)
                rows.append({
                    "account": account_label,
                    "conversation_id": conv_id,
                    "conversation_title": title,
                    "message_id": m.get("id"),
                    "role": role,
                    "author": author_name,
                    "text": text,
                    "create_time": dt
                })
        else:
            # Fallback: try best-effort common fields
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = df["create_time"].dt.date
        df["month"] = df["create_time"].dt.to_period("M").astype(str)
    return df

def month_range(year: int, month: int):
    start = datetime(year, month, 1, tzinfo=LOCAL_TZ)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=LOCAL_TZ)
    else:
        end = datetime(year, month + 1, 1, tzinfo=LOCAL_TZ)
    return start, end

def filter_month(df, year, month):
    if df.empty:
        return df
    start, end = month_range(year, month)
    return df[(df["create_time"] >= start) & (df["create_time"] < end)].copy()

# ====== PROMPT QUALITY HEURISTICS ======
def score_prompt_quality(text: str):
    """
    Lightweight heuristic: 0-100.
    Criteria:
      - length (tokens ~ words): too short -> low; concise but structured -> good
      - has context markers (background, constraints, examples, role)
      - clear ask verbs (write, summarize, generate, translate, explain, list, compare…)
      - has evaluation/acceptance criteria (format, length caps, style)
    """
    if not text:
        return 0
    t = text.strip()
    words = len(t.split())
    score = 0

    # length
    if words < 5:
        score += 5
    elif words < 15:
        score += 20
    elif words < 60:
        score += 35
    elif words < 200:
        score += 30
    else:
        score += 25  # very long can be noisy, but still ok

    # context signals
    keywords_context = ["bối cảnh", "context", "dữ liệu", "input", "tham số", "giả định", "assume", "role:", "giả sử","bạn là"]
    if any(k.lower() in t.lower() for k in keywords_context):
        score += 15

    # constraints / format
    keywords_constraints = ["định dạng", "format", "tiêu đề", "mục", "bullet", "json", "bảng", "số từ", "ký tự", "tone"]
    if any(k.lower() in t.lower() for k in keywords_constraints):
        score += 15

    # clear ask verbs
    verbs = ["viết", "tóm tắt", "dịch", "giải thích", "liệt kê", "so sánh", "phân tích", "đưa ra", "tạo", "generate", "summarize", "translate", "explain", "viết code"]
    if any(v.lower() in t.lower() for v in verbs):
        score += 15

    # examples
    if "ví dụ" in t.lower() or "example" in t.lower() or "mẫu" in t.lower():
        score += 10

    # domain terms hint (often better prompts)
    domain_hints = ["hợp đồng", "báo cáo","phân tích","mã code", "fix code", "biên bản", "đấu thầu", "ngân sách", "marketing", "CRM", "quy trình", "SOP"]
    if any(d.lower() in t.lower() for d in domain_hints):
        score += 10

    return int(max(0, min(100, score)))

# ====== TOPIC TAGGING (EDITABLE RULES) ======
DEFAULT_TOPIC_RULES = {
    "Kinh doanh": ["doanh thu", "doanh số", "P&L", "chốt deal", "CRM", "báo cáo kinh doanh"," nhà cung cấp"],
    "Nội dung/Kế hoạch": ["quảng cáo", "chiến dịch", "meta ads", "google ads", "seo", "content", "nội dung", "thương hiệu", "branding", "viết", "văn bản"],
    "Tuyển dụng/HR": ["jd", "tuyển", "phỏng vấn", "chế độ", "lương", "onboarding", "OKR", "đánh giá", "performance", "kpi", "phúc lợi", "benefits", "hợp đồng lao động", "hợp đồng thử việc", "hợp đồng chính thức", "hợp đồng lao động thời vụ", "hợp đồng cộng tác viên", "hợp đồng khoán việc","ứng viên", "candidate", "recruitment", "hồ sơ ứng viên", "job description", "employee handbook"],
    "Pháp lý": ["hợp đồng","nhà thầu","chủ đầu tư","gói thầu", "bên a", "bên b" ,"phụ lục", "điều khoản", "pháp lý", "luật", "tuân thủ", "compliance", "kiện","tranh chấp", "thỏa thuận", "agreement", "legal","khoản","điều","nghị định", "nghị quyết", ],
    "Tài chính/Kế toán": ["hóa đơn", "ngân sách", "bút toán", "báo cáo tài chính", "thuế", "vat","invoice", "budget", "accounting", "financial report", "tax", "kiểm toán", "audit", "financial statement", "balance sheet", "profit and loss", "cash flow"],
    "Vận hành": ["quy trình", "SOP", "quy định", "quản trị", "biểu mẫu", "quản lý xây dựng"],
    "Học thuật": ["học thuật","so sánh" ,"check lại","nghiên cứu","bài báo", "tài liệu học thuật", "academic", "research", "paper","xử lý", "thế nào","lesson plan", "công thức", "cách tính"],
    "Tra cứu": ["trích","tra cứu","viết tắt","bao nhiêu","vậy còn","làm sao để " ,"là gì", "là sao", "tại sao" ,"đọc file","tìm kiếm", "lookup", "thông tin", "data", "cơ sở dữ liệu", "database", "tài liệu", "hướng dẫn", "là gì?"],
    "Dịch thuật": ["dịch","dich", "dich sang","chuyển ngữ", "Chuyển sang song ngữ Anh - Việt", "translate", "phiên dịch", "bản dịch", "ngôn ngữ", "tiếng anh", "tiếng việt"],
    "CSKH": ["hỗ trợ", "khách hàng", "phản hồi", "ticket", "trợ giúp", "giải đáp"],
    "Xây dựng": ["thi công", "công trình", "dự án", "xây dựng", "kiến trúc","bê tông", "kỹ sư", "giám sát", "thiết kế kiến trúc", "construction", "project management", "architectural design", "engineering"],
    "Báo cáo": ["báo cáo", "report", "thống kê", "phân tích", "dashboard", "biểu đồ", "chart", "data analysis"],
    "Hỗ trợ Thiết kế": ["thiết kế", "đồ họa", "sketchup", "autocad", "bản vẽ","mockup", "prototype", "edit", "viền mỏng", "viền dày", "màu sắc", "font chữ", "logo", "branding", "design", "graphic design", "sketchup", "autocad", "photoshop", "illustrator","mm", "cm", "inch", "pixel", "dpi", "resolution", "vector", "raster", "layout", "composition", "typography"],
    "Email": ["email", "thư điện tử", "gửi mail", "trả lời email", "hộp thư", "outlook", "gmail"],
    "QLHC" : ["mua", "chỗ mua", "tìm mua", "ncc", "thiết bị", "vật tư", "văn phòng phẩm", "đặt mua", "đặt hàng", "nhà cung cấp", "vendor", "procurement", "supply", "supply chain", "NCC"],
    "Kỹ thuật/Phần mềm/IT": ["api","sơ đồ","giả lập","giao diện","học sâu","học máy","chức năng","window","file","code", "router","nguồn","excel", "word", "pdf", "sharepoint", "database", "script", "server", "deploy", "bug", "log", "python", "sql","in ấn", "sửa lỗi", "sửa chữa", "bảo trì", "cài đặt", "hệ thống", "mạng", "network","AI HUb", "chatgpt", "openai", "gpt-3.5", "gpt-4", "llm", "machine learning", "deep learning", "trí tuệ nhân tạo", "artificial intelligence", "chatbot", "model", "training", "fine-tuning", "prompt engineering", "android", "ios", "app development", "web development", "frontend", "backend", "fullstack", "javascript", "react", "vue", "angular", "nodejs", "express", "django", "flask", "ruby on rails", "kích thước", "file excel"],
}

def tag_topic(text, rules):
    t = text.lower()
    for topic, kws in rules.items():
        if any(kw.lower() in t for kw in kws):
            return topic
    return "Khác"

# ====== PURPOSE TAGGING (additional heuristics for report template) ======
# The template requires a breakdown of "Mục đích sử dụng" (purpose of use) such as
# Tra cứu, tóm tắt, viết mail, học tập,…  Because the raw data does not
# explicitly encode purpose, we define a simple heuristic mapping from
# keywords to broad purpose categories.  You can extend or adjust this
# dictionary to better fit your data.  Each entry maps a purpose name to a
# list of keywords; if any keyword appears in the prompt text, that purpose
# will be assigned.  If no keywords match, the prompt is tagged as "Khác".
PURPOSE_RULES = {
    "Tra cứu": [
        "tra cứu", "tra cuu", "lookup", "search", "tìm", "tìm kiếm", "là gì", "bao nhiêu", "tại sao", "how", "what",
    ],
    "Tóm tắt": [
        "tóm tắt", "tóm tat", "summary", "summarize", "tổng kết", "phân tách", "đúc kết", "tóm lược",
    ],
    "Viết mail": [
        "email", "mail", "thư", "gửi mail", "viết mail", "trả lời email", "reply email", "forward",
    ],
    "Học tập": [
        "học", "học tập", "learn", "study", "bài học", "lesson", "giảng", "nghiên cứu",
    ],
    "Dịch thuật": [
        "dịch", "dich", "translate", "dịch sang", "chuyển ngữ", "phiên dịch",
    ],
    "Viết code": [
        "code", "python", "java", "c++", "javascript", "program", "script", "sql", "lập trình",
    ],
    "Báo cáo": [
        "báo cáo", "report", "thống kê", "dashboard", "biểu đồ", "chart",
    ],
    "Kế hoạch": [
        "kế hoạch", "plan", "planning", "project", "dự án", "chiến dịch",
    ],
}

def tag_purpose(text: str, rules: dict = PURPOSE_RULES) -> str:
    """
    Assign a high‑level purpose category to a user prompt based on keyword
    matching.  Returns the first matching purpose name, or "Khác" if none
    match.  This is a simple heuristic and can be tuned by editing
    PURPOSE_RULES above.
    """
    if not text:
        return "Khác"
    t_lower = str(text).lower()
    for purpose, kws in rules.items():
        if any(kw.lower() in t_lower for kw in kws):
            return purpose
    return "Khác"

# ====== KPI RATING ======
def evaluate_kpi(
    prompts: int,
    active_days: int,
    unique_topics: int = 0,
    avr_messages: float = 0.0,
    avr_wordprompts: float = 0.0,
) -> tuple[str, int]:
    """
    Đánh giá mức độ sử dụng ChatGPT và trả về tên xếp hạng cùng điểm KPI.

    Thang điểm xét trên các yếu tố:
      - prompts: tổng số lượng prompts (user) của phòng ban trong kỳ.
      - active_days: số ngày sử dụng khác nhau trong kỳ.
      - unique_topics: số chủ đề khác nhau.
      - avr_messages: số tin nhắn user trung bình mỗi cuộc hội thoại.
      - avr_wordprompts: số từ trung bình của mỗi prompt.

    Kết quả trả về là cặp (xếp hạng, điểm) với các mốc 0, 25, 50, 80, 85, 90, 95, 100.

    Quy tắc sử dụng các ngưỡng sau (có thể điều chỉnh tùy theo thực tế):
      - Xuất sắc (100%): prompts ≥ 300, active_days ≥ 24, unique_topics ≥ 20,
        avr_messages ≥ 10 và avr_wordprompts ≥ 40.
      - Rất tốt (95%): prompts ≥ 250, active_days ≥ 20, unique_topics ≥ 18,
        avr_messages ≥ 8 và avr_wordprompts ≥ 35.
      - Tốt+ (90%): prompts ≥ 200, active_days ≥ 17, unique_topics ≥ 15,
        avr_messages ≥ 6 và avr_wordprompts ≥ 30.
      - Tốt (85%): prompts ≥ 170, active_days ≥ 14, unique_topics ≥ 10,
        avr_messages ≥ 5 và avr_wordprompts ≥ 25.
      - Khá (80%): prompts ≥ 150, active_days ≥ 6,
        avr_messages ≥ 4 và avr_wordprompts ≥ 20.
      - Trung bình (50%): (prompts ≥ 100 or active_days ≥ 5) and
        avr_messages ≥ 3 and avr_wordprompts ≥ 15.
      - Thấp (25%): có sử dụng nhưng không đạt các ngưỡng trên.
      - Không sử dụng (0%): không có prompt nào trong kỳ.
    """
    # Xuất sắc nhất: sử dụng gần như hàng ngày, nhiều chủ đề và nội dung dài
    if (
        prompts >= 200
        and active_days >= 25
        and unique_topics >= 17
        and avr_messages >= 10
        and avr_wordprompts >= 40
    ):
        return "Xuất sắc", 100
    # Rất tốt: dùng nhiều, đa dạng với nội dung khá dài
    if (
        prompts >= 150
        and active_days >= 22
        and unique_topics >= 14
        and avr_messages >= 8
        and avr_wordprompts >= 35
    ):
        return "Rất tốt", 95
    # Tốt+: dùng thường xuyên, đa dạng vừa phải
    if (
        prompts >= 120
        and active_days >= 16
        and unique_topics >= 12
        and avr_messages >= 6
        and avr_wordprompts >= 30
    ):
        return "Tốt+", 90
    # Tốt: đáp ứng yêu cầu “Tốt”
    if (
        prompts >= 100
        and active_days >= 14
        and unique_topics >= 10
        and avr_messages >= 5
        and avr_wordprompts >= 25
    ):
        return "Tốt", 85
    # Khá: sử dụng đủ thường xuyên, nội dung tương đối
    if (
        prompts >= 100
        and active_days >= 12
        and avr_messages >= 4
        and avr_wordprompts >= 20
    ):
        return "Khá", 80
    # Trung bình: có sử dụng nhưng chưa đều hoặc nội dung ngắn
    if (
        (prompts >= 100 or active_days >= 5)
        and avr_messages >= 3
        and avr_wordprompts >= 15
    ):
        return "Trung bình", 50
    # Thấp: rất ít dùng hoặc nội dung quá ngắn
    if prompts > 0:
        return "Thấp", 25
    # Không sử dụng
    return "Không sử dụng", 0


# ====== OPTIONAL: CLUSTERING (if sklearn available) ======
def try_cluster(df_user_prompts, n_clusters=15):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        texts = df_user_prompts["text"].fillna("").tolist()
        if len(texts) < n_clusters:
            return None
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X = vec.fit_transform(texts)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        return labels
    except Exception:
        return None

# ====== UI ======
st.title("📊 ChatGPT Usage Analyzer for TTG ")

st.sidebar.subheader("Tải dữ liệu")
uploaded_files = st.sidebar.file_uploader(
    "Kéo thả nhiều file conversations.json (mỗi tài khoản/phòng ban một file)",
    type=["json"], accept_multiple_files=True
)

dept_map_file = st.sidebar.file_uploader(
    "Tùy chọn: CSV map tài khoản ➜ Phòng ban (cột: account, department)",
    type=["csv"]
)

col1, col2 = st.sidebar.columns(2)
with col1:
    year = st.number_input("Năm", min_value=2023, max_value=2100, value=datetime.now().year, step=1)
with col2:
    month = st.number_input("Tháng", min_value=1, max_value=12, value=datetime.now().month, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Cài đặt phân tích")
use_clustering = st.sidebar.checkbox("Bật clustering chủ đề (nếu có scikit‑learn)", value=False)

# Editable rules
st.sidebar.markdown("**Rules phân loại chủ đề (có thể chỉnh):**")
rules_text = st.sidebar.text_area(
    "Dạng JSON (topic -> list từ khóa)",
    value=json.dumps(DEFAULT_TOPIC_RULES, ensure_ascii=False, indent=2),
    height=220
)
try:
    topic_rules = json.loads(rules_text)
except Exception:
    st.sidebar.error("Rules JSON không hợp lệ. Dùng mặc định.")
    topic_rules = DEFAULT_TOPIC_RULES

# ====== PARSE & COMBINE ======
if uploaded_files:
    frames = []
    for f in uploaded_files:
        try:
            content = json.load(f)
            # Derive a base name from the filename (without extension).  This name
            # will be used as both the account label and the default department
            # name.  Administrators can rename the JSON file to reflect the
            # actual department before uploading.
            base_name = f.name.rsplit(".", 1)[0]
            df = parse_conversation_json(content, account_label=base_name)
            if not df.empty:
                df["department"] = base_name
            frames.append(df)
        except Exception as e:
            st.error(f"Không đọc được {f.name}: {e}")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # Filter to selected month
        month_df = filter_month(all_df, int(year), int(month))

        # ====== INPUT NUMBER OF STAFF PER DEPARTMENT ======
        # Before performing aggregations, allow the user to specify how many
        # employees are in each department.  This helps compute average
        # prompts per user for fair KPI scoring.
        dept_staff_counts = {}
        department_options = sorted([d for d in month_df["department"].dropna().unique()])
        if department_options:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Số lượng nhân sự theo phòng ban")
            for dep in department_options:
                # Default to 1 employee if not specified
                dept_staff_counts[dep] = st.sidebar.number_input(
                    f"{dep}", min_value=1, value=1, step=1, key=f"staff_{dep}"
                )

        # Load department map if given.  The CSV must contain two columns: account and
        # department.  If provided, the department values from the CSV will
        # override the automatically derived department names (taken from the
        # filename).  Rows without a mapping will retain their original
        # department value.  If the CSV cannot be read, we log an error and
        # leave the existing department values untouched.
        if dept_map_file is not None:
            try:
                map_df = pd.read_csv(dept_map_file)
                # Normalize column names to lowercase for matching
                map_df.columns = [c.lower() for c in map_df.columns]
                if not {"account", "department"}.issubset(set(map_df.columns)):
                    raise ValueError("CSV cần 2 cột: account, department")
                # Rename the department column to avoid clobbering the existing one during merge
                map_df = map_df.rename(columns={"department": "department_override"})
                month_df = month_df.merge(map_df[["account", "department_override"]], on="account", how="left")
                # Use the override department where available; otherwise keep the original
                month_df["department"] = month_df["department_override"].combine_first(month_df["department"])
                month_df.drop(columns=["department_override"], inplace=True)
            except Exception as e:
                st.error(f"Không đọc được map CSV: {e}")
                # Keep existing department values
                pass

        # Basic derived fields
        month_df["is_user"] = month_df["role"].eq("user")
        month_df["is_assistant"] = month_df["role"].eq("assistant")

        # Count per conversation
        conv_stats = month_df.groupby(["account","department","conversation_id","conversation_title"], dropna=False).agg(
            first_time=("create_time","min"),
            last_time=("create_time","max"),
            user_prompts=("is_user","sum"),
            assistant_msgs=("is_assistant","sum"),
            total_msgs=("message_id","count"),
            active_days=("date", lambda s: s.nunique())
        ).reset_index()

        # Topic tagging, purpose tagging & prompt quality on user prompts
        user_prompts_df = month_df[month_df["is_user"]].copy()
        user_prompts_df["topic"] = user_prompts_df["text"].apply(lambda t: tag_topic(t, topic_rules))
        user_prompts_df["purpose"] = user_prompts_df["text"].apply(lambda t: tag_purpose(t))
        user_prompts_df["prompt_quality"] = user_prompts_df["text"].apply(score_prompt_quality)
        # compute word count for summary metrics
        user_prompts_df["word_count"] = user_prompts_df["text"].apply(lambda t: len(str(t).split()) if pd.notna(t) else 0)

        # Try clustering
        if use_clustering:
            labels = try_cluster(user_prompts_df)
            if labels is not None:
                user_prompts_df["cluster"] = labels
            else:
                user_prompts_df["cluster"] = None
        else:
            user_prompts_df["cluster"] = None

        # ====== OVERVIEW ======
        st.header("Tổng quan")
        total_convs = conv_stats["conversation_id"].nunique()
        total_prompts = int(user_prompts_df.shape[0])
        total_msgs = int(month_df.shape[0])
        dep_count = month_df["department"].nunique() if month_df["department"].notna().any() else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Số cuộc hội thoại", f"{total_convs:,}")
        kpi2.metric("Tổng số prompts (user)", f"{total_prompts:,}")
        kpi3.metric("Tổng số message", f"{total_msgs:,}")
        kpi4.metric("Số phòng ban", f"{dep_count if dep_count else '-'}")

        # Aggregations
        st.subheader("Theo phòng ban")
        by_dep = conv_stats.groupby("department", dropna=False).agg(
            conversations=("conversation_id","nunique"),
            user_prompts=("user_prompts","sum"),
            assistant_msgs=("assistant_msgs","sum"),
            total_msgs=("total_msgs","sum")
        ).reset_index().rename(columns={"department":"Phòng ban"})
        st.dataframe(by_dep)

        st.subheader("Chủ đề (dựa trên rules)")
        by_topic = user_prompts_df.groupby(["department","topic"], dropna=False).size().reset_index(name="prompts")
        st.dataframe(by_topic)

        # The detailed statistics of prompt quality (mean/median/min/max) have been
        # removed per user request.  Prompt quality is now reflected in the KPI
        # compliance section below.

        # ====== SUMMARY DATA FOR EXCEL REPORT ======
        # Prepare a per‑department summary with metrics required by the template
        report_month_str = f"{int(month):02d}-{int(year)}"
        dept_summary_rows = []
        all_departments = sorted([d for d in month_df["department"].dropna().unique()])
        for idx, dep in enumerate(all_departments, start=1):
            convs = conv_stats[conv_stats["department"] == dep]
            prompts_dep = user_prompts_df[user_prompts_df["department"] == dep]
            if convs.empty:
                continue
            # Số cuộc hội thoại
            conv_count = convs["conversation_id"].nunique()
            # Số tin nhắn user trung bình mỗi cuộc hội thoại (avr_messages)
            avr_messages = float(convs["user_prompts"].mean()) if not convs.empty else 0.0
            # Số ngày hoạt động trong kỳ
            active_days = month_df[month_df["department"] == dep]["date"].nunique()
            # Độ dài trung bình câu prompt (số từ)
            avr_wordprompts = float(prompts_dep["word_count"].mean()) if not prompts_dep.empty else 0.0
            # Phân bổ mục đích sử dụng
            purpose_counts = prompts_dep["purpose"].value_counts()
            total_p = purpose_counts.sum()
            purpose_lines = []
            if total_p > 0:
                for p_name, count in purpose_counts.items():
                    pct = count / total_p
                    purpose_lines.append(f"+ {p_name} ({pct:.0%})")
            purpose_str = "\n".join(purpose_lines)
            # Phân bổ chủ đề
            topic_counts = prompts_dep["topic"].value_counts()
            total_t = topic_counts.sum()
            topic_lines = []
            if total_t > 0:
                for t_name, count in topic_counts.items():
                    pct = count / total_t
                    topic_lines.append(f"+ {t_name} ({pct:.0%})")
            topic_str = "\n".join(topic_lines)
            # Tổng số prompt trong phòng ban
            prompts_count = len(prompts_dep)
            # Số chủ đề khác nhau
            unique_topics_count = int(prompts_dep["topic"].nunique()) if not prompts_dep.empty else 0
            # Số nhân sự do người dùng cấu hình cho phòng ban này; mặc định 1
            user_num = dept_staff_counts.get(dep, 1)
            # Đánh giá KPI thô dựa trên tổng số prompt, ngày hoạt động, chủ đề, số tin nhắn và độ dài prompt
            rating_name, raw_score = evaluate_kpi(
                prompts_count,
                active_days,
                unique_topics_count,
                avr_messages,
                avr_wordprompts,
            )
            # Tính điểm cuối cùng chia cho số nhân sự để công bằng giữa các phòng ban
            final_score = 0
            if user_num > 0:
                final_score = int(round(raw_score / user_num))
            dept_summary_rows.append({
                "Tiêu đề": dep,
                "#": idx,
                "Tháng": report_month_str,
                "Active Days (số ngày có sử dụng)": active_days,
                "Số lượng hội thoại": conv_count,
                # Use average messages per conversation for this metric
                "Số lượng tin nhắn trung bình trong mỗi hội thoại": avr_messages,
                # Use average word count per prompt
                "Độ dài trung bình câu prompt (từ)": avr_wordprompts,
                "Mục đích sử dụng (Tra cứu, tóm tắt, Viết mail, học tập,...) kèm %": purpose_str,
                "Chủ đề (Tuyển dụng, Thuế, ...) kèm %": topic_str,
                "Đánh giá chất lượng theo A3 (Xuất sắc, Rất tốt, Tốt+, Tốt, Khá, Trung bình, Thấp, Không sử dụng)": rating_name,
                "Điểm KPIs %": final_score,
            })
        dept_summary_df = pd.DataFrame(dept_summary_rows)

        # ====== DETAIL VIEWS BY DEPARTMENT ======
        st.header("Chi tiết theo phòng ban")
        departments = sorted([d for d in month_df["department"].dropna().unique()])
        if not departments:
            st.info("Không có phòng ban nào được tìm thấy trong dữ liệu.")
        else:
            selected_dep = st.selectbox("Chọn phòng ban", options=departments, index=0)
            # Filter dataframes for selected department
            conv_stats_dep = conv_stats[conv_stats["department"] == selected_dep]
            user_prompts_dep = user_prompts_df[user_prompts_df["department"] == selected_dep]
            assistant_dep = month_df[(month_df["department"] == selected_dep) & (month_df["is_assistant"])]
            raw_dep = month_df[month_df["department"] == selected_dep]
            # Tabs for detailed views
            tab_a, tab_b, tab_c, tab_d = st.tabs([
                "Cuộc hội thoại", "Prompts (user)", "Assistant trả lời", "Dữ liệu gốc"
            ])
            with tab_a:
                st.subheader(f"Danh sách hội thoại của phòng ban {selected_dep}")
                st.dataframe(
                    conv_stats_dep.sort_values("first_time").rename(
                        columns={
                            "conversation_id": "ID hội thoại",
                            "conversation_title": "Tiêu đề hội thoại",
                            "first_time": "Bắt đầu",
                            "last_time": "Kết thúc",
                            "user_prompts": "Số prompt",
                            "assistant_msgs": "Số trả lời",
                            "total_msgs": "Tổng tin nhắn",
                            "active_days": "Số ngày hoạt động",
                        }
                    )
                )
                if not conv_stats_dep.empty:
                    conv_choices = conv_stats_dep[["conversation_id", "conversation_title"]].apply(
                        lambda r: f"{r['conversation_id']} – {r['conversation_title']}", axis=1
                    ).tolist()
                    selected_conv_label = st.selectbox(
                        "Chọn hội thoại để xem chi tiết", options=conv_choices
                    )
                    selected_conv_id = selected_conv_label.split("–", 1)[0].strip()
                    if st.button("Xem chi tiết"):
                        show_in_modal = hasattr(st, "modal")
                        container_ctx = st.modal if show_in_modal else st.expander
                        with container_ctx(f"Chi tiết hội thoại: {selected_conv_label}"):
                            conv_msgs = raw_dep[(raw_dep["conversation_id"].astype(str) == selected_conv_id)].sort_values("create_time")
                            if conv_msgs.empty:
                                st.info("Không có dữ liệu cho hội thoại này.")
                            else:
                                for _, row in conv_msgs.iterrows():
                                    ts = row["create_time"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["create_time"]) else ""
                                    if row["role"] == "user":
                                        st.markdown(f"**User ({ts})**: {row['text']}")
                                    else:
                                        st.markdown(f"**Assistant ({ts})**: {row['text']}")
            with tab_b:
                st.subheader(f"Danh sách prompt của phòng ban {selected_dep}")
                show_cols = [
                    "create_time", "conversation_title", "text", "topic", "purpose", "prompt_quality", "word_count", "cluster"
                ]
                st.dataframe(
                    user_prompts_dep[show_cols]
                    .sort_values("create_time")
                    .rename(
                        columns={
                            "create_time": "Thời gian",
                            "conversation_title": "Tiêu đề hội thoại",
                            "text": "Nội dung prompt",
                            "topic": "Chủ đề",
                            "purpose": "Mục đích",
                            "prompt_quality": "Điểm chất lượng",
                            "word_count": "Số từ",
                            "cluster": "Nhóm",
                        }
                    )
                )
            with tab_c:
                st.subheader(f"Câu trả lời của Assistant – phòng ban {selected_dep}")
                st.dataframe(
                    assistant_dep[["create_time", "conversation_title", "text"]]
                    .sort_values("create_time")
                    .rename(
                        columns={
                            "create_time": "Thời gian",
                            "conversation_title": "Tiêu đề hội thoại",
                            "text": "Nội dung",
                        }
                    )
                )
            with tab_d:
                st.subheader(f"Dữ liệu gốc – phòng ban {selected_dep}")
                st.dataframe(raw_dep.sort_values("create_time"))

        # ====== KPI COMPLIANCE OVERVIEW ======
        st.subheader("Tuân thủ quy định về ChatGPT")
        st.markdown(
            """
            **Hướng dẫn chấm điểm KPIs**
            
            Hệ thống đánh giá tuân thủ ChatGPT dựa trên tổng số prompt, số ngày sử dụng, độ đa dạng chủ đề,
            số tin nhắn trung bình mỗi cuộc hội thoại và độ dài trung bình của prompt.  Điểm chia nhỏ thành các mốc sau:
            
            - **Xuất sắc (100%)**: sử dụng rất thường xuyên, nhiều chủ đề (≥20) và nội dung dài (≥10 tin nhắn mỗi cuộc, ≥40 từ mỗi prompt).
            - **Rất tốt (95%)**: sử dụng thường xuyên, đa dạng (≥18 chủ đề), nội dung khá dài (≥8 tin nhắn, ≥35 từ).
            - **Tốt+ (90%)**: sử dụng đều đặn, khá đa dạng (≥15 chủ đề), nội dung vừa đủ (≥6 tin nhắn, ≥30 từ).
            - **Tốt (85%)**: đáp ứng yêu cầu tốt với ≥170 prompt, ≥14 ngày sử dụng và ≥10 chủ đề, nội dung có chiều sâu (≥5 tin nhắn, ≥25 từ).
            - **Khá (80%)**: có mức sử dụng đáng kể (≥150 prompt, ≥6 ngày), nội dung trung bình (≥4 tin nhắn, ≥20 từ).
            - **Trung bình (50%)**: có sử dụng nhưng chưa đều (≥100 prompt hoặc ≥5 ngày) với nội dung tương đối (≥3 tin nhắn, ≥15 từ).
            - **Thấp (25%)**: có sử dụng nhưng rất ít hoặc nội dung ngắn.
            - **Không sử dụng (0%)**: không phát sinh prompt trong kỳ báo cáo.
            
            Điểm KPI cuối cùng được chia theo số lượng nhân sự trong phòng ban để đảm bảo công bằng giữa các phòng quy mô khác nhau.
            """,
            unsafe_allow_html=True,
        )
        # Display KPI rating per department
        if not dept_summary_df.empty:
            kpi_table = dept_summary_df[[
                "Tiêu đề",
                "Đánh giá chất lượng theo A3 (Xuất sắc, Rất tốt, Tốt+, Tốt, Khá, Trung bình, Thấp, Không sử dụng)",
                "Điểm KPIs %",
            ]].rename(columns={
                "Tiêu đề": "Phòng ban",
                "Đánh giá chất lượng theo A3 (Xuất sắc, Rất tốt, Tốt+, Tốt, Khá, Trung bình, Thấp, Không sử dụng)": "Đánh giá",
                "Điểm KPIs %": "KPIs (%)",
            })
            st.dataframe(kpi_table)

        # ====== EXPORTS ======
        st.header("Xuất báo cáo")
        try:
            import xlsxwriter
            buffer = io.BytesIO()
            workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
            worksheet = workbook.add_worksheet("By Dept")
            # Define formats
            header_fmt = workbook.add_format({
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "text_wrap": True,
            })
            text_fmt = workbook.add_format({
                "align": "left",
                "valign": "top",
                "border": 1,
                "text_wrap": True,
            })
            num_fmt = workbook.add_format({
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "num_format": "0.00",
            })
            int_fmt = workbook.add_format({
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "num_format": "0",
            })
            # Header labels
            headers = [
                "Tiêu đề",
                "#",
                "Tháng",
                "Active Days (số ngày có sử dụng)",
                "Số lượng hội thoại",
                "Số lượng tin nhắn trung bình trong mỗi hội thoại",
                "Độ dài trung bình câu prompt (từ)",
                "Mục đích sử dụng (Tra cứu, tóm tắt, Viết mail, học tập,...) kèm %",
                "Chủ đề (Tuyển dụng, Thuế, ...) kèm %",
                "Đánh giá chất lượng theo A3 (Xuất sắc, Rất tốt, Tốt+, Tốt, Khá, Trung bình, Thấp, Không sử dụng)",
                "Điểm KPIs %",
            ]
            for col_idx, col_name in enumerate(headers):
                worksheet.write(0, col_idx, col_name, header_fmt)
            # Column widths
            col_widths = [25, 5, 10, 20, 20, 25, 25, 40, 40, 25, 15]
            for i, width in enumerate(col_widths):
                worksheet.set_column(i, i, width)
            # Write data
            for row_idx, row_data in dept_summary_df.iterrows():
                worksheet.write(row_idx + 1, 0, row_data["Tiêu đề"], text_fmt)
                worksheet.write(row_idx + 1, 1, row_data["#"], int_fmt)
                worksheet.write(row_idx + 1, 2, row_data["Tháng"], text_fmt)
                worksheet.write(row_idx + 1, 3, row_data["Active Days (số ngày có sử dụng)"], int_fmt)
                worksheet.write(row_idx + 1, 4, row_data["Số lượng hội thoại"], int_fmt)
                worksheet.write(row_idx + 1, 5, row_data["Số lượng tin nhắn trung bình trong mỗi hội thoại"], num_fmt)
                worksheet.write(row_idx + 1, 6, row_data["Độ dài trung bình câu prompt (từ)"], num_fmt)
                worksheet.write(row_idx + 1, 7, row_data["Mục đích sử dụng (Tra cứu, tóm tắt, Viết mail, học tập,...) kèm %"], text_fmt)
                worksheet.write(row_idx + 1, 8, row_data["Chủ đề (Tuyển dụng, Thuế, ...) kèm %"], text_fmt)
                worksheet.write(row_idx + 1, 9, row_data["Đánh giá chất lượng theo A3 (Xuất sắc, Rất tốt, Tốt+, Tốt, Khá, Trung bình, Thấp, Không sử dụng)"], text_fmt)
                worksheet.write(row_idx + 1, 10, row_data["Điểm KPIs %"], int_fmt)
            workbook.close()
            buffer.seek(0)
            st.download_button(
                label="⬇️ Tải Excel báo cáo",
                data=buffer.getvalue(),
                file_name=f"ai_usage_report_{int(year)}_{int(month):02d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(f"Lỗi khi tạo file Excel: {exc}")

    else:
        st.info("Chưa có dữ liệu hợp lệ.")
else:
    st.info("Hãy tải lên một hoặc nhiều file conversations.json để bắt đầu.")
