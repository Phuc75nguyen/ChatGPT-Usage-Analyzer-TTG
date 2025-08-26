import io
import json
import math
import zipfile
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
    "Hỗ trợ Thiết kế": ["thiết kế", "đồ họa", "sketchup", "autocad", "bản vẽ", "mockup", "prototype", "edit", "viền mỏng", "viền dày", "màu sắc", "font chữ", "logo", "branding", "design", "graphic design", "sketchup", "autocad", "photoshop", "illustrator","mm", "cm", "inch", "pixel", "dpi", "resolution", "vector", "raster", "layout", "composition", "typography"],
    "Email": ["email", "thư điện tử", "gửi mail", "trả lời email", "hộp thư", "outlook", "gmail"],
    "Kỹ thuật/Phần mềm/IT": ["api","sơ đồ","giả lập","giao diện","học sâu","học máy","chức năng","window","file","code", "router","nguồn","excel", "word", "pdf", "sharepoint", "database", "script", "server", "deploy", "bug", "log", "python", "sql","in ấn", "sửa lỗi", "sửa chữa", "bảo trì", "cài đặt", "hệ thống", "mạng", "network","AI HUb", "chatgpt", "openai", "gpt-3.5", "gpt-4", "llm", "machine learning", "deep learning", "trí tuệ nhân tạo", "artificial intelligence", "chatbot", "model", "training", "fine-tuning", "prompt engineering", "android", "ios", "app development", "web development", "frontend", "backend", "fullstack", "javascript", "react", "vue", "angular", "nodejs", "express", "django", "flask", "ruby on rails", "kích thước", "file excel"],
}

def tag_topic(text, rules):
    t = text.lower()
    # Choose the topic with the highest number of matching keywords.  This helps
    # reduce the number of prompts falling into the "Khác" category by selecting
    # the most relevant rule set instead of returning the first match.
    best_topic = None
    best_count = 0
    for topic, kws in rules.items():
        # Count how many keywords from this topic appear in the text
        count = sum(1 for kw in kws if kw.lower() in t)
        if count > best_count:
            best_count = count
            best_topic = topic
    return best_topic if best_count > 0 else "Khác"

# ====== PURPOSE TAGGING (EDITABLE RULES) ======
# Similar to topic tagging, we maintain a separate dictionary of rules for
# identifying the purpose of a prompt.  These are high‑level categories
# reflecting how staff typically use ChatGPT, such as looking up information
# (Tra cứu), drafting emails (Viết mail), summarising documents (Tóm tắt),
# learning (Học tập), translation (Dịch thuật) and so on.  The aim is to
# minimise the number of prompts falling into the "Khác" category by
# providing comprehensive keyword lists.  Users can override these rules
# via the sidebar.
DEFAULT_PURPOSE_RULES = {
    "Tra cứu": [
        "tra cứu", "lookup", "tìm kiếm", "trích", "bao nhiêu", "là gì", "là sao", "tại sao", "hướng dẫn", "thông tin", "data", "cơ sở dữ liệu"
    ],
    "Tóm tắt": [
        "tóm tắt", "summary", "tóm lược", "condense", "viết ngắn", "paraphrase", "rút gọn", "gist"
    ],
    "Viết mail": [
        "email", "mail", "thư điện tử", "soạn mail", "gửi mail", "trả lời mail", "viết email", "thư", "outlook", "gmail"
    ],
    "Học tập": [
        "học", "bài học", "lesson", "nghiên cứu", "đọc", "học thuật", "tài liệu", "ngôn ngữ", "tiếng anh", "tiếng việt", "học tập", "tự học"
    ],
    "Dịch thuật": [
        "dịch", "chuyển ngữ", "dịch sang", "translate", "phiên dịch", "song ngữ", "dịch thuật"
    ],
    "Báo cáo": [
        "báo cáo", "report", "thống kê", "dashboard", "phân tích", "biểu đồ", "data analysis"
    ],
    "Lập kế hoạch": [
        "kế hoạch", "plan", "lịch trình", "timeline", "sắp xếp", "planning", "schedule"
    ],
    "Tư vấn": [
        "nên làm", "nên chọn", "tư vấn", "gợi ý", "recommend", "advice", "giải pháp", "hướng dẫn"
    ],
    "Soạn thảo văn bản": [
        "soạn thảo", "viết", "văn bản", "document", "draft", "report", "memo", "lời thoại"
    ],
    "Khác": []
}

def tag_purpose(text: str, rules: dict) -> str:
    """Assign a purpose category to a prompt based on keyword rules.

    Returns the category with the highest number of matching keywords or
    "Khác" if none match.
    """
    t = text.lower() if isinstance(text, str) else ""
    best_purpose = None
    best_count = 0
    for purpose, kws in rules.items():
        if not kws:
            continue
        count = sum(1 for kw in kws if kw.lower() in t)
        if count > best_count:
            best_count = count
            best_purpose = purpose
    return best_purpose if best_count > 0 else "Khác"


# ====== KPI RATING ======
def evaluate_kpi(prompts: int, active_days: int) -> tuple[str, int]:
    """Return a qualitative rating and KPI percentage based on usage.

    The rules below approximate the organisation's guidelines for
    assessing AI usage.  They classify usage as:
      • Xuất sắc (100%): frequent daily use with diverse content
      • Tốt (80%): regular use with several substantial contributions
      • Khá (50%): some interaction but not yet frequent or focused
      • Thấp (0%): little interaction or low application in work

    Parameters
    ----------
    prompts : int
        Number of user prompts in the period.
    active_days : int
        Number of distinct days on which ChatGPT was used.

    Returns
    -------
    tuple[str, int]
        A pair of (rating_name, kpi_score_percent).
    """
    # These heuristics can be tuned based on actual KPI guidelines
    if prompts >= 20 or active_days >= 15:
        return "Xuất sắc", 100
    elif prompts >= 10 or active_days >= 7:
        return "Tốt", 80
    elif prompts >= 5 or active_days >= 3:
        return "Khá", 50
    else:
        return "Thấp", 0

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

    # update

    # Purpose rules editable
st.sidebar.markdown("**Rules phân loại mục đích (có thể chỉnh):**")
purpose_rules_text = st.sidebar.text_area(
    "Dạng JSON (purpose -> list từ khóa)",
    value=json.dumps(DEFAULT_PURPOSE_RULES, ensure_ascii=False, indent=2),
    height=220
)
try:
    purpose_rules = json.loads(purpose_rules_text)
except Exception:
    st.sidebar.error("Rules JSON không hợp lệ. Dùng mặc định cho mục đích.")
    purpose_rules = DEFAULT_PURPOSE_RULES

# ====== PARSE & COMBINE ======
if uploaded_files:
    frames = []
    for f in uploaded_files:
        try:
            content = json.load(f)
            account_label = f.name.rsplit(".", 1)[0]  # filename as account label
            df = parse_conversation_json(content, account_label=account_label)
            frames.append(df)
        except Exception as e:
            st.error(f"Không đọc được {f.name}: {e}")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # Filter to selected month
        month_df = filter_month(all_df, int(year), int(month))

        # Load department map if given
        if dept_map_file is not None:
            try:
                map_df = pd.read_csv(dept_map_file)
                if not {"account","department"}.issubset(set(map_df.columns.str.lower())):
                    # try non-lowercase
                    need = {"account","department"}
                    if not need.issubset(set(map_df.columns)):
                        raise ValueError("CSV cần 2 cột: account, department")
                # normalize cols
                map_df.columns = [c.lower() for c in map_df.columns]
                month_df = month_df.merge(map_df[["account","department"]], on="account", how="left")
            except Exception as e:
                st.error(f"Không đọc được map CSV: {e}")
                month_df["department"] = None
        else:
            month_df["department"] = None

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
        # assign topic
        user_prompts_df["topic"] = user_prompts_df["text"].apply(lambda t: tag_topic(t, topic_rules))
        # assign purpose
        user_prompts_df["purpose"] = user_prompts_df["text"].apply(lambda t: tag_purpose(t, purpose_rules))
        # prompt quality score
        user_prompts_df["prompt_quality"] = user_prompts_df["text"].apply(score_prompt_quality)

        # Try clustering for exploratory analysis (optional)
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

        # --- Topic overview by department with interactive selection ---
        st.subheader("Chủ đề theo rules")
        # aggregate topic counts per department
        by_topic = user_prompts_df.groupby(["department","topic"], dropna=False).size().reset_index(name="prompts")
        # Provide a selection box for departments
        dept_options = ["Tất cả"] + sorted([d for d in by_topic["department"].dropna().unique()])
        selected_dep = st.selectbox("Chọn phòng ban để xem chủ đề", dept_options, index=0)
        if selected_dep == "Tất cả":
            # summarise across all departments
            agg_topic = by_topic.groupby("topic")[["prompts"]].sum().reset_index()
            agg_topic["%"] = (agg_topic["prompts"] / agg_topic["prompts"].sum() * 100).round(1)
            agg_topic = agg_topic.sort_values("prompts", ascending=False)
            st.dataframe(agg_topic.rename(columns={"prompts":"Số prompts"}))
        else:
            df_topic_dep = by_topic[by_topic["department"] == selected_dep]
            if not df_topic_dep.empty:
                df_topic_dep = df_topic_dep.sort_values("prompts", ascending=False)
                df_topic_dep["%"] = (df_topic_dep["prompts"] / df_topic_dep["prompts"].sum() * 100).round(1)
                st.dataframe(df_topic_dep[["topic","prompts","%"]].rename(columns={"topic":"Chủ đề","prompts":"Số prompts"}))
            else:
                st.info(f"Không có dữ liệu chủ đề cho phòng ban '{selected_dep}'")

        # --- KPI evaluation section ---
        st.subheader("Đánh giá chất lượng theo A3 (KPIs)")
        # compute number of active days per department based on user prompts
        active_days_by_dep = user_prompts_df.groupby("department", dropna=False)["date"].nunique().reset_index(name="active_days")
        prompt_counts_by_dep = user_prompts_df.groupby("department", dropna=False).size().reset_index(name="prompts")
        kpi_df = by_dep.merge(active_days_by_dep, left_on="Phòng ban", right_on="department", how="left")
        kpi_df = kpi_df.merge(prompt_counts_by_dep, left_on="Phòng ban", right_on="department", how="left", suffixes=("_drop","_drop2"))
        kpi_df.drop(columns=["department_drop","department_drop2"], inplace=True, errors="ignore")
        # fill NaNs with zeros
        kpi_df[["active_days","prompts"]] = kpi_df[["active_days","prompts"]].fillna(0).astype(int)
        # apply evaluation function
        ratings = kpi_df.apply(lambda row: evaluate_kpi(row["prompts"], row["active_days"]), axis=1)
        kpi_df[["Đánh giá","Điểm KPIs %"]] = pd.DataFrame(ratings.tolist(), index=kpi_df.index)
        # show selected columns
        st.dataframe(kpi_df[["Phòng ban","conversations","user_prompts","active_days","Đánh giá","Điểm KPIs %"]].rename(columns={
            "conversations":"Số cuộc hội thoại",
            "user_prompts":"Số prompts (user)"
        }))

        # ====== DETAILS TABS ======
        tab1, tab2, tab3, tab4 = st.tabs(["Cuộc hội thoại", "Prompts (user)", "Assistant trả lời", "Dữ liệu gốc"])

        with tab1:
            st.dataframe(conv_stats.sort_values("first_time"))

        with tab2:
            show_cols = ["account","department","conversation_title","create_time","text","topic","prompt_quality","cluster"]
            st.dataframe(user_prompts_df[show_cols].sort_values("create_time"))

        with tab3:
            assistant_df = month_df[month_df["is_assistant"]][["account","department","conversation_title","create_time","text"]]
            st.dataframe(assistant_df.sort_values("create_time"))

        with tab4:
            st.dataframe(month_df.sort_values("create_time"))

        # ====== EXPORTS ======
        st.header("Xuất báo cáo")
        # Build summary table matching the provided template
        summary_rows = []
        # Determine the month string (e.g., "08-2025")
        month_str = f"{int(month):02d}-{int(year)}"
        # Generate summary per department (or unknown)
        departments_list = by_dep["Phòng ban"].tolist()
        for idx, dep in enumerate(departments_list, start=1):
            # filter for this department
            df_dep_prompts = user_prompts_df[user_prompts_df["department"] == dep] if pd.notna(dep) else user_prompts_df[user_prompts_df["department"].isna()]
            # active days
            active_days = df_dep_prompts["date"].nunique() if not df_dep_prompts.empty else 0
            # number of conversations
            conv_count = int(by_dep.loc[by_dep["Phòng ban"] == dep, "conversations"].values[0])
            # average messages per conversation
            avg_msgs = None
            conv_msgs = conv_stats[conv_stats["department"] == dep] if pd.notna(dep) else conv_stats[conv_stats["department"].isna()]
            if not conv_msgs.empty:
                avg_msgs = conv_msgs["total_msgs"].mean()
                avg_msgs = round(avg_msgs, 1)
            else:
                avg_msgs = 0
            # average prompt length (words)
            if not df_dep_prompts.empty:
                word_lengths = df_dep_prompts["text"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
                avg_prompt_len = round(word_lengths.mean(), 1) if not word_lengths.empty else 0
            else:
                avg_prompt_len = 0
            # purpose distribution
            purpose_counts = df_dep_prompts["purpose"].value_counts()
            total_purpose = purpose_counts.sum() if not purpose_counts.empty else 0
            purpose_strs = []
            if total_purpose > 0:
                for purpose, count in purpose_counts.items():
                    percent = round(count / total_purpose * 100)
                    purpose_strs.append(f"+ {purpose} ({percent}%)")
            purpose_cell = "\n".join(purpose_strs) if purpose_strs else ""
            # topic distribution
            topic_counts = df_dep_prompts["topic"].value_counts()
            total_topic = topic_counts.sum() if not topic_counts.empty else 0
            topic_strs = []
            if total_topic > 0:
                for topic, count in topic_counts.items():
                    percent = round(count / total_topic * 100)
                    topic_strs.append(f"+ {topic} ({percent}%)")
            topic_cell = "\n".join(topic_strs) if topic_strs else ""
            # evaluation and KPI
            prompts_count = len(df_dep_prompts)
            rating, kpi_score = evaluate_kpi(prompts_count, active_days)
            summary_rows.append({
                "#": idx,
                "Tháng": month_str,
                "Active Days (số ngày có sử dụng)": active_days,
                "Số lượng hội thoại": conv_count,
                "Số lượng tin nhắn trung bình trong mỗi hội thoại": avg_msgs,
                "Độ dài trung bình câu prompt (từ)": avg_prompt_len,
                "Mục đích sử dụng kèm %": purpose_cell,
                "Chủ đề kèm %": topic_cell,
                "Đánh giá chất lượng theo A3": rating,
                "Điểm KPIs %": kpi_score
            })
        summary_df = pd.DataFrame(summary_rows)

        # Write to Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter", datetime_format="yyyy-mm-dd HH:MM") as writer:
            # Summary sheet matching template
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            # Write other sheets for detailed analysis
            by_dep.to_excel(writer, index=False, sheet_name="Overview_Departments")
            # collapse multiindex for topics and purposes for export
            by_topic.to_excel(writer, index=False, sheet_name="Topics_By_Department")
            # Purpose counts by department
            by_purpose = user_prompts_df.groupby(["department","purpose"], dropna=False).size().reset_index(name="prompts")
            by_purpose.to_excel(writer, index=False, sheet_name="Purposes_By_Department")
            conv_stats.to_excel(writer, index=False, sheet_name="Conversations")
            user_prompts_df.to_excel(writer, index=False, sheet_name="User_Prompts")
            assistant_df.to_excel(writer, index=False, sheet_name="Assistant_Replies")
            month_df.to_excel(writer, index=False, sheet_name="Raw")

        st.download_button(
            label="⬇️ Tải Excel tổng hợp",
            data=buffer.getvalue(),
            file_name=f"chatgpt_report_{int(year)}_{int(month):02d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.info("Chưa có dữ liệu hợp lệ.")
else:
    st.info("Hãy tải lên một hoặc nhiều file conversations.json để bắt đầu.")
