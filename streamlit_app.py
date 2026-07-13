from __future__ import annotations

import os
import glob
import logging
import html
import re
import asyncio
from turtle import position, width
from typing import Any, List, Optional, Dict
import base64
from pathlib import Path
import pandas as pd
import json
import streamlit as st

import streamlit.components.v1 as components
from sympy import content
from app.config.settings import settings
from app.ingestion.topic_index_store import load_topics_index
from app.services.rag_pipeline import run_rag_pipeline
from app.agents.evaluation_agent import EvaluationAgent
logger = logging.getLogger(__name__)

APP_TITLE = "Book Navigator"
DEFAULT_BOOK_OPTION = "(انتخاب کتاب)"
DEFAULT_TOPIC_OPTION = "(بدون فیلتر)"

@st.cache_resource
def get_evaluation_agent():
    return EvaluationAgent()

evaluation_agent = get_evaluation_agent()

# -------------------------------------------------------------------
# Fonts / Styling
# -------------------------------------------------------------------
def _font_to_base64(font_path: str) -> str:
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def inject_custom_fonts() -> None:
    base_dir = Path(__file__).resolve().parent

    inter_path = base_dir / "fonts" / "Inter-Regular.ttf"
    vazir_regular_path = base_dir / "fonts" / "Vazirmatn-Regular.ttf"
    vazir_light_path = base_dir / "fonts" / "Vazirmatn-Light.ttf"
    vazir_black_path = base_dir / "fonts" / "Vazirmatn-Black.ttf"

    for p in [inter_path, vazir_regular_path, vazir_light_path, vazir_black_path]:
        if not p.exists():
            st.error(f"Font file not found: {p}")
            return

    inter_b64 = _font_to_base64(str(inter_path))
    vazir_regular_b64 = _font_to_base64(str(vazir_regular_path))
    vazir_light_b64 = _font_to_base64(str(vazir_light_path))
    vazir_black_b64 = _font_to_base64(str(vazir_black_path))

    css = f"""
    <style>
    @font-face {{
        font-family: 'InterCustom';
        src: url(data:font/ttf;base64,{inter_b64}) format('truetype');
        font-weight: 400;
        font-style: normal;
    }}

    @font-face {{
        font-family: 'VazirmatnCustom';
        src: url(data:font/ttf;base64,{vazir_regular_b64}) format('truetype');
        font-weight: 400;
        font-style: normal;
    }}

    @font-face {{
        font-family: 'VazirmatnCustom';
        src: url(data:font/ttf;base64,{vazir_light_b64}) format('truetype');
        font-weight: 300;
        font-style: normal;
    }}

    @font-face {{
        font-family: 'VazirmatnCustom';
        src: url(data:font/ttf;base64,{vazir_black_b64}) format('truetype');
        font-weight: 900;
        font-style: normal;
    }}

    html, body, [class*="css"] {{
        font-family: 'VazirmatnCustom', sans-serif;
    }}

    .persian-text {{
        font-family: 'VazirmatnCustom', sans-serif !important;
        direction: rtl;
        text-align: right;
        line-height: 2;
        font-size: 17px;
    }}

    .english-text {{
        font-family: 'InterCustom', sans-serif !important;
        direction: ltr;
        text-align: left;
        line-height: 1.8;
        font-size: 16px;
    }}

    .app-title {{
        font-family: 'InterCustom', 'VazirmatnCustom', sans-serif !important;
        font-weight: 900;
    }}
    .app-title:hover {{
    background: linear-gradient(
        90deg,
        #ff2d55 0%,
        #ff7a18 16%,
        #ffd60a 32%,
        #32d74b 48%,
        #0a84ff 64%,
        #5e5ce6 82%,
        #bf5af2 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    filter: brightness(1.22) saturate(1.35);

    text-shadow:
        0 0 10px rgba(255,255,255,0.18),
        0 0 22px rgba(255,122,24,0.20),
        0 0 28px rgba(255,214,10,0.20),
        0 0 34px rgba(50,215,75,0.18),
        0 0 40px rgba(10,132,255,0.18),
        0 0 46px rgba(191,90,242,0.20);

    transform: translateY(-1px);

    }}
    .title-wrapper {{
    width: 100%;
    min-height: 145px;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px 0 12px 0;
    margin: 0;
    position: relative;
    }}

    .title-wrapper::before {{
    content: "";
    position: absolute;
    width: 70%;
    height: 90px;
    border-radius: 999px;
    background: radial-gradient(
        ellipse at center,
        rgba(255,255,255,0.14) 0%,
        rgba(210,220,230,0.08) 35%,
        rgba(180,190,205,0.04) 60%,
        rgba(0,0,0,0) 100%
    );
    filter: blur(24px);
    z-index: 0;
    }}

    .app-title {{
    position: relative;
    z-index: 1;
    cursor: default;
    }}


    .topic-path {{
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}

    .dashboard-card {{
        padding: 1rem;
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 12px;
        margin-bottom: 1rem;
        background-color: rgba(250,250,250,0.02);
    }}

    .meta-chip {{
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border-radius: 999px;
        border: 1px solid rgba(120,120,120,0.25);
        font-size: 0.85rem;
    }}

    section[data-testid="stSidebar"] * {{
        font-family: 'VazirmatnCustom', sans-serif !important;
    }}

    textarea, input, button, select {{
        font-family: 'VazirmatnCustom', sans-serif !important;
    }}

    .rag-response-card {{
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(120,120,120,0.22);
        border-radius: 14px;
        margin-bottom: 1rem;
        background: rgba(250,250,250,0.035);
    }}

    .rag-answer-card {{
        border-left: 4px solid #4f8cff;
        direction: ltr;
    }}

    .rag-guidance-card {{
        border-left: 4px solid #22c55e;
        background: rgba(34,197,94,0.04);
        direction: ltr;
    }}

    .rag-sources-card {{
        border-left: 4px solid #f59e0b;
        direction: ltr;
    }}

    .rag-related-card {{
        border-left: 4px solid #a855f7;
        direction: ltr;
    }}

    .rag-card-title {{
        font-weight: 900;
        margin-bottom: 0.65rem;
        font-size: 1.05rem;
    }}

    .rag-muted {{
        color: #777;
        font-size: 0.9rem;
    }}

    .related-topic-chip {{
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem 0.25rem 0.25rem 0;
        border-radius: 999px;
        border: 1px solid rgba(120,120,120,0.25);
        background: rgba(168,85,247,0.06);
        font-size: 0.9rem;
    }}

    .writer-status-ok {{
        color: #16a34a;
        font-weight: 700;
    }}

    .writer-status-warning {{
        color: #ca8a04;
        font-weight: 700;
    }}

    .writer-status-error {{
        color: #dc2626;
        font-weight: 700;
    }}

    .persian-answer-container {{
        direction: rtl;
        text-align: right;
        font-family: 'VazirmatnCustom', sans-serif !important;
        line-height: 2.15;
        font-size: 16.5px;
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(120,120,120,0.16);
        background: rgba(255,255,255,0.03);
        max-height: 420px;
        overflow-y: auto;
        overflow-x: hidden;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.03);
    }}

    .persian-answer-title {{
        font-family: 'VazirmatnCustom', sans-serif !important;
        font-weight: 900;
        font-size: 1rem;
        margin: 0 0 0.75rem 0;
        color: inherit;
    }}

    .persian-answer-paragraph {{
        margin: 0 0 0.9rem 0;
        white-space: normal;
        word-break: break-word;
    }}

    .persian-answer-list {{
        margin: 0.15rem 0 0.9rem 0;
        padding-right: 1.25rem;
        padding-left: 0;
    }}

    .persian-answer-list li {{
        margin-bottom: 0.45rem;
    }}

    .persian-answer-muted {{
        color: #777;
        font-size: 0.92rem;
    }}
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] *,
    section[data-testid="stSidebar"] * {{
    font-family: 'VazirmatnCustom', sans-serif !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
def _font_to_base64(font_path: str) -> str:
    with open(font_path, "rb") as font_file:
        return base64.b64encode(font_file.read()).decode("utf-8")
# -------------------------------------------
#  Language Detection
# -------------------------------------------
def contains_persian(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

# -------------------------------------------
#  Typewriter Animation Title With Light Beam
# -------------------------------------------

def get_embedded_title_font_css() -> str:
    base_dir = Path(__file__).resolve().parent

    inter_regular_path = base_dir / "fonts" / "Inter-Regular.ttf"
    vazir_regular_path = base_dir / "fonts" / "Vazirmatn-Regular.ttf"
    vazir_black_path = base_dir / "fonts" / "Vazirmatn-Black.ttf"

    fallback_css = """
    .app-title {
    font-family: 'InterCustom', 'VazirmatnCustom', sans-serif;
    font-size: 120px;
    font-weight: 900;
    line-height: 1.0;
    letter-spacing: -0.04em;
    text-align: center;
    white-space: nowrap;
    margin: 0;

    /* Silver / white metallic base */
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f7f7f7 18%,
        #e8ebef 38%,
        #ffffff 52%,
        #cfd5dc 68%,
        #fefefe 82%,
        #d9dee5 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    filter: brightness(1.08) saturate(1.05);

    text-shadow:
        0 0 1px rgba(255,255,255,0.95),
        0 0 8px rgba(255,255,255,0.40),
        0 0 18px rgba(214,224,235,0.28),
        0 0 34px rgba(180,195,210,0.22),
        0 10px 30px rgba(160,170,185,0.18);

    transition:
        background 320ms ease,
        filter 320ms ease,
        text-shadow 320ms ease,
        transform 320ms ease;
}

    """

    if not inter_regular_path.exists() or not vazir_regular_path.exists():
        return fallback_css

    inter_b64 = _font_to_base64(str(inter_regular_path))
    vazir_regular_b64 = _font_to_base64(str(vazir_regular_path))

    if vazir_black_path.exists():
        vazir_black_b64 = _font_to_base64(str(vazir_black_path))
    else:
        vazir_black_b64 = vazir_regular_b64

    return f"""
    @font-face {{
        font-family: 'InterCustom';
        src: url(data:font/ttf;base64,{inter_b64}) format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }}

    @font-face {{
        font-family: 'VazirmatnCustom';
        src: url(data:font/ttf;base64,{vazir_regular_b64}) format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }}

    @font-face {{
        font-family: 'VazirmatnCustom';
        src: url(data:font/ttf;base64,{vazir_black_b64}) format('truetype');
        font-weight: 900;
        font-style: normal;
        font-display: swap;
    }}

    .app-title {{
        font-family: 'InterCustom', 'VazirmatnCustom', sans-serif !important;
        font-weight: 900;
        font-size: 120px;
        line-height: 1.05;
        margin: 0;
        letter-spacing: -3px;
        color: var(--title-color, #5AA0FF);

        filter: brightness(1.12) saturate(1.15);

        text-shadow:
            0 0 10px color-mix(in srgb, var(--title-color, #5AA0FF) 35%, transparent),
            0 0 24px color-mix(in srgb, var(--title-color, #5AA0FF) 22%, transparent),
            0 8px 34px color-mix(in srgb, var(--title-color, #5AA0FF) 18%, transparent);
    }}
    """

# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------
def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value

    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result

    if result is None:
        return {}

    if hasattr(result, "model_dump"):
        try:
            dumped = result.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(result, "dict"):
        try:
            dumped = result.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    return {"final_answer": str(result)}


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _html_text(text: str) -> str:
    escaped = html.escape(text.strip())
    return escaped.replace("\n", "<br>")


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
def _prettify_book_name(filename: str) -> str:
    name = filename.replace("_topics_index.json", "")
    name = name.replace("-", " ").replace("_", " ")
    return name.strip()


def _book_title_from_data(book_data: dict[str, Any], fallback_filename: str) -> str:
    title = _safe_str(book_data.get("book_title"))
    if title:
        return title
    return _prettify_book_name(fallback_filename)


def _book_summary(book_data: dict[str, Any]) -> str:
    return _safe_str(book_data.get("summary"))


def _book_main_ideas(book_data: dict[str, Any]) -> list[str]:
    value = book_data.get("main_ideas")
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if _safe_str(x)]


def _book_study_path(book_data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = book_data.get("suggested_study_path")
    if not isinstance(raw, list):
        raw = book_data.get("suggested_path")

    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []

    for idx, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            path_value = item.get("path")
            normalized.append(
                {
                    "step": _safe_int(item.get("step")) or idx,
                    "topic_id": _safe_str(item.get("topic_id")),
                    "title": _safe_str(item.get("title")),
                    "path": path_value if isinstance(path_value, list) else [],
                }
            )
        else:
            title = _safe_str(item)
            if title:
                normalized.append(
                    {
                        "step": idx,
                        "topic_id": "",
                        "title": title,
                        "path": [],
                    }
                )

    return normalized


def _flatten_topics(
    topics: list[dict[str, Any]],
    out: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    if out is None:
        out = []

    for topic in topics:
        if not isinstance(topic, dict):
            continue

        out.append(topic)
        subtopics = _safe_list(topic.get("subtopics"))
        _flatten_topics(subtopics, out)

    return out


def _book_total_topics(book_data: dict[str, Any]) -> int:
    value = _safe_int(book_data.get("total_topics"))
    if value is not None:
        return value
    return len(_flatten_topics(_safe_list(book_data.get("topics"))))


def _book_top_level_count(book_data: dict[str, Any]) -> int:
    value = _safe_int(book_data.get("top_level_topic_count"))
    if value is not None:
        return value
    return len(_safe_list(book_data.get("topics")))


def _page_range_text(page_start: Any, page_end: Any) -> str:
    ps = _safe_int(page_start)
    pe = _safe_int(page_end)

    if ps is not None and pe is not None:
        return f"{ps}-{pe}"
    if ps is not None:
        return str(ps)
    return ""


def _study_step_label(step: dict[str, Any]) -> str:
    title = _safe_str(step.get("title")) or "Untitled Step"
    path = step.get("path")

    if isinstance(path, list) and path:
        cleaned = [str(x).strip() for x in path if _safe_str(x)]
        if cleaned:
            return f"{title}  —  {' > '.join(cleaned)}"

    return title


def _topic_label(topic: dict[str, Any]) -> str:
    title = _safe_str(topic.get("title"), "Untitled Topic")
    depth = _safe_int(topic.get("depth")) or 1
    prefix = "— " * max(0, depth - 1)
    return f"{prefix}{title}"


def _topic_path_text(topic: dict[str, Any]) -> str:
    path = topic.get("path", [])
    if isinstance(path, list) and path:
        return " > ".join(str(p) for p in path if _safe_str(p))
    return _safe_str(topic.get("title"))


def _collect_topic_options(book_data: dict[str, Any]) -> list[dict[str, str]]:
    root_topics = _safe_list(book_data.get("topics"))
    flat_topics = _flatten_topics(root_topics)

    options: list[dict[str, str]] = []
    seen_ids = set()

    for topic in flat_topics:
        topic_id = _safe_str(topic.get("topic_id"))
        title = _safe_str(topic.get("title"))

        if not topic_id or not title or topic_id in seen_ids:
            continue

        options.append(
            {
                "topic_id": topic_id,
                "title": title,
                "label": _topic_label(topic),
                "path_text": _topic_path_text(topic),
            }
        )
        seen_ids.add(topic_id)

    return options


def _find_topic_by_id(
    book_data: dict[str, Any],
    topic_id: Optional[str],
) -> Optional[dict[str, Any]]:
    if not topic_id:
        return None

    flat_topics = _flatten_topics(_safe_list(book_data.get("topics")))
    for topic in flat_topics:
        if _safe_str(topic.get("topic_id")) == topic_id:
            return topic

    return None
def _extract_answer(result: dict) -> str:
    """
    Extract answer from normalized pipeline result.
    """
    for key in ["answer", "response", "output", "final_answer", "generated_answer"]:
        if key in result and result[key]:
            return str(result[key])

    return ""


def _extract_contexts(result: dict) -> list[str]:
    """
    Extract retrieved contexts from pipeline result.
    """
    raw_contexts = (
        result.get("contexts")
        or result.get("retrieved_docs")
        or result.get("documents")
        or result.get("sources")
        or result.get("chunks")
        or []
    )

    contexts: list[str] = []

    for doc in raw_contexts:
        if isinstance(doc, str):
            contexts.append(doc)

        elif isinstance(doc, dict):
            if "content" in doc:
                contexts.append(str(doc["content"]))
            elif "text" in doc:
                contexts.append(str(doc["text"]))
            elif "page_content" in doc:
                contexts.append(str(doc["page_content"]))
            else:
                contexts.append(str(doc))

        else:
            contexts.append(str(doc))

    return contexts




def _safe_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []

    result: List[str] = []

    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
            continue

        if hasattr(item, "page_content"):
            text = str(getattr(item, "page_content", "")).strip()
            if text:
                result.append(text)
            continue

        if isinstance(item, dict):
            text = str(item.get("text") or item.get("page_content") or "").strip()
            if text:
                result.append(text)
            continue

        text = str(item).strip()
        if text:
            result.append(text)

    return result

# -------------------------------------
#    Evaluation Block
# -------------------------------------
def _run_pipeline_evaluation(
    question: str,
    contexts: List[str],
    answer: str,
) -> Optional[Dict[str, Any]]:
    logger.info("EVAL HOOK ENTERED")
    logger.info("Answer preview: %r", str(answer)[:200])
    logger.info("Raw sources type: %s", type(contexts))
    logger.info("Raw sources preview: %r", contexts[:1] if isinstance(contexts, list) else contexts)

    normalized_contexts = _safe_str_list(contexts)
    logger.info("Normalized contexts count: %s", len(normalized_contexts))
    if normalized_contexts:
        logger.info("First normalized context preview: %r", normalized_contexts[0][:200])

    if not normalized_contexts:
        logger.warning("Evaluation skipped: normalized contexts are empty")
        return None

    if not evaluation_agent.should_evaluate():
        logger.warning("Evaluation skipped: should_evaluate returned False")
        return None

    try:
        evaluation_result = evaluation_agent.evaluate(
            question=str(question).strip(),
            answer=str(answer).strip(),
            contexts=normalized_contexts,
            ground_truth="",
            async_mode=False,
        )
        logger.info("Evaluation agent returned type: %s", type(evaluation_result))

        if isinstance(evaluation_result, asyncio.Task):
            logger.error(
                "Evaluation agent returned asyncio.Task when async_mode=False. "
                "Expected EvaluationResult or dict."
            )
            return None

        if evaluation_result is None:
            logger.warning("Evaluation returned None from agent.")
            return None

        if hasattr(evaluation_result, "to_dict") and callable(evaluation_result.to_dict):
            evaluation_dict = evaluation_result.to_dict()
        elif isinstance(evaluation_result, dict):
            evaluation_dict = evaluation_result
        else:
            logger.warning(
                "Unexpected evaluation result type: %s. Expected EvaluationResult or dict.",
                type(evaluation_result),
            )
            return None

        logger.info("Evaluation dict keys: %s", list(evaluation_dict.keys()))
        logger.info("Evaluation scores keys: %s", list(evaluation_dict.get("scores", {}).keys()))
        logger.info("Evaluation metrics keys: %s", list(evaluation_dict.get("metrics", {}).keys()))

        return evaluation_dict

    except Exception:
        logger.exception("Error during pipeline evaluation")
        return None


# -------------------------------------------------------------------
# Index loading
# -------------------------------------------------------------------
def _load_single_book_topics_index() -> dict[str, Any]:
    try:
        path = os.path.join(
            settings.TOPICS_INDEX_DIR,
            settings.DEFAULT_BOOK_TOPICS_FILE,
        )

        if os.path.exists(path):
            data = load_topics_index(path)
            if isinstance(data, dict):
                return data

    except Exception as exc:
        logger.exception("Failed to load single-book topics index: %s", exc)

    return {
        "book_id": "unknown-book",
        "book_title": "Unknown Book",
        "topics": [],
    }


def _load_books_topics_map() -> dict[str, dict[str, Any]]:
    books_map: dict[str, dict[str, Any]] = {}

    try:
        pattern = os.path.join(settings.TOPICS_INDEX_DIR, "*_topics_index.json")
        files = sorted(glob.glob(pattern))

        for path in files:
            try:
                data = load_topics_index(path)
                if not isinstance(data, dict):
                    continue

                filename = os.path.basename(path)
                books_map[filename] = data

            except Exception as exc:
                logger.exception("Failed to load topics index file %s: %s", path, exc)

    except Exception as exc:
        logger.exception("Failed to scan TOPICS_INDEX_DIR: %s", exc)

    return books_map


def _load_all_topics_indices() -> dict[str, Any]:
    combined: dict[str, Any] = {
        "topics": [],
        "books": {},
    }

    books_map = _load_books_topics_map()
    combined["books"] = books_map

    for _, data in books_map.items():
        topics = data.get("topics", [])
        if isinstance(topics, list):
            combined["topics"].extend(topics)

    return combined


def _load_topics_index() -> dict[str, Any]:
    if settings.USE_MULTI_BOOK:
        return _load_all_topics_indices()
    return _load_single_book_topics_index()


# -------------------------------------------------------------------
# Render helpers
# -------------------------------------------------------------------
def _render_topic_tree(topics: list[dict[str, Any]], level: int = 0) -> None:
    for topic in topics:
        if not isinstance(topic, dict):
            continue

        title = _safe_str(topic.get("title"), "Untitled Topic")
        page_start = topic.get("page_start")
        page_end = topic.get("page_end")

        indent = "    " * level
        page_text = ""

        if page_start is not None and page_end is not None:
            page_text = f" (pp. {page_start}-{page_end})"
        elif page_start is not None:
            page_text = f" (p. {page_start})"

        st.markdown(f"{indent}- **{title}**{page_text}")

        subtopics = _safe_list(topic.get("subtopics"))
        if subtopics:
            _render_topic_tree(subtopics, level + 1)


def _render_book_dashboard(book_data: dict[str, Any]) -> None:
    book_title = _safe_str(book_data.get("book_title"), "Untitled Book")
    book_id = _safe_str(book_data.get("book_id"), "unknown-book")
    summary = _book_summary(book_data)
    main_ideas = _book_main_ideas(book_data)
    study_path = _book_study_path(book_data)
    topics = _safe_list(book_data.get("topics"))

    total_topics = _book_total_topics(book_data)
    top_level_count = _book_top_level_count(book_data)

    author = _safe_str(book_data.get("author"))
    language = _safe_str(book_data.get("language"))
    source = _safe_str(book_data.get("source") or book_data.get("source_file"))
    page_range = _page_range_text(book_data.get("page_start"), book_data.get("page_end"))
    has_semantic = bool(book_data.get("has_semantic_enrichment", False))

    st.subheader("Book Dashboard")

    meta_lines = [
        f"<b>Title:</b> {html.escape(book_title)}",
        f"<b>Book ID:</b> {html.escape(book_id)}",
        f"<b>Total Topics:</b> {total_topics}",
        f"<b>Top-level Sections:</b> {top_level_count}",
    ]

    if author:
        meta_lines.append(f"<b>Author:</b> {html.escape(author)}")
    if language:
        meta_lines.append(f"<b>Language:</b> {html.escape(language)}")
    if page_range:
        meta_lines.append(f"<b>Pages:</b> {html.escape(page_range)}")
    if source:
        meta_lines.append(f"<b>Source:</b> {html.escape(source)}")

    meta_lines.append(f"<b>Semantic Enrichment:</b> {'Yes' if has_semantic else 'No'}")

    st.markdown(
        f"""
        <div class="dashboard-card">
            {'<br>'.join(meta_lines)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Summary")
        if summary:
            st.markdown(
                f'<div class="persian-text">{_html_text(summary)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No book summary available yet.")

        st.markdown("### Main Ideas")
        if main_ideas:
            for idea in main_ideas:
                st.markdown(f"- {idea}")
        else:
            st.info("No main ideas available yet.")

    with col2:
        st.markdown("### Suggested Study Path")
        if study_path:
            for step in study_path:
                step_no = _safe_int(step.get("step")) or 0
                st.markdown(f"{step_no}. {_study_step_label(step)}")
        else:
            st.info("No suggested study path available yet.")

        st.markdown("### Structural Overview")
        if topics:
            for topic in topics[:12]:
                st.markdown(f"- {_safe_str(topic.get('title'), 'Untitled Topic')}")
            if len(topics) > 12:
                st.caption(f"... and {len(topics) - 12} more top-level sections")
        else:
            st.info("No topics available.")

    st.divider()
    st.markdown("### Topic Tree / Mind Map Preview")
    if topics:
        _render_topic_tree(topics)
    else:
        st.info("No topic tree available for this book.")


def _render_topic_details(topic: Optional[dict[str, Any]]) -> None:
    st.subheader("Topic Details")

    if not topic:
        st.info("یک موضوع انتخاب کنید تا جزئیات آن نمایش داده شود.")
        return

    title = _safe_str(topic.get("title"), "Untitled Topic")
    topic_id = _safe_str(topic.get("topic_id"))
    summary_text = _safe_str(topic.get("summary_text"))
    keywords = _safe_list(topic.get("keywords"))
    chunk_ids = _safe_list(topic.get("chunk_ids"))
    main_ideas = _safe_list(topic.get("main_ideas"))
    merge_source = _safe_str(topic.get("merge_source"))
    semantic_match_score = topic.get("semantic_match_score")
    path_text = _topic_path_text(topic)
    page_start = topic.get("page_start")
    page_end = topic.get("page_end")

    st.markdown(f"### {title}")
    st.caption(f"Topic ID: {topic_id}")
    st.markdown(
        f'<div class="topic-path">Path: {html.escape(path_text)}</div>',
        unsafe_allow_html=True,
    )

    meta_parts = []

    if page_start is not None and page_end is not None:
        meta_parts.append(f"Pages: {page_start} - {page_end}")
    elif page_start is not None:
        meta_parts.append(f"Page: {page_start}")

    if merge_source:
        meta_parts.append(f"Merge Source: {merge_source}")

    if isinstance(semantic_match_score, (int, float)):
        meta_parts.append(f"Semantic Match Score: {semantic_match_score:.3f}")

    if meta_parts:
        st.write(" | ".join(meta_parts))

    if summary_text:
        st.markdown("#### Summary")
        st.markdown(
            f'<div class="persian-text">{_html_text(summary_text)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No topic summary available yet.")

    st.markdown("#### Main Ideas")
    if main_ideas:
        for idea in main_ideas:
            st.markdown(f"- {idea}")
    else:
        st.info("No main ideas available for this topic.")

    st.markdown("#### Keywords")
    if keywords:
        st.write(", ".join(str(k) for k in keywords))
    else:
        st.info("No keywords available.")

    st.markdown("#### Chunk References")
    if chunk_ids:
        st.code("\n".join(str(c) for c in chunk_ids))
    else:
        st.info("No chunk references available.")


# -------------------------------------------------------------------
# RAG response render helpers
# -------------------------------------------------------------------
def _render_html_text_block(
    text: str,
    *,
    direction_class: str,
) -> None:
    if not text.strip():
        return

    st.markdown(
        f'<div class="{direction_class}">{_html_text(text)}</div>',
        unsafe_allow_html=True,
    )


def _split_nonempty_lines(text: Any) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _render_persian_text_as_html(text: str) -> str:
    lines = _split_nonempty_lines(text)

    if not lines:
        return '<div class="persian-answer-muted">متنی برای نمایش وجود ندارد.</div>'

    paragraph_blocks: list[str] = []

    for line in lines:
        escaped = html.escape(line)

        if re.match(r"^(\-|\*|•|\d+[\.\)\-])\s+", line):
            escaped = re.sub(r"^(\-|\*|•|\d+[\.\)\-])\s+", "", escaped)
            paragraph_blocks.append(
                f'<ul class="persian-answer-list"><li>{escaped}</li></ul>'
            )
        else:
            paragraph_blocks.append(
                f'<div class="persian-answer-paragraph">{escaped}</div>'
            )

    return "".join(paragraph_blocks)


def _render_persian_lines_as_list_html(text: str) -> str:
    lines = _split_nonempty_lines(text)

    if not lines:
        return '<div class="persian-answer-muted">موردی برای نمایش وجود ندارد.</div>'

    items = []
    for line in lines:
        cleaned = re.sub(r"^(\-|\*|•|\d+[\.\)\-])\s+", "", line).strip()
        if cleaned:
            items.append(f"<li>{html.escape(cleaned)}</li>")

    if not items:
        return '<div class="persian-answer-muted">موردی برای نمایش وجود ندارد.</div>'

    return f'<ul class="persian-answer-list">{"".join(items)}</ul>'


def _normalize_related_topics(value: Any) -> list[str]:
    return _split_related_topics(value)


def _build_persian_answer_box_html(text: str) -> str:
    return f"""
    <div class="persian-answer-container">
        {_render_persian_text_as_html(text)}
    </div>
    """


def _split_related_topics(value: Any) -> list[str]:
    """
    Supports:
    - list[str]
    - bullet string
    - comma-separated string
    - newline-separated string
    """
    if isinstance(value, list):
        cleaned_list = []
        seen = set()

        for item in value:
            item_str = str(item).strip(" -•\t\r\n")
            if not item_str:
                continue

            key = item_str.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned_list.append(item_str)

        return cleaned_list[:12]

    if not isinstance(value, str) or not value.strip():
        return []

    text = value.strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) > 1:
        raw_parts = lines
    elif "," in text:
        raw_parts = text.split(",")
    else:
        raw_parts = re.split(r"\s*[-•]\s+", text)

    cleaned: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        item = part.strip(" -•\t\r\n")
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(item)

    return cleaned[:12]


def _render_related_topics(value: Any) -> None:
    topics = _normalize_related_topics(value)

    if not topics:
        st.caption("موضوع مرتبطی از خروجی استخراج نشد.")
        return

    chips_html = "".join(
        f'<span class="related-topic-chip">{html.escape(topic)}</span>'
        for topic in topics
    )

    st.markdown(chips_html, unsafe_allow_html=True)


def _writer_status_class(status: str, ok: bool) -> str:
    status = (status or "").lower()

    if not ok or status in {"failed", "error", "no_context", "empty_query"}:
        return "writer-status-error"

    if status in {"ok_with_section_recovery", "insufficient_info"}:
        return "writer-status-warning"

    return "writer-status-ok"


def _render_rag_response(result: dict[str, Any]) -> None:
    # Persian-first fields from translator
    translated_answer = _first_text(
        result.get("translated_answer"),
        result.get("persian_structured_response"),
    )
    persian_final_answer = _first_text(result.get("persian_final_answer"))
    persian_book_sources = _first_text(result.get("persian_book_sources"))
    persian_search_guidance = _first_text(result.get("persian_search_guidance"))
    persian_related_topics = result.get("persian_related_topics")

    # Original English writer fields
    final_answer = _first_text(result.get("final_answer"))
    book_sources = _first_text(result.get("book_sources"))
    search_guidance = _first_text(result.get("search_guidance"))
    related_topics = result.get("related_topics")

    # UI-facing values: prefer Persian, fallback to English
    final_answer_view = _first_text(
        persian_final_answer,
        translated_answer,
        final_answer,
    )
    book_sources_view = _first_text(
        persian_book_sources,
        book_sources,
    )
    search_guidance_view = _first_text(
        persian_search_guidance,
        search_guidance,
    )
    related_topics_view = persian_related_topics or related_topics

    has_persian_answer = _has_text(persian_final_answer) or _has_text(translated_answer)
    has_persian_sources = _has_text(persian_book_sources)
    has_persian_guidance = _has_text(persian_search_guidance)
    has_persian_related = bool(_normalize_related_topics(persian_related_topics))

    writer_status = _safe_str(result.get("writer_status"))
    writer_ok = _safe_bool(result.get("writer_ok", True), default=True)
    writer_sections_ok = _safe_bool(result.get("writer_sections_ok", False), default=False)
    writer_fallback_used = _safe_bool(result.get("writer_fallback_used", False), default=False)
    writer_answer_insufficient = _safe_bool(
        result.get("writer_answer_insufficient", False),
        default=False,
    )

    st.markdown("### پاسخ")

    st.markdown(
        """
        <div class="rag-response-card rag-answer-card">
            <div class="rag-card-title">پاسخ اصلی</div>
        """,
        unsafe_allow_html=True,
    )

    if writer_answer_insufficient:
        st.warning(
            "پاسخ مستقیم و کافی از متن کتاب استخراج نشده است. "
            "منابع، مسیر ادامه‌ی جست‌وجو، و موضوعات مرتبط در پایین نمایش داده شده‌اند."
        )

    if final_answer_view:
        if has_persian_answer:
            st.markdown(
                _build_persian_answer_box_html(final_answer_view),
                unsafe_allow_html=True,
            )
        else:
            _render_html_text_block(
                final_answer_view,
                direction_class="english-text",
            )
    else:
        st.markdown(
            '<div class="persian-text">پاسخی تولید نشد.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("English Answer", expanded=False):
        if final_answer:
            st.markdown(
                f'<div class="english-text">{_html_text(final_answer)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write("No English answer generated.")

    if search_guidance_view:
        st.markdown(
            """
            <div class="rag-response-card rag-guidance-card">
                <div class="rag-card-title">چگونه جست‌وجو در کتاب را ادامه دهید</div>
            """,
            unsafe_allow_html=True,
        )

        if has_persian_guidance:
            st.markdown(
                f'<div class="persian-answer-container">{_render_persian_text_as_html(search_guidance_view)}</div>',
                unsafe_allow_html=True,
            )
        else:
            _render_html_text_block(
                search_guidance_view,
                direction_class="english-text",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if book_sources_view:
        with st.expander("منابع در کتاب / Sources in the Book", expanded=True):
            st.markdown(
                """
                <div class="rag-response-card rag-sources-card">
                    <div class="rag-card-title">منابع در کتاب</div>
                """,
                unsafe_allow_html=True,
            )

            if has_persian_sources:
                st.markdown(
                    f'<div class="persian-answer-container">{_render_persian_lines_as_list_html(book_sources_view)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                _render_html_text_block(
                    book_sources_view,
                    direction_class="english-text",
                )

            st.markdown("</div>", unsafe_allow_html=True)

    if related_topics_view:
        st.markdown(
            """
            <div class="rag-response-card rag-related-card">
                <div class="rag-card-title">موضوعات مرتبط</div>
            """,
            unsafe_allow_html=True,
        )

        _render_related_topics(related_topics_view)

        st.markdown("</div>", unsafe_allow_html=True)
    elif has_persian_related:
        st.markdown(
            """
            <div class="rag-response-card rag-related-card">
                <div class="rag-card-title">موضوعات مرتبط</div>
            """,
            unsafe_allow_html=True,
        )

        _render_related_topics(persian_related_topics)

        st.markdown("</div>", unsafe_allow_html=True)

    if writer_status:
        status_class = _writer_status_class(writer_status, writer_ok)

        status_html = (
            f"Writer status: "
            f"<span class='{status_class}'>{html.escape(writer_status)}</span>"
            f" | sections_ok: <code>{writer_sections_ok}</code>"
            f" | fallback_used: <code>{writer_fallback_used}</code>"
            f" | answer_insufficient: <code>{writer_answer_insufficient}</code>"
        )

        st.markdown(
            f'<div class="rag-muted">{status_html}</div>',
            unsafe_allow_html=True,
        )


def _render_debug_info(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Debug Info")

    retrieved_docs = _safe_list(result.get("retrieved_docs"))
    reranked_docs = _safe_list(result.get("reranked_docs"))
    compressed_context = result.get("compressed_context")
    final_answer_debug = result.get("final_answer")
    translated_answer_debug = result.get("translated_answer")

    debug_payload = {
        "config": {
            "use_multi_book": settings.USE_MULTI_BOOK,
            "topics_index_dir": settings.TOPICS_INDEX_DIR,
            "default_book_topics_file": settings.DEFAULT_BOOK_TOPICS_FILE,
        },
        "ui_selection": {
            "selected_book_file": result.get("ui_selected_book_file"),
            "selected_book_id": result.get("ui_selected_book_id"),
            "selected_book_title": result.get("ui_selected_book_title"),
            "selected_topic_id": result.get("ui_selected_topic_id"),
            "selected_topic_title": result.get("ui_selected_topic_title"),
        },
        "query": {
            "user_query": result.get("user_query"),
            "translated_query": result.get("translated_query"),
            "enhanced_query": result.get("enhanced_query"),
            "retrieval_query": result.get("retrieval_query"),
            "rerank_query": result.get("rerank_query"),
            "compression_query": result.get("compression_query"),
        },
        "routing": {
            "selected_topic": result.get("selected_topic"),
            "topic_filter": _safe_dict(result.get("topic_filter")),
            "routing": _safe_dict(result.get("routing")),
            "routing_confidence": float(result.get("routing_confidence") or 0.0),
            "should_retrieve": bool(result.get("should_retrieve", True)),
        },
        "retrieval": {
            "retrieved_docs_count": len(retrieved_docs),
            "reranked_docs_count": len(reranked_docs),
            "applied_filters": _safe_dict(result.get("applied_filters")),
            "fallback_used": bool(result.get("fallback_used", False)),
            "retrieval_status": result.get("retrieval_status"),
        },
        "output": {
            "compressed_context_present": _has_text(compressed_context),
            "compressed_context_length": (
                len(compressed_context) if isinstance(compressed_context, str) else 0
            ),
            "final_answer_present": _has_text(final_answer_debug),
            "translated_answer_present": _has_text(translated_answer_debug),
            "persian_final_answer_present": _has_text(result.get("persian_final_answer")),
            "persian_book_sources_present": _has_text(result.get("persian_book_sources")),
            "persian_search_guidance_present": _has_text(result.get("persian_search_guidance")),
            "persian_related_topics_present": bool(result.get("persian_related_topics")),
            "book_sources_present": _has_text(result.get("book_sources")),
            "search_guidance_present": _has_text(result.get("search_guidance")),
            "related_topics_present": bool(result.get("related_topics")),
            "done": bool(result.get("done", False)),
        },
        "writer": {
            "writer_status": result.get("writer_status"),
            "writer_ok": bool(result.get("writer_ok", True)),
            "writer_sections_ok": bool(result.get("writer_sections_ok", False)),
            "writer_fallback_used": bool(result.get("writer_fallback_used", False)),
            "writer_answer_insufficient": bool(result.get("writer_answer_insufficient", False)),
            "writer_query": result.get("writer_query"),
            "writer_query_source": result.get("writer_query_source"),
            "writer_context_source": result.get("writer_context_source"),
            "writer_model": result.get("writer_model"),
            "writer_input_doc_count": result.get("writer_input_doc_count"),
            "writer_context_chars": result.get("writer_context_chars"),
            "writer_section_flags": _safe_dict(result.get("writer_section_flags")),
            "writer_error": result.get("writer_error"),
        },
    }

    st.json(debug_payload)

    writer_structured_response = result.get("writer_structured_response")
    writer_raw_response = result.get("writer_raw_response")

    if _has_text(writer_structured_response):
        with st.expander("Writer Structured Response", expanded=False):
            st.code(str(writer_structured_response))

    if _has_text(writer_raw_response):
        with st.expander("Writer Raw Response", expanded=False):
           st.code(str(writer_raw_response))
# -------------------------------------------------------------------
# Evaluation Dashboard Helpers
# -------------------------------------------------------------------
def _load_evaluation_files() -> list[dict]:
    results_dir = settings.EVAL_RESULTS_DIR

    if not os.path.exists(results_dir):
        return []

    files = sorted(glob.glob(os.path.join(results_dir, "evaluations_*.json")))
    all_items: list[dict] = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
                if isinstance(items, list):
                    all_items.extend([item for item in items if isinstance(item, dict)])
        except Exception:
            logger.exception("Failed to load evaluation file: %s", path)
            continue

    return all_items

def _flatten_evaluation_item(item: dict) -> dict:
    row: dict = {}

    for key, value in item.items():
        if key in ("scores", "metrics") and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                row[f"{key}_{sub_key}"] = sub_value
        else:
            row[key] = value

    return row

    # ------------------------------------------------------------------
    # Raw Table
    # ------------------------------------------------------------------
    st.markdown("### Evaluation Records")
    st.dataframe(df, width="stretch")

    # ------------------------------------------------------------------
    # Drilldown
    # ------------------------------------------------------------------
    id_column = None
    for candidate in ["evaluation_id", "id"]:
        if candidate in df.columns:
            id_column = candidate
            break

    if id_column:
        st.markdown("### Inspect Single Evaluation")

        selected_id = st.selectbox(
            "Select evaluation ID",
            df[id_column].astype(str).tolist(),
        )

        selected_row = df[df[id_column].astype(str) == str(selected_id)].iloc[0]

        st.markdown("#### Question")
        st.code(selected_row.get("question", ""))

        st.markdown("#### Answer")
        st.code(selected_row.get("answer", ""))

        contexts = selected_row.get("contexts", [])
        if isinstance(contexts, list) and contexts:
            st.markdown("#### Contexts Used")
            for ctx in contexts:
                st.markdown(f"- {ctx}")

        st.markdown("#### Scores")
        for key in metric_cols:
            if key in selected_row:
                try:
                    st.metric(key, f"{float(selected_row[key]):.3f}")
                except Exception:
                    st.write(f"{key}: {selected_row[key]}")

        # Threshold alerts
        st.markdown("#### Threshold Alerts")
        if "faithfulness" in selected_row and float(selected_row["faithfulness"]) < settings.EVAL_FAITHFULNESS_THRESHOLD:
            st.error("Faithfulness below threshold")

        if "answer_relevancy" in selected_row and float(selected_row["answer_relevancy"]) < settings.EVAL_RELEVANCY_THRESHOLD:
            st.error("Answer relevancy below threshold")

        if "context_precision" in selected_row and float(selected_row["context_precision"]) < settings.EVAL_CONTEXT_PRECISION_THRESHOLD:
            st.error("Context precision below threshold")


import streamlit.components.v1 as components
import html

def render_cosmic_typewriter_title(text="Book Navigator", speed=70):
    google_blue = "#4285F4"

    safe_text = html.escape(text)

    html_code = f"""
    <html>
    <head>
        <style>
            :root {{
                --letter-height: 165px;
                --font-size: 96px;
                --google-blue: {google_blue};
            }}

            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: visible;
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                font-family: "Poppins", "Segoe UI", sans-serif;
            }}

            .title-wrap {{
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 18px 0 30px;
                overflow: visible;
            }}

            .name {{
                display: flex;
                align-items: center;
                justify-content: center;
                flex-wrap: nowrap;
                white-space: nowrap;
                overflow: visible;
                border-right: 3px solid rgba(66, 133, 244, 0.85);
                padding-right: 6px;
            }}

            .cosmic {{
                position: relative;
                cursor: pointer;
                height: var(--letter-height);
                min-width: 0.62em;
                display: inline-flex;
                justify-content: center;
                align-items: center;
                overflow: visible;
            }}

            .cosmic span {{
                font-size: var(--font-size);
                font-weight: 800;
                line-height: 1;
                color: transparent;
                -webkit-text-stroke: 2px var(--google-blue);
                transition: opacity 0.45s ease;
                user-select: none;
            }}

            .cosmic.hover-enabled:hover span {{
                opacity: 0;
            }}

            .cosmic::before {{
                content: attr(data-text);
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 100%;
                height: 0;
                overflow: hidden;
                text-align: center;
                font-size: var(--font-size);
                font-weight: 800;
                line-height: var(--letter-height);
                color: var(--google-blue);
                white-space: nowrap;
                pointer-events: none;
                transition: height 0.55s ease, filter 0.55s ease;
            }}

            .cosmic.hover-enabled:hover::before {{
                height: 100%;

                /*
                   Reduced glow:
                   Only one soft Google-blue shadow.
                */
                filter:
                    drop-shadow(0 0 20px rgba(66, 133, 244, 0.45));
            }}

            .space {{
                width: 28px;
                height: var(--letter-height);
                flex: 0 0 28px;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="title-wrap">
            <div class="name" id="typed"></div>
        </div>

        <script>
            const text = {safe_text!r};
            const speed = {speed};

            const root = document.getElementById("typed");
            let i = 0;

            function escapeHtml(str) {{
                return str
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }}

            function createLetter(ch) {{
                if (ch === " ") {{
                    const space = document.createElement("span");
                    space.className = "space";
                    return space;
                }}

                const safe = escapeHtml(ch);

                const wrapper = document.createElement("div");
                wrapper.className = "cosmic";
                wrapper.setAttribute("data-text", ch);

                const span = document.createElement("span");
                span.innerHTML = safe;

                wrapper.appendChild(span);
                return wrapper;
            }}

            function typeWriter() {{
                if (i < text.length) {{
                    const node = createLetter(text[i]);
                    root.appendChild(node);
                    i++;
                    setTimeout(typeWriter, speed);
                }} else {{
                    setTimeout(() => {{
                        root.style.borderRight = "none";
                        document.querySelectorAll(".cosmic").forEach(el => {{
                            el.classList.add("hover-enabled");
                        }});
                    }}, 350);
                }}
            }}

            typeWriter();

            const blink = setInterval(() => {{
                if (root.style.borderRight === "none") {{
                    clearInterval(blink);
                    return;
                }}
                root.style.borderRightColor =
                    root.style.borderRightColor === "transparent"
                        ? "rgba(66, 133, 244, 0.85)"
                        : "transparent";
            }}, 500);
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=245)


# -------------------------------------------------------------------
# Spinner Effdect
# -------------------------------------------------------------------
def render_generating_spinner(text="Generating answer..."):
    html_code = f"""
    <html>
    <head>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                display: grid;
                background: transparent;
                overflow: hidden;
            }}

            .loading-wrap {{
                display: grid;
                place-items: center;
                width: 100%;
                height: 100%;
                padding: 12px 0;
                background: transparent;
            }}

            .loading-text {{
                place-self: center;
                background: linear-gradient(
                    90deg,
                    #3b82f6,
                    #22d3ee,
                    #e0f2fe,
                    #22d3ee,
                    #3b82f6
                ) -100% / 200%;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                font: 900 clamp(1.2em, 3.8vw, 2.4em) "Inter", "Segoe UI", sans-serif;
                letter-spacing: 0.01em;
                animation: shimmer 2s linear infinite;
                text-align: center;
                white-space: nowrap;
            }}

            @keyframes shimmer {{
                to {{
                    background-position: 100%;
                }}
            }}

            @media (forced-colors: active) {{
                .loading-text {{
                    background: none;
                    color: aquamarine;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="loading-wrap">
            <div class="loading-text">{text}</div>
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=70)

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if "last_evaluation" not in st.session_state:
        st.session_state["last_evaluation"] = None

    inject_custom_fonts()

    render_cosmic_typewriter_title(text="Book Navigator", speed=70)


    st.caption("سامانه هدایت دانش‌محور برای کتاب‌ها با ساختار Book Dashboard + Topic Navigation")


    all_topics_index = _load_topics_index()
    books_map = all_topics_index.get("books", {}) if settings.USE_MULTI_BOOK else {}

    selected_book_file: Optional[str] = None
    selected_book_data: Optional[dict[str, Any]] = None
    selected_book_label: str = DEFAULT_BOOK_OPTION

    topic_options: list[dict[str, str]] = []
    selected_topic_id: Optional[str] = None
    selected_topic_data: Optional[dict[str, Any]] = None

    with st.sidebar:
        st.header("Book Navigator")

        if settings.USE_MULTI_BOOK:
            book_files = sorted(list(books_map.keys()))
            label_to_file: dict[str, str] = {}

            book_labels = [DEFAULT_BOOK_OPTION]
            for file_name in book_files:
                label = _book_title_from_data(books_map.get(file_name, {}), file_name)
                book_labels.append(label)
                label_to_file[label] = file_name

            selected_book_label = st.selectbox("Book", book_labels)

            if selected_book_label != DEFAULT_BOOK_OPTION:
                selected_book_file = label_to_file.get(selected_book_label)
                if selected_book_file:
                    selected_book_data = books_map.get(selected_book_file)
        else:
            selected_book_file = settings.DEFAULT_BOOK_TOPICS_FILE
            selected_book_data = all_topics_index
            selected_book_label = _safe_str(
                selected_book_data.get("book_title"),
                _prettify_book_name(selected_book_file),
            )
            st.selectbox("Book", [selected_book_label], disabled=True)

        if selected_book_data:
            topic_options = _collect_topic_options(selected_book_data)

        topic_labels = [DEFAULT_TOPIC_OPTION] + [t["label"] for t in topic_options]
        selected_topic_label = st.selectbox(
            "Topic",
            topic_labels,
            disabled=(len(topic_options) == 0),
        )

        if selected_topic_label != DEFAULT_TOPIC_OPTION:
            matched = next(
                (t for t in topic_options if t["label"] == selected_topic_label),
                None,
            )
            if matched:
                selected_topic_id = matched["topic_id"]
                selected_topic_data = _find_topic_by_id(
                    selected_book_data or {},
                    selected_topic_id,
                )

        st.caption("اکنون می‌توانید کتاب، موضوع، و مسیر یادگیری را به‌صورت ساختاری مرور کنید.")

        st.divider()
        st.subheader("Settings")
        st.write(f"Multi-book mode: `{settings.USE_MULTI_BOOK}`")
        st.write(f"Topics dir: `{settings.TOPICS_INDEX_DIR}`")

        if not settings.USE_MULTI_BOOK:
            st.write(f"Default book file: `{settings.DEFAULT_BOOK_TOPICS_FILE}`")

        if selected_book_file:
            st.write(f"Selected book file: `{selected_book_file}`")

        if selected_book_data:
            st.write(f"Selected book id: `{selected_book_data.get('book_id', '')}`")

        show_debug = st.checkbox("نمایش جزئیات فنی (Debug mode)", value=False)

    if not selected_book_data:
        st.info("برای شروع، یک کتاب انتخاب کنید تا داشبورد آن نمایش داده شود.")
        return

    tab_dashboard, tab_topic, tab_ask, tab_map, tab_plan, tab_eval = st.tabs(
    ["Book Dashboard", "Topic Details", "Ask Book", "Mind Map", "Study Plan", "Quality Dashboard"]
)

    with tab_dashboard:
        _render_book_dashboard(selected_book_data)

    with tab_topic:
        _render_topic_details(selected_topic_data)

    with tab_ask:
        st.subheader("Ask This Book")
        st.caption("UI اکنون برای retrieval کتاب‌محور، topic-aware و خروجی ساختاریافته آماده است.")

        if selected_topic_data:
            st.info(
                f"فیلتر موضوع فعال است: {selected_topic_data.get('title', 'Untitled Topic')}"
            )

        query = st.text_area(
            "سؤال (ترجیحاً فارسی):",
            height=120,
            key="ask_book_query",
        )

        ask = st.button("پرسیدن", type="primary")

        if ask:
            cleaned_query = _safe_str(query)

            if not cleaned_query:
                st.warning("لطفاً ابتدا سؤال خود را وارد کنید.")
            else:
                selected_topic_title = None
                if selected_topic_data:
                    selected_topic_title = selected_topic_data.get("title")

                with st.spinner("در حال تولید پاسخ..."):
                    try:
                                    pipeline_result = run_rag_pipeline(
                                        cleaned_query,
                                        book_id=selected_book_data.get("book_id"),
                                        topic_id=selected_topic_id,
                                    )

                                    result = _normalize_result(pipeline_result)

                                    # -------------------------------
                                    # RAG Evaluation Hook
                                    # -------------------------------
                                    evaluation_dict: Optional[Dict[str, Any]] = None

                                    try:
                                        answer: str = str(
                                            result.get("final_answer")
                                            or result.get("answer")
                                            or result.get("response")
                                            or ""
                                        ).strip()

                                        raw_sources: Any = (
                                            result.get("source_documents")
                                            or result.get("retrieved_docs")
                                            or result.get("documents")
                                            or result.get("sources")
                                            or result.get("chunks")
                                            or []
                                        )

                                        logger.warning("EVAL HOOK ENTERED")
                                        logger.warning("Answer preview: %s", repr(answer)[:200])
                                        logger.warning("Raw sources type: %s", type(raw_sources))
                                        logger.warning("Raw sources preview: %s", repr(raw_sources)[:500])

                                        if raw_sources is None:
                                            raw_sources = []
                                        elif not isinstance(raw_sources, list):
                                            raw_sources = [raw_sources]

                                        source_documents_list: List[str] = []

                                        for item in raw_sources:
                                            if item is None:
                                                continue

                                            text: str = ""

                                            if isinstance(item, str):
                                                text = item.strip()
                                            elif hasattr(item, "page_content"):
                                                text = str(getattr(item, "page_content", "")).strip()
                                            elif isinstance(item, dict):
                                                text = str(
                                                    item.get("page_content")
                                                    or item.get("content")
                                                    or item.get("text")
                                                    or item.get("chunk")
                                                    or ""
                                                ).strip()
                                            else:
                                                text = str(item).strip()

                                            if text:
                                                source_documents_list.append(text)

                                        logger.warning("Normalized contexts count: %d", len(source_documents_list))
                                        if source_documents_list:
                                            logger.warning(
                                                "First normalized context preview: %s",
                                                repr(source_documents_list[0])[:300],
                                            )
                                        else:
                                            logger.warning("No valid normalized contexts extracted")

                                        translated_query: str = str(
                                            result.get("translated_query")
                                            or ""
                                        ).strip()

                                        evaluation_question: str = str(
                                            translated_query or cleaned_query or ""
                                        ).strip()

                                        evaluation_dict = _run_pipeline_evaluation(
                                            question=evaluation_question,
                                            contexts=source_documents_list,
                                            answer=answer,
                                        )

                                        logger.info("Evaluation dict returned: %r", evaluation_dict)
                                        logger.warning("Evaluation dict returned: %s", repr(evaluation_dict)[:500])

                                        if evaluation_dict:
                                            st.session_state["last_evaluation"] = evaluation_dict

                                            if "eval_results" not in st.session_state:
                                                st.session_state["eval_results"] = []

                                            st.session_state["eval_results"].append(evaluation_dict)
                                        else:
                                            logger.warning("No evaluation data available to render.")

                                    except Exception as exc:
                                        logger.exception("RAG evaluation hook failed: %s", exc)
                                        if st.session_state.get("show_debug", False):
                                            st.warning(f"Evaluation failed: {exc}")

                                    result["evaluation"] = evaluation_dict

                                    if evaluation_dict is not None:
                                        st.session_state["last_evaluation"] = evaluation_dict


                                    # -------------------------------

                                    result["ui_selected_book_file"] = selected_book_file
                                    result["ui_selected_book_id"] = selected_book_data.get("book_id")
                                    result["ui_selected_book_title"] = selected_book_data.get("book_title")
                                    result["ui_selected_topic_id"] = selected_topic_id
                                    result["ui_selected_topic_title"] = selected_topic_title


                    except Exception as exc:
                        logger.exception("Pipeline execution failed: %s", exc)
                        st.error("در اجرای pipeline خطایی رخ داد. لطفاً دوباره تلاش کنید.")
                        if show_debug:
                            st.exception(exc)
                        return

                _render_rag_response(result)

                if show_debug:
                    _render_debug_info(result)

    with tab_map:
        st.subheader("Mind Map Preview")
        topics = _safe_list(selected_book_data.get("topics"))

        if topics:
            _render_topic_tree(topics)
        else:
            st.info("No topic structure available.")

    with tab_plan:
        st.subheader("Study Plan")
        study_path = _book_study_path(selected_book_data)

        if study_path:
            st.markdown("### Recommended Reading / Learning Order")
            for step in study_path:
                step_no = _safe_int(step.get("step")) or 0
                st.markdown(f"{step_no}. {_study_step_label(step)}")
        else:
            st.info("No suggested study plan available yet.")

        if selected_topic_data:
            st.divider()
            st.markdown("### Focus Topic")
            st.write(selected_topic_data.get("title", "Untitled Topic"))

            topic_summary = _safe_str(selected_topic_data.get("summary_text"))
            if topic_summary:
                st.markdown(
                    f'<div class="persian-text">{_html_text(topic_summary)}</div>',
                    unsafe_allow_html=True,
                )
    with tab_eval:
        st.subheader("Evaluation Dashboard")
        # Render latest evaluation result
        evaluation = st.session_state.get("last_evaluation")

        if not evaluation:
            st.info("No evaluation result available yet.")
        elif not isinstance(evaluation, dict):
            st.warning("Latest evaluation result has an invalid format.")
            if st.session_state.get("show_debug", False):
                st.write("Raw value:")
                st.write(evaluation)
        else:
            st.subheader("RAG Evaluation")

            scores = evaluation.get("scores") or {}

            with st.expander("Evaluation Results", expanded=True):
                # Scores section
                st.markdown("### Scores")
                if isinstance(scores, dict) and scores:
                    try:
                        st.dataframe(
                            pd.DataFrame([scores]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    except Exception as exc:
                        logger.exception("Failed to render evaluation scores: %s", exc)
                        st.warning("Scores could not be displayed as a table.")
                        if st.session_state.get("show_debug", False):
                            st.json(scores)
                else:
                    st.info("No scores returned.")

                # Optional summary metadata
                question = str(evaluation.get("question") or "").strip()
                answer = str(evaluation.get("answer") or "").strip()
                contexts = evaluation.get("contexts") or []

                if question or answer or contexts:
                    st.markdown("### Evaluation Inputs")

                    if question:
                        st.write("**Question:**")
                        st.code(question, language="text")

                    if answer:
                        st.write("**Answer:**")
                        st.code(answer, language="text")

                    if isinstance(contexts, list) and contexts:
                        st.write(f"**Contexts:** {len(contexts)} item(s)")
                        if st.session_state.get("show_debug", False):
                            st.json(contexts)

                st.write("---")
                st.markdown("### Full Evaluation Payload")
                st.json(evaluation)




if __name__ == "__main__":
    main()
