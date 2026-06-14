from __future__ import annotations

import io
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
os.environ["LOCALAPPDATA"] = str(PROJECT_DIR / ".localappdata")
os.environ["APPDATA"] = str(PROJECT_DIR / ".appdata")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")

import crewai_core.paths as crewai_paths


def _local_crewai_storage() -> str:
    storage_dir = PROJECT_DIR / ".crewai_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


crewai_paths.db_storage_path = _local_crewai_storage

import pandas as pd
import streamlit as st
from cachetools import TTLCache, cached
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv
from google import genai
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

APP_TITLE = "Multi-Agent AI Research Studio"
DEFAULT_TEXT_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_VISION_MODEL = "gemini-2.5-flash"
LEGACY_TEXT_MODELS = {"gemini-1.5-flash", "gemini/gemini-1.5-flash"}
LEGACY_VISION_MODELS = {"gemini-1.5-flash", "models/gemini-1.5-flash"}
TEXT_MODEL = DEFAULT_TEXT_MODEL if os.getenv("GEMINI_TEXT_MODEL") in LEGACY_TEXT_MODELS else os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL)
VISION_MODEL = DEFAULT_VISION_MODEL if os.getenv("GEMINI_VISION_MODEL") in LEGACY_VISION_MODELS else os.getenv("GEMINI_VISION_MODEL", DEFAULT_VISION_MODEL)
MAX_TEXT_CHARS = 12000
CSV_PREVIEW_ROWS = 30
MAX_IMAGE_SIDE = 1400
SUPPORTED_FILE_TYPES = ["csv", "txt", "md", "pdf", "xlsx", "xls"]


class Source(BaseModel):
    title: str = ""
    url: str
    content: str = ""


class ResearchBundle(BaseModel):
    topic: str
    query: str
    sources: list[Source] = Field(default_factory=list)


@dataclass(frozen=True, repr=False)
class AppSecrets:
    google_api_key: str | None
    tavily_api_key: str | None

    @property
    def missing(self) -> list[str]:
        missing_keys = []
        if not self.google_api_key:
            missing_keys.append("GOOGLE_API_KEY")
        if not self.tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
        return missing_keys


def get_secrets() -> AppSecrets:
    return AppSecrets(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str) -> LLM:
    return LLM(model=model, api_key=api_key, temperature=0.25)


@st.cache_resource(show_spinner=False)
def get_vision_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_tavily_tool(api_key: str) -> TavilySearchTool:
    return TavilySearchTool(api_key=api_key, max_results=5)


@cached(cache=TTLCache(maxsize=128, ttl=1800))
def cached_search(topic: str, tavily_api_key: str, max_results: int = 5) -> tuple[dict[str, Any], ...]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=tavily_api_key)
    response = client.search(
        query=topic,
        search_depth="advanced" if max_results > 5 else "basic",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )
    return tuple(response.get("results", []))


def format_sources(results: tuple[dict[str, Any], ...]) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Source"
        url = result.get("url") or ""
        content = (result.get("content") or "")[:650]
        lines.append(f"{index}. {title}\nURL: {url}\nEvidence: {content}")
    return "\n\n".join(lines) or "No sources returned."


def source_markdown(results: tuple[dict[str, Any], ...]) -> str:
    rows = []
    seen = set()
    for result in results:
        url = result.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        title = result.get("title") or url
        rows.append(f"- [{title}]({url})")
    return "\n".join(rows) or "- No source URLs available."


def create_agent(role: str, goal: str, backstory: str, llm: LLM, tools: list[Any] | None = None) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools or [],
        allow_delegation=False,
        verbose=False,
    )


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def kickoff_crew(agents: list[Agent], tasks: list[Task]) -> str:
    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)
    return str(crew.kickoff())


def run_single_agent(prompt: str, role: str, goal: str, backstory: str, llm: LLM, tools: list[Any]) -> str:
    agent = create_agent(role, goal, backstory, llm, tools)
    task = Task(description=prompt, expected_output="A clear, well-structured Markdown response.", agent=agent)
    return kickoff_crew([agent], [task])


def quick_answer(topic: str, llm: LLM, tool: TavilySearchTool, api_key: str) -> tuple[str, str]:
    results = cached_search(topic, api_key, 4)
    evidence = format_sources(results)
    prompt = f"""
Answer this question concisely using the provided Tavily evidence.

Question: {topic}

Evidence:
{evidence}

Return exactly these sections:
## Concise Answer
## Key Insights
## Sources
Use source URLs from the evidence.
"""
    answer = run_single_agent(
        prompt,
        "Fast Research Specialist",
        "Find the most useful evidence and answer user questions quickly.",
        "You are a concise research agent that prioritizes speed, clarity, and cited facts.",
        llm,
        [tool],
    )
    return answer, source_markdown(results)


def full_report(topic: str, llm: LLM, tool: TavilySearchTool, api_key: str) -> tuple[str, str]:
    results = cached_search(topic, api_key, 6)
    evidence = format_sources(results)
    researcher = create_agent(
        "Research Agent",
        "Gather relevant evidence from reliable web sources.",
        "You collect facts, source URLs, and context without over-interpreting the evidence.",
        llm,
        [tool],
    )
    analyst = create_agent(
        "Analysis Agent",
        "Convert evidence into useful insights, opportunities, and risks.",
        "You are a strategic analyst who separates facts from interpretation.",
        llm,
    )
    writer = create_agent(
        "Report Writer Agent",
        "Write polished reports with practical recommendations and citations.",
        "You turn research and analysis into executive-ready Markdown.",
        llm,
    )
    research_task = Task(
        description=f"Review this topic and evidence. Topic: {topic}\n\nEvidence:\n{evidence}",
        expected_output="Research notes with source URLs and key evidence.",
        agent=researcher,
    )
    analysis_task = Task(
        description="Analyze the research notes. Identify key findings, opportunities, risks, and practical insights.",
        expected_output="Structured analysis with opportunities, risks, and insights.",
        agent=analyst,
        context=[research_task],
    )
    writing_task = Task(
        description="""Create a polished Markdown report with exactly these sections:
## Executive Summary
## Key Findings
## Analysis
## Recommendations
## Sources
Include source URLs.""",
        expected_output="A complete Markdown report.",
        agent=writer,
        context=[research_task, analysis_task],
    )
    tasks = [research_task, analysis_task, writing_task]
    return kickoff_crew([researcher, analyst, writer], tasks), source_markdown(results)


def two_outputs(topic: str, llm: LLM) -> tuple[str, str]:
    glossary_prompt = f"Create a concise glossary of the most important terms for this topic: {topic}"
    checklist_prompt = f"Create a practical checklist someone can use for this topic: {topic}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        glossary_future = executor.submit(
            run_single_agent,
            glossary_prompt,
            "Glossary Creator",
            "Define important terms in plain language.",
            "You make complex topics approachable with precise definitions.",
            llm,
            [],
        )
        checklist_future = executor.submit(
            run_single_agent,
            checklist_prompt,
            "Checklist Creator",
            "Create practical, action-oriented checklists.",
            "You translate ideas into steps people can use immediately.",
            llm,
            [],
        )
        return glossary_future.result(), checklist_future.result()


def deep_research(topic: str, llm: LLM, tool: TavilySearchTool, api_key: str) -> tuple[str, str]:
    results = cached_search(topic, api_key, 8)
    evidence = format_sources(results)
    senior = create_agent(
        "Senior Researcher",
        "Gather evidence from multiple viewpoints and identify useful sources.",
        "You are a senior researcher who looks for agreement, disagreement, and missing context.",
        llm,
        [tool],
    )
    strategist = create_agent(
        "Strategic Analyst",
        "Assess credibility, uncertainty, evidence quality, and confidence.",
        "You evaluate evidence carefully and assign High, Medium, or Low confidence.",
        llm,
    )
    author = create_agent(
        "Research Author",
        "Produce professional cited research reports.",
        "You write clear, balanced, source-grounded reports for decision makers.",
        llm,
    )
    research_task = Task(
        description=f"Conduct deep research on: {topic}\n\nUse this Tavily evidence:\n{evidence}",
        expected_output="Evidence notes covering multiple perspectives and source URLs.",
        agent=senior,
    )
    strategy_task = Task(
        description="Assess credibility, uncertainties, limitations, and confidence levels from the research notes.",
        expected_output="Credibility assessment with High, Medium, and Low confidence ratings.",
        agent=strategist,
        context=[research_task],
    )
    author_task = Task(
        description="""Write a final report containing EXACTLY these six Markdown sections:
## 1. Topic Overview
## 2. Evidence and Findings
## 3. Multiple Perspectives
## 4. Risks and Limitations
## 5. Confidence Ratings
## 6. Final Recommendations

Include citations and source URLs. Confidence ratings must use High, Medium, or Low.""",
        expected_output="A professional final report with exactly six sections.",
        agent=author,
        context=[research_task, strategy_task],
    )
    tasks = [research_task, strategy_task, author_task]
    return kickoff_crew([senior, strategist, author], tasks), source_markdown(results)


def read_txt(uploaded_file: Any) -> str:
    content = uploaded_file.getvalue()[: MAX_TEXT_CHARS * 2]
    return content.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]


def read_csv(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), nrows=CSV_PREVIEW_ROWS)
    summary = [
        f"Preview rows loaded: {len(df)}",
        f"Columns: {', '.join(map(str, df.columns[:25]))}",
        f"Preview:\n{df.head(10).to_markdown(index=False)}",
    ]
    return df, "\n\n".join(summary)


def read_excel(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    try:
        workbook = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
    except ImportError as exc:
        raise RuntimeError("Excel uploads require openpyxl. Install requirements.txt, then restart Streamlit.") from exc

    sheet_name = workbook.sheet_names[0]
    df = workbook.parse(sheet_name=sheet_name, nrows=CSV_PREVIEW_ROWS)
    summary = [
        f"Workbook sheets: {', '.join(workbook.sheet_names[:10])}",
        f"Preview sheet: {sheet_name}",
        f"Preview rows loaded: {len(df)}",
        f"Columns: {', '.join(map(str, df.columns[:25]))}",
        f"Preview:\n{df.head(10).to_markdown(index=False)}",
    ]
    return df, "\n\n".join(summary)


def read_pdf(uploaded_file: Any) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF uploads require pypdf. Install requirements.txt, then restart Streamlit.") from exc

    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for page_number, page in enumerate(reader.pages[:10], start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"Page {page_number}:\n{text.strip()}")
        if sum(len(page_text) for page_text in pages) >= MAX_TEXT_CHARS:
            break
    extracted = "\n\n".join(pages)[:MAX_TEXT_CHARS]
    if not extracted.strip():
        raise RuntimeError("No selectable text was found in this PDF. Try a text-based PDF or convert it to TXT.")
    return extracted


def read_uploaded_file(uploaded_file: Any, suffix: str) -> tuple[str, str, pd.DataFrame | None]:
    if suffix == "csv":
        df, file_text = read_csv(uploaded_file)
        return file_text, "CSV", df
    if suffix in {"xlsx", "xls"}:
        df, file_text = read_excel(uploaded_file)
        return file_text, "Excel", df
    if suffix == "pdf":
        return read_pdf(uploaded_file), "PDF", None
    if suffix in {"txt", "md"}:
        return read_txt(uploaded_file), suffix.upper(), None
    raise RuntimeError(f"Unsupported file type: {suffix}")


def file_analysis(file_text: str, file_kind: str, llm: LLM) -> str:
    compact_text = textwrap.shorten(file_text, width=MAX_TEXT_CHARS, placeholder="\n\n[Content truncated for analysis.]")
    prompt = f"""
Analyze this {file_kind} content. Keep the response practical and concise.

Content:
{compact_text}

Return these sections:
## Summary
## Observations
## Trends or Themes
## Action Items
"""
    return run_single_agent(
        prompt,
        "File Analysis Agent",
        "Analyze uploaded CSV and text files efficiently.",
        "You extract useful observations from compact file previews without inventing unavailable facts.",
        llm,
        [],
    )


def optimize_image(uploaded_file: Any) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def analyze_image(image: Image.Image, vision_client: genai.Client) -> str:
    prompt = """
Analyze this image for a business user. Return Markdown with these sections:
## Description
## Objects Detected
## Visible Text
## Contextual Interpretation
## Unusual Details
"""
    response = vision_client.models.generate_content(model=VISION_MODEL, contents=[prompt, image])
    return response.text or "No image analysis was returned."


def validate_ready(secrets: AppSecrets) -> bool:
    if secrets.missing:
        st.error(f"Missing required environment variables: {', '.join(secrets.missing)}")
        st.info("Create a .env file from .env.example, add your API keys, then restart Streamlit.")
        return False
    return True


def render_download(label: str, content: str, filename: str) -> None:
    st.download_button(label, content, file_name=filename, mime="text/markdown", use_container_width=True)


def friendly_error_message(exc: Exception) -> str:
    error_text = str(exc).lower()
    if "not found" in error_text and "models/" in error_text:
        return "The selected Gemini model is not available to this API key. Check GEMINI_TEXT_MODEL or GEMINI_VISION_MODEL in .env."
    if "excel uploads require" in error_text or "pdf uploads require" in error_text:
        return str(exc)
    if "no selectable text" in error_text or "unsupported file type" in error_text:
        return str(exc)
    if "emptydataerror" in error_text or "no columns to parse" in error_text:
        return "The uploaded file could not be read as tabular data. Check that it is not empty and is saved in the expected format."
    if "parsererror" in error_text:
        return "The uploaded CSV could not be parsed. Check the delimiter/quotes, or save it again as a standard CSV."
    if "api key" in error_text or "unauthorized" in error_text or "permission" in error_text:
        return "The API request was rejected. Check that your Google and Tavily keys are active and copied correctly."
    if "connect" in error_text or "timeout" in error_text:
        return "The request could not reach an external API. Check your internet connection and try again."
    return "Something went wrong while running this workflow. Please check your keys, connection, and input, then try again."


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="AI", layout="wide")
    st.title(APP_TITLE)
    st.caption("Research, analyze, summarize, and generate reports with coordinated AI agents.")

    secrets = get_secrets()

    with st.sidebar:
        st.header("Navigation")
        mode = st.selectbox(
            "Mode",
            [
                "Quick Answer",
                "Full Report",
                "Two Outputs at Once",
                "File Analysis",
                "Deep Research",
                "Image Analysis",
            ],
        )
        st.divider()
        if secrets.missing:
            st.warning("API keys needed")
        else:
            st.success("API keys loaded")

    if not validate_ready(secrets):
        return

    llm = get_llm(secrets.google_api_key or "", TEXT_MODEL)
    tavily_tool = get_tavily_tool(secrets.tavily_api_key or "")
    vision_client = get_vision_client(secrets.google_api_key or "")

    try:
        if mode in {"Quick Answer", "Full Report", "Two Outputs at Once", "Deep Research"}:
            topic = st.text_area("Topic or question", height=120, placeholder="Example: How will AI agents change customer support?")
            run = st.button("Run", type="primary", use_container_width=False)
            if run:
                if not topic.strip():
                    st.error("Please enter a topic or question.")
                    return
                with st.spinner("Agents are working..."):
                    if mode == "Quick Answer":
                        output, sources = quick_answer(topic.strip(), llm, tavily_tool, secrets.tavily_api_key or "")
                        st.success("Answer ready")
                        st.markdown(output)
                        st.markdown("### Source URLs")
                        st.markdown(sources)
                    elif mode == "Full Report":
                        output, sources = full_report(topic.strip(), llm, tavily_tool, secrets.tavily_api_key or "")
                        st.success("Report ready")
                        st.markdown(output)
                        st.markdown("### Source URLs")
                        st.markdown(sources)
                        render_download("Download Markdown Report", output, "full_report.md")
                    elif mode == "Two Outputs at Once":
                        glossary, checklist = two_outputs(topic.strip(), llm)
                        st.success("Parallel outputs ready")
                        left, right = st.columns(2)
                        with left:
                            st.subheader("Glossary")
                            st.markdown(glossary)
                        with right:
                            st.subheader("Checklist")
                            st.markdown(checklist)
                    else:
                        output, sources = deep_research(topic.strip(), llm, tavily_tool, secrets.tavily_api_key or "")
                        st.success("Deep research report ready")
                        st.markdown(output)
                        st.markdown("### Source URLs")
                        st.markdown(sources)
                        render_download("Download Deep Research Report", output, "deep_research_report.md")

        elif mode == "File Analysis":
            uploaded_file = st.file_uploader("Upload a file", type=SUPPORTED_FILE_TYPES)
            if uploaded_file:
                suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
                if suffix not in SUPPORTED_FILE_TYPES:
                    st.error("Unsupported file type. Please upload CSV, TXT, Markdown, PDF, XLSX, or XLS.")
                    return
                with st.spinner("Reading file preview..."):
                    file_text, file_kind, df = read_uploaded_file(uploaded_file, suffix)
                    if df is not None:
                        left, right = st.columns([1, 2])
                        with left:
                            st.metric("Preview Rows", len(df))
                            st.metric("Columns", len(df.columns))
                        with right:
                            st.dataframe(df.head(10), use_container_width=True)
                    else:
                        st.text_area("Text preview", file_text[:3000], height=220)
                if st.button("Analyze File", type="primary"):
                    with st.spinner("Analysis agent is reviewing the file..."):
                        output = file_analysis(file_text, file_kind, llm)
                        st.success("File analysis ready")
                        st.markdown(output)

        else:
            uploaded_image = st.file_uploader("Upload PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                with st.spinner("Optimizing image..."):
                    image = optimize_image(uploaded_image)
                st.image(image, caption="Uploaded image", use_container_width=True)
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Gemini Vision is analyzing the image..."):
                        output = analyze_image(image, vision_client)
                        st.success("Image analysis ready")
                        st.markdown(output)

    except Exception as exc:
        logger.error("Application error: {}", exc.__class__.__name__)
        st.error(friendly_error_message(exc))
        st.caption(f"Error summary: {exc.__class__.__name__}")


if __name__ == "__main__":
    main()
