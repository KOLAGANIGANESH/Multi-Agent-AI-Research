# Multi-Agent AI Research Studio

Multi-Agent AI Research Studio is a local Streamlit application where specialized AI agents collaborate to research, analyze, summarize, and generate reports on any topic. It uses CrewAI for agent workflows, Google Gemini for language and image understanding, and Tavily for web research.

## Features

- Quick answers with cited web sources.
- Full multi-agent reports with research, analysis, recommendations, and sources.
- Parallel glossary and checklist generation.
- CSV and TXT file analysis with lightweight previews.
- Deep research reports with confidence ratings and multiple perspectives.
- Gemini Vision image analysis for PNG, JPG, and JPEG uploads.
- Cached shared Gemini and Tavily resources for faster repeated use.
- Friendly validation for empty prompts, missing API keys, unsupported files, and runtime errors.
- Markdown download buttons for generated reports.

## Architecture Overview

The app keeps the architecture intentionally simple:

- `app.py` contains the Streamlit UI, cached resource setup, reusable agent factory, workflow functions, file helpers, and image analysis.
- CrewAI agents are created through one helper function to avoid duplicate definitions.
- Tavily searches are cached with `cachetools.TTLCache` to reduce latency and duplicate web calls.
- Gemini is initialized once through Streamlit caching and reused across workflows.
- Gemini Vision is used directly for image understanding because image uploads do not need a full CrewAI workflow.
- Tenacity retries temporary failures in CrewAI and Gemini Vision calls.
- Loguru records runtime exceptions without exposing stack traces in the UI.

## Folder Structure

```text
multi_agent_research_studio/
├── app.py
├── app.ipynb
├── requirements.txt
├── .env.example
└── README.md
```

## Installation Steps

1. Create and enter a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variable Setup

1. Copy `.env.example` to `.env`.
2. Add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
GEMINI_TEXT_MODEL=gemini/gemini-2.5-flash
GEMINI_VISION_MODEL=gemini-2.5-flash
```

Keep `.env` private. Do not commit it to source control.

The Gemini model variables are optional. The app defaults to `gemini-2.5-flash` for both text and vision workflows.

## Running the Application

Start Streamlit from the project folder:

```bash
streamlit run app.py
```

The app will open in your browser. If it does not, Streamlit will print a local URL such as `http://localhost:8501`.

## Example Use Cases

- Research a market, competitor, regulation, technology, or trend.
- Generate an executive-ready report with recommendations.
- Build a glossary and checklist for client onboarding.
- Summarize survey exports or small CSV previews.
- Extract themes and action items from meeting notes.
- Analyze screenshots, product images, whiteboards, or scanned visuals.

## Troubleshooting

- Missing API keys: confirm `.env` exists and contains `GOOGLE_API_KEY` and `TAVILY_API_KEY`.
- Authentication errors: verify both keys are active and copied without extra spaces.
- Gemini 404 model errors: set `GEMINI_TEXT_MODEL` and `GEMINI_VISION_MODEL` to a model available to your API key, such as `gemini/gemini-2.5-flash` for CrewAI text workflows and `gemini-2.5-flash` for Gemini Vision.
- Slow responses: use Quick Answer first, or choose a narrower topic.
- CSV upload issues: confirm the file is valid CSV and not an Excel workbook.
- TXT upload issues: save the file as UTF-8 when possible.
- Image upload issues: use PNG, JPG, or JPEG and avoid extremely large files.
- CrewAI dependency conflicts: create a fresh virtual environment and reinstall from `requirements.txt`.

## Future Enhancements

- Add persistent report history.
- Add selectable Gemini model names.
- Export reports to PDF or DOCX.
- Add richer source quality scoring.
- Add organization-specific prompt templates.
- Support more file formats such as PDF and XLSX.
