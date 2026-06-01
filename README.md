# Credit Risk RAG

## Run

uv sync

## Ingest

from app.ingestion import ingest_pdf
ingest_pdf("data/Capstone_Project_3_Intelligent_Credit_Risk_Assessment_FAQ.pdf")

python app/ingestion.py

## Start API

uv run uvicorn main:app --reload

## Start UI

uv run streamlit run ui/streamlit_app.py
