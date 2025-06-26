# RAG-Based Agent for Operational Data Analysis

## Project Overview
This project aims to create an intelligent RAG-based agent system for operational data analysis. The agent will support multiple data formats, integrate with various systems, and provide insightful, context-aware responses.

## Architecture

### Core Technologies
- **Orchestration**: n8n workflows
- **Database**: Supabase (PostgreSQL + pgvector for Vector Store)
- **AI Models**: OpenAI GPT-4 + Embeddings / Google Gemini
- **Trigger**: WhatsApp Integration
- **Future Integrations**: AWS, SAP, AXIS

### System Components
1.  **Data Ingestion Pipeline**: Processes and vectorizes data from Excel, PDF, Word, and audio files.
2.  **RAG Agent Core**: Performs vector similarity searches, generates context-aware responses, and can create new documents.
3.  **Integration Layer**: Connects with operational data sources, with future plans for financial (SAP) and HR (AXIS) systems.

## Data Schema
The primary data source is an Excel file with 39 columns, including `CONTENEDOR`, `CLIENTE`, `CONDUCTOR`, `FECHA_VIAJE`, and `ESTADO`.

## Project Structure
The project is organized into the following main directories:
- `src/`: Contains all the Python source code for data processing, the RAG engine, API, and integrations.
- `n8n-workflows/`: Holds the JSON definitions for n8n automation workflows.
- `database/`: Includes the Supabase SQL schema and sample data.
- `data/`: Storage for raw data files like the main Excel sheet.

## Getting Started

### Prerequisites
- Python 3.9+
- Pip
- A Supabase account
- API keys for OpenAI/Google Gemini

### Setup
1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file from the `.env.example` and fill in your credentials.
5.  Set up the database schema by running the script in `database/supabase_schema.sql` in your Supabase SQL Editor. **Note**: You must enable the `vector` extension in Supabase first.
# ragpv
