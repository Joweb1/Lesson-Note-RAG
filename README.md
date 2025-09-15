
# AI Lesson Generator

This is a single-page Flask web application that uses a Retrieval-Augmented Generation (RAG) pipeline to create AI-generated lesson notes from uploaded PDF, DOCX, audio, and image sources.

## Features

- Create, manage, and delete lessons.
- Upload multiple source files (PDF, DOCX, audio, images) for each lesson.
- Ingest sources into a vector database (ChromaDB).
- Generate and edit lesson content using a large language model (Gemini or DeepSeek).
- Single-page application (SPA) feel with a modern UI.

## Tech Stack

- **Backend**: Flask, Gunicorn, SQLAlchemy, MariaDB/MySQL
- **Frontend**: HTML, Tailwind CSS, vanilla JavaScript
- **Vector Database**: ChromaDB
- **RAG Pipeline**: `google-generativeai`, `sentence-transformers`, `pypdf2`, `python-docx`, `pydub`
- **Containerization**: Docker, Docker Compose

## Setup and Running the Application

### Prerequisites

- Docker
- Docker Compose

### 1. Clone the repository

```bash
# This step is not needed as you are already in the project directory
```

### 2. Configure Environment Variables

Create a `.env` file by copying the example file:

```bash
cp .env.example .env
```

Now, edit the `.env` file and add your API keys for Gemini and/or DeepSeek. You can also change the database credentials if needed.

```
GEMINI_API_KEY="your_gemini_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"

MYSQL_HOST="db"
MYSQL_USER="user"
MYSQL_PASSWORD="password"
MYSQL_DATABASE="lesson_generator"
```

### 3. Build and Run with Docker Compose

From the project root directory, run:

```bash
docker-compose up --build
```

This will build the Docker image for the Flask application and start the `web` and `db` services.

The application will be accessible at [http://localhost:8000](http://localhost:8000).

### 4. Running Tests

To run the backend tests, you can exec into the running `web` container:

```bash
docker-compose exec web pytest
```

**Note**: The `test_generate` test is skipped by default as it makes a live call to the LLM API and requires API keys to be set.

## How to Use

1.  **Create a new lesson**: Click the "+ New Lesson" button and enter a title.
2.  **Open a lesson**: Click on a lesson card in the "Highlights" screen.
3.  **Upload sources**: In the "Sources" screen, click the floating action button to open the upload modal. Select your files and click "Upload".
4.  **Ingest sources**: After uploading, an "Ingest Sources" button will appear. Click it to start the ingestion process.
5.  **Generate content**: In the "Workspace" screen, enter a prompt in the input field at the bottom and click the send button. The AI will generate content based on your prompt and the ingested sources.
6.  **Edit content**: You can directly edit the title and content in the "Workspace". The changes are saved automatically when you click away from the input field or textarea.

## Project Structure

```
project-root/
├─ backend/             # Flask backend
│  ├─ app.py            # Flask app, routes
│  ├─ rag_pipeline.py   # RAG code
│  ├─ models.py         # SQLAlchemy models
│  ├─ db.py             # DB session setup
│  ├─ chroma_adapter.py # ChromaDB manager
│  ├─ storage.py        # File storage helpers
│  ├─ requirements.txt
│  └─ tests/            # Pytest tests
├─ frontend/            # Frontend files
│  ├─ index.html
│  ├─ main.js
│  ├─ ui.css
├─ uploads/             # Persistent file storage
├─ chroma_db/           # Persistent folder for chromadb
├─ .env.example
├─ Dockerfile
└─ docker-compose.yml
```
