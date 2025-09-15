# AI Lesson Note Generator (RAG)

A full-stack web application that allows educators to generate comprehensive lesson notes from various source materials (PDFs, DOCX, audio, images) using a Retrieval-Augmented Generation (RAG) pipeline.

This application provides a single-page app (SPA) experience for creating lessons, uploading source materials, ingesting them into a vector database, and generating detailed, editable lesson plans using large language models.

## Key Features

- **Lesson Management**: Create, list, update, and delete lessons.
- **Multi-format Source Upload**: Upload sources in various formats including PDF, DOCX, MP3, WAV, and plain text.
- **AJAX Uploads with Progress**: Files are uploaded asynchronously with a visual progress bar.
- **Per-Lesson Data Silos**: Each lesson has its own dedicated vector collection (using ChromaDB), ensuring content remains isolated.
- **RAG-based Content Generation**:
    - **Targeted Generation**: Use a simple prompt to generate or refine specific parts of a lesson note.
    - **Full Lesson Generation**: A dedicated button synthesizes *all* ingested content for a lesson into a complete, well-structured lesson plan based on a comprehensive prompt.
- **Markdown Editor**: Lesson notes are rendered from Markdown in a "view mode" and can be edited in a raw Markdown "edit mode".
- **API-driven Backend**: A robust Flask backend provides a full suite of API endpoints for all frontend functionality.

## How It Works: The RAG Pipeline

The application's core is a Retrieval-Augmented Generation pipeline:

1.  **Upload**: A user uploads source files (e.g., a PDF textbook chapter, a lecture audio). The files are stored and associated with a specific lesson.
2.  **Ingestion**: When ingestion is triggered, the backend processes each file:
    - It extracts raw text (from PDFs, DOCX), transcribes audio, or performs OCR on images.
    - The extracted text is split into smaller, manageable chunks.
    - These chunks are converted into vector embeddings (numerical representations) using a sentence-transformer model.
    - The embeddings are stored in a lesson-specific ChromaDB collection.
3.  **Generation (Retrieval + Augmentation)**:
    - When a user requests a lesson note, a prompt is created.
    - For targeted generation, the system searches the vector database for the most relevant chunks of information related to the user's prompt.
    - For full lesson generation, the system retrieves *all* chunks of information.
    - This retrieved content (the "context") is combined with the user's prompt and sent to a Large Language Model (like Google Gemini).
    - The LLM generates the lesson note based on the rich context provided, and the result is saved and displayed to the user.

## Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **Database**: MariaDB / MySQL
- **Vector Database**: ChromaDB (local persistent storage)
- **RAG & AI**: `google-generativeai`, `sentence-transformers`, `langchain` (for text splitting)
- **Frontend**: Vanilla JavaScript (ES6), Tailwind CSS, Marked.js (for Markdown rendering)
- **Deployment**: Docker & Docker Compose (optional, for containerized setup)

---

## Installation and Setup

You can run this project either locally on your machine or using Docker.

### Method 1: Local Machine Setup (Recommended)

**Prerequisites:**
- Python 3.9+
- A running MariaDB or MySQL server
- `ffmpeg` for audio processing (`pkg install ffmpeg` on Termux, `brew install ffmpeg` on macOS, `sudo apt-get install ffmpeg` on Debian/Ubuntu)

**1. Clone & Setup Environment:**
   - Create a `.env` file from the example: `cp .env.example .env`
   - Edit the `.env` file:
     - Add your `GEMINI_API_KEY`.
     - Set `DATABASE_URL` to point to your local database. Example:
       ```
       DATABASE_URL="mysql+mysqlconnector://YOUR_USER:YOUR_PASSWORD@127.0.0.1:3306/lesson_generator"
       ```

**2. Setup Database:**
   - Log into your MySQL/MariaDB server as a root/admin user.
   - Create the database and grant permissions:
     ```sql
     CREATE DATABASE lesson_generator;
     GRANT ALL PRIVILEGES ON lesson_generator.* TO 'YOUR_USER'@'localhost';
     FLUSH PRIVILEGES;
     ```

**3. Install Dependencies & Run:**
   - Install Python packages:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - Run the Flask application:
     ```bash
     python backend/app.py
     ```

**4. Access the App:**
   - Open your browser and go to `http://localhost:8000`.

### Method 2: Docker Setup

**Prerequisites:**
- Docker
- Docker Compose (v2, i.e., `docker compose`)

**1. Setup Environment:**
   - Create a `.env` file from the example: `cp .env.example .env`
   - Edit the `.env` file with your `GEMINI_API_KEY` and any desired changes to the default MySQL credentials.

**2. Build and Run Containers:**
   ```bash
   docker compose up --build -d
   ```

**3. Access the App:**
   - Open your browser and go to `http://localhost:8000`.
   - The database will be running in a container and accessible to the Flask app at the hostname `db`.

---

## API Endpoints

All endpoints are prefixed with `/api`.

| Method | Endpoint                             | Description                                         |
|--------|--------------------------------------|-----------------------------------------------------|
| GET    | `/lessons`                           | Get a list of all lessons.                          |
| POST   | `/lessons`                           | Create a new lesson.                                |
| GET    | `/lessons/<id>`                      | Get full details for a single lesson.               |
| PUT    | `/lessons/<id>`                      | Update a lesson's title or content.                 |
| DELETE | `/lessons/<id>`                      | Delete a lesson and all its associated data.        |
| GET    | `/lessons/<id>/sources`              | Get a list of uploaded sources for a lesson.        |
| POST   | `/lessons/<id>/sources`              | Upload a new source file.                           |
| DELETE | `/lessons/<id>/sources/<source_id>`  | Delete a source file.                               |
| POST   | `/lessons/<id>/ingest`               | Trigger the ingestion process for uploaded sources. |
| POST   | `/lessons/<id>/generate`             | Generate/refine content with a simple prompt.       |
| POST   | `/lessons/<id>/generate-full`        | Generate a full lesson from all ingested content.   |