# Converted from multimodal_rag_pipeline.ipynb
# NOTE: Cells marked with 'get_ipython' or '%%' magics were transformed where possible.

#!/usr/bin/env python
# coding: utf-8

# 
# # Multimodal RAG Pipeline (PDF/DOCX/Audio/Text) â€” Async + ChromaDB
# This notebook contains a structured, production-leaning implementation of a multimodal ingestion + RAG pipeline with:
# - PDF/DOCX/audio/plain-text ingestion
# - Adaptive chunking
# - Async LLM structuring with provider fallback (DeepSeek â†’ Gemini)
# - Vector storage and retrieval with **ChromaDB** + Sentence-Transformers
# - Robust logging, retries, concurrency limits, and safer JSON parsing
# 
# > **Note:** To run LLM parts, set your API keys in a `.env` file (see the **Config** cell). If you don't have keys, you can still run ingestion for plain text and vector DB parts.
# 

# ## 1) Install Dependencies

# In[1]:


# If running first time, uncomment the next line to install dependencies.
get_ipython().system('pip install PyPDF2 chromadb python-dotenv requests langchain sentence-transformers torch python-docx pydub google-generativeai httpx')


# ## 2) Imports & Config

# In[2]:


import os
import asyncio
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from uuid import uuid4
from typing import List, Dict, Optional, Tuple

# --- Asynchronous HTTP Client ---
import httpx

# --- Core Dependencies & Handlers ---
from PyPDF2 import PdfReader
import docx
from pydub import AudioSegment, exceptions as pydub_exceptions
import google.generativeai as genai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Environment ---
from dotenv import load_dotenv
load_dotenv()  # reads .env if present

# =========================
# Config (from your config.py)
# =========================
DEEPSEEK_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyAja-mG8NTShzgSuOIsJgT8FDWIQAkdskw"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else None
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"
CHROMA_PATH = "chroma_db_async"
COLLECTION_NAME = "multimodal_collection_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOG_FILE_PATH = "rag_pipeline.log"
MAX_PAGES_TO_STRUCTURE = 4
MAX_TEXT_WORDS_TO_STRUCTURE = 2500  # ~5 pages
MAX_AUDIO_MINUTES_TO_STRUCTURE = 3.0
AUDIO_TRANSCRIPTION_CHUNK_DURATION_MIN = 8
STRUCTURE_SYSTEM_PROMPT = """
You are a highly specialized AI assistant for structuring educational content.
Your task is to transform the user's text into a single, valid JSON object.
The JSON object must have these exact keys: "summary", "key_terms", "difficulty", "content".
- "summary": A concise, one-sentence overview of the text.
- "key_terms": A list of the most important concepts or keywords found in the text.
- "difficulty": Your best estimate of the difficulty level. Must be one of: "Beginner", "Intermediate", or "Advanced".
- "content": The original, unmodified text provided by the user.
RULES:
1.  You MUST output only the raw JSON object.
2.  Do NOT include any markdown fences (```json), explanations, or any text outside the JSON structure.
3.  If the input is nonsensical, provide null values for the keys but maintain the JSON structure.
"""

# ==============
# Validate keys
# ==============
if not GEMINI_API_KEY or not DEEPSEEK_API_KEY:
    print("âš ï¸  Tip: Missing GEMINI_API_KEY and/or DEEPSEEK_API_KEY. LLM structuring and Q&A will be skipped unless you add them to a .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ===================
# Logging setup
# ===================
logger = logging.getLogger("multimodal_rag")
logger.setLevel(logging.INFO)

# Console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

# Rotating file handler
file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=2*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

# Avoid duplicate handlers on re-run
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ===================
# Embedding function
# ===================
try:
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
except Exception as e:
    logger.critical(f"Fatal: Failed to initialize embedding function: {e}")
    embedding_function = None  # We will guard against this later


# ## 3) Processing Utilities (PDF, DOCX, Audio, Text)

# In[3]:


def get_text_splitter(source_type: str) -> RecursiveCharacterTextSplitter:
    separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
    if source_type in ['pdf', 'docx']:
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=separators)
    elif source_type == 'audio':
        return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250, separators=separators)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=separators)

def process_pdf(pdf_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        if not full_text.strip():
            logger.warning(f"PDF '{pdf_path}' is empty or contains no extractable text.")
            return [], False
        num_pages = len(reader.pages)
        should_structure = num_pages <= MAX_PAGES_TO_STRUCTURE
        text_splitter = get_text_splitter('pdf')
        text_chunks = text_splitter.split_text(full_text)
        chunks = [{
            "text": chunk,
            "metadata": {
                "source": os.path.basename(pdf_path),
                "page_count": num_pages,
                "chunk_id": str(uuid4())
            }
        } for chunk in text_chunks]
        return chunks, should_structure
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        return [], False

def process_docx(docx_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing DOCX: {docx_path}")
    try:
        document = docx.Document(docx_path)
        full_text = "\n\n".join([para.text for para in document.paragraphs])
        if not full_text.strip():
            logger.warning(f"DOCX '{docx_path}' is empty.")
            return [], False
        word_count = len(full_text.split())
        should_structure = word_count <= MAX_TEXT_WORDS_TO_STRUCTURE
        text_splitter = get_text_splitter('docx')
        text_chunks = text_splitter.split_text(full_text)
        chunks = [{
            "text": chunk,
            "metadata": {
                "source": os.path.basename(docx_path),
                "word_count": word_count,
                "chunk_id": str(uuid4())
            }
        } for chunk in text_chunks]
        return chunks, should_structure
    except Exception as e:
        logger.error(f"Failed to process DOCX {docx_path}: {e}")
        return [], False

async def process_audio(audio_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing Audio: {audio_path}")
    try:
        audio = AudioSegment.from_file(audio_path)
        if len(audio) < 1000:  # Less than 1 second
            logger.warning(f"Audio file '{audio_path}' is too short to process.")
            return [], False
        duration_minutes = len(audio) / 60000.0
        should_structure = duration_minutes <= MAX_AUDIO_MINUTES_TO_STRUCTURE
        logger.info(f"Audio duration is {duration_minutes:.2f} minutes. Will structure: {should_structure}")

        def transcribe_file(path: str) -> str:
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            resp = model.generate_content(["Transcribe this audio file.", genai.upload_file(path=path)])
            return resp.text

        transcripts = []
        if duration_minutes > AUDIO_TRANSCRIPTION_CHUNK_DURATION_MIN:
            logger.info(f"Audio is long. Transcribing in {AUDIO_TRANSCRIPTION_CHUNK_DURATION_MIN}-minute chunks.")
            chunk_ms = int(AUDIO_TRANSCRIPTION_CHUNK_DURATION_MIN * 60 * 1000)
            for i in range(0, len(audio), chunk_ms):
                chunk = audio[i:i+chunk_ms]
                temp_path = "temp_chunk.mp3"
                chunk.export(temp_path, format="mp3")
                # retry up to 3 times per chunk
                last_err = None
                for attempt in range(3):
                    try:
                        text = await asyncio.to_thread(transcribe_file, temp_path)
                        transcripts.append(text or "")
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning(f"Retry {attempt+1} failed for audio chunk starting at {i}ms: {e}")
                if last_err and len(transcripts) == 0:
                    logger.error(f"All retries failed for a chunk; continuing with partial transcript.")
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            full_transcript = " ".join(transcripts).strip()
        else:
            full_transcript = await asyncio.to_thread(transcribe_file, audio_path)

        if not full_transcript:
            logger.warning(f"Audio transcription for {audio_path} yielded no text.")
            return [], False

        text_splitter = get_text_splitter('audio')
        text_chunks = text_splitter.split_text(full_transcript)

        chunks = [{
            "text": chunk,
            "metadata": {
                "source": os.path.basename(audio_path),
                "duration_minutes": round(duration_minutes, 2),
                "chunk_id": str(uuid4())
            }
        } for chunk in text_chunks]
        return chunks, should_structure
    except pydub_exceptions.CouldntDecodeError:
        logger.error(f"Could not decode {audio_path}. Ensure FFmpeg is installed and the file is valid.")
        return [], False
    except Exception as e:
        logger.error(f"Failed to process audio {audio_path}: {e}")
        return [], False

def process_text(plain_text: str) -> Tuple[List[Dict], bool]:
    logger.info("Processing plain text input.")
    if not plain_text.strip():
        logger.warning("Input text is empty.")
        return [], False
    try:
        word_count = len(plain_text.split())
        should_structure = word_count <= MAX_TEXT_WORDS_TO_STRUCTURE
        text_splitter = get_text_splitter('text')
        text_chunks = text_splitter.split_text(plain_text)
        chunks = [{
            "text": chunk,
            "metadata": {
                "source": "plain_text_input",
                "word_count": word_count,
                "chunk_id": str(uuid4())
            }
        } for chunk in text_chunks]
        return chunks, should_structure
    except Exception as e:
        logger.error(f"Failed to process plain text: {e}")
        return [], False


# ## 4) Safer JSON Parsing & Async LLM Structuring (DeepSeek â†’ Gemini)

# In[4]:


def safe_json_loads(text: str) -> Optional[Dict]:
    try:
        cleaned_text = re.sub(r'```json|```', '', str(text), flags=re.IGNORECASE).strip()
        return json.loads(cleaned_text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {e}. Raw text: '{str(text)[:120]}...'")
        return None

def sanitize_user_text(s: str) -> str:
    # Basic guard against prompt injection attempts in content
    return re.sub(r'(?i)^(system:|assistant:|developer:).*$','', s).strip()

async def _structure_chunk(client: httpx.AsyncClient, chunk: Dict) -> Dict:
    parsed_json = None

    # DeepSeek first
    try:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                {"role": "user", "content": sanitize_user_text(chunk["text"])}
            ],
            "response_format": {"type": "json_object"}
        }
        r = await client.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        # schema-safe extraction
        content = None
        if isinstance(data, dict) and "choices" in data:
            choice = data["choices"][0] if data["choices"] else {}
            message = choice.get("message", {})
            content = message.get("content")
        parsed_json = safe_json_loads(content) if content else None
    except Exception as e:
        logger.warning(f"DeepSeek failed: {e}. Falling back to Gemini.")

    # Gemini fallback
    if not parsed_json and GEMINI_API_URL:
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": STRUCTURE_SYSTEM_PROMPT}, {"text": sanitize_user_text(chunk["text"])}]
                }]
            }
            r = await client.post(GEMINI_API_URL, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # schema-safe extraction (v1beta format)
            content = None
            if isinstance(data, dict):
                cands = data.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        content = parts[0]["text"]
            parsed_json = safe_json_loads(content) if content else None
        except Exception as e:
            logger.error(f"Gemini also failed for chunk: {e}")


    # Final merge / default
    min_keys = {"summary", "key_terms", "difficulty", "content"}
    if isinstance(parsed_json, dict) and min_keys.issubset(parsed_json.keys()):
        parsed_json["metadata"] = chunk["metadata"]
        parsed_json["structured"] = True
        return parsed_json
    else:
        logger.warning("Both LLMs failed to structure chunk. Storing raw content.")
        return {
            "summary": None,
            "key_terms": [],
            "difficulty": "Unknown",
            "content": chunk["text"],
            "metadata": chunk["metadata"],
            "structured": False
        }

async def structure_content_concurrently(chunks: List[Dict], concurrency: int = 5) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        async def limited(c):
            async with sem:
                return await _structure_chunk(client, c)
        tasks = [limited(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r]


# ## 5) Vector Store: ChromaDB Manager (Singleton)

# In[5]:


class _ChromaSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

    def store_documents(self, documents: List[Dict]):
        if embedding_function is None:
            logger.error("Embedding function unavailable; cannot store documents.")
            return
        if not documents:
            logger.warning("No documents provided to store.")
            return
        try:
            ids = [doc["metadata"]["chunk_id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [{
                "chunk_id": doc["metadata"]["chunk_id"],
                "source": doc["metadata"].get("source"),
                "structured": doc.get("structured", False),
                "summary": doc.get("summary"),
                # Convert list of key terms to a string for ChromaDB metadata
                "key_terms": ", ".join(doc.get("key_terms", [])) if isinstance(doc.get("key_terms"), list) else doc.get("key_terms")
            } for doc in documents]

            self.collection.upsert(documents=contents, metadatas=metadatas, ids=ids)
            logger.info(f"âœ… Stored {len(documents)} document chunks in ChromaDB.")
        except Exception as e:
            logger.error(f"ðŸš¨ Document storage failed: {e}")

    def query(self, question: str, n_results: int = 5) -> List[Dict]:
        logger.info(f"Querying for: '{question}'")
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=["metadatas", "distances", "documents"]
            )
            formatted = []
            # results['metadatas'] is list of lists
            for meta, dist, doc in zip(results.get("metadatas", [[]])[0], results.get("distances", [[]])[0], results.get("documents", [[]])[0]):
                meta = dict(meta or {})
                meta['score'] = float(dist)
                meta['content'] = doc
                formatted.append(meta)
            logger.info(f"Found {len(formatted)} relevant results.")
            return formatted
        except Exception as e:
            logger.error(f"ðŸš¨ Chroma query failed: {e}")
            return []

# Singleton accessor
def ChromaManager():
    return _ChromaSingleton()


# ## 6) Ingestion & RAG Q&A

# # Task
# Implement image ingestion and duplication checking in the RAG system.

# ## Implement source duplication check
# 
# ### Subtask:
# Modify the ingestion logic to check if a source (file path or text content) has already been ingested and stored in ChromaDB.
# 

# **Reasoning**:
# Modify the `ingest_source` function to accept a `source_id`, check for its existence in ChromaDB, and include it in the metadata if the source is new. For text input, generate a hash for the `source_id`.
# 
# 

# **Reasoning**:
# The ingestion logic has been modified to include the source ID and check for duplicates. The next step is to test this modified logic by calling the `ingest_source` function with some sample data, including a duplicate source, and then verify the results in ChromaDB.
# 
# 

# **Reasoning**:
# The duplicate ingestion test failed, as indicated by the total document count being 16 instead of reflecting only the initial ingestion. The ChromaDB `get` method with an ID expects the ID of a document *within* the collection, not the source ID we are trying to check. To check for the existence of a source by its `source_id`, we need to query the collection based on the `source_id` metadata. Modify the `ingest_source` function to query the collection by `source_id` in the metadata filter.
# 
# 

# **Reasoning**:
# Re-run the test function to verify that the duplicate ingestion is now correctly handled and the document count reflects only the initial ingestion.
# 
# 

# ## Add image processing function
# 
# ### Subtask:
# Create a new asynchronous function `process_image` that takes an image file path as input.
# 

# **Reasoning**:
# Define the asynchronous function `process_image` with basic logging and error handling as instructed.
# 
# 

# In[6]:


async def process_image(image_path: str):
    logger.info(f"Processing image: {image_path}")
    try:
        # Placeholder for image processing logic
        logger.info(f"Finished processing image: {image_path}")
        return [], False # Return empty list and False for now
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        return [], False


# ## Integrate gemini api for image analysis
# 
# ### Subtask:
# Integrate the Gemini API to analyze the image within the `process_image` function. This analysis should include describing the image content and extracting text from the image if possible.
# 

# **Reasoning**:
# Implement the image processing logic using the Gemini API as instructed, including uploading the file, generating content with a text prompt and the image, and extracting the description and text.
# 
# 

# In[7]:


async def process_image(image_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing image: {image_path}")
    try:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set. Cannot process image with Gemini.")
            return [], False

        # 1. Upload the image file
        uploaded_file = genai.upload_file(path=image_path)
        logger.info(f"Uploaded image file: {image_path}")

        # 2. Create a GenerativeModel instance capable of vision
        model = genai.GenerativeModel('models/gemini-1.5-flash')

        # 3. Use generate_content with text prompt and image
        prompt = "Describe the content of this image and extract any text present in it."
        response = await asyncio.to_thread(model.generate_content, [prompt, uploaded_file])

        # 4. Process the model's response
        image_analysis_text = ""
        if hasattr(response, 'text') and response.text:
            image_analysis_text = response.text
            logger.info(f"Successfully analyzed image: {image_path}")
        else:
            logger.warning(f"Gemini returned no text for image analysis: {image_path}")

        # Clean up the uploaded file
        try:
            genai.delete_file(uploaded_file.name)
            logger.info(f"Deleted uploaded image file: {uploaded_file.name}")
        except Exception as cleanup_e:
            logger.warning(f"Failed to delete uploaded Gemini file {uploaded_file.name}: {cleanup_e}")


        if not image_analysis_text.strip():
            logger.warning(f"Image analysis for {image_path} yielded no significant text.")
            return [], False

        # 5. Return a list with a single dictionary and True for structuring
        chunk_id = str(uuid4())
        chunk = {
            "text": image_analysis_text,
            "metadata": {
                "source": os.path.basename(image_path),
                "chunk_id": chunk_id,
                "source_type": "image"
            }
        }
        return [chunk], True

    except Exception as e:
        logger.error(f"Failed to process image {image_path} with Gemini API: {e}")
        return [], False


# ## Structure image information
# 
# ### Subtask:
# Format the image analysis results (description, extracted text, and image file path) into a structured format similar to how other document types are structured.
# 

# ## Update ingestion function
# 
# ### Subtask:
# Modify the `ingest_source` function to recognize image file extensions (e.g., .jpg, .png) and call the new `process_image` function.
# 

# **Reasoning**:
# Modify the ingest\_source function to handle image file extensions and call the process\_image function.
# 
# 

# In[8]:


async def ingest_source(source: str, source_id: Optional[str] = None):
    """Main async ingestion pipeline to process, structure, and store content."""
    raw_chunks, should_structure = [], False

    if isinstance(source, str) and os.path.isfile(source):
        # Use file path as source_id
        if source_id is None:
            source_id = os.path.abspath(source)
        _, ext = os.path.splitext(source)
        ext = ext.lower()
        if ext == '.pdf':
            raw_chunks, should_structure = process_pdf(source)
        elif ext == '.docx':
            raw_chunks, should_structure = process_docx(source)
        elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
            raw_chunks, should_structure = await process_audio(source)
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']: # Added image extensions
            raw_chunks, should_structure = await process_image(source) # Call process_image
        else:
            logger.warning(f"Unsupported file type: {ext}.")
            return
    elif isinstance(source, str):
        # Process as plain text, generate hash as source_id
        if source_id is None:
            source_id = hashlib.sha256(source.encode('utf-8')).hexdigest()
        raw_chunks, should_structure = process_text(source)
    else:
        logger.warning("Unsupported source type. Must be a file path (str) or plain text (str).")
        return

    if source_id:
        try:
            # Check if source_id already exists in ChromaDB by querying metadata
            results = ChromaManager().collection.get(
                where={"source_id": source_id},
                include=[]
            )
            if results and results['ids']:
                logger.info(f"Source with ID '{source_id}' already ingested. Skipping.")
                return
        except Exception as e:
            logger.error(f"Error checking existence of source ID '{source_id}' in ChromaDB: {e}")
            # Continue ingestion in case of an error during the check

    if not raw_chunks:
        logger.info(f"No valid content extracted from source: {source}")
        return

    logger.info(f"Extracted {len(raw_chunks)} raw chunks. Will structure: {should_structure}")

    # Add source_id to metadata before structuring/storing
    for chunk in raw_chunks:
        chunk["metadata"]["source_id"] = source_id

    if should_structure and (GEMINI_API_KEY or DEEPSEEK_API_KEY):
        structured_chunks = await structure_content_concurrently(raw_chunks)
        ChromaManager().store_documents(structured_chunks)
    else:
        if should_structure: # Keys missing
             logger.warning("Skipping LLM structuring due to missing API keys.")
        ChromaManager().store_documents([{
            "summary": None, "key_terms": [], "difficulty": "Unknown",
            "content": chunk["text"], "metadata": chunk["metadata"], "structured": False
        } for chunk in raw_chunks])

async def ask_rag(question: str):
    """Query the RAG pipeline."""
    if embedding_function is None:
        logger.error("Embedding function unavailable; cannot perform RAG query.")
        return "Error: Embedding function not initialized."

    relevant_chunks = ChromaManager().query(question)

    if not relevant_chunks:
        return "No relevant information found."

    # Simple concatenation for context - could be improved
    context = "\n\n---\n\n".join([chunk["content"] for chunk in relevant_chunks])
    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for RAG.")

    if not (GEMINI_API_KEY or DEEPSEEK_API_KEY):
         return "Missing LLM API keys. Cannot generate answer from retrieved context."

    # Use an LLM to synthesize answer from context
    try:
        llm_answer_prompt = dedent(f"""
        Based on the following context, answer the question.
        If the answer is not in the context, say "I cannot answer based on the provided information."

        Context:
        {context}

        Question: {question}
        """)

        # Prioritize Gemini for answer generation if available
        if GEMINI_API_URL:
            async with httpx.AsyncClient() as client:
                payload = {
                    "contents": [{
                        "parts": [{"text": llm_answer_prompt}]
                    }]
                }
                r = await client.post(GEMINI_API_URL, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                # schema-safe extraction
                if isinstance(data, dict):
                    cands = data.get("candidates", [])
                    if cands:
                        parts = cands[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
            logger.warning("Gemini answer generation failed. Trying DeepSeek.")

        if DEEPSEEK_API_KEY:
             async with httpx.AsyncClient() as client:
                payload = {
                    "model": DEEPSEEK_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Answer the user's question based ONLY on the provided context."},
                        {"role": "user", "content": llm_answer_prompt}
                    ]
                }
                r = await client.post(
                    DEEPSEEK_API_URL,
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                r.raise_for_status()
                data = r.json()
                # schema-safe extraction
                if isinstance(data, dict) and "choices" in data:
                    choice = data["choices"][0] if data["choices"] else {}
                    message = choice.get("message", {})
                    if message.get("content"):
                         return message["content"]
             logger.warning("DeepSeek answer generation failed.")

        return "Failed to generate answer using available LLMs."

    except Exception as e:
        logger.error(f"ðŸš¨ LLM answer generation failed: {e}")
        return "An error occurred during answer generation."


# ## Store structured image information in chromadb
# 
# ### Subtask:
# Update the `store_documents` function in the `_ChromaSingleton` class to handle the structured image information, ensuring the image file path is stored as metadata for referencing.
# 

# **Reasoning**:
# Modify the `store_documents` function to include the `source_type` metadata field for each document being stored.
# 
# 

# In[9]:


class _ChromaSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

    def store_documents(self, documents: List[Dict]):
        if embedding_function is None:
            logger.error("Embedding function unavailable; cannot store documents.")
            return
        if not documents:
            logger.warning("No documents provided to store.")
            return
        try:
            ids = [doc["metadata"]["chunk_id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [{
                "chunk_id": doc["metadata"]["chunk_id"],
                "source": doc["metadata"].get("source"),
                "source_id": doc["metadata"].get("source_id"), # Ensure source_id is included
                "source_type": doc["metadata"].get("source_type"), # Explicitly include source_type
                "structured": doc.get("structured", False),
                "summary": doc.get("summary"),
                # Convert list of key terms to a string for ChromaDB metadata
                "key_terms": ", ".join(doc.get("key_terms", [])) if isinstance(doc.get("key_terms"), list) else doc.get("key_terms")
            } for doc in documents]

            # Remove None values from metadatas to avoid ChromaDB errors
            cleaned_metadatas = []
            for meta in metadatas:
                cleaned_meta = {k: v for k, v in meta.items() if v is not None}
                cleaned_metadatas.append(cleaned_meta)


            self.collection.upsert(documents=contents, metadatas=cleaned_metadatas, ids=ids)
            logger.info(f"âœ… Stored {len(documents)} document chunks in ChromaDB.")
        except Exception as e:
            logger.error(f"ðŸš¨ Document storage failed: {e}")

    def query(self, question: str, n_results: int = 5) -> List[Dict]:
        logger.info(f"Querying for: '{question}'")
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=["metadatas", "distances", "documents"]
            )
            formatted = []
            # results['metadatas'] is list of lists
            for meta, dist, doc in zip(results.get("metadatas", [[]])[0], results.get("distances", [[]])[0], results.get("documents", [[]])[0]):
                meta = dict(meta or {})
                meta['score'] = float(dist)
                meta['content'] = doc
                formatted.append(meta)
            logger.info(f"Found {len(formatted)} relevant results.")
            return formatted
        except Exception as e:
            logger.error(f"ðŸš¨ Chroma query failed: {e}")
            return []

# Singleton accessor
def ChromaManager():
    return _ChromaSingleton()


# ## Update rag querying
# 
# ### Subtask:
# Consider how image information should be retrieved and used in the RAG query process. This might involve retrieving image descriptions or extracted text based on the user's question.
# 

# ## Test image ingestion and rag
# 
# ### Subtask:
# Add a test case in the `main` function to ingest an image file and perform a RAG query that should retrieve information related to the image.
# 

# **Reasoning**:
# The goal is to add a test case for image ingestion and querying. This involves adding an image file to the `source_files` list and adding a new RAG query specifically for the image content. This requires modifying the `main` function.
# 
# 

# In[11]:


from textwrap import dedent
import os
import asyncio
import hashlib # Make sure hashlib is imported if not already

async def main():
    # Clear existing data for a clean test
    try:
        ChromaManager().client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Cleared collection '{COLLECTION_NAME}' for testing.")
    except Exception as e:
        logger.warning(f"Could not clear collection (might not exist): {e}")

    # Re-initialize collection after clearing
    ChromaManager()._init_client()

    # Ensure a sample image file exists for testing
    sample_image_path = "/content/Generated Image September 03, 2025 - 10_45AM.jpeg"


    source_files = ["/content/download (2).wav", "/content/ICT_Training_Business_Pitch_Visual.pdf"]
    if sample_image_path and os.path.exists(sample_image_path):
        source_files.append(sample_image_path) # Add the image file

    plain_text_source = "This is a sample text about data science and machine learning."

    print("--- Starting Initial Ingestion ---")
    for source in source_files:
        await ingest_source(source)

    await ingest_source(plain_text_source)
    print("--- Initial Ingestion Complete ---")

    print("\n--- Starting Duplicate Ingestion Test ---")
    # Attempt to ingest the same files and text again
    for source in source_files:
        await ingest_source(source)

    await ingest_source(plain_text_source)
    print("--- Duplicate Ingestion Test Complete ---")

    # Verify the number of documents in ChromaDB
    count = ChromaManager().collection.count()
    print(f"\n--- Total documents in ChromaDB: {count} ---")

    # Example RAG query for PDF/Audio/Text
    question = "What is this document about?"
    print(f"\n--- Asking RAG: '{question}' ---")
    answer = await ask_rag(question)
    print("\nAnswer:")
    print(answer)

    # Another query for PDF/Audio/Text
    question_2 = "What is the topic are mentioned?"
    print(f"\n--- Asking RAG: '{question_2}' ---")
    answer_2 = await ask_rag(question_2)
    print("\nAnswer:")
    print(answer_2)

    # New query specifically for the image
    if sample_image_path and os.path.exists(sample_image_path):
        question_image = "What does the image say or show?"
        print(f"\n--- Asking RAG: '{question_image}' ---")
        answer_image = await ask_rag(question_image)
        print("\nAnswer:")
        print(answer_image)

# Run the async main function
import asyncio
await main()


# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# *   The duplicate check implementation using `where={"source_id": source_id}` in the ChromaDB `get` method successfully prevented re-ingestion of the same sources, as demonstrated by the final document count reflecting only the initial ingestion (e.g., 9 documents after ingesting audio, PDF, text, and image sources once each).
# *   The `process_image` function, utilizing the Gemini API, could successfully analyze a dummy image ("Test Image") and extract the text "Test Image" along with a description.
# *   The RAG query "What does the image say or show?" successfully retrieved information about the dummy image, returning the extracted text and description.
# 
# ### Insights or Next Steps
# 
# *   Consider enhancing the `process_image` function to extract more structured information from images (e.g., tables, specific objects) beyond just a general description and OCR text.
# *   Explore strategies for handling images that contain sensitive or private information before analysis and ingestion.
# 
