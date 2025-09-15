
# ---- RAG pipeline (extracted from provided notebook) ----
# WARNING: I redacted hard-coded API keys that were present in the notebook.
# Please set these in a .env file: GEMINI_API_KEY, DEEPSEEK_API_KEY
import os
import asyncio
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from uuid import uuid4
from typing import List, Dict, Optional, Tuple
import httpx
from PyPDF2 import PdfReader
import docx
from pydub import AudioSegment, exceptions as pydub_exceptions
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from textwrap import dedent
import hashlib 

# Import the new ChromaManager
from chroma_adapter import ChromaManager

# load .env
load_dotenv()

# Redacted keys — originally in the notebook as string literals; use env variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Other notebook configuration variables (example values)
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"

# Logging (as in notebook)
logger = logging.getLogger("multimodal_rag")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("rag_pipeline.log", maxBytes=5*1024*1024, backupCount=2)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---- Text splitting and processing helpers ----
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
            logger.warning("PDF extracted no text.")
            return [], False
        plain_text = full_text
        word_count = len(plain_text.split())
        MAX_TEXT_WORDS_TO_STRUCTURE = 12000
        should_structure = word_count <= MAX_TEXT_WORDS_TO_STRUCTURE
        text_splitter = get_text_splitter('text')
        text_chunks = text_splitter.split_text(plain_text)
        chunks = [{
            "text": chunk,
            "metadata": {
                "source": os.path.basename(pdf_path),
                "word_count": word_count,
                "chunk_id": str(uuid4())
            }
        } for chunk in text_chunks]
        return chunks, should_structure
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        return [], False

def process_docx(docx_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing DOCX: {docx_path}")
    try:
        doc = docx.Document(docx_path)
        full_text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
        if not full_text.strip():
            logger.warning("DOCX extracted no text.")
            return [], False
        plain_text = full_text
        word_count = len(plain_text.split())
        MAX_TEXT_WORDS_TO_STRUCTURE = 12000
        should_structure = word_count <= MAX_TEXT_WORDS_TO_STRUCTURE
        text_splitter = get_text_splitter('text')
        text_chunks = text_splitter.split_text(plain_text)
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
        logger.error(f"Failed to process DOCX: {e}")
        return [], False

async def process_audio(audio_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing audio: {audio_path}")
    try:
        audio = AudioSegment.from_file(audio_path)
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        resp = model.generate_content(["Transcribe this audio file.", genai.upload_file(path=audio_path)])
        transcript = None
        try:
            transcript = resp.text if hasattr(resp, 'text') else None
        except Exception:
            transcript = None
        if not transcript:
            logger.warning("Transcription returned no text.")
            return [], False
        plain_text = transcript
        word_count = len(plain_text.split())
        should_structure = word_count <= 12000
        text_splitter = get_text_splitter('audio')
        text_chunks = text_splitter.split_text(plain_text)
        chunks = [{
            "text": chunk,
            "metadata": {
                "source": os.path.basename(audio_path),
                "word_count": word_count,
                "chunk_id": str(uuid4())
            }
        } for chunk in text_chunks]
        return chunks, should_structure
    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
        return [], False

def process_text(plain_text: str) -> Tuple[List[Dict], bool]:
    try:
        word_count = len(plain_text.split())
        MAX_TEXT_WORDS_TO_STRUCTURE = 12000
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

# ---- JSON parsing and structure helpers ----
def safe_json_loads(text: str) -> Optional[Dict]:
    try:
        cleaned_text = re.sub(r'```json|```', '', str(text), flags=re.IGNORECASE).strip()
        return json.loads(cleaned_text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {e}. Raw text: '{str(text)[:120]}...'")
        return None

def sanitize_user_text(s: str) -> str:
    return re.sub(r'(?i)^(system:|assistant:|developer:).*$','', s).strip()

async def _structure_chunk(client: httpx.AsyncClient, chunk: Dict) -> Dict:
    parsed_json = None
    try:
        system_prompt = dedent('''
        You are a content-structuring assistant. Given an input text, return JSON with:
        {
          "title": "...",
          "summary": "...",
          "learning_objectives": ["..."],
          "difficulty": "Easy|Medium|Hard",
          "content": "..."
        }
        ''')
        payload = {"model": DEEPSEEK_MODEL, "messages": [{"role":"system", "content": system_prompt}, {"role":"user", "content": chunk["text"]}]}
        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                resp = model.generate_content([system_prompt, chunk["text"]])
                answer = resp.text if hasattr(resp, "text") else None
                parsed_json = safe_json_loads(answer) if answer else None
            except Exception as e:
                logger.warning(f"Gemini call failed in structure step: {e}")

        if parsed_json is None and DEEPSEEK_API_KEY:
            try:
                r = await client.post(
                    DEEPSEEK_API_URL,
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                r.raise_for_status()
                data = r.json()
                content = None
                if isinstance(data, dict) and "choices" in data:
                    choice = data["choices"][0] if data["choices"] else {}
                    message = choice.get("message", {})
                    content = message.get("content")
                parsed_json = safe_json_loads(content) if content else None
            except Exception as e:
                logger.warning(f"DeepSeek failed: {e}. Falling back or leaving unstructured.")
    except Exception as e:
        logger.error(f"Structure chunk error: {e}")
    if not parsed_json:
        return {
            "title": chunk["text"][:60],
            "summary": chunk["text"][:200],
            "learning_objectives": [],
            "difficulty": "Unknown",
            "content": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "structured": False
        }
    else:
        parsed_json.setdefault("metadata", chunk.get("metadata", {}))
        parsed_json["structured"] = True
        return parsed_json

async def structure_content_concurrently(chunks: List[Dict], concurrency: int = 5) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        async def limited(c):
            async with sem:
                return await _structure_chunk(client, c)
        tasks = [limited(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r]

# ---- Image processing (simplified from notebook) ----
async def process_image(image_path: str) -> Tuple[List[Dict], bool]:
    logger.info(f"Processing image: {image_path}")
    try:
        # Placeholder for OCR logic
        return [], False
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return [], False

# ---- Main ingestion pipeline (from notebook) ----
async def ingest_source(source_path: str, collection_name: str, source_id: Optional[str] = None):
    """Main async ingestion pipeline to process, structure, and store content."""
    raw_chunks, should_structure = [], False
    
    if not os.path.isfile(source_path):
        logger.warning(f"ingest_source called with non-file source: {source_path}; expecting file path.")
        return

    if source_id is None:
        source_id = os.path.abspath(source_path)

    _, ext = os.path.splitext(source_path)
    ext = ext.lower()
    if ext == '.pdf':
        raw_chunks, should_structure = process_pdf(source_path)
    elif ext == '.docx':
        raw_chunks, should_structure = process_docx(source_path)
    elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
        raw_chunks, should_structure = await process_audio(source_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']:
        raw_chunks, should_structure = await process_image(source_path)
    else:
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                raw_chunks, should_structure = process_text(f.read())
        except Exception as e:
            logger.error(f"Failed to read text file {source_path}: {e}")
            return

    if not raw_chunks:
        logger.warning(f"No chunks extracted from {source_path}; skipping ingestion.")
        return

    chroma_manager = ChromaManager(collection_name=collection_name)

    # Prevent duplication if source_id already exists
    if source_id:
        try:
            results = chroma_manager.collection.get(where={"source_id": source_id}, include=[])
            if results and results['ids']:
                logger.info(f"Source with ID '{source_id}' already ingested in collection '{collection_name}'. Skipping.")
                return
        except Exception as e:
            logger.error(f"Error checking existence of source ID '{source_id}' in ChromaDB: {e}")

    if should_structure and (GEMINI_API_KEY or DEEPSEEK_API_KEY):
        structured = await structure_content_concurrently(raw_chunks, concurrency=4)
    else:
        structured = raw_chunks

    docs = [c.get("content", c.get("text", "")) for c in structured]
    metadatas = []
    ids = []
    for c in structured:
        meta = c.get("metadata", {})
        meta["source_id"] = source_id
        meta["chunk_id"] = meta.get("chunk_id", str(uuid4()))
        metadatas.append(meta)
        ids.append(meta["chunk_id"])

    try:
        chroma_manager.add_documents(docs=docs, metadatas=metadatas, ids=ids)
    except Exception as e:
        logger.error(f"Failed to add documents to chroma for collection {collection_name}: {e}")
        return

# ---- RAG ask function (from notebook) ----
async def ask_rag(question: str, collection_name: str, n_results: int = 5) -> str:
    """
    Query chroma for relevant chunks and call the LLM (Gemini or DeepSeek).
    """
    chroma_manager = ChromaManager(collection_name=collection_name)
    hits = chroma_manager.query(question, n_results=n_results)
    context = "\n\n".join([f"Source: {h.get('source','unknown')} Content: {h.get('content','')}" for h in hits])
    prompt = dedent(f'''
    Use the context below to answer the user's question. Be concise and present results as structured lesson material.
    CONTEXT:
    {context}

    QUESTION:
    {question}
    ''')

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            resp = model.generate_content([prompt])
            return resp.text if hasattr(resp, 'text') else str(resp)
        except Exception as e:
            logger.warning("Gemini answer generation failed. Trying DeepSeek.")

    if DEEPSEEK_API_KEY:
        try:
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role":"user", "content": prompt}]
            }
            async with httpx.AsyncClient() as client:
                r = await client.post(DEEPSEEK_API_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "choices" in data:
                    choice = data["choices"][0] if data["choices"] else {}
                    message = choice.get("message", {})
                    return message.get("content", "")
                return ""
        except Exception as e:
            logger.warning("DeepSeek answer generation failed.")
            return ""
    return "No LLM API key configured or both LLMs failed."

async def ask_rag_with_full_context(question: str, collection_name: str) -> str:
    """
    Retrieves all documents from a collection and uses them as context to call the LLM.
    """
    chroma_manager = ChromaManager(collection_name=collection_name)
    
    try:
        # The get() method with no filter retrieves all items.
        all_docs = chroma_manager.collection.get(include=["documents"])
        
        if not all_docs or not all_docs.get("documents"):
            return "Error: No content has been ingested for this lesson yet."

        # Concatenate all document texts into one large context
        context = "\n\n---\n\n".join(all_docs["documents"])

    except Exception as e:
        logger.error(f"Failed to retrieve all documents from chroma for collection {collection_name}: {e}")
        return f"Error: Could not retrieve content from database: {e}"

    # The 'question' here is the comprehensive prompt from the frontend
    prompt = f"""
    {question}

    CONTEXT:
    {context}
    """

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            resp = await model.generate_content_async([prompt])
            return resp.text if hasattr(resp, 'text') else str(resp)
        except Exception as e:
            logger.warning(f"Gemini answer generation failed for full context: {e}. Trying DeepSeek.")

    if DEEPSEEK_API_KEY:
        try:
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role":"user", "content": prompt}]
            }
            async with httpx.AsyncClient() as client:
                r = await client.post(DEEPSEEK_API_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=180) # Longer timeout for full context
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "choices" in data:
                    choice = data["choices"][0] if data["choices"] else {}
                    message = choice.get("message", {})
                    return message.get("content", "")
                return ""
        except Exception as e:
            logger.warning(f"DeepSeek answer generation failed for full context: {e}")
            return "Error: DeepSeek failed to generate a response."
            
    return "No LLM API key configured or both LLMs failed."
