
import os
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from db import get_db, engine
from models import Base, Lesson, Source, SourceStatus
from storage import save_file, delete_file
import rag_pipeline

# Create tables
Base.metadata.create_all(bind=engine)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app) # Allow all origins for simplicity in this example

# Ensure the upload folder exists
if not os.path.exists('uploads/lessons'):
    os.makedirs('uploads/lessons')

@app.route("/api/lessons", methods=["GET"])
def get_lessons():
    db = next(get_db())
    lessons = db.query(Lesson).order_by(Lesson.created_at.desc()).all()
    return jsonify([{"id": l.id, "title": l.title, "created_at": l.created_at} for l in lessons])

@app.route("/api/lessons", methods=["POST"])
def create_lesson():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({"error": "Title is required"}), 400

    db = next(get_db())
    new_lesson = Lesson(title=data['title'], content="")
    db.add(new_lesson)
    db.commit()
    db.refresh(new_lesson)

    # Set chroma_collection name
    new_lesson.chroma_collection = f"lesson_{new_lesson.id}"
    db.commit()

    return jsonify({"id": new_lesson.id, "title": new_lesson.title, "content": new_lesson.content, "chroma_collection": new_lesson.chroma_collection}), 201

@app.route("/api/lessons/<int:lesson_id>", methods=["GET"])
def get_lesson(lesson_id):
    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404
    
    sources = db.query(Source).filter(Source.lesson_id == lesson_id).all()
    return jsonify({
        "id": lesson.id,
        "title": lesson.title,
        "content": lesson.content,
        "sources": [
            {
                "id": s.id,
                "filename": s.filename,
                "file_type": s.file_type,
                "status": s.status.value,
                "uploaded_at": s.uploaded_at
            } for s in sources
        ]
    })

@app.route("/api/lessons/<int:lesson_id>", methods=["PUT"])
def update_lesson(lesson_id):
    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    data = request.get_json()
    if 'title' in data:
        lesson.title = data['title']
    if 'content' in data:
        lesson.content = data['content']
    
    db.commit()
    return jsonify({"id": lesson.id, "title": lesson.title, "content": lesson.content})

@app.route("/api/lessons/<int:lesson_id>", methods=["DELETE"])
def delete_lesson(lesson_id):
    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    # Delete sources and files
    sources = db.query(Source).filter(Source.lesson_id == lesson_id).all()
    for source in sources:
        delete_file(source.storage_path)
        db.delete(source)

    # Delete chroma collection
    if lesson.chroma_collection:
        try:
            chroma_manager = rag_pipeline.ChromaManager(collection_name=lesson.chroma_collection)
            chroma_manager.clear_collection()
        except Exception as e:
            app.logger.error(f"Could not delete chroma collection {lesson.chroma_collection}: {e}")

    db.delete(lesson)
    db.commit()
    return jsonify({"message": "Lesson deleted"}), 200

@app.route("/api/lessons/<int:lesson_id>/sources", methods=["GET"])
def get_sources(lesson_id):
    db = next(get_db())
    sources = db.query(Source).filter(Source.lesson_id == lesson_id).all()
    return jsonify([{
        "id": s.id,
        "filename": s.filename,
        "file_type": s.file_type,
        "status": s.status.value,
        "uploaded_at": s.uploaded_at
    } for s in sources])

@app.route("/api/lessons/<int:lesson_id>/sources", methods=["POST"])
def upload_source(lesson_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    storage_path, filename = save_file(file, lesson_id)
    file_type = filename.split('.')[-1]

    new_source = Source(
        lesson_id=lesson_id,
        filename=filename,
        file_type=file_type,
        storage_path=storage_path,
        status=SourceStatus.UPLOADED
    )
    db.add(new_source)
    db.commit()
    db.refresh(new_source)

    return jsonify({
        "id": new_source.id,
        "filename": new_source.filename,
        "file_type": new_source.file_type,
        "status": new_source.status.value
    }), 201

@app.route("/api/lessons/<int:lesson_id>/sources/<int:source_id>", methods=["DELETE"])
def delete_source(lesson_id, source_id):
    db = next(get_db())
    source = db.query(Source).filter(Source.id == source_id, Source.lesson_id == lesson_id).first()
    if not source:
        return jsonify({"error": "Source not found"}), 404

    delete_file(source.storage_path)
    db.delete(source)
    db.commit()

    return jsonify({"message": "Source deleted"}), 200

@app.route("/api/lessons/<int:lesson_id>/ingest", methods=["POST"])
def ingest_sources(lesson_id):
    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    sources = db.query(Source).filter(Source.lesson_id == lesson_id, Source.status == SourceStatus.UPLOADED).all()
    if not sources:
        return jsonify({"message": "No new sources to ingest"}), 200

    async def run_ingestion():
        for source in sources:
            try:
                await rag_pipeline.ingest_source(source.storage_path, lesson.chroma_collection, source_id=str(source.id))
                source.status = SourceStatus.INGESTED
                db.commit()
            except Exception as e:
                source.status = SourceStatus.FAILED
                db.commit()
                app.logger.error(f"Ingestion failed for source {source.id}: {e}")

    asyncio.run(run_ingestion())
    return jsonify({"message": "Ingestion process started"})

@app.route("/api/lessons/<int:lesson_id>/generate", methods=["POST"])
def generate_lesson_content(lesson_id):
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400

    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    prompt = data['prompt']
    if lesson.content:
        prompt = f"{lesson.content}\n\n---\n\nUser prompt: {prompt}"

    async def run_generation():
        return await rag_pipeline.ask_rag(prompt, lesson.chroma_collection)

    generated_content = asyncio.run(run_generation())

    if generated_content:
        lesson.content = generated_content
        db.commit()

    return jsonify({"content": generated_content})

# Endpoint for generating a full lesson from all context
@app.route("/api/lessons/<int:lesson_id>/generate-full", methods=["POST"])
def generate_full_lesson(lesson_id):
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400

    db = next(get_db())
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    prompt = data['prompt']

    async def run_generation():
        return await rag_pipeline.ask_rag_with_full_context(prompt, lesson.chroma_collection)

    generated_content = asyncio.run(run_generation())

    if generated_content and not generated_content.startswith("Error:"):
        lesson.content = generated_content
        db.commit()

    return jsonify({"content": generated_content})


@app.route('/')
def serve_index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
