
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/lessons'

def save_file(file, lesson_id):
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, str(lesson_id))):
        os.makedirs(os.path.join(UPLOAD_FOLDER, str(lesson_id)))

    filename = secure_filename(file.filename)
    storage_path = os.path.join(UPLOAD_FOLDER, str(lesson_id), filename)
    file.save(storage_path)
    return storage_path, filename

def delete_file(storage_path):
    if os.path.exists(storage_path):
        os.remove(storage_path)
