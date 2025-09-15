
import pytest
import requests
import os

API_URL = "http://localhost:8000/api"

@pytest.fixture(scope="module")
def test_lesson():
    # Create a lesson for testing
    response = requests.post(f"{API_URL}/lessons", json={"title": "Test Lesson"})
    assert response.status_code == 201
    lesson = response.json()
    yield lesson
    # Teardown: delete the lesson
    requests.delete(f"{API_URL}/lessons/{lesson['id']}")

def test_create_lesson():
    response = requests.post(f"{API_URL}/lessons", json={"title": "My Test Lesson"})
    assert response.status_code == 201
    data = response.json()
    assert data['title'] == "My Test Lesson"
    # cleanup
    requests.delete(f"{API_URL}/lessons/{data['id']}")

def test_get_lessons():
    response = requests.get(f"{API_URL}/lessons")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_and_ingest(test_lesson):
    lesson_id = test_lesson['id']
    
    # Create a dummy file to upload
    with open("test.txt", "w") as f:
        f.write("This is a test file for ingestion.")

    with open("test.txt", "rb") as f:
        response = requests.post(f"{API_URL}/lessons/{lesson_id}/sources", files={"file": f})
    
    os.remove("test.txt")
    assert response.status_code == 201

    # Ingest
    response = requests.post(f"{API_URL}/lessons/{lesson_id}/ingest")
    assert response.status_code == 200
    assert response.json()['message'] == "Ingestion process started"


@pytest.mark.skip(reason="This test makes a call to a live LLM API and requires API keys")
def test_generate(test_lesson):
    lesson_id = test_lesson['id']
    # This test assumes ingestion has been run and there are documents in the collection.
    # For a real test, you would mock the LLM call.
    response = requests.post(f"{API_URL}/lessons/{lesson_id}/generate", json={"prompt": "What is this about?"})
    assert response.status_code == 200
    assert "content" in response.json()

