import pytest
from httpx import Client

# The base URL of your running FastAPI application
BASE_URL = "http://127.0.0.1:8000"

# List of questions to test
test_questions = [
    "What is SageMaker?",
    "What are all AWS regions where SageMaker is available?",
    "How to check if an endpoint is KMS encrypted?",
    "What are SageMaker Geospatial capabilities?",
]


@pytest.mark.parametrize("question", test_questions)
def test_chat_endpoint_questions(question):
    """
    Tests the /chat endpoint with a list of questions.
    Asserts that the response is successful and has the correct data structure.
    """
    with Client() as client:
        # Define the request payload
        payload = {"question": question, "chat_history": []}

        # Send the POST request to the /chat endpoint
        response = client.post(f"{BASE_URL}/chat", json=payload, timeout=60)  # Increased timeout for model inference

        # 1. Assert that the request was successful
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"

        # 2. Assert that the response body is valid JSON
        data = response.json()

        # 3. Assert that the response has the expected keys and types
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert "citations" in data
        assert isinstance(data["citations"], list)

        print(f"\nQuestion: {question}\nAnswer: {data['answer'][:100]}...")  # Print a snippet for context


def test_update_embeddings_endpoint():
    """
    Tests the /update-embeddings endpoint.
    """
    with Client() as client:
        response = client.post(f"{BASE_URL}/update-embeddings", json={}, timeout=120)  # Longer timeout for indexing

        # Assert the request was successful and the message is correct
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
