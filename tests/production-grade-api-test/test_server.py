import os
import re
import requests
import logging
import uuid
import jwt
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")
server_url = os.getenv("SERVER_URL", "http://localhost:8080")

# Generate a unique test person ID
test_person_id = "automation-test-user-" + uuid.uuid4().hex


def generate_auth_header(person_id: str) -> dict:
    """Generate an authorization header for the test."""
    auth_dict = {"person_id": person_id}
    auth_token = jwt.encode(auth_dict, client_secret, algorithm="HS256")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }


def clean_response(response_content: str) -> str:
    """Clean padded response content."""
    buffer_padding_template = re.compile(r"(<<<) *(>>>)|(<<<) *| *(>>>)")
    return re.sub(buffer_padding_template, "", response_content)


def test_chat_process():
    """Test the /chat_process endpoint."""
    headers = generate_auth_header(test_person_id)
    data = {"UserInput": "Hello there!"}

    # Make a POST request to /chat_process
    response = requests.post(server_url + "/chat_process", headers=headers, json=data)

    assert response.status_code == 200
    assert len(response.content.decode("utf-8")) > 0


def test_save_feedback():
    """Test the /save_feedback endpoint."""
    headers = generate_auth_header(test_person_id)

    # Step 1: Send a message to /chat_process
    message_data = {"UserInput": "Hello there!"}
    message_response = requests.post(
        server_url + "/chat_process", headers=headers, json=message_data
    )
    assert message_response.status_code == 200

    # Clean the response content
    parsed_response = clean_response(message_response.content.decode("utf-8"))

    # Step 2: Send feedback to /save_feedback
    feedback_data = {
        "bot_message": parsed_response,
        "user_feedback": "test",
    }
    feedback_response = requests.post(
        server_url + "/save_feedback", headers=headers, json=feedback_data
    )

    # Assert success response
    assert feedback_response.status_code == 200
