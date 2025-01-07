import os
import re
import requests
import logging
import uuid
import json
import jwt
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")
server_url = os.getenv("SERVER_URL", "0.0.0.0:8080")

test_person_id = "automation-test-user-" + uuid.uuid4().hex


def test_chat_process():
    auth_dict = {"person_id": test_person_id}
    auth_header = jwt.encode(auth_dict, client_secret, algorithm="HS256")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(auth_header),
    }
    data = {
        "UserInput": "Hello there!",
    }
    response = requests.post(server_url + "/chat_process", headers=headers, json=data)
    typed_response = response.content.decode("utf-8")

    assert response.status_code == 200
    assert len(typed_response) > 0


def test_save_feedback():
    buffer_padding_template = re.compile(r"(<<<) *(>>>)|(<<<) *| *(>>>)")

    # Generate an authorization header
    auth_dict = {"person_id": test_person_id}
    auth_header = jwt.encode(auth_dict, client_secret, algorithm="HS256")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(auth_header),
    }

    message_data = {
        "UserInput": "Hello there!",
    }
    message_response = requests.post(
        server_url + "/chat_process", headers=headers, json=message_data
    )
    decoded_message_response = message_response.content.decode("utf-8")
    parsed_response = re.sub(buffer_padding_template, "", decoded_message_response)

    feedback_data = {
        "bot_message": parsed_response,
        "user_feedback": "test",
    }
    response = requests.post(
        server_url + "/save_feedback", headers=headers, json=feedback_data
    )

    assert response.status_code == 200
